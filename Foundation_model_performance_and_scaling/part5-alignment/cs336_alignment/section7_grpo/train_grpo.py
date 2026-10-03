"""GRPO training on MATH with verified rewards.

Run via part_5_7.sh or directly:
    uv run python cs336_alignment/section7_grpo/train_grpo.py [args]
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import signal
import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import torch
import torch.nn.functional as F

from cs336_alignment.section7_grpo.run_utils import (
    accumulation_weight, create_run_directory, environment_versions,
    evaluation_indices, file_fingerprint, resolve_rollout_device, should_evaluate, training_microbatches, validate_frozen_policy_agreement,
)

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_seed
    vllm_set_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm(policy: torch.nn.Module, llm: LLM) -> None:
    if hasattr(llm, "sync_policy_weights"):
        llm.sync_policy_weights(policy)
        return
    state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def get_ground_truth(ex: dict) -> str:
    if "solution" in ex:
        return str(ex["solution"])
    raw = str(ex.get("answer", ex.get("ground_truth", "")))
    return raw.split("####")[-1].strip() if "####" in raw else raw


# ---------------------------------------------------------------------------
# Old log-prob precomputation
# ---------------------------------------------------------------------------

def precompute_old_log_probs(
    policy: torch.nn.Module,
    rollout_prompts: list[str],
    rollout_responses: list[str],
    tokenizer,
    micro_batch_size: int,
    device: str,
    output_token_ids: list[list[int]] | None = None,
    log_probs_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute log-probs for all rollout examples under the current (frozen) policy.

    Returns a tensor of shape (rollout_batch_size, max_seq_len) padded with zeros
    at positions beyond each sequence's actual length.
    """
    from cs336_alignment.section4_sft.helpers import get_response_log_probs, tokenize_prompt_and_output

    chunks: list[torch.Tensor] = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(rollout_prompts), micro_batch_size):
            end = min(start + micro_batch_size, len(rollout_prompts))
            tok = tokenize_prompt_and_output(
                rollout_prompts[start:end], rollout_responses[start:end], tokenizer,
                output_token_ids=None if output_token_ids is None else output_token_ids[start:end],
            )
            lp = get_response_log_probs(
                policy, tok["input_ids"].to(device), tok["labels"].to(device),
                log_probs_dtype=log_probs_dtype,
            )["log_probs"]          # (mb, seq_len)
            chunks.append(lp.cpu())
    policy.train()

    # Pad each chunk to a common max length and concatenate
    max_sl = max(c.shape[1] for c in chunks)
    padded = [F.pad(c, (0, max_sl - c.shape[1])) for c in chunks]
    return torch.cat(padded, dim=0)   # (rollout_batch_size, max_sl)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    policy: torch.nn.Module,
    vllm_model: LLM,
    tokenizer,
    val_examples: list[dict],
    prompt_template: str,
    eval_sampling_params: SamplingParams,
    device: str,
    reward_fn=None,
) -> dict:
    from cs336_alignment.section4_sft.helpers import log_generations

    if reward_fn is None:
        from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
        reward_fn = r1_zero_reward_fn

    subset = val_examples  # Already selected once and recorded for this run.
    prompts = [prompt_template.format(question=ex.get("problem", ex.get("question", ""))) for ex in subset]
    ground_truths = [get_ground_truth(ex) for ex in subset]

    load_policy_into_vllm(policy, vllm_model)
    return log_generations(
        vllm_model=vllm_model,
        policy_model=policy,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        sampling_params=eval_sampling_params,
        device=device,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # TERM from the cloud runner's timeout unwinds the owned server/resources.
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        with ExitStack() as cleanup:
            _train(args, cleanup)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _train(args: argparse.Namespace, cleanup: ExitStack) -> None:
    from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams

    from cs336_alignment.section4_sft.helpers import get_response_log_probs, tokenize_prompt_and_output
    from cs336_alignment.section7_grpo.helpers import (
        grpo_microbatch_train_step,
        masked_mean,
    )

    from cs336_alignment.section7_grpo.estimators import resolve_estimator, score_rewards, build_advantages
    estimator = resolve_estimator(args)
    microbatch_layout = args.microbatch_layout or ("stable" if args.rollout_backend == "server" else "reshuffle")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for name in ("n_grpo_steps", "group_size", "rollout_batch_size", "train_batch_size",
                 "gradient_accumulation_steps", "epochs_per_rollout_batch", "max_response_tokens",
                 "eval_interval", "n_eval_examples"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.rollout_batch_size % args.group_size or args.train_batch_size % args.gradient_accumulation_steps:
        raise ValueError("Rollout/group and train/accumulation batch sizes must divide evenly")
    if args.train_batch_size < args.group_size:
        raise ValueError("train_batch_size must be at least group_size")

    # --- GPU setup ---
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available.")
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    train_device = args.train_device
    vllm_device, shares_device = resolve_rollout_device(n_gpus, train_device, args.vllm_device)
    if args.rollout_backend == "server":
        from importlib.metadata import version
        if version("vllm") != "0.19.1" or version("torch").split("+")[0] != "2.10.0":
            raise ValueError("Server backend requires the locked .venv-server environment")
        if shares_device or n_gpus < 2:
            raise ValueError("Server backend requires separate training/inference GPUs")
    if args.verify_weight_sync and args.rollout_backend != "server":
        raise ValueError("--verify_weight_sync requires --rollout_backend server")
    vllm_mem = 0.30 if shares_device else args.gpu_memory_utilization
    if shares_device:
        print(f"INFO: vLLM shares {train_device} with policy (vllm_mem=0.30).")

    output_path = create_run_directory(Path(args.output), args.run_name)
    print(f"Run outputs: {output_path}")
    project_root = Path(__file__).resolve().parents[2]
    prompt_file = "question_only.prompt" if args.prompt_type == "question_only" else "r1_zero.prompt"
    prompt_path = Path(__file__).parent.parent / "prompts" / prompt_file
    train_examples = load_jsonl(Path(args.data))
    if args.max_train_examples is not None:
        if args.max_train_examples <= 0:
            raise ValueError("max_train_examples must be positive")
        train_examples = train_examples[:args.max_train_examples]
    if not train_examples:
        raise ValueError("Training data is empty")
    eval_seed = args.eval_seed
    val_examples = []
    eval_record = {"protocol": "disabled" if args.skip_eval else "fixed_subset_v1",
                   "seed": eval_seed, "indices": []}
    if not args.skip_eval:
        full_validation = load_jsonl(Path(args.val_data))
        if not full_validation:
            raise ValueError("Validation data is empty; use --skip_eval to disable evaluation")
        indices = evaluation_indices(len(full_validation), args.n_eval_examples, eval_seed)
        val_examples = [full_validation[i] for i in indices]
        eval_record.update({"indices": indices, "dataset": file_fingerprint(Path(args.val_data))})
    (output_path / "evaluation_subset.json").write_text(json.dumps(eval_record, indent=2) + "\n")
    manifest = {
        "args": vars(args), "estimator": estimator.to_dict(), "environment": environment_versions(),
        "train_data": file_fingerprint(Path(args.data)), "train_examples": len(train_examples),
        "prompt": file_fingerprint(prompt_path), "evaluation": eval_record,
        "resolved": {"train_device": train_device, "vllm_device": vllm_device,
                     "gpu_memory_utilization": vllm_mem,
                     "microbatch_layout": microbatch_layout,
                     "micro_batch_size": args.train_batch_size // args.gradient_accumulation_steps},
        "cuda_runtime": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpus)],
        "sources": [file_fingerprint(p) for p in (
            Path(__file__), Path(__file__).with_name("helpers.py"),
            Path(__file__).with_name("run_utils.py"), Path(__file__).with_name("estimators.py"),
            Path(__file__).with_name("policy_diagnostics.py"),
            project_root / "pyproject.toml",
            project_root / "cs336_alignment/section4_sft/helpers.py",
            project_root / "cs336_alignment/drgrpo_grader.py",
            project_root / "uv.lock", Path(__file__).with_name("rollout_backend.py"),
            Path(__file__).with_name("weight_sync_probe.py"),
            Path(__file__).parent / "server_environment/pyproject.toml",
            Path(__file__).parent / "server_environment/uv.lock") if p.exists()],
    }
    (output_path / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- wandb ---
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        cleanup.callback(wandb.finish)
        wandb.define_metric("grpo_step")
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("grpo/*", step_metric="grpo_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # --- Model ---
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    attention = args.attn_implementation or ("sdpa" if args.rollout_backend == "server" else "flash_attention_2")
    manifest["resolved"]["attn_implementation"] = attention
    manifest["resolved"]["rollout_backend"] = args.rollout_backend
    manifest["resolved"]["training_tokens"] = "generated_ids" if args.rollout_backend == "server" else "retokenized_text"
    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=attention
    ).to(train_device)
    manifest["model"] = {"source": args.model, "config": policy.config.to_dict(),
                         "tokenizer_source": tokenizer.name_or_path}
    (output_path / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    policy.train()

    # --- vLLM ---
    print(f"Initializing vLLM on {vllm_device} ...")
    if args.rollout_backend == "server":
        from cs336_alignment.section7_grpo.rollout_backend import VLLMServerBackend
        vllm_model = cleanup.enter_context(VLLMServerBackend(
            args.model, vllm_device, args.seed, output_path, vllm_mem,
            port=args.server_port, startup_timeout=args.server_startup_timeout,
            request_timeout=args.server_request_timeout, max_model_len=args.server_max_model_len,
            enforce_eager=args.server_enforce_eager,
        ))
        vllm_model.init_weight_sync(train_device)
        if args.verify_weight_sync:
            from cs336_alignment.section7_grpo.weight_sync_probe import verify_weight_transfer
            verify_weight_transfer(policy, tokenizer, vllm_model, output_path)
    else:
        vllm_model = init_vllm(args.model, vllm_device, args.seed, vllm_mem)
    print("vLLM ready")

    # --- Prompt template and reward function ---
    prompt_template = prompt_path.read_text()
    reward_fn = question_only_reward_fn if args.prompt_type == "question_only" else r1_zero_reward_fn
    if args.prompt_type == "question_only":
        print("Using question_only prompt and reward function.")

    # --- Data ---
    print(f"Train examples: {len(train_examples)}")
    print(f"Fixed evaluation examples: {len(val_examples)}")

    # --- Sampling params ---
    stop_tokens = ["</answer>"] if args.prompt_type == "r1_zero" else []
    rollout_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_response_tokens,
        min_tokens=4,
        n=args.group_size,
        stop=stop_tokens,
        include_stop_str_in_output=True,
    )
    eval_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_response_tokens,
        stop=stop_tokens,
        include_stop_str_in_output=True,
    )

    if args.rollout_backend == "server":
        eval_params.seed = args.eval_seed

    # --- Derived hyperparameters ---
    n_prompts = args.rollout_batch_size // args.group_size
    micro_bs = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0
    assert args.train_batch_size % args.gradient_accumulation_steps == 0
    assert args.train_batch_size >= args.group_size
    print(
        f"n_prompts/step={n_prompts}, group_size={args.group_size}, "
        f"rollout_bs={args.rollout_batch_size}, micro_bs={micro_bs}, "
        f"grad_accum={args.gradient_accumulation_steps}"
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )
    optimizer.zero_grad()

    # --- Metrics file ---
    metrics_path = output_path / f"eval_metrics_{args.run_name}.jsonl"
    metrics_file = cleanup.enter_context(open(metrics_path, "x"))
    system_file = cleanup.enter_context((output_path / "system_metrics.jsonl").open("x"))
    rollout_file = cleanup.enter_context((output_path / "rollouts.jsonl").open("x")) if args.rollout_backend == "server" else None

    train_step = 0
    eval_step = 0
    needs_old_lp = estimator.importance_reweighting != "none"

    # -----------------------------------------------------------------------
    # GRPO loop
    # -----------------------------------------------------------------------
    for grpo_step in range(1, args.n_grpo_steps + 1):
        print(f"\n=== GRPO step {grpo_step}/{args.n_grpo_steps} ===")

        # --- Sync policy weights into vLLM ---
        policy.eval()
        torch.cuda.synchronize(train_device)
        sync_start = time.perf_counter()
        load_policy_into_vllm(policy, vllm_model)
        sync_seconds = time.perf_counter() - sync_start

        # --- Sample questions and generate rollouts ---
        batch_qs = random.sample(train_examples, min(n_prompts, len(train_examples)))
        prompts = [prompt_template.format(question=ex.get("problem", ex.get("question", ""))) for ex in batch_qs]
        ground_truths = [get_ground_truth(ex) for ex in batch_qs]

        if args.rollout_backend == "server":
            rollout_params.seed = args.seed + grpo_step
        rollout_start = time.perf_counter()
        vllm_outputs = vllm_model.generate(prompts, rollout_params)
        rollout_seconds = time.perf_counter() - rollout_start

        # Flatten: [q1_r1, q1_r2, ..., q1_rG, q2_r1, ...]
        rollout_prompts: list[str] = []
        rollout_responses: list[str] = []
        rollout_gts: list[str] = []
        generated_ids: list[list[int]] | None = [] if args.rollout_backend == "server" else None
        for prompt, gt, out in zip(prompts, ground_truths, vllm_outputs):
            if len(out.outputs) != args.group_size:
                raise ValueError("Rollout response group size mismatch")
            for completion in out.outputs:
                rollout_prompts.append(prompt)
                rollout_responses.append(completion.text)
                rollout_gts.append(gt)
                if generated_ids is not None:
                    if out.prompt_token_ids != tokenizer.encode(prompt, add_special_tokens=True):
                        raise ValueError("Server/trainer prompt token alignment mismatch")
                    generated_ids.append(completion.token_ids)
                    rollout_file.write(json.dumps({"grpo_step": grpo_step,
                        "weight_version": vllm_model.weight_version, "index": len(rollout_responses)-1,
                        "text": completion.text, "token_ids": completion.token_ids,
                        "prompt_token_ids": out.prompt_token_ids,
                        "finish_reason": completion.finish_reason}) + "\n")
        if len(vllm_outputs) != len(prompts):
            raise ValueError("Rollout prompt count mismatch")
        if rollout_file is not None:
            rollout_file.flush()
        rollout_bs = len(rollout_responses)

        # --- Compute rewards and advantages ---
        raw_rewards, reward_meta = score_rewards(reward_fn, rollout_responses, rollout_gts)
        advantages = build_advantages(raw_rewards, args.group_size, estimator.baseline,
            estimator.advantage_normalizer, args.advantage_eps)

        print(
            f"  rewards: mean={reward_meta['mean_reward']:.3f} "
            f"correct={reward_meta['fraction_correct']:.3f} "
            f"format={reward_meta['mean_format_reward']:.3f}"
        )

        if not args.no_wandb:
            wandb.log({
                "grpo/mean_reward": reward_meta["mean_reward"],
                "grpo/fraction_correct": reward_meta["fraction_correct"],
                "grpo/mean_format_reward": reward_meta["mean_format_reward"],
                "grpo/mean_answer_reward": reward_meta["mean_answer_reward"],
                "grpo/rollout_batch_size": rollout_bs,
                "grpo_step": grpo_step,
            })

        # --- Precompute old log-probs (for off-policy or grpo_clip) ---
        all_old_log_probs: torch.Tensor | None = None
        if needs_old_lp:
            all_old_log_probs = precompute_old_log_probs(
                policy, rollout_prompts, rollout_responses, tokenizer, micro_bs, train_device, generated_ids,
                log_probs_dtype=torch.float32 if args.log_prob_precision == "fp32" else None,
            )

        policy.train()

        # Accumulators across all epochs in this GRPO step (for JSONL logging)
        agreement = []
        initial_train_step = train_step
        step_grad_norms: list[float] = []
        optimizer_batch_sizes: list[int] = []
        step_clip_fracs: list[float] = []
        sequence_clips: list[float] = []
        active_clips: list[float] = []
        clipped_token_count = 0.
        response_token_count = 0
        entropy_sum = 0.
        micro_losses: list[float] = []

        # --- Training epochs on this rollout batch ---
        for epoch in range(args.epochs_per_rollout_batch):
            epoch_batches = training_microbatches(rollout_bs, micro_bs,
                args.gradient_accumulation_steps, microbatch_layout)
            perm = [i for _, indices, _ in epoch_batches for i in indices]

            optimizer.zero_grad()
            microbatch_count = 0
            epoch_loss = 0.0
            epoch_clip_frac = 0.0
            epoch_entropy = 0.0
            n_mb = 0

            for mb_start, mb_idx, update_size in epoch_batches:

                mb_p = [rollout_prompts[i] for i in mb_idx]
                mb_r = [rollout_responses[i] for i in mb_idx]
                mb_adv = advantages[mb_idx].unsqueeze(1).to(train_device)
                mb_raw = raw_rewards[mb_idx].unsqueeze(1).to(train_device)

                tok = tokenize_prompt_and_output(mb_p, mb_r, tokenizer,
                    output_token_ids=None if generated_ids is None else [generated_ids[i] for i in mb_idx])
                input_ids = tok["input_ids"].to(train_device)
                labels = tok["labels"].to(train_device)
                response_mask = tok["response_mask"].to(train_device)

                if args.scoring_probe_only and epoch == 0 and mb_start == 0:
                    if all_old_log_probs is None:
                        raise ValueError("Scoring probe requires an importance-weighted estimator")
                    from cs336_alignment.section7_grpo.policy_diagnostics import investigate_rollout_scoring
                    probe = investigate_rollout_scoring(
                        policy, rollout_prompts, rollout_responses, tokenizer, generated_ids,
                        all_old_log_probs, perm, micro_bs, args.scoring_probe_fp32_model)
                    probe["grpo_step"] = grpo_step
                    (output_path / "scoring_parity_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
                    print("Scoring parity probe saved; no optimizer updates or checkpoint saves.")
                    return

                try:
                    lp_out = get_response_log_probs(
                        policy, input_ids, labels, return_token_entropy=True,
                        log_probs_dtype=torch.float32 if args.log_prob_precision == "fp32" else None,
                    )
                    policy_lp = lp_out["log_probs"]       # (mb, seq_len)
                    token_ent = lp_out.get("token_entropy")

                    mb_old_lp: torch.Tensor | None = None
                    if all_old_log_probs is not None:
                        curr_sl = policy_lp.shape[1]
                        mb_old_lp = all_old_log_probs[mb_idx, :curr_sl].to(train_device)
                        if epoch == 0 and train_step == initial_train_step:
                            from cs336_alignment.section7_grpo.policy_diagnostics import policy_agreement
                            diagnostic = policy_agreement(policy_lp, mb_old_lp, response_mask)
                            if microbatch_layout == "stable":
                                validate_frozen_policy_agreement([diagnostic])
                            agreement.append(diagnostic)

                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs=policy_lp,
                        response_mask=response_mask,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        loss_type=args.loss_type,
                        raw_rewards=mb_raw if args.loss_type == "no_baseline" else None,
                        advantages=mb_adv,
                        estimator_config=estimator,
                        old_log_probs=mb_old_lp,
                        cliprange=args.cliprange,
                        cliprange_low=args.cliprange_low, cliprange_high=args.cliprange_high,
                        length_norm=args.length_norm,
                        max_response_tokens=args.max_response_tokens,
                        loss_scale=accumulation_weight(len(mb_idx), update_size),
                    )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                        # Dropping a microbatch also drops accumulated gradients
                        # and changes the effective batch/objective silently.
                        raise RuntimeError(
                            f"GPU failure at grpo_step={grpo_step}, epoch={epoch}, "
                            f"mb_start={mb_start}; run aborted without skipping data. "
                            "Reduce cloud microbatch size or use more GPU memory."
                        ) from e
                    raise

                micro_losses.append(loss.item() / accumulation_weight(len(mb_idx), update_size))
                response_token_count += int(response_mask.sum())
                if "is_clipped" in meta:
                    clipped_token_count += float((meta["is_clipped"] * response_mask).sum())
                if token_ent is not None:
                    entropy_sum += float(torch.where(response_mask, token_ent.detach(), 0.).sum())
                if "sequence_is_clipped" in meta:
                    sequence_clips.extend(meta["sequence_is_clipped"].detach().cpu().tolist())
                if "clip_active" in meta:
                    active_clips.extend(masked_mean(meta["clip_active"], response_mask, dim=1).detach().cpu().tolist())
                epoch_loss += loss.item() / accumulation_weight(len(mb_idx), update_size)
                if "is_clipped" in meta and response_mask.any():
                    clip_frac = masked_mean(meta["is_clipped"], response_mask.float()).item()
                    epoch_clip_frac += clip_frac
                if token_ent is not None and response_mask.any():
                    epoch_entropy += masked_mean(token_ent, response_mask.float()).item()
                n_mb += 1
                microbatch_count += 1

                if microbatch_count % args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_step += 1
                    microbatch_count = 0
                    step_grad_norms.append(grad_norm)
                    optimizer_batch_sizes.append(update_size)

                    avg_loss = epoch_loss / max(n_mb, 1)
                    avg_clip = epoch_clip_frac / max(n_mb, 1)
                    avg_ent = epoch_entropy / max(n_mb, 1)
                    step_clip_fracs.append(avg_clip)
                    print(
                        f"  train_step={train_step} loss={avg_loss:.4f} "
                        f"grad_norm={grad_norm:.3f} entropy={avg_ent:.3f}"
                        + (f" clip_frac={avg_clip:.3f}" if estimator.importance_reweighting in ("grpo", "gspo", "noclip") else "")
                    )

                    if not args.no_wandb:
                        log_dict: dict = {
                            "train/loss": avg_loss,
                            "train/grad_norm": grad_norm,
                            "train/token_entropy": avg_ent,
                            "train/mean_reward": reward_meta["mean_reward"],
                            "train/fraction_correct": reward_meta["fraction_correct"],
                            "train_step": train_step,
                        }
                        if estimator.importance_reweighting in ("grpo", "gspo", "noclip"):
                            log_dict["train/clip_fraction"] = avg_clip
                        wandb.log(log_dict)

            # Flush any remaining accumulated gradients
            if microbatch_count > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
                optimizer.step()
                optimizer.zero_grad()
                train_step += 1
                step_grad_norms.append(grad_norm)
                optimizer_batch_sizes.append(update_size)

        system_file.write(json.dumps({"grpo_step": grpo_step,
            "backend": args.rollout_backend, "sync_seconds": sync_seconds,
            "rollout_seconds": rollout_seconds, "responses": rollout_bs,
            "generated_tokens": sum(map(len, generated_ids)) if generated_ids is not None else None,
            "truncated": sum(c.finish_reason == "length" for o in vllm_outputs for c in o.outputs),
            "weight_version": getattr(vllm_model, "weight_version", None),
            "grad_norms": step_grad_norms, "optimizer_batch_sizes": optimizer_batch_sizes, "reward": reward_meta,
            "policy_agreement_before_first_update": agreement,
            "mean_microbatch_loss": sum(micro_losses)/len(micro_losses),
            "token_entropy": entropy_sum/max(response_token_count,1),
            "response_token_weighted_ratio_outside_clip_fraction": clipped_token_count/max(response_token_count,1),
            "mean_response_tokens": sum(map(len, generated_ids))/rollout_bs if generated_ids is not None else None,
            "mean_sequence_clip_fraction": sum(sequence_clips)/len(sequence_clips) if sequence_clips else None,
            "mean_active_clip_fraction": sum(active_clips)/len(active_clips) if active_clips else None}) + "\n")
        system_file.flush()

        # --- Periodic evaluation ---
        if should_evaluate(args.skip_eval, len(val_examples), grpo_step, args.eval_interval):
            print(f"  Running eval on {args.n_eval_examples} val examples ...")
            policy.eval()
            with torch.no_grad():
                metrics = run_eval(
                    policy, vllm_model, tokenizer,
                    val_examples, prompt_template, eval_params,
                    train_device,
                    reward_fn=reward_fn,
                )
            policy.train()
            eval_step += 1
            print(
                f"  [eval] step={eval_step} "
                f"accuracy={metrics['accuracy']:.3f} "
                f"format={metrics['format_rate']:.3f} "
                f"reward={metrics['avg_reward']:.3f} "
                f"entropy={metrics['avg_token_entropy']:.3f}"
            )
            metrics_file.write(json.dumps({
                "grpo_step": grpo_step,
                "train_step": train_step,
                "eval_step": eval_step,
                "timestamp": time.time(),
                "accuracy": metrics["accuracy"],
                "format_rate": metrics["format_rate"],
                "avg_reward": metrics["avg_reward"],
                "avg_token_entropy": metrics["avg_token_entropy"],
                "avg_response_length": metrics["avg_response_length"],
                "avg_grad_norm": float(sum(step_grad_norms) / len(step_grad_norms)) if step_grad_norms else 0.0,
                "avg_clip_frac": float(sum(step_clip_fracs) / len(step_clip_fracs)) if step_clip_fracs else 0.0,
            }) + "\n")
            metrics_file.flush()
            if not args.no_wandb:
                wandb.log({
                    "eval/accuracy": metrics["accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval/avg_reward": metrics["avg_reward"],
                    "eval/avg_token_entropy": metrics["avg_token_entropy"],
                    "eval/avg_response_length": metrics["avg_response_length"],
                    "eval_step": eval_step,
                })

    # -----------------------------------------------------------------------
    # Final eval and save
    # -----------------------------------------------------------------------
    if val_examples and not args.skip_eval:
        print("\nRunning final evaluation ...")
        policy.eval()
        with torch.no_grad():
            final = run_eval(
                policy, vllm_model, tokenizer,
                val_examples, prompt_template, eval_params,
                train_device,
                reward_fn=reward_fn,
            )
        print(f"Final accuracy: {final['accuracy']:.4f}")
        (output_path / "final_eval.json").write_text(
            json.dumps({k: v for k, v in final.items() if k != "examples"}, indent=2)
        )

    import os
    user = os.environ.get("USER", "user")
    save_name = output_path.name
    cluster_dir = Path(f"/data/{user}/{save_name}")
    local_dir = Path(__file__).parent.parent.parent / "assets" / save_name
    save_dir = cluster_dir if cluster_dir.parent.exists() else local_dir
    save_dir.mkdir(parents=True, exist_ok=False)
    manifest["checkpoint"] = str(save_dir.resolve())
    (output_path / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    metrics_file.close()
    print(f"Eval metrics saved to {metrics_path}")
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training on MATH with verified rewards")
    # Paths
    parser.add_argument("--model", default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--data", default="/data/a5-alignment/MATH/train.jsonl")
    parser.add_argument("--val_data", default="/data/a5-alignment/MATH/validation.jsonl")
    parser.add_argument("--output", default="results/section7")
    parser.add_argument("--max_train_examples", type=int, default=None)
    # GRPO hyperparameters
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to reduce activation memory (~single-GPU runs)")
    parser.add_argument("--use_std_normalization", action="store_true", default=True)
    parser.add_argument("--no_std_normalization", dest="use_std_normalization", action="store_false")
    parser.add_argument(
        "--loss_type",
        default="reinforce_with_baseline",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    )
    parser.add_argument("--microbatch_layout", choices=["stable", "reshuffle"], default=None,
                        help="Default: stable original microbatches on server; legacy example reshuffle otherwise")
    parser.add_argument("--scoring_probe_only", action="store_true",
                        help="Diagnose first rollout scoring and exit before optimizer updates")
    parser.add_argument("--scoring_probe_fp32_model", action="store_true",
                        help="Also test FP32 model forwards in the scoring-only probe")
    parser.add_argument("--log_prob_precision", choices=["fp32", "model"], default="fp32",
                        help="Normalize old/current training log probabilities in FP32 (model reproduces legacy precision)")
    parser.add_argument("--cliprange_low", type=float, default=None)
    parser.add_argument("--cliprange_high", type=float, default=None)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument(
        "--length_norm",
        default="masked_mean",
        choices=["masked_mean", "masked_normalize"],
        help="Per-example loss aggregation: masked_mean or masked_normalize by max_response_tokens",
    )
    parser.add_argument(
        "--prompt_type",
        default="r1_zero",
        choices=["r1_zero", "question_only"],
        help="Prompt template and reward function to use",
    )
    # Generation
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_response_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--rollout_backend", choices=["legacy", "server"], default="legacy")
    parser.add_argument("--attn_implementation", choices=["sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--server_port", type=int, default=0)
    parser.add_argument("--server_startup_timeout", type=float, default=600)
    parser.add_argument("--server_request_timeout", type=float, default=300)
    parser.add_argument("--server_max_model_len", type=int, default=4096)
    parser.add_argument("--server_enforce_eager", action="store_true")
    parser.add_argument("--verify_weight_sync", action="store_true",
                        help="Cloud-only controlled weight perturbation and prefix-cache probe")
    parser.add_argument("--baseline", choices=["mean", "none"], default=None)
    parser.add_argument("--advantage_normalizer", choices=["std", "none", "mean"], default=None)
    parser.add_argument("--importance_reweighting", choices=["none", "noclip", "grpo", "gspo"], default=None)
    parser.add_argument("--loss_normalization", choices=["sequence", "constant"], default=None)
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--n_eval_examples", type=int, default=1024)
    parser.add_argument("--eval_seed", type=int, default=12345,
                        help="Independent seed for the saved periodic/final evaluation subset")
    parser.add_argument("--skip_eval", action="store_true")
    # Devices
    parser.add_argument("--train_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    # Logging
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="cs336-alignment-grpo")
    parser.add_argument("--run_name", default="grpo_reinforce")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
