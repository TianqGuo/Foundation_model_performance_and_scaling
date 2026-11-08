"""Native Ray Train/FSDP2 SFT worker; all heavy imports/execution stay on cloud."""
from importlib import import_module
import json
import math
import os
from pathlib import Path
import random
import socket

settings = import_module("infrastructure.7e_sft.config")
workload = import_module("workloads.sft_workload")


def tensor_batch(rows, pad_id, device):
    """Pad validated examples and create CUDA long tensors; return native model inputs."""
    import torch
    return {key: torch.tensor(value, dtype=torch.long, device=device)
            for key, value in workload.collate_lists(rows, pad_token_id=pad_id).items()}


def evaluate(model, rows, rank, world, pad_id, device):
    """Return global held-out response-token mean loss using equal collective counts.

    Every rank performs ceil(examples/world) forwards. A trailing rank executes a
    zero-weight duplicate rather than skipping a collective or biasing the metric.
    Uses eval/no_grad, restores train mode and does not generate or score answers.
    """
    import torch
    import torch.distributed as dist
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    model.eval()
    with torch.no_grad():
        for index, include_metric in workload.evaluation_slots(len(rows), rank, world):
            batch = tensor_batch([rows[index]], pad_id, device)
            loss = model(**batch, use_cache=False).loss
            count = int((batch["labels"][:, 1:] != -100).sum().item()) if include_metric else 0
            if not torch.isfinite(loss):
                raise ValueError("Held-out loss is nonfinite")
            totals[0] += loss.detach().double()*count
            totals[1] += count
    dist.all_reduce(totals)
    model.train()
    if totals[1] <= 0:
        raise ValueError("Held-out set has no response targets")
    return float((totals[0]/totals[1]).item())


def sharding_evidence(model, optimizer):
    """Inspect real post-update DTensor parameter/gradient/Adam local/global shapes.

    Returns per-parameter diagnostic rows and counts, without gathering tensors.
    Requires actual sharded parameters/gradients and two sharded moment slots for
    every trainable parameter; flags alone never count as successful sharding.
    """
    from torch.distributed.tensor import DTensor, Shard
    rows = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        values = dict(parameter=parameter, gradient=parameter.grad)
        values.update({key: optimizer.state[parameter].get(key) for key in ("exp_avg", "exp_avg_sq")})
        evidence = {}
        for key, value in values.items():
            if not isinstance(value, DTensor) or not any(isinstance(p, Shard) for p in value.placements):
                raise ValueError(f"Expected a real sharded DTensor for {name}/{key}")
            local = value.to_local()
            if local.numel() >= value.numel():
                raise ValueError(f"Expected a smaller local shard for {name}/{key}")
            evidence[key] = dict(global_shape=list(value.shape), local_shape=list(local.shape),
                                 placements=[str(p) for p in value.placements])
        rows.append(dict(name=name, **evidence))
    if not rows:
        raise ValueError("No trainable sharded parameters found")
    return dict(parameter_tensors=len(rows), gradient_tensors=len(rows),
                adam_moment_tensors=2*len(rows), tensors=rows)


def train_loop_per_worker(options):
    """Execute bounded native SFT updates inside a two-GPU Ray Train worker group.

    Inputs: validated config, hash-verified data and complete local model snapshot,
    plus fresh shared evidence directory. Ray owns process groups/devices; native
    FSDP2 owns sharding. Writes per-rank token/placement/sharding/metric evidence,
    reports global losses on every rank and raises on divergence or no update.
    This increment has no checkpoint save/restore; launcher explicitly rejects
    resume input. Cloud execution and native imports remain unverified.
    """
    launcher = import_module("infrastructure.7e_sft.run_7e")
    if launcher.require_cloud_pod() != options["image"]:
        raise ValueError("Worker image identity differs from launcher")
    import numpy as np
    import ray
    from ray import train
    from ray.train.torch import get_device
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = settings.SFTConfig(**options["config"]).validate(cloud=True)
    context = train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    if world != config.world_size or dist.get_world_size() != world or not torch.cuda.is_available():
        raise ValueError("Require two real GPU ranks with Ray-managed process groups")
    device = get_device()
    torch.cuda.set_device(device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    data, _, bundle_hash = launcher.read_bundle(options["bundle"])
    if bundle_hash != options["bundle_sha256"]:
        raise ValueError("Worker bundle identity differs from launcher")
    launcher.verify_snapshot(options["model_path"], config)
    output = Path(options["output"])/f"rank_{rank}"
    output.mkdir(exist_ok=False)
    location = dict(rank=rank, world_size=world, host=socket.gethostname(),
        ray_node_id=str(ray.get_runtime_context().get_node_id()),
        ray_gpu_ids=ray.get_runtime_context().get_accelerator_ids().get("GPU", []),
        device=str(device), gpu_name=torch.cuda.get_device_name(device), pid=os.getpid(), pod_name=os.environ.get("POD_NAME"),
        pod_uid=os.environ["POD_UID"], node_name=os.environ["NODE_NAME"])
    locations = [None]*world
    dist.all_gather_object(locations, location)
    gpu_ids = [item["ray_gpu_ids"] for item in locations]
    if len({item["ray_node_id"] for item in locations}) != 1 or any(len(ids) != 1 for ids in gpu_ids) or len({str(ids[0]) for ids in gpu_ids}) != world:
        raise ValueError("Initial smoke requires two distinct whole GPUs on one Ray node")
    (output/"placement.json").write_text(json.dumps(location, indent=2)+"\n")
    tokenizer = AutoTokenizer.from_pretrained(options["model_path"], local_files_only=True, trust_remote_code=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenized, token_receipt = {}, {}
    for split, rows in data.items():
        tokenized[split], token_receipt[split] = workload.tokenize_pairs(rows, tokenizer,
            max_length=config.max_sequence_length)
    token_receipt["tokenizer"] = dict(class_name=type(tokenizer).__name__,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=pad_id,
        bos_token_id=tokenizer.bos_token_id, special_tokens_added=False)
    receipts = [None]*world
    dist.all_gather_object(receipts, token_receipt)
    if any(item != token_receipt for item in receipts):
        raise ValueError("Tokenization differs between ranks")
    required = config.total_steps*config.global_batch_size()
    if len(tokenized["train"]) < required or not tokenized["evaluation"]:
        raise ValueError("Not enough usable tokenized pairs after overlength exclusions")
    (output/"tokenization.json").write_text(json.dumps(token_receipt, indent=2)+"\n")
    model = AutoModelForCausalLM.from_pretrained(options["model_path"], dtype=torch.float32,
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not len(layers):
        raise ValueError("Model needs a reviewed transformer-block adapter before switching architecture")
    model.to(device)
    mesh = init_device_mesh("cuda", (world,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, foreach=False)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=config.total_steps)
    held_out = evaluate(model, tokenized["evaluation"], rank, world, pad_id, device)
    train.report(dict(step=0, held_out_loss=held_out, recovery_verified=False))
    records = []
    for step in range(config.total_steps):
        indices = workload.update_indices(step, rank, world_size=world,
            micro_batch_size=config.micro_batch_size, accumulation_steps=config.gradient_accumulation_steps)
        microbatches = [[tokenized["train"][index] for index in batch] for batch in indices]
        local_tokens = sum(sum(value != -100 for value in row["labels"][1:]) for batch in microbatches for row in batch)
        denominator = torch.tensor(local_tokens, dtype=torch.long, device=device)
        dist.all_reduce(denominator)
        global_tokens = int(denominator.item())
        optimizer.zero_grad(set_to_none=True)
        numerator = torch.zeros((), dtype=torch.float64, device=device)
        for rows in microbatches:
            batch = tensor_batch(rows, pad_id, device)
            count = int((batch["labels"][:, 1:] != -100).sum().item())
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Training loss is nonfinite")
            numerator += loss.detach().double()*count
            (loss*workload.loss_scale(count, global_tokens, world)).backward()
        squared_norm = torch.zeros((), dtype=torch.float64, device=device)
        probe = None
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.to_local().detach()
            squared_norm += grad.double().square().sum()
            if probe is None and torch.count_nonzero(grad).item():
                index = int(grad.abs().reshape(-1).argmax().item())
                local = parameter.detach().to_local().reshape(-1)
                probe = (parameter, index, float(local[index].item()))
        dist.all_reduce(squared_norm)
        if not torch.isfinite(squared_norm) or squared_norm <= 0 or probe is None:
            raise ValueError("Expected nonzero finite sharded gradients on both ranks")
        optimizer.step()
        scheduler.step()
        parameter, index, before = probe
        delta = float(parameter.detach().to_local().reshape(-1)[index].item())-before
        if not math.isfinite(delta) or delta == 0:
            raise ValueError("Optimizer did not change the probed parameter")
        if step == 0:
            (output/"sharding.json").write_text(json.dumps(sharding_evidence(model, optimizer), indent=2)+"\n")
        dist.all_reduce(numerator)
        record = dict(step=step+1, train_loss=float((numerator/global_tokens).item()),
            global_response_tokens=global_tokens, gradient_norm=float(squared_norm.sqrt().item()),
            probed_update=delta, local_source_indices=[row["source_index"] for batch in microbatches for row in batch])
        if step+1 == config.total_steps:
            record["held_out_loss"] = evaluate(model, tokenized["evaluation"], rank, world, pad_id, device)
        records.append(record)
        (output/"metrics.json").write_text(json.dumps(records, indent=2)+"\n")
        train.report({key: value for key, value in record.items() if key != "local_source_indices"})
    dist.barrier()
