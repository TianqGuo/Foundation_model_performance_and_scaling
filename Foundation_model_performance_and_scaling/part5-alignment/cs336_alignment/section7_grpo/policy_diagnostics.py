"""Policy agreement and controlled scoring probes; no vLLM scoring parity claim."""
from __future__ import annotations
import torch
import torch.nn.functional as F

from cs336_alignment.section4_sft.helpers import get_response_log_probs, tokenize_prompt_and_output


@torch.no_grad()
def policy_agreement(current, old, mask):
    if current.shape != old.shape or mask.shape != current.shape:
        raise ValueError("Matching log-probability and mask shapes required")
    mask = mask.bool()
    counts = mask.sum(1)
    if (counts == 0).any():
        raise ValueError("Every response requires tokens")
    delta = torch.where(mask, current.float() - old.float(), 0.)
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite policy agreement")
    tokens = delta[mask]
    sequences = delta.sum(1) / counts
    return {
        "responses": len(counts), "tokens": int(counts.sum()),
        "current_dtype": str(current.dtype), "old_dtype": str(old.dtype),
        "mean_abs_token_log_ratio": float(tokens.abs().mean()),
        "max_abs_token_log_ratio": float(tokens.abs().max()),
        "max_abs_sequence_log_ratio": float(sequences.abs().max()),
        "max_abs_sequence_ratio_minus_one": float(torch.expm1(sequences).abs().max()),
    }


def probe_scoring_parity(policy, batch, original_batches, frozen_old, pad_id,
                         include_fp32_model=False):
    """Compare the same responses under different forward modes and layouts.

    original_batches contains (tokenized batch, [(current_row, original_row)]).
    Input batches are on the policy device. Results are detached CPU tensors.
    The probe restores model mode, floating parameter dtype and RNG state.
    """
    parameter = next(policy.parameters())
    original_dtype, original_mode = parameter.dtype, policy.training
    if any(p.is_floating_point() and p.dtype != original_dtype for p in policy.parameters()):
        raise ValueError("Whole-model precision probe requires uniform parameter dtype")
    original_buffers = {name: buffer.detach().clone() for name, buffer in policy.named_buffers()}
    devices = [parameter.device.index] if parameter.device.type == 'cuda' else []
    mask = batch['response_mask'].cpu()

    def score(tokens, training=False, gradients=False):
        policy.train(training)
        with torch.set_grad_enabled(gradients):
            out = get_response_log_probs(policy, tokens['input_ids'], tokens['labels'],
                                         log_probs_dtype=torch.float32)['log_probs']
            return out.detach().cpu()

    def original_layout():
        width = batch['input_ids'].shape[1]
        result = torch.zeros_like(batch['labels'], dtype=torch.float32, device='cpu')
        for tokens, positions in original_batches:
            scored = score(tokens)
            for current_row, original_row in positions:
                length = min(width, scored.shape[1])
                result[current_row, :length] = scored[original_row, :length]
        return result

    def collect():
        reference = original_layout()
        current = score(batch)
        wider = {key: F.pad(value, (0, 8), value=pad_id if key != 'response_mask' else False)
                 for key, value in batch.items()}
        return {
            'original_layout_vs_frozen_old': policy_agreement(reference, frozen_old.cpu(), mask),
            'changed_batch_layout': policy_agreement(current, reference, mask),
            'repeat_same_forward': policy_agreement(score(batch), current, mask),
            'gradient_tracking_only': policy_agreement(score(batch, gradients=True), current, mask),
            'training_mode_only': policy_agreement(score(batch, training=True), current, mask),
            'training_mode_and_gradients': policy_agreement(score(batch, training=True, gradients=True), current, mask),
            'wider_padding_only': policy_agreement(score(wider)[:, :current.shape[1]], current, mask),
        }

    try:
        with torch.random.fork_rng(devices=devices):
            result = {'model_dtype': str(original_dtype), 'optimizer_updates': 0,
                      'input_shape': list(batch['input_ids'].shape),
                      'float32_matmul_precision': torch.get_float32_matmul_precision(),
                      'original_input_shapes': [list(tokens['input_ids'].shape) for tokens, _ in original_batches],
                      'model_precision': collect()}
            if include_fp32_model:
                policy.float()
                # Compare each FP32 forward against its own FP32 layout reference.
                fp32 = collect()
                fp32.pop('original_layout_vs_frozen_old')
                result['fp32_model'] = fp32
            return result
    finally:
        policy.to(dtype=original_dtype)
        for name, buffer in original_buffers.items():
            parent, _, leaf = name.rpartition(".")
            policy.get_submodule(parent)._buffers[leaf] = buffer
        policy.train(original_mode)


def select_probe_batch(policy, candidates):
    """Select the largest frozen-policy sequence-ratio discrepancy, before updates."""
    original_mode = policy.training
    device = next(policy.parameters()).device
    diagnostics = []
    try:
        policy.eval()
        with torch.no_grad():
            for candidate in candidates:
                tokens = candidate['tokens']
                scored = get_response_log_probs(
                    policy, tokens['input_ids'].to(device), tokens['labels'].to(device),
                    log_probs_dtype=torch.float32)['log_probs'].cpu()
                diagnostics.append(policy_agreement(scored, candidate['old'], tokens['response_mask']))
        selected = max(range(len(candidates)),
                       key=lambda i: diagnostics[i]['max_abs_sequence_ratio_minus_one'])
        return selected, diagnostics
    finally:
        policy.train(original_mode)


def investigate_rollout_scoring(policy, prompts, responses, tokenizer, generated_ids,
                                old_log_probs, permutation, micro_batch_size,
                                include_fp32_model=False):
    """Build aligned batches and run the worst-case scoring probe for one rollout."""
    device = next(policy.parameters()).device

    def tokenize(indices):
        return tokenize_prompt_and_output(
            [prompts[i] for i in indices], [responses[i] for i in indices], tokenizer,
            output_token_ids=None if generated_ids is None else [generated_ids[i] for i in indices])

    candidates = []
    for offset in range(0, len(permutation), micro_batch_size):
        indices = permutation[offset:offset + micro_batch_size]
        tokens = tokenize(indices)
        candidates.append({"indices": indices, "tokens": tokens,
                           "old": old_log_probs[indices, :tokens["input_ids"].shape[1]].cpu()})
    selected, diagnostics = select_probe_batch(policy, candidates)
    candidate = candidates[selected]
    indices = candidate["indices"]
    originals = []
    for start in sorted({(i // micro_batch_size) * micro_batch_size for i in indices}):
        end = min(start + micro_batch_size, len(prompts))
        tokens = {key: value.to(device) for key, value in tokenize(range(start, end)).items()}
        positions = [(row, i-start) for row, i in enumerate(indices) if start <= i < end]
        originals.append((tokens, positions))
    result = probe_scoring_parity(
        policy, {key: value.to(device) for key, value in candidate["tokens"].items()},
        originals, candidate["old"], tokenizer.pad_token_id, include_fp32_model)
    result.update(candidate_diagnostics=diagnostics, selected_candidate=selected,
                  example_indices=indices)
    return result
