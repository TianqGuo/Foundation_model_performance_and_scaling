"""Controlled scoring comparisons on an already loaded policy; no optimizer updates."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cs336_alignment.section4_sft.helpers import get_response_log_probs
from cs336_alignment.section7_grpo.policy_diagnostics import policy_agreement


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
