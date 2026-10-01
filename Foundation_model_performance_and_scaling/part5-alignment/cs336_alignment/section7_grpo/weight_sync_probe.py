"""Cloud-only transfer/cache probe; temporarily alters one untied output row.

No checkpoint loading occurs here. The caller owns an already loaded policy.
Restore the row before ordinary rollouts. This is an execution check, never an
accuracy measurement or a training algorithm change.
"""
from pathlib import Path
import json


def verify_weight_transfer(policy, tokenizer, backend, output_dir: Path) -> None:
    import torch

    if policy.config.tie_word_embeddings:
        raise ValueError("Controlled cache probe requires untied output embeddings (Qwen/MATH)")
    prompt = "The sum of two and three is"
    inputs = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)],
                          device=next(policy.parameters()).device)
    was_training = policy.training
    policy.eval()
    params = {"temperature": 0., "max_tokens": 1, "n": 1, "seed": 12345, "ignore_eos": True}
    row_backup = None
    target = None
    try:
        with torch.no_grad():
            output = policy(inputs, output_hidden_states=True, use_cache=False)
            before = int(output.logits[0, -1].argmax())
            hidden = output.hidden_states[-1][0, -1].float()
            max_logit = float(output.logits[0, -1].float().abs().max())
            target = (before + 1) % output.logits.shape[-1]
            head = policy.get_output_embeddings().weight
            row_backup = head[target].clone()  # GPU row only; no CPU policy copy.
            del output
            backend.sync_policy_weights(policy)
            initial = backend.generate([prompt], params)[0].outputs[0].token_ids
            if initial != [before]:
                raise AssertionError(f"Initial trainer/server greedy mismatch: {before}, {initial}")
            # The same prompt is now cached. Make a different token dominate.
            head[target].copy_((hidden * (4 * (max_logit + 10) / hidden.square().sum())).to(head.dtype))
            changed_expected = int(policy(inputs, use_cache=False).logits[0, -1].argmax())
            if changed_expected != target:
                raise AssertionError("Controlled weight perturbation did not change the trainer prediction")
            backend.sync_policy_weights(policy)
            changed = backend.generate([prompt], params)[0].outputs[0].token_ids
            if changed != [target]:
                raise AssertionError(f"Updated weights/cache mismatch: expected {target}, got {changed}")
            head[target].copy_(row_backup)
            row_backup = None
            backend.sync_policy_weights(policy)
            restored = backend.generate([prompt], params)[0].outputs[0].token_ids
            if restored != [before]:
                raise AssertionError(f"Restored weights/cache mismatch: expected {before}, got {restored}")
            (output_dir / "weight_sync_probe.json").write_text(json.dumps({
                "passed": True, "initial_token": before, "changed_token": target,
                "restored_token": restored[0], "prompt": prompt,
                "checks": ["initial_parity", "changed_weights", "cache_invalidation", "restoration"],
            }, indent=2) + "\n")
    finally:
        if row_backup is not None:
            with torch.no_grad():
                policy.get_output_embeddings().weight[target].copy_(row_backup)
        policy.train(was_training)
