"""Cloud-only hooks: delegate the objective to verl; retain rank/state evidence.

Loaded through model.external_lib. No optimizer, synchronization or training loop
is implemented here. A frozen-policy discrepancy stops the first update.
"""

import json
import os
from pathlib import Path
import socket
import time

import torch
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import register_optimizer_step_post_hook
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, register_policy_loss

_updates = 0


def emit(event, **fields):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    root = Path(os.environ["PART7_RUN_DIR"]) / "diagnostics"
    root.mkdir(exist_ok=True)
    record = dict(event=event, time=time.time(), rank=rank, host=socket.gethostname(),
                  pid=os.getpid(), world_size=torch.distributed.get_world_size(),
                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), **fields)
    with (root / f"rank_{rank}_pid_{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def tensor_description(value):
    local = value.to_local() if isinstance(value, DTensor) else value
    return dict(global_shape=list(value.shape), local_shape=list(local.shape),
                dtype=str(value.dtype), device=str(local.device),
                placements=[str(p) for p in value.placements] if isinstance(value, DTensor) else [],
                mesh=value.device_mesh.mesh.tolist() if isinstance(value, DTensor) else None)


@register_policy_loss("part7_grpo")
def diagnostic_loss(old_log_prob, log_prob, advantages, response_mask,
                    loss_agg_mode="token-mean", config=None, rollout_is_weights=None):
    if _updates == 0:
        valid = response_mask.bool()
        delta = (log_prob.detach().float() - old_log_prob.detach().float())[valid]
        maximum = delta.abs().max().item() if delta.numel() else 0.0
        tolerance = float(os.environ.get("PART7_FROZEN_LOGPROB_TOLERANCE", "0.02"))
        emit("frozen_policy", tokens=delta.numel(), max_abs_logprob_delta=maximum,
             mean_abs_logprob_delta=delta.abs().mean().item() if delta.numel() else 0.0,
             tolerance=tolerance, finite=bool(torch.isfinite(delta).all()))
        # Both ranks fail together before backward; no rank waits in FSDP collectives.
        bad = torch.tensor(int(not torch.isfinite(delta).all() or maximum > tolerance),
                           device=log_prob.device)
        torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bad.item():
            raise RuntimeError("Frozen-policy log probabilities differ; review diagnostics before training")
    return compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask,
                                      loss_agg_mode, config, rollout_is_weights)


def optimizer_evidence(optimizer, args, kwargs):
    global _updates
    if not torch.distributed.is_initialized():
        return
    _updates += 1
    emit("optimizer_update", number=_updates)
    if _updates != 1:
        return
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    sharded = [p for p in parameters if isinstance(p, DTensor)
               and any(placement.is_shard() for placement in p.placements)]
    samples = []
    for parameter in sharded[:3]:
        state = optimizer.state.get(parameter, {})
        samples.append(dict(parameter=tensor_description(parameter),
                            gradient=tensor_description(parameter.grad) if parameter.grad is not None else None,
                            optimizer={name: tensor_description(value) for name, value in state.items()
                                       if isinstance(value, torch.Tensor)}))
    emit("first_optimizer_update", parameter_tensors=len(parameters), sharded_tensors=len(sharded),
         samples=samples, cuda_device=torch.cuda.current_device(),
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
         peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if torch.distributed.get_world_size() != 2 or not sharded:
        raise RuntimeError("7A smoke requires two training ranks and actual DTensor sharding")


_optimizer_hook = register_optimizer_step_post_hook(optimizer_evidence)
