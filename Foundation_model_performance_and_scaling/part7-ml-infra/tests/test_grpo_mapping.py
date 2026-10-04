"""Check pinned upstream math against Part 5 on tiny CPU tensors.

Set VERL_SOURCE to a checkout of the selected commit. Extract only the named
functions with AST to avoid importing verl/Ray/CUDA. This checks equations and
scaling, not framework integration, FSDP execution or checkpoint/model behavior.
"""

import ast
from collections import defaultdict
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from workloads.math_workload import PART5_ROOT, load_reference


def extract_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Required functions missing from {path}")
    for node in selected:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


@unittest.skipUnless(os.environ.get("VERL_SOURCE"), "Set VERL_SOURCE to the pinned source checkout")
class GrpoMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ["VERL_SOURCE"])
        # Exact files from bec9ef74768dd201881cd4e54cd0385e87caae27.
        for name, expected in {
            "verl/trainer/ppo/core_algos.py": "bd5c2a262df55ec143af4392ae3f702c56497a76c6e6b5b641cc5e67760fd432",
            "verl/utils/torch_functional.py": "cb37d045dec2161d637e2633032cc97c97334e5547cea62dace42630c9683259",
        }.items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
                raise AssertionError(f"Wrong or modified upstream source: {name}")
        functional = {"torch": torch}
        extract_functions(source / "verl/utils/torch_functional.py", ["masked_mean", "masked_sum"], functional)
        namespace = {"torch": torch, "defaultdict": defaultdict,
                     "verl_F": SimpleNamespace(**{k: functional[k] for k in ["masked_mean", "masked_sum"]}),
                     "AlgoConfig": type("UnusedAlgoConfig", (), {})}
        extract_functions(source / "verl/trainer/ppo/core_algos.py",
                          ["compute_grpo_outcome_advantage", "agg_loss", "compute_policy_loss_vanilla"], namespace)
        cls.upstream = namespace
        cls.reference = load_reference(str((PART5_ROOT / "cs336_alignment/section7_grpo/estimators.py").resolve()))

    def config(self, batch_size=4, dp_size=1, dual_clip=1e10):
        return Config(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2,
                      clip_ratio_c=dual_clip, global_batch_info={"global_batch_size": batch_size, "dp_size": dp_size})

    def test_group_sample_std_and_zero_variance_match(self):
        rewards = torch.tensor([0., 1., 0., 1., 1., 1., 1., 1.], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 0]] * 8, dtype=torch.float64)
        token_rewards = torch.zeros_like(mask)
        token_rewards[:, 1] = rewards
        advantages, _ = self.upstream["compute_grpo_outcome_advantage"](
            token_rewards, mask, [0] * 4 + [1] * 4)
        expected = self.reference.build_advantages(rewards, 4)
        torch.testing.assert_close(advantages, expected[:, None] * mask)

    def test_vanilla_loss_and_gradient_match_with_dual_clip_inactive(self):
        old = torch.zeros((4, 3), dtype=torch.float64)
        current = torch.tensor([[.3, -.4, 0.], [2., .1, -.1], [-.2, .3, .1], [.5, 0., 0.]],
                               dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=torch.float64)
        advantages = torch.tensor([1., -1., 0., .5], dtype=torch.float64)
        expected = self.reference.aggregate_loss(
            self.reference.estimator_loss(advantages, current, mask, method="grpo", old_log_probs=old)[0], mask)
        actual, _ = self.upstream["compute_policy_loss_vanilla"](
            old, current, advantages[:, None] * mask, mask, "seq-mean-token-mean", self.config())
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        expected_grad = torch.autograd.grad(expected, current, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual, current)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-7)

    def test_default_dual_clip_changes_negative_advantage_loss(self):
        old = torch.zeros((1, 1), dtype=torch.float64)
        current = torch.full((1, 1), 2., dtype=torch.float64)
        advantages = -torch.ones_like(old)
        mask = torch.ones_like(old)
        default, _ = self.upstream["compute_policy_loss_vanilla"](
            old, current, advantages, mask, "seq-mean-token-mean", self.config(1, dual_clip=3.))
        selected, _ = self.upstream["compute_policy_loss_vanilla"](
            old, current, advantages, mask, "seq-mean-token-mean", self.config(1))
        self.assertLess(default.item(), selected.item())

    def test_global_sequence_scaling_matches_two_rank_microbatch_accumulation(self):
        losses = torch.tensor([[1., 3., 0.], [4., 0., 0.], [2., 3., 7.], [8., 2., 0.]],
                              dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.float64)
        expected = self.reference.aggregate_loss(losses, mask)
        # Each rank accumulates two one-example microbatches; FSDP averages rank gradients.
        actual = sum(self.upstream["agg_loss"](
            losses[i:i+1], mask[i:i+1], "seq-mean-token-mean", global_batch_size=4, dp_size=2)
            for i in range(4)) / 2
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        expected_grad = torch.autograd.grad(expected, losses, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual, losses)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
