"""Exercise the cloud hook's numerical gate on CPU, without importing verl."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class FrozenPolicyGateTests(unittest.TestCase):
    def gate(self, peer_failed=False):
        source = Path(__file__).resolve().parents[1] / "infrastructure/7a_verl/verl_diagnostics.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "diagnostic_loss")
        function.decorator_list = []
        events, calls = [], []
        def reduce(tensor, op):
            if peer_failed:
                tensor.fill_(1)
        def objective(*args):
            calls.append(args)
            return "upstream result"
        namespace = dict(_updates=0, emit=lambda event, **fields: events.append((event, fields)),
            os=SimpleNamespace(environ={}), compute_policy_loss_vanilla=objective,
            torch=SimpleNamespace(isfinite=torch.isfinite, tensor=torch.tensor,
                distributed=SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MAX="max"))))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        return namespace["diagnostic_loss"], events, calls

    def test_valid_tokens_delegate_unchanged_and_ignore_padding(self):
        loss, events, calls = self.gate()
        old, new = torch.tensor([[1., 0.]]), torch.tensor([[1.001, float("nan")]])
        advantage, mask = torch.ones_like(old), torch.tensor([[1, 0]])
        self.assertEqual(loss(old, new, advantage, mask, "seq-mean-token-mean", "config"), "upstream result")
        self.assertIs(calls[0][0], old)
        self.assertIs(calls[0][1], new)
        self.assertEqual(events[0][1]["tokens"], 1)

    def test_discrepancy_and_nonfinite_abort_before_objective(self):
        for value in (0.03, float("nan"), float("inf")):
            with self.subTest(value=value):
                loss, events, calls = self.gate()
                with self.assertRaisesRegex(RuntimeError, "Frozen-policy"):
                    loss(torch.zeros(1, 1), torch.tensor([[value]]), torch.ones(1, 1), torch.ones(1, 1))
                self.assertFalse(calls)

    def test_peer_failure_also_aborts_local_rank(self):
        loss, events, calls = self.gate(peer_failed=True)
        with self.assertRaisesRegex(RuntimeError, "Frozen-policy"):
            loss(torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, 1))
        self.assertFalse(calls)
