"""Lightweight state/hook/evidence tests; no Torch, checkpoints or GPU loads."""
from importlib import import_module
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

audit = import_module("infrastructure.7b_ray.recovery_audit")
experiment = import_module("infrastructure.7b_ray.experiment")


class RecoveryTests(unittest.TestCase):
    def test_fingerprint_detects_values_types_and_boundaries(self):
        self.assertEqual(audit.fingerprint({1: "a", "1": "b"}), audit.fingerprint({"1": "b", 1: "a"}))
        samples = [1, True, 1.0, "1", [1], (1,), ["ab", "c"], ["a", "bc"]]
        self.assertEqual(len({audit.fingerprint(value) for value in samples}), len(samples))
        with self.assertRaises(TypeError):
            audit.fingerprint(object())

    def test_full_array_payload_and_dtype_change_fingerprint(self):
        class Array:
            def __init__(self, dtype, data):
                self.dtype, self.data = dtype, data
        def encode(value):
            return ((value.dtype, len(value.data)), value.data) if isinstance(value, Array) else None
        original = audit.fingerprint(Array("fp32", b"abcdefgh"), encode)
        self.assertNotEqual(original, audit.fingerprint(Array("fp32", b"abcdefgX"), encode))
        self.assertNotEqual(original, audit.fingerprint(Array("fp16", b"abcdefgh"), encode))

    def test_corrupted_component_rejects_restore(self):
        result = audit.compare_states({"optimizer": {"step": 2}}, {"optimizer": {"step": 1}})
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            audit.require_matches(result)
        with self.assertRaisesRegex(ValueError, "inventory"):
            audit.compare_states({"rng": 1}, {})

    def test_rng_restored_when_audit_raises(self):
        manager = SimpleNamespace(get_rng_state=Mock(return_value={"random": "saved"}), load_rng_state=Mock())
        with self.assertRaises(RuntimeError):
            with audit.preserve_rng(manager):
                raise RuntimeError("audit failed")
        manager.load_rng_state.assert_called_once_with({"random": "saved"})

    def test_worker_load_is_delegated_once_and_audited_afterward(self):
        calls = []
        class Manager:
            rank = 0
            def load_checkpoint(self, path, *args, **kwargs):
                calls.append("native")
                return "loaded"
        audit.install_worker_hooks(Manager)
        audit.install_worker_hooks(Manager)
        with patch.object(audit, "audit_rank", side_effect=lambda *args: calls.append("audit")), patch.object(audit, "write_event"):
            manager = Manager()
            self.assertEqual(manager.load_checkpoint("actor"), "loaded")
            self.assertEqual(calls, ["native", "audit"])
            manager.load_checkpoint(None)
            self.assertEqual(calls, ["native", "audit", "native"])
            with self.assertRaisesRegex(ValueError, "keeping"):
                manager.load_checkpoint("actor", del_local_after_load=True)

    def test_failed_native_load_never_reports_audit_success(self):
        class Manager:
            def load_checkpoint(self, *args, **kwargs):
                raise RuntimeError("native failed")
        audit.install_worker_hooks(Manager)
        with patch.object(audit, "audit_rank") as compare:
            with self.assertRaisesRegex(RuntimeError, "native failed"):
                Manager().load_checkpoint("actor")
            compare.assert_not_called()

    def test_interruption_requires_native_checkpoint_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            marker = checkpoints / "latest_checkpointed_iteration.txt"
            class Trainer:
                global_steps = 2
                config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(checkpoints)))
                def _save_checkpoint(self):
                    marker.write_text("2")
                def _load_checkpoint(self):
                    pass
                def _get_gen_batch(self, value):
                    return value
            audit.install_trainer_hooks(Trainer)
            with patch.dict("os.environ", PART7_RUN_DIR=str(root), PART7_INTERRUPT_STEP="2"), \
                 patch.object(experiment.bundle, "layout"), patch.object(audit, "write_event"):
                with self.assertRaises(audit.PlannedInterruption):
                    Trainer()._save_checkpoint()
                self.assertTrue(json.loads((root / "planned_interruption.json").read_text())["completed"])
            (root / "planned_interruption.json").unlink()
            marker.write_text("1")
            # A partial checkpoint cannot authorize an expected interruption.
            with patch.dict("os.environ", PART7_RUN_DIR=str(root), PART7_INTERRUPT_STEP="2"), \
                 patch.object(experiment.bundle, "layout", side_effect=ValueError("missing shard")):
                with self.assertRaisesRegex(ValueError, "missing shard"):
                    Trainer()._save_checkpoint()
                self.assertFalse((root / "planned_interruption.json").exists())

    def make_evidence(self, root):
        """Write synthetic valid evidence for policy checks, without tensor files."""
        runs = [root / name for name in ("baseline", "interrupt", "resume")]
        for run in runs:
            (run / "recovery").mkdir(parents=True)
            (run / "job_status.json").write_text(json.dumps(dict(success=True, planned_interruption=run.name == "interrupt")))
        baseline, interrupted, resumed = runs
        (resumed / "diagnostics").mkdir()
        (resumed / "diagnostics/events.jsonl").write_text("\n".join(json.dumps(dict(
            event="first_optimizer_update", rank=rank, world_size=2, sharded_tensors=3,
            host="cloud", visible_devices="0,1", cuda_device=rank, time=20)) for rank in (0, 1)))
        match = dict(expected="a", actual="a")
        records = [dict(event="rank_restore", rank=rank, world_size=2,
                        components={name: match for name in ("model", "optimizer", "scheduler", "rng")})
                   for rank in (0, 1)]
        records += [dict(event="rank_load_complete", rank=rank, time=10) for rank in (0, 1)]
        records += [dict(event="trainer_restore", step=2, components={name: match for name in ("data", "step")}),
                    dict(event="ray_placement", resources={"GPU": 2}, placement_groups={"group": {}}),
                    dict(event="training_batch", step=3, source_indices=[17, 23]),
                    dict(event="checkpoint_complete", step=3)]
        (resumed / "recovery/events.jsonl").write_text("\n".join(json.dumps(r) for r in records))
        (baseline / "recovery/events.jsonl").write_text(json.dumps(dict(event="training_batch", step=3, source_indices=[17, 23])))
        return runs, records

    def test_review_rejects_missing_rank_and_state_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, records = self.make_evidence(Path(directory))
            self.assertTrue(experiment.review_recovery(*runs, 2)["state_restore_verified"])
            path = runs[2] / "recovery/events.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records[1:]))
            with self.assertRaisesRegex(ValueError, "two rank"):
                experiment.review_recovery(*runs, 2)
            records[0]["components"]["model"] = dict(expected="a", actual="b")
            path.write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                experiment.review_recovery(*runs, 2)

    def test_review_rejects_missing_actual_sharded_update(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, records = self.make_evidence(Path(directory))
            (runs[2] / "diagnostics/events.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "sharded optimizer"):
                experiment.review_recovery(*runs, 2)

    def test_foreground_auth_does_not_create_repo_or_launch_training(self):
        """Verify first-login/cached paths return before any timed GPU experiment."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new_run"
            for cached in (None, "synthetic-credential"):
                api = Mock()
                api.whoami.return_value = {"name": "test-owner"}
                login = Mock()
                fake_hub = SimpleNamespace(HfApi=Mock(return_value=api), get_token=Mock(return_value=cached), login=login)
                stream = io.StringIO()
                with patch("sys.argv", ["experiment", "--cloud", "--authenticate-only", "--output", str(output)]), \
                     patch.dict("sys.modules", huggingface_hub=fake_hub), \
                     patch.object(experiment.bundle, "cloud_guard"), \
                     patch.object(experiment.shutil, "disk_usage", return_value=SimpleNamespace(free=200 * 1024**3)), \
                     patch("sys.stdin.isatty", return_value=True), \
                     patch.object(experiment.getpass, "getpass", return_value="synthetic-credential") as prompt, \
                     patch.object(experiment, "run_phase") as launch, redirect_stdout(stream):
                    experiment.main()
                    launch.assert_not_called()
                    api.create_repo.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertNotIn("synthetic-credential", stream.getvalue())
                    self.assertIn("test-owner", stream.getvalue())
                    self.assertEqual(prompt.call_count, int(cached is None))
                    self.assertEqual(login.call_count, int(cached is None))

    def test_evidence_upload_excludes_full_state_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("resume/recovery/events.jsonl", "interrupt/checkpoints/global_step_2/actor/model.pt",
                             "restored/global_step_2/data.pt", "interrupt/checkpoints/.cache/token", "timings.json"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic")
            self.assertEqual(experiment.evidence_files(root), ["resume/recovery/events.jsonl", "timings.json"])

    def test_review_rejects_changed_next_data_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, records = self.make_evidence(Path(directory))
            for record in records:
                if record["event"] == "training_batch":
                    record["source_indices"] = [99]
            (runs[2] / "recovery/events.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaisesRegex(ValueError, "source batch"):
                experiment.review_recovery(*runs, 2)


if __name__ == "__main__":
    unittest.main()
