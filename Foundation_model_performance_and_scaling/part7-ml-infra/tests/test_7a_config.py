"""CPU checks against upstream YAML; no verl/model imports or GPU operations."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from importlib import import_module

runner = import_module("infrastructure.7a_verl.run_7a")
ROOT = runner.ROOT
compose_config = runner.compose_config
verify_bundle = runner.verify_bundle
ensure_model_snapshot = runner.ensure_model_snapshot


class SmokePreparationTests(unittest.TestCase):
    def test_missing_model_downloads_on_cloud_and_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            def download(**kwargs):
                model.mkdir()
                (model / "config.json").write_text("{}")
            callback = Mock(side_effect=download)
            with patch.dict("sys.modules", huggingface_hub=SimpleNamespace(snapshot_download=callback)):
                ensure_model_snapshot(model, "Qwen/example", "commit", cloud=True)
                callback.assert_called_once_with(repo_id="Qwen/example", revision="commit", local_dir=str(model))
                ensure_model_snapshot(model, "Qwen/example", None, cloud=True)
                self.assertEqual(callback.call_count, 1)

    def test_local_preparation_never_downloads(self):
        callback = Mock()
        with patch.dict("sys.modules", huggingface_hub=SimpleNamespace(snapshot_download=callback)):
            with self.assertRaisesRegex(RuntimeError, "cloud-only"):
                ensure_model_snapshot(Path("/missing"), "Qwen/example", None, cloud=False)
            callback.assert_not_called()

    def test_missing_explicit_path_does_not_download_a_different_model(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = Mock()
            with patch.dict("sys.modules", huggingface_hub=SimpleNamespace(snapshot_download=callback)):
                with self.assertRaises(FileNotFoundError):
                    ensure_model_snapshot(Path(directory), "Qwen/example", None, cloud=True, explicit_path=True)
                callback.assert_not_called()

    def test_bundle_verification_survives_transfer_and_detects_corruption(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload").write_bytes(b"original")
            manifest = {"outputs": {"payload": {"path": "/old/host/payload",
                        "sha256": hashlib.sha256(b"original").hexdigest()}}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_bundle(root), manifest)
            (root / "payload").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_bundle(root)

    @unittest.skipUnless(os.environ.get("VERL_SOURCE"), "set VERL_SOURCE to pinned upstream checkout")
    def test_real_upstream_composition_and_global_batch_mapping(self):
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw_prompt.jinja").write_text("{{ messages[0]['content'] }}")
            config = compose_config(Path(os.environ["VERL_SOURCE"]), root, root / "model", root / "output")
            actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
            self.assertEqual(actor.strategy, "fsdp2")
            self.assertEqual(actor.fsdp_config.strategy, "fsdp2")
            self.assertEqual(config.trainer.use_legacy_worker_impl, "disable")
            self.assertEqual(actor.ppo_mini_batch_size * rollout.n, 16)
            self.assertEqual((config.data.train_batch_size // actor.ppo_mini_batch_size)
                             * actor.ppo_epochs * config.trainer.total_training_steps, 6)
            self.assertEqual(actor.fsdp_config.fsdp_size, config.trainer.n_gpus_per_node)
            self.assertEqual(rollout.tensor_model_parallel_size, 1)
            self.assertEqual(config.reward.custom_reward_function.reward_kwargs.grader_path,
                             str(root / "reference/drgrpo_grader.py"))
            self.assertEqual(config.data.truncation, "error")
            self.assertFalse(config.algorithm.rollout_correction.bypass_mode)
            self.assertFalse(config.algorithm.use_kl_in_reward)
            self.assertEqual(config.trainer.resume_mode, "disable")
            self.assertEqual(OmegaConf.to_container(config, resolve=True)["trainer"]["logger"], ["console"])


if __name__ == "__main__":
    unittest.main()
