"""Synthetic byte files only; no model loads or real Hub transfers."""
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

bundle = import_module("infrastructure.7b_ray.checkpoint_bundle")
runner = import_module("infrastructure.7b_ray.run_7b")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "source/global_step_2"
        self.root.mkdir(parents=True)
        names = ["data.pt", "actor/huggingface/config.json", "actor/huggingface/tokenizer_config.json",
                 "actor/huggingface/tokenizer.json"]
        names += [f"actor/{kind}_world_size_2_rank_{rank}.pt"
                  for rank in range(2) for kind in ("model", "optim", "extra_state")]
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic-payload")
        (self.root / "actor/fsdp_config.json").write_text(json.dumps(dict(FSDP_version=2, world_size=2)))
        (self.root.parent / "latest_checkpointed_iteration.txt").write_text("2")
        self.identity = dict(verl_commit=bundle.reference.VERL_COMMIT,
                             model_files={"config.json": {"sha256": "a" * 64}},
                             data_manifest={"outputs": {"train.parquet": {"sha256": "b" * 64}}})

    def create(self):
        return bundle.create_manifest(self.root, self.identity)

    def test_portable_manifest_after_producer_removal(self):
        expected = self.create()
        destination = Path(self.directory.name) / "restored/global_step_2"
        shutil.copytree(self.root, destination)
        shutil.rmtree(self.root.parent)
        self.assertEqual(bundle.verify_manifest(destination), expected)

    def test_missing_training_state_and_tokenizer_rejected(self):
        for name in ["data.pt", "actor/optim_world_size_2_rank_1.pt",
                     "actor/extra_state_world_size_2_rank_0.pt", "actor/model_world_size_2_rank_1.pt",
                     "actor/huggingface/tokenizer.json"]:
            path = self.root / name
            original = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                self.create()
            path.write_bytes(original)

    def test_incomplete_save_marker_rejected(self):
        (self.root.parent / "latest_checkpointed_iteration.txt").write_text("1")
        with self.assertRaisesRegex(ValueError, "completion marker"):
            self.create()

    def test_same_size_corruption_detected(self):
        self.create()
        (self.root / "data.pt").write_bytes(b"x" * len(b"synthetic-payload"))
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            bundle.verify_manifest(self.root)

    def test_extra_shard_and_wrong_topology_rejected(self):
        path = self.root / "actor/model_world_size_2_rank_2.pt"
        path.write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "Unexpected rank"):
            self.create()
        path.unlink()
        (self.root / "actor/fsdp_config.json").write_text(json.dumps(dict(FSDP_version=2, world_size=4)))
        with self.assertRaisesRegex(ValueError, "world-size mismatch"):
            self.create()

    def test_manifest_traversal_rejected(self):
        manifest = self.create()
        manifest["files"]["../outside.pt"] = dict(bytes=1, sha256="a" * 64)
        (self.root / bundle.MANIFEST).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            bundle.verify_manifest(self.root)

    def test_unlisted_actor_payload_rejected(self):
        self.create()
        (self.root / "actor/notes.json").write_text("extra")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            bundle.verify_manifest(self.root)

    def test_symlink_and_empty_file_rejected(self):
        path = self.root / "link"
        path.symlink_to(self.root / "data.pt")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.create()
        path.unlink()
        (self.root / "data.pt").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Empty"):
            self.create()

    def test_config_resume_step_limit_and_topology(self):
        manifest = self.create()
        config = dict(trainer=dict(nnodes=1, n_gpus_per_node=2),
                      actor_rollout_ref=dict(actor=dict(strategy="fsdp2", fsdp_config=dict(fsdp_size=2))))
        result = runner.apply_recovery_config(deepcopy(config), manifest, self.root, 4)
        self.assertEqual(result["trainer"]["resume_mode"], "resume_path")
        self.assertEqual(result["trainer"]["resume_from_path"], str(self.root))
        self.assertFalse(result["trainer"]["val_before_train"])
        self.assertFalse(result["trainer"]["del_local_ckpt_after_load"])
        with self.assertRaisesRegex(ValueError, "exceed"):
            runner.apply_recovery_config(deepcopy(config), manifest, self.root, 2)
        config["trainer"]["nnodes"] = 2
        with self.assertRaisesRegex(ValueError, "topology"):
            runner.apply_recovery_config(config, manifest, self.root, 4)

    def test_wrong_verl_identity_rejected(self):
        self.identity["verl_commit"] = "bad"
        with self.assertRaisesRegex(ValueError, "verl commit"):
            self.create()

    def test_restore_pins_revision_and_requires_fresh_destination(self):
        self.create()
        destination = Path(self.directory.name) / "download/global_step_2"
        callback = Mock(side_effect=lambda **kw: shutil.copytree(self.root, kw["local_dir"]))
        with patch.object(bundle, "cloud_guard"), patch.dict("sys.modules", huggingface_hub=SimpleNamespace(snapshot_download=callback)):
            manifest = bundle.restore("owner/checkpoint", "a" * 40, destination)
            self.assertEqual(manifest["step"], 2)
            self.assertEqual(callback.call_args.kwargs["repo_type"], "dataset")
            self.assertEqual(callback.call_args.kwargs["revision"], "a" * 40)
            with self.assertRaises(FileExistsError):
                bundle.restore("owner/checkpoint", "a" * 40, destination)
            with self.assertRaisesRegex(ValueError, "immutable"):
                bundle.restore("owner/checkpoint", "main", destination)
            self.assertEqual(callback.call_count, 1)


if __name__ == "__main__":
    unittest.main()
