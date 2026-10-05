"""CPU checks for artifact selection and privacy; no Hub writes or model loads."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

artifacts = importlib.import_module("infrastructure.7a_verl.hf_artifacts")


class ArtifactTests(unittest.TestCase):
    def test_auto_destinations_use_verified_token_account(self):
        api = Mock()
        api.whoami.return_value = {"name": "sclion"}
        destinations = artifacts.upload_destinations(api, Path("smoke_001"), "auto", "auto")
        self.assertEqual(destinations["model_repo"], "sclion/part7-7a-smoke-001")
        self.assertEqual(destinations["artifact_repo"], "sclion/part7-7a-smoke-001-artifacts")
        api.whoami.assert_called_once()

    def test_explicit_destination_is_preserved(self):
        api = Mock()
        api.whoami.return_value = {"name": "sclion"}
        destinations = artifacts.upload_destinations(api, Path("smoke_001"), "team/custom", None)
        self.assertEqual(destinations["model_repo"], "team/custom")
        self.assertIsNone(destinations["artifact_repo"])

    def test_small_evidence_excludes_weights_and_upload_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            names = ["job_status.json", "diagnostics/rank0.jsonl", "checkpoints/global_step_3/actor/model.pt",
                     "checkpoints/global_step_3/data.pt", "hf_model/model.safetensors",
                     ".cache/huggingface/upload/state", "hf_upload.json"]
            for name in names:
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(set(artifacts.artifact_files(run, False)), set(names[:2]))
            self.assertEqual(set(artifacts.artifact_files(run, True)), set(names[:4]))

    def test_latest_checkpoint_uses_numeric_step(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for step in (3, 12):
                (run / f"checkpoints/global_step_{step}/actor").mkdir(parents=True)
            self.assertEqual(artifacts.checkpoint_actor(run).parent.name, "global_step_12")

    def test_public_destination_is_rejected(self):
        api = Mock()
        api.repo_info.return_value.private = False
        with self.assertRaises(ValueError):
            artifacts.private_repo(api, "owner/run", "dataset")
        api.create_repo.assert_called_once_with(repo_id="owner/run", repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
