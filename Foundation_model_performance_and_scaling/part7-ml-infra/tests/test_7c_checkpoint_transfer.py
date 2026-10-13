"""Mocked private Hub transfers; no network, model loading or GPU execution."""
from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
transfer=import_module("infrastructure.7c_kuberay.checkpoint_transfer")
render=import_module("infrastructure.7c_kuberay.render")
controller=import_module("infrastructure.7c_kuberay.job_control")
REVISION="a"*40


class CheckpointTransferTests(unittest.TestCase):
    def api(self,private=True):
        """Return a fake authenticated Hub API with complete inventory and commit."""
        api=Mock(token="fake-secret")
        api.repo_info.return_value=SimpleNamespace(private=private,sha=REVISION)
        api.whoami.return_value={"name":"account"}
        api.list_repo_files.return_value=[transfer.bundle.MANIFEST,"actor/state.pt"]
        return api

    def test_options_require_immutable_paired_restore(self):
        """Reject missing revisions, mutable branches and malformed repository IDs."""
        transfer.validate_options("auto","account/checkpoint",REVISION)
        for args in ((None,"auto",REVISION),(None,"account/checkpoint","main"),(None,"account/checkpoint",None),(None,None,REVISION),("bad;command",None,None)):
            with self.assertRaises(ValueError):transfer.validate_options(*args)

    def test_standalone_credentials_rejected_before_hub_import(self):
        """Block transfer outside a Kubernetes pod before any token/network access."""
        with patch.dict("os.environ",{},clear=True):
            with self.assertRaisesRegex(RuntimeError,"Kubernetes pod"):transfer.credentials()

    def test_upload_is_private_and_records_pinned_revision(self):
        """Verify auto naming, private repository and complete-state receipt."""
        api=self.api();manifest=dict(step=2,files={"actor/state.pt":{}})
        with patch.object(transfer,"credentials",return_value=api),patch.object(transfer.bundle,"verify_manifest",return_value=manifest):
            receipt=transfer.upload(Path("/workspace/checkpoints/global_step_2"),"auto","run-unique")
        self.assertEqual(receipt["repo_id"],"account/part7-7c-run-unique")
        self.assertEqual(receipt["revision"],REVISION)
        self.assertTrue(receipt["full_checkpoint"])
        self.assertTrue(api.create_repo.call_args.kwargs["private"])
        api.upload_large_folder.assert_called_once()

    def test_public_destination_or_missing_inventory_fails(self):
        """Do not report retention success for public or incomplete repositories."""
        manifest=dict(step=2,files={"actor/state.pt":{}})
        api=self.api(private=False)
        with patch.object(transfer,"credentials",return_value=api),patch.object(transfer.bundle,"verify_manifest",return_value=manifest):
            with self.assertRaisesRegex(ValueError,"private"):transfer.upload(Path("/workspace/saved"),"account/repo","run")
        api.upload_large_folder.assert_not_called()
        api=self.api();api.list_repo_files.return_value=[]
        with patch.object(transfer,"credentials",return_value=api),patch.object(transfer.bundle,"verify_manifest",return_value=manifest):
            with self.assertRaisesRegex(RuntimeError,"inventory"):transfer.upload(Path("/workspace/saved"),"account/repo","run")

    def test_restore_downloads_pinned_state_then_verifies(self):
        """Use Secret token and full commit for a fresh PVC download, then hash-check."""
        api=self.api();snapshot=Mock();manifest={"step":2}
        with patch.object(transfer,"credentials",return_value=api),patch.dict(sys.modules,{"huggingface_hub":SimpleNamespace(snapshot_download=snapshot)}),patch.object(transfer.bundle,"verify_manifest",return_value=manifest) as verify:
            result=transfer.restore("account/repo",REVISION,"/workspace/restored/run/global_step_2")
        self.assertEqual(result,manifest)
        self.assertEqual(snapshot.call_args.kwargs["revision"],REVISION)
        self.assertEqual(snapshot.call_args.kwargs["token"],"fake-secret")
        verify.assert_called_once_with(Path("/workspace/restored/run/global_step_2"))

    def test_restore_rejects_ephemeral_or_reused_path(self):
        """Fail unsafe/reused destinations before reading credentials or downloading."""
        with patch.object(transfer,"credentials") as credentials:
            with self.assertRaises(ValueError):transfer.restore("account/repo",REVISION,"/tmp/global_step_2")
            with patch.object(Path,"exists",return_value=True):
                with self.assertRaises(FileExistsError):transfer.restore("account/repo",REVISION,"/workspace/restored/global_step_2")
            credentials.assert_not_called()

    def test_manifest_and_controller_preserve_transfer_options(self):
        """Pass repository/commit through the RayJob command without token contents."""
        m=render.render_job("registry/image@sha256:"+"a"*64,"part7","resume-hub","pvc","hf","node",phase="resume",
            checkpoint="/workspace/restored/global_step_2",upload_checkpoint_repo="auto",restore_repo="account/repo",restore_revision=REVISION)
        self.assertEqual(controller.validate_manifest(m),("part7","resume-hub","resume"))
        self.assertIn("--restore-revision "+REVISION,m["spec"]["entrypoint"])
        with self.assertRaises(ValueError):render.render_job("registry/image@sha256:"+"a"*64,"part7","bad","pvc","hf","node",upload_checkpoint_repo="auto")
