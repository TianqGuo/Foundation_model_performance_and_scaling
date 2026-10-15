"""CPU orchestration tests; cloud installation/GPU/training calls are simulated."""
import argparse
import contextlib
from importlib import import_module
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
run = import_module("infrastructure.7c_kuberay.e2e")


class E2ERunnerTests(unittest.TestCase):
    def arguments(self, folder, fresh=False):
        """Return isolated CLI-like arguments for orchestration tests."""
        return argparse.Namespace(output=Path(folder)/"run", fresh_setup=fresh,
            image=run.IMAGE, part5_root=Path(folder)/"part5", data=Path(folder)/"bundle", name="test-e2e")

    def test_preview_never_executes_cloud_operations(self):
        """A full-flow preview must not invoke setup, Docker, Kubernetes or training."""
        with patch("sys.argv",["e2e","--check","--fresh-setup"]), patch.object(run,"verify_references"), patch.object(run,"execute") as execute, contextlib.redirect_stdout(io.StringIO()):
            run.main()
        execute.assert_not_called()

    def test_fresh_full_orchestration_and_receipt(self):
        """Check normal setup-to-recovery ordering and actual overall receipt writes."""
        stages=[]
        with tempfile.TemporaryDirectory() as folder, patch.object(run,"cloud_cluster_present",return_value=False), patch.object(run,"verify_references"), patch.object(run.setup,"install",side_effect=lambda *a,**k:stages.append("setup")), patch.object(run.cluster,"inspect",side_effect=lambda *a:stages.append("check") or {}), patch.object(run.cluster.control,"kubectl",return_value="secret/name"), patch.object(run,"prepare_bundle",side_effect=lambda *a:stages.append("data")), patch.object(run.experiment,"execute",side_effect=lambda *a:stages.append("experiment")), patch.dict(run.os.environ):
            args=self.arguments(folder, fresh=True)
            run.execute(args)
            report=json.loads((args.output/"e2e_review.json").read_text())
            self.assertEqual(stages,["setup","check","data","experiment"])
            self.assertTrue(report["success"])
            self.assertTrue(report["fresh_setup_verified"])

    def test_existing_cluster_is_not_fresh_install_evidence(self):
        """A requested clean-install test cannot silently reuse an existing cluster."""
        with tempfile.TemporaryDirectory() as folder, patch.object(run,"cloud_cluster_present",return_value=True), patch.object(run.setup,"install") as install:
            with self.assertRaisesRegex(ValueError,"clean VM"):
                run.execute(self.arguments(folder, fresh=True))
            install.assert_not_called()

    def test_failure_keeps_overall_receipt_unsuccessful(self):
        """A failed recovery must retain progress and cannot mark the full run passed."""
        with tempfile.TemporaryDirectory() as folder, patch.object(run,"cloud_cluster_present",return_value=True), patch.object(run,"verify_references"), patch.object(run.cluster,"inspect",return_value={}), patch.object(run.cluster.control,"kubectl",return_value="secret/name"), patch.object(run,"prepare_bundle"), patch.object(run.experiment,"execute",side_effect=RuntimeError("failed")), patch.dict(run.os.environ):
            args=self.arguments(folder)
            with self.assertRaises(RuntimeError):run.execute(args)
            report=json.loads((args.output/"e2e_review.json").read_text())
            self.assertFalse(report["success"])
            self.assertFalse(report["fresh_setup_verified"])
            self.assertEqual(report["stages"],["cluster_ready","data_ready"])

    def test_cloud_bundle_preparation_uses_image_and_readonly_reference(self):
        """Prepare missing data from cloud clone without local transfer/model/GPU flags."""
        manifest=dict(train_count=1023,validation_count=32,training_prefix_count=1024,
            evaluation=dict(seed=12345), inputs={name:dict(sha256=value[1]) for name,value in run.INPUT_HASHES.items()})
        with tempfile.TemporaryDirectory() as folder, patch.object(run.subprocess,"run",return_value=run.subprocess.CompletedProcess([],0)) as execute, patch.object(run.experiment.staging.reference,"verify_bundle",return_value=manifest):
            run.prepare_bundle(run.IMAGE,Path(folder)/"part5",Path(folder)/"data/bundle")
            command=execute.call_args.args[0]
            self.assertIn("NVIDIA_VISIBLE_DEVICES=void",command)
            self.assertIn("none",command)
            self.assertTrue(any("target=/part5,readonly" in value for value in command))
            self.assertIn("--drop-empty-solutions",command)
