"""CPU provenance and simulated E2E orchestration tests; no cloud/model execution."""
import argparse
from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
module=import_module("infrastructure.7d_serve.e2e")
IMAGE="registry.example/serve@sha256:"+"b"*64


class ServingE2ETests(unittest.TestCase):
    def test_actual_reference_prompts_and_selection(self):
        """Reuse real hash-verified source text without importing model/Parquet libraries."""
        with tempfile.TemporaryDirectory() as folder:
            prompts,receipt=module.prepare_prompts(module.ROOT.parent/"part5-alignment",Path(folder)/"data")
            self.assertEqual(len(prompts),32)
            self.assertEqual(receipt["selection"]["seed"],12345)
            helper=module.workload.load_reference(str(module.ROOT.parent/"part5-alignment"/module.platform.INPUT_HASHES["selection_helper"][0]))
            examples=module.workload.read_examples(module.ROOT.parent/"part5-alignment/data/math/validation.jsonl")
            self.assertEqual(receipt["selection"]["indices"],helper.evaluation_indices(len(examples),32,12345))
            rows=[json.loads(line) for line in (Path(folder)/"data/prompts.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["prompt"],prompts[0])
            self.assertNotIn("solution",rows[0])
            with self.assertRaises(FileExistsError):
                module.prepare_prompts(module.ROOT.parent/"part5-alignment",Path(folder)/"data")

    def test_image_gate_refuses_stale_source_and_import_failure(self):
        """A resolvable image digest cannot bypass native import/source compatibility gates."""
        def run(args,**kwargs):
            """Pretend Docker operations succeeded but return mismatched source identities."""
            if "run" in args:
                return argparse.Namespace(returncode=0,stdout='PART7_RECEIPT='+json.dumps(dict(
                    packages=dict(ray=module.control.settings.RAY_VERSION),source_hashes={})),stderr="")
            return argparse.Namespace(returncode=0)
        with tempfile.TemporaryDirectory() as folder,patch.object(module.subprocess,"run",side_effect=run):
            with self.assertRaisesRegex(ValueError,"differs"):
                module.image_gate(IMAGE,Path(folder))
            self.assertTrue((Path(folder)/"image_gate.log").exists())

    def test_fresh_flag_rejects_existing_cluster_before_mutation(self):
        """Fresh mode never uninstalls or silently reuses an existing Kubernetes cluster."""
        args=argparse.Namespace(fresh_setup=True)
        with patch.object(module.control,"require_cloud_host"),patch.object(module.platform,"cloud_cluster_present",return_value=True),patch.object(
                module.platform.setup,"install") as install:
            with self.assertRaises(ValueError):module.execute(args)
            install.assert_not_called()

    def test_image_failure_blocks_setup_and_retains_receipt(self):
        """Failing native imports stop before Kubernetes install or service creation."""
        with tempfile.TemporaryDirectory() as folder:
            args=argparse.Namespace(fresh_setup=True,part5_root=Path(folder),output=Path(folder)/"run",image=IMAGE)
            with patch.object(module.control,"require_cloud_host"),patch.object(module.platform,"cloud_cluster_present",return_value=False),patch.object(
                    module.platform,"verify_references"),patch.object(module,"image_gate",side_effect=RuntimeError("import failed")),patch.object(
                    module.platform.setup,"install") as install:
                with self.assertRaises(RuntimeError):module.execute(args)
                install.assert_not_called()
            receipt=json.loads((args.output/"e2e_review.json").read_text())
            self.assertFalse(receipt["success"])
            self.assertEqual(receipt["stages"],[])

    def test_full_reuse_sequence_requires_review_and_cleanup(self):
        """Simulate the composed normal path and retain a receipt only after cleanup verification."""
        with tempfile.TemporaryDirectory() as folder:
            args=argparse.Namespace(fresh_setup=False,part5_root=Path(folder),output=Path(folder)/"run",
                                    image=IMAGE,context="default",name="part7-7d-test")
            def service(document,context,output,prompts,**kwargs):
                """Save synthetic phase receipts without touching a cluster."""
                self.assertTrue(kwargs["cleanup_on_success"])
                output.mkdir()
                for filename in ("review.json","cleanup.json"):
                    module.control.save_json(output/filename,dict(success=True,service_uid="owned-service"))
            with patch.object(module.control,"require_cloud_host"),patch.object(module.platform,"cloud_cluster_present",return_value=True),patch.object(
                    module.platform,"verify_references"),patch.object(module,"image_gate"),patch.object(module.platform.cluster,"inspect",return_value={}),patch.object(
                    module.control,"api"),patch.object(module,"prepare_prompts",return_value=(["raw prompt"],{})),patch.object(
                    module.control,"execute",side_effect=service),patch.dict(module.os.environ,{"KUBECONFIG":"explicit-config"}):
                module.execute(args)
            receipt=json.loads((args.output/"e2e_review.json").read_text())
            self.assertTrue(receipt["success"])
            self.assertFalse(receipt["fresh_setup_verified"])
            self.assertEqual(receipt["stages"][-1],"owned_cleanup_verified")
