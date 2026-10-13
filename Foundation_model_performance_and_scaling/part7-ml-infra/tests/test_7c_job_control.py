"""CPU controller tests with synthetic Kubernetes responses; no cluster contact."""
from copy import deepcopy
from importlib import import_module
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
controller=import_module("infrastructure.7c_kuberay.job_control")
render=import_module("infrastructure.7c_kuberay.render")


class JobControlTests(unittest.TestCase):
    def manifest(self,phase="baseline"):
        """Return a synthetic generated job with a fake immutable registry digest."""
        return render.render_job("registry/image@sha256:"+"a"*64,"part7","test-job","workspace","hf","node",
            phase=phase,checkpoint="/workspace/saved/global_step_2" if phase=="resume" else None)

    def test_only_generated_manifests_accepted(self):
        """Validate each phase and reject changed execution/resource/credential fields."""
        for phase in ("preflight","baseline","interrupt","resume"):
            self.assertEqual(controller.validate_manifest(self.manifest(phase)),("part7","test-job",phase))
        bad=self.manifest();bad["spec"]["entrypoint"]="echo replaced"
        with self.assertRaises(ValueError):controller.validate_manifest(bad)
        bad=self.manifest();bad["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]["env"].append(dict(name="HF_TOKEN",value="fake"))
        with self.assertRaises(ValueError):controller.validate_manifest(bad)

    def test_terminal_status_precedence(self):
        """Require RayJob success and treat deployment failure as overriding success."""
        self.assertEqual(controller.terminal_state({}),"pending")
        self.assertEqual(controller.terminal_state(dict(jobStatus="RUNNING")),"pending")
        self.assertEqual(controller.terminal_state(dict(jobStatus="SUCCEEDED")),"success")
        for state in ("FAILED","STOPPED"):
            self.assertEqual(controller.terminal_state(dict(jobStatus=state)),"failure")
        self.assertEqual(controller.terminal_state(dict(jobStatus="SUCCEEDED",jobDeploymentStatus="Failed")),"failure")

    def successful_collection(self, *args):
        """Write the required empty error receipt and return a synthetic head pod."""
        (args[-1] / "log_capture_errors.json").write_text('[]')
        return [dict(metadata=dict(name="head",labels={"ray.io/node-type":"head"}))]

    def fake_api(self,job_status="SUCCEEDED"):
        """Return a kubectl substitute for create, existence and one terminal poll."""
        def call(context,namespace,args,**kwargs):
            """Return synthetic JSON without executing kubectl or reading credentials."""
            if args[:3]==["get","rayjob","test-job"] and "--ignore-not-found" in args:return ""
            if args[0]=="create":return json.dumps(dict(metadata=dict(uid="owned")))
            return json.dumps(dict(metadata=dict(uid="owned"),status=dict(jobStatus=job_status,rayClusterName="owned-cluster",jobId="ray-submission")))
        return call

    def test_success_retains_before_cleanup_and_holds_cluster(self):
        """Keep cluster alive for evidence and clean only after retention succeeds."""
        calls=[]
        head=dict(metadata=dict(name="head",labels={"ray.io/node-type":"head"}))
        with tempfile.TemporaryDirectory() as d, patch.object(controller,"kubectl",side_effect=self.fake_api()), patch.object(controller,"collect",side_effect=self.successful_collection), patch.object(controller,"retain_workspace",side_effect=lambda *args:calls.append("retain")), patch.object(controller,"cleanup",side_effect=lambda *args:calls.append("cleanup")):
            out=Path(d)/"evidence"
            controller.run(self.manifest(),"explicit",out,True)
            self.assertEqual(calls,["retain","cleanup"])
            self.assertFalse(json.loads((out/"submitted_rayjob.json").read_text())["spec"]["shutdownAfterJobFinishes"])
            self.assertTrue(json.loads((out/"submission.json").read_text())["success"])

    def test_failed_job_keeps_resources_and_receipt(self):
        """Do not delete failed GPU jobs even when cleanup-on-success was selected."""
        with tempfile.TemporaryDirectory() as d, patch.object(controller,"kubectl",side_effect=self.fake_api("FAILED")), patch.object(controller,"collect",return_value=[]), patch.object(controller,"cleanup") as cleanup:
            out=Path(d)/"evidence"
            with self.assertRaisesRegex(RuntimeError,"failed"):controller.run(self.manifest(),"explicit",out,True)
            cleanup.assert_not_called()
            self.assertFalse(json.loads((out/"submission.json").read_text())["success"])

    def test_retention_failure_blocks_cleanup(self):
        """Prevent deletion when workspace evidence cannot be retained."""
        head=dict(metadata=dict(name="head",labels={"ray.io/node-type":"head"}))
        with tempfile.TemporaryDirectory() as d, patch.object(controller,"kubectl",side_effect=self.fake_api()), patch.object(controller,"collect",side_effect=self.successful_collection), patch.object(controller,"retain_workspace",side_effect=RuntimeError("retention failed")), patch.object(controller,"cleanup") as cleanup:
            with self.assertRaisesRegex(RuntimeError,"retention failed"):controller.run(self.manifest(),"explicit",Path(d)/"out",True)
            cleanup.assert_not_called()

    def test_cleanup_refuses_reused_job_name(self):
        """Compare UID before deletion and reject an unrelated replacement job."""
        with tempfile.TemporaryDirectory() as d, patch.object(controller,"kubectl",return_value=json.dumps(dict(metadata=dict(uid="other")))) as api:
            with self.assertRaisesRegex(RuntimeError,"UID"):controller.cleanup("ctx","part7","test-job","owned","cluster",Path(d))
            self.assertEqual(api.call_count,1)

    def test_cleanup_verifies_cluster_and_pod_absence(self):
        """Verify only scoped objects disappear, without PVC/Secret delete calls."""
        replies=[json.dumps(dict(metadata=dict(uid="owned"))),"","",json.dumps(dict(items=[]))]
        with tempfile.TemporaryDirectory() as d, patch.object(controller,"kubectl",side_effect=replies) as api:
            controller.cleanup("ctx","part7","test-job","owned","cluster",Path(d))
            self.assertTrue(json.loads((Path(d)/"cleanup.json").read_text())["verified"])
            mutations=[c.args[2] for c in api.call_args_list if c.args[2][0]=="delete"]
            self.assertEqual(mutations,[["delete","rayjob","test-job","--wait=true","--timeout=90s"]])

    def test_lightweight_archive_excludes_tensors_but_keeps_manifest(self):
        """Execute only the CPU archive script on tiny files, never Torch/model code."""
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"workspace";root.mkdir();out=Path(d)/"out";out.mkdir()
            (root/"job_status.json").write_text('{"success":true}')
            ckpt=root/"checkpoints";ckpt.mkdir()
            (ckpt/"checkpoint_manifest.json").write_text('{}')
            (ckpt/"model.pt").write_bytes(b'fake tensor bytes')
            (root/"weights.safetensors").write_bytes(b'fake weights')
            def local_archive(context,namespace,args,**kwargs):
                """Run the remote tar script locally against synthetic CPU files only."""
                return subprocess.check_output(["python","-c",args[-2],str(root)])
            with patch.object(controller,"kubectl",side_effect=local_archive):controller.retain_workspace("ctx","part7","head","test-job",out)
            with tarfile.open(out/"workspace_evidence.tar.gz") as archive:
                self.assertEqual(set(archive.getnames()),{"job_status.json","checkpoints/checkpoint_manifest.json"})

    def test_driver_log_identity_required(self):
        """Block cleanup when the Ray submission ID needed for logs is missing."""
        with tempfile.TemporaryDirectory() as d, patch.object(controller,"kubectl") as api:
            with self.assertRaisesRegex(RuntimeError,"submission ID"):
                controller.retain_driver_logs("ctx","part7","head",None,Path(d))
            api.assert_not_called()

    def test_pod_capture_omits_environment_values(self):
        """Retain placement/resources/status without recording pod credentials."""
        pod=dict(metadata=dict(name="worker",uid="uid",labels={}),spec=dict(nodeName="node",containers=[dict(name="ray-worker",image="digest",resources={},env=[dict(name="secret",value="do-not-retain")])]),status=dict(phase="Running"))
        with tempfile.TemporaryDirectory() as d, patch.object(controller,"kubectl",side_effect=[json.dumps(dict(items=[pod])),"logs"]):
            controller.collect("ctx","part7","cluster",Path(d))
            self.assertNotIn("do-not-retain",(Path(d)/"pods.json").read_text())
