"""Synthetic Kubernetes lifecycle tests; no kubectl, network or models are executed."""
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from contextlib import nullcontext
module=import_module("infrastructure.7d_serve.service_control")
IMAGE="registry.example/serve@sha256:"+"a"*64


def ready_service():
    """Return a minimal KubeRay 1.4.2 ready status with observed generation."""
    return dict(metadata=dict(uid="service-uid",generation=1),status=dict(
        observedGeneration=1,numServeEndpoints=1,conditions=[dict(type="Ready",status="True",observedGeneration=1)],
        activeServiceStatus=dict(rayClusterName="cluster",applicationStatuses={"part7-llm":dict(
            status="RUNNING",serveDeploymentStatuses={"engine":dict(status="HEALTHY")})})))


def owned_pod(name="worker",uid="worker-uid",kind="worker"):
    """Return one ready pod with a controller owner, role label and node placement."""
    return dict(metadata=dict(name=name,uid=uid,ownerReferences=[dict(uid="cluster-uid",controller=True)],
                             labels={"ray.io/node-type":kind}),
                spec=dict(nodeName="node",containers=[dict(name="ray-worker",image=IMAGE)]),
                status=dict(conditions=[dict(type="Ready",status="True")]))


class ServiceControlTests(unittest.TestCase):
    def session(self, output, call):
        """Create a synthetic already-owned Session without contacting Kubernetes."""
        result=module.Session(module.renderer.render_service(IMAGE,"node"),"explicit",output,call=call)
        result.uid="service-uid"
        result.generation=1
        result.cluster="cluster"
        result.cluster_uid="cluster-uid"
        return result

    def test_cluster_binding_waits_only_for_not_found(self):
        """Allow delayed cluster creation while preserving API and ownership failures."""
        with tempfile.TemporaryDirectory() as folder:
            call=MagicMock(return_value="")
            session=self.session(Path(folder),call)
            session.cluster=session.cluster_uid=None
            self.assertFalse(session.bind_cluster("cluster"))
            self.assertIn("--ignore-not-found=true",call.call_args.args[0])
            call.return_value=json.dumps(dict(metadata=dict(uid="cluster-uid",
                ownerReferences=[dict(uid="service-uid",controller=True)])))
            self.assertTrue(session.bind_cluster("cluster"))
            call.side_effect=RuntimeError("API unavailable")
            with self.assertRaises(RuntimeError):session.bind_cluster("cluster")

    def test_exact_manifest_gate(self):
        """Reject resource/command modifications before any cluster operations."""
        doc=module.renderer.render_service(IMAGE,"node")
        namespace,name,config=module.validate_manifest(doc)
        self.assertEqual(namespace,"part7")
        self.assertEqual(config.model_revision,module.settings.DEFAULT_REVISION)
        doc["spec"]["rayClusterConfig"]["headGroupSpec"]["template"]["spec"]["containers"][0]["command"]=["bad"]
        with self.assertRaises(ValueError):module.validate_manifest(doc)

    def test_readiness_requires_current_operator_and_deployment_status(self):
        """A running pod or stale Running status alone cannot satisfy service readiness."""
        document=ready_service()
        self.assertEqual(module.service_ready(document),"cluster")
        for change in (lambda d:d["status"].update(observedGeneration=0),
                       lambda d:d["status"].update(numServeEndpoints=0),
                       lambda d:d["status"].update(conditions=[]),
                       lambda d:d["status"]["activeServiceStatus"]["applicationStatuses"]["part7-llm"].update(serveDeploymentStatuses={})):
            copy=deepcopy(document);change(copy)
            self.assertIsNone(module.service_ready(copy))
        document["status"]["activeServiceStatus"]["applicationStatuses"]["part7-llm"]["status"]="DEPLOY_FAILED"
        with self.assertRaises(RuntimeError):module.service_ready(document)

    def test_uid_precondition_is_atomic_delete_body(self):
        """Send API DeleteOptions UID instead of a race-prone name-only delete."""
        calls=[]
        def call(args,**kwargs):
            """Capture a synthetic kubectl invocation."""
            calls.append((args,kwargs));return "{}"
        module.delete_owned(call,"part7","pods","worker","old-uid")
        args,kwargs=calls[0]
        self.assertIn("/api/v1/namespaces/part7/pods/worker",args)
        self.assertEqual(kwargs["body"]["preconditions"],dict(uid="old-uid"))
        with self.assertRaises(ValueError):module.delete_owned(call,"part7","secrets","worker","uid")

    def test_owner_and_service_generation_guards(self):
        """Labels are insufficient ownership; name reuse/spec edits stop operations."""
        pod=owned_pod()
        module.require_owner(pod,"cluster-uid")
        with self.assertRaises(ValueError):module.require_owner(pod,"unrelated")
        with tempfile.TemporaryDirectory() as folder:
            doc=ready_service()
            doc["metadata"]["uid"]="replacement"
            session=self.session(Path(folder),lambda args:json.dumps(doc))
            with self.assertRaises(ValueError):session.current()
            doc["metadata"].update(uid="service-uid",generation=2)
            with self.assertRaises(ValueError):session.current()

    def test_restart_only_single_owned_ready_worker(self):
        """Delete only the recorded worker and save intent before mutation."""
        calls=[]
        def call(args,**kwargs):
            """Return controlled status/pods or record the deletion."""
            calls.append((args,kwargs))
            if args[:2]==["get","rayservice"]:return json.dumps(ready_service())
            if args[:2]==["get","pods"]:return json.dumps(dict(items=[owned_pod()]))
            return "{}"
        with tempfile.TemporaryDirectory() as folder:
            session=self.session(Path(folder),call)
            old=session.restart_worker()
            self.assertEqual(old["uid"],"worker-uid")
            self.assertTrue((Path(folder)/"restart_intent.json").exists())
            self.assertEqual(calls[-1][1]["body"]["preconditions"]["uid"],"worker-uid")

    def test_recovery_requires_new_uid_and_working_completion(self):
        """Retain real probe failures; a ready replacement alone is insufficient."""
        pods=[[],[owned_pod(uid="new-uid")]]
        with tempfile.TemporaryDirectory() as folder:
            session=self.session(Path(folder),lambda args:json.dumps(ready_service()))
            with patch.object(session,"pods",side_effect=pods),patch.object(module.time,"sleep"),patch.object(
                    module.benchmark,"request_completion",side_effect=[dict(success=False,error_type="TimeoutError"),dict(success=True)]):
                receipt=session.recover_worker(dict(name="worker",uid="worker-uid",node="node"),"http://localhost","raw")
            self.assertEqual(receipt["new_worker"]["uid"],"new-uid")
            self.assertEqual(receipt["http_probe_failures"],1)
            self.assertTrue(receipt["pod_readiness_loss_observed"])

    def test_terminating_old_worker_cannot_prove_recovery(self):
        """A successful response during overlap may come from the old worker."""
        old=owned_pod()
        old["metadata"]["deletionTimestamp"]="2026-10-25T00:00:00Z"
        replacement=owned_pod(uid="new-uid")
        with tempfile.TemporaryDirectory() as folder:
            session=self.session(Path(folder),lambda args:json.dumps(ready_service()))
            with patch.object(session,"pods",side_effect=[[old,replacement],[replacement]]),patch.object(
                    module.time,"sleep"),patch.object(module.benchmark,"request_completion",return_value=dict(success=True)) as requests:
                receipt=session.recover_worker(dict(name="worker",uid="worker-uid",node="node"),"http://localhost","raw")
            self.assertEqual(requests.call_count,2)
            self.assertTrue(receipt["old_worker_absent"])
            observations=json.loads((Path(folder)/"recovery_observations.json").read_text())
            self.assertTrue(observations[0]["old_worker_present"])
            self.assertFalse(observations[1]["old_worker_present"])

    def test_cleanup_requires_review_and_verifies_absence(self):
        """Passing review gates UID-protected cleanup; storage is never deleted."""
        calls=[]
        def call(args,**kwargs):
            """Simulate deletion followed by absence of owned descendants."""
            calls.append((args,kwargs))
            if "--ignore-not-found" in args:return ""
            if args[:2]==["get","pods"]:return '{"items":[]}'
            if args[:2]==["get","rayservice"]:return json.dumps(ready_service())
            return "{}"
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);session=self.session(output,call)
            module.save_json(output/"review.json",dict(success=False,service_uid="service-uid"))
            with self.assertRaises(ValueError):session.cleanup()
            self.assertFalse(calls)
            module.save_json(output/"review.json",dict(success=True,service_uid="service-uid"))
            session.cleanup()
            self.assertTrue(json.loads((output/"cleanup.json").read_text())["success"])
            deletes=[entry for entry in calls if entry[0][0]=="delete"]
            self.assertEqual(len(deletes),1)
            self.assertIn("/rayservices/part7-7d",deletes[0][0][2])

    def test_create_uses_create_and_refuses_output_reuse(self):
        """Never silently adopt/apply an existing service or overwrite old run evidence."""
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            def call(args,**kwargs):
                """Capture only the explicit create mutation."""
                calls.append(args);return json.dumps(ready_service())
            output=Path(folder)/"new"
            session=module.Session(module.renderer.render_service(IMAGE,"node"),"context",output,call=call)
            session.create()
            self.assertEqual(calls[0][0],"create")
            with self.assertRaises(FileExistsError):session.create()
            self.assertEqual(len(calls),1)


class LifecycleWorkflowTests(unittest.TestCase):
    def fake_session(self, output):
        """Return a controlled session whose methods perform no cluster/model work."""
        session=MagicMock()
        session.uid="service-uid"
        session.cluster_uid="cluster-uid"
        session.config=module.settings.ServingConfig()
        session.create.side_effect=lambda:output.mkdir()
        session.collect.return_value=True
        session.restart_worker.return_value=dict(name="old",uid="old-uid",node="node")
        return session

    def test_success_review_precedes_cleanup(self):
        """All phases, retained evidence and successful review must precede deletion."""
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"evidence"
            session=self.fake_session(output)
            def cleanup():
                """Assert the passing receipt exists at the moment cleanup is requested."""
                self.assertTrue(json.loads((output/"review.json").read_text())["success"])
            session.cleanup.side_effect=cleanup
            with patch.object(module,"require_cloud_host"),patch.object(module,"Session",return_value=session),patch.object(
                    module,"port_forward",return_value=nullcontext("http://localhost")),patch.object(
                    module,"http_ready",return_value=True),patch.object(module.benchmark,"run",return_value=dict(summary=dict(errors=0))):
                module.execute({},"context",output,["raw prompt"],cleanup_on_success=True)
            session.restart_worker.assert_called_once()
            session.recover_worker.assert_called_once()
            session.cleanup.assert_called_once()
            self.assertTrue((output/"benchmark_before.json").exists())
            self.assertTrue((output/"benchmark_after.json").exists())

    def test_failure_retains_resources_and_receipt(self):
        """A deployment failure captures evidence and never requests cleanup."""
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"evidence"
            session=self.fake_session(output)
            session.wait_ready.side_effect=RuntimeError("synthetic failure")
            with patch.object(module,"require_cloud_host"),patch.object(module,"Session",return_value=session):
                with self.assertRaises(RuntimeError):
                    module.execute({},"context",output,["prompt"],cleanup_on_success=True)
            session.cleanup.assert_not_called()
            session.restart_worker.assert_not_called()
            receipt=json.loads((output/"failure.json").read_text())
            self.assertTrue(receipt["resources_retained"])
            self.assertNotIn("synthetic failure",json.dumps(receipt))

    def test_transport_is_stopped_after_body_failure(self):
        """Stop only the local kubectl tunnel when a benchmark raises an exception."""
        with tempfile.TemporaryDirectory() as folder:
            session=MagicMock(context="context",namespace="part7",output=Path(folder))
            session.pods.return_value=[owned_pod(name="head",kind="head")]
            process=MagicMock()
            process.poll.return_value=None
            def start(*args,**kwargs):
                """Emit the same dynamic-port announcement as kubectl's port-forward."""
                kwargs["stdout"].write("Forwarding from 127.0.0.1:32123 -> 8000\n")
                kwargs["stdout"].flush()
                return process
            with patch.object(module.subprocess,"Popen",side_effect=start):
                with self.assertRaisesRegex(RuntimeError,"body failed"):
                    with module.port_forward(session) as url:
                        self.assertEqual(url,"http://127.0.0.1:32123")
                        raise RuntimeError("body failed")
            process.terminate.assert_called_once()
            process.wait.assert_called_once()


class ServingDriverTests(unittest.TestCase):
    def test_cuda13_driver_baseline(self):
        """Reject unsupported/unknown driver rows before any expensive cloud operation."""
        self.assertEqual(module.validate_driver_versions("580.105.08\n595.71.05\n"),["580.105.08","595.71.05"])
        for rows in ("", "unknown", "570.86.15", "595.71.05\n550.1.0"):
            with self.assertRaises(ValueError):module.validate_driver_versions(rows)

    def test_non_amd64_host_is_rejected_before_gpu_checks(self):
        """The digest-pinned base is amd64 and must not be mistaken for an ARM profile."""
        with patch.object(module.platform,"release",return_value="6.8.0-generic"),patch.object(
                module.platform,"machine",return_value="aarch64"),patch.object(module.subprocess,"run") as run:
            with self.assertRaisesRegex(ValueError,"x86_64"):module.require_cloud_host()
            run.assert_not_called()
