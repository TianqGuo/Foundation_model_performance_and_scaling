"""CPU-only RayJob/placement validation; no cluster contacts, Torch or models."""
from copy import deepcopy
from importlib import import_module
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

render = import_module("infrastructure.7c_kuberay.render")
preflight = import_module("infrastructure.7c_kuberay.preflight")
IMAGE = "registry.example/part7@sha256:" + "a" * 64


class KubernetesPreparationTests(unittest.TestCase):
    def manifest(self):
        """Return a synthetic digest/node manifest; never applied to Kubernetes."""
        return render.render_job(IMAGE, "part7", "test-preflight", "workspace", "hf-secret", "gpu-node")

    def test_mutable_or_injected_images_rejected(self):
        for image in ("repo/image:latest", "repo/image@sha256:123", IMAGE + " ; id", IMAGE + "\n"):
            with self.subTest(image=image), self.assertRaises(ValueError):
                render.pinned_image(image)
        self.assertEqual(render.pinned_image(IMAGE), IMAGE)

    def test_names_cannot_inject_entrypoint(self):
        for name in ("Bad", "job; echo nope", "../path", "a" * 64):
            with self.assertRaises(ValueError):
                render.render_job(IMAGE, "part7", name, "workspace", "hf-secret", "gpu-node")

    def test_cpu_head_and_one_whole_two_gpu_worker(self):
        cluster = self.manifest()["spec"]["rayClusterSpec"]
        head = cluster["headGroupSpec"]["template"]["spec"]["containers"][0]
        worker = cluster["workerGroupSpecs"][0]
        gpu = worker["template"]["spec"]["containers"][0]
        self.assertNotIn("nvidia.com/gpu", head["resources"]["limits"])
        self.assertEqual(gpu["resources"]["requests"]["nvidia.com/gpu"], 2)
        self.assertEqual(gpu["resources"]["requests"], gpu["resources"]["limits"])
        self.assertEqual((worker["replicas"], worker["minReplicas"], worker["maxReplicas"]), (1, 1, 1))
        self.assertEqual(cluster["rayVersion"], "2.49.2")
        self.assertFalse(cluster["enableInTreeAutoscaling"])

    def test_workspace_shm_and_secret_file_are_mounted_in_both_pods(self):
        cluster = self.manifest()["spec"]["rayClusterSpec"]
        pods = [cluster["headGroupSpec"]["template"]["spec"], cluster["workerGroupSpecs"][0]["template"]["spec"]]
        for pod in pods:
            self.assertEqual(pod["nodeSelector"], {"kubernetes.io/hostname": "gpu-node"})
            self.assertFalse(pod["automountServiceAccountToken"])
            volumes = {v["name"]: v for v in pod["volumes"]}
            self.assertEqual(volumes["workspace"]["persistentVolumeClaim"]["claimName"], "workspace")
            self.assertEqual(volumes["hub-token"]["secret"]["secretName"], "hf-secret")
            self.assertEqual(volumes["dshm"]["emptyDir"]["medium"], "Memory")
            self.assertNotIn("HF_TOKEN", json.dumps(pod["containers"][0]["env"]))

    def test_optional_registry_secret_does_not_contain_credentials(self):
        document = render.render_job(IMAGE, "part7", "job", "workspace", "hf-secret", "node", "registry-auth")
        pod = document["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
        self.assertEqual(pod["imagePullSecrets"], [{"name": "registry-auth"}])
        self.assertEqual(document["spec"]["backoffLimit"], 0)
        self.assertTrue(document["spec"]["shutdownAfterJobFinishes"])
        self.assertEqual(document["spec"]["activeDeadlineSeconds"], 900)

    def test_renderer_writes_fresh_file_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            argv = ["render", "--image", IMAGE, "--node-name", "node", "--output", str(output)]
            with patch("sys.argv", argv):
                render.main()
                with self.assertRaises(FileExistsError):
                    render.main()
            self.assertEqual(json.loads(output.read_text())["kind"], "RayJob")

    def placement(self):
        """Return fabricated CPU/GPU Ray nodes and two single-GPU actor probes."""
        nodes = [dict(NodeID="head", Alive=True, Resources={"CPU": 2}),
                 dict(NodeID="worker", Alive=True, Resources={"CPU": 24, "GPU": 2})]
        probes = [dict(node_id="worker", pod_uid="gpu-pod", gpu_ids=[gpu],
                       cuda_available=True, cuda_device_count=1, cuda_probe_value=1.0) for gpu in (0, 1)]
        return nodes, probes

    def test_actual_gpu_assignment_and_cuda_kernel_result_required(self):
        nodes, probes = self.placement()
        preflight.check_placement(nodes, probes)
        invalid = deepcopy(probes)
        invalid[1]["gpu_ids"] = [0]
        with self.assertRaisesRegex(ValueError, "distinct"):
            preflight.check_placement(nodes, invalid)
        invalid = deepcopy(probes)
        invalid[0]["cuda_probe_value"] = 0
        with self.assertRaisesRegex(ValueError, "exposure"):
            preflight.check_placement(nodes, invalid)
        invalid = deepcopy(probes)
        invalid[0]["cuda_device_count"] = 2
        with self.assertRaisesRegex(ValueError, "exposure"):
            preflight.check_placement(nodes, invalid)

    def test_wrong_cluster_and_cross_pod_placement_rejected(self):
        nodes, probes = self.placement()
        nodes[1]["Resources"]["GPU"] = 4
        with self.assertRaisesRegex(ValueError, "exactly two"):
            preflight.check_placement(nodes, probes)
        nodes, probes = self.placement()
        probes[1]["pod_uid"] = "other-pod"
        with self.assertRaisesRegex(ValueError, "one GPU worker pod"):
            preflight.check_placement(nodes, probes)
        nodes, probes = self.placement()
        probes[1]["node_id"] = "head"
        with self.assertRaisesRegex(ValueError, "same GPU worker"):
            preflight.check_placement(nodes, probes)

    def test_standalone_preflight_fails_before_ray_or_torch_import(self):
        with patch.dict("os.environ", {}, clear=True), patch("sys.argv", ["preflight", "--cloud", "--output", "/unused"]):
            with self.assertRaisesRegex(RuntimeError, "Kubernetes pod"):
                preflight.main()

    @unittest.skipUnless(os.environ.get("KUBERAY_CRD"), "set KUBERAY_CRD for pinned offline schema check")
    def test_pinned_rayjob_openapi_schema(self):
        """Validate generated fields against the actual v1.4.2 CRD, without a server."""
        import yaml
        import jsonschema
        crd = yaml.safe_load(Path(os.environ["KUBERAY_CRD"]).read_text())
        schema = next(v for v in crd["spec"]["versions"] if v["name"] == "v1")["schema"]["openAPIV3Schema"]
        for phase in ("preflight", "baseline", "interrupt", "resume"):
            with self.subTest(phase=phase):
                manifest = render.render_job(IMAGE, "part7", "test-phase", "workspace", "hf-secret", "gpu-node",
                    phase=phase, checkpoint="/workspace/saved/global_step_2" if phase == "resume" else None)
                errors = list(jsonschema.Draft7Validator(schema).iter_errors(manifest))
                self.assertFalse(errors, [(list(e.path), e.message) for e in errors])


if __name__ == "__main__":
    unittest.main()
