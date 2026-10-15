"""CPU tests for bootstrap boundaries and same-node storage; no host installation."""
from importlib import import_module
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
setup = import_module("infrastructure.7c_kuberay.setup_cluster")
render = import_module("infrastructure.7c_kuberay.render")


class SetupTests(unittest.TestCase):
    def test_preview_never_inspects_or_mutates_host(self):
        """Check mode must only print objects, including when Secret flag is present."""
        with patch("sys.argv", ["setup", "--check", "--with-hf-secret"]), patch.object(setup, "host_check") as host, patch.object(setup, "run") as run, contextlib.redirect_stdout(io.StringIO()) as out:
            setup.main()
        host.assert_not_called()
        run.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())["kuberay"], "1.4.2")

    def test_host_rejection_precedes_output_or_installation(self):
        """An unsuitable host cannot create evidence directories or run installers."""
        with tempfile.TemporaryDirectory() as folder, patch.object(setup, "host_check", side_effect=ValueError("container")), patch.object(setup, "run") as run:
            destination = Path(folder) / "output"
            with self.assertRaises(ValueError):
                setup.install(destination)
            self.assertFalse(destination.exists())
            run.assert_not_called()

    def test_storage_is_prebound_same_node_retained(self):
        """PV/PVC must share one selected node and preserve host data on claim deletion."""
        objects = setup.documents()
        pv = next(obj for obj in objects if obj["kind"] == "PersistentVolume")
        pvc = next(obj for obj in objects if obj["kind"] == "PersistentVolumeClaim")
        self.assertEqual(pv["spec"]["persistentVolumeReclaimPolicy"], "Retain")
        self.assertEqual(pvc["spec"]["volumeName"], pv["metadata"]["name"])
        self.assertEqual(pv["spec"]["claimRef"]["name"], pvc["metadata"]["name"])
        self.assertEqual(pv["spec"]["nodeAffinity"]["required"]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"], [setup.NODE])

    def test_subprocess_errors_do_not_echo_stdin(self):
        """A failed Secret creation cannot expose its stdin through our error text."""
        with patch.object(setup.subprocess, "run", return_value=setup.subprocess.CompletedProcess([], 1, "", "secret-content")):
            with self.assertRaises(RuntimeError) as caught:
                setup.run(["kubectl", "create"], input_text="secret-content")
        self.assertNotIn("secret-content", str(caught.exception))

    def test_cpu_head_does_not_inherit_visible_gpus(self):
        """CPU head must hide base-image GPU defaults; worker uses plugin allocation."""
        job = render.render_job("registry.example/image@sha256:" + "0" * 64, "part7", "check", "part7-workspace", "part7-huggingface", setup.NODE)
        cluster = job["spec"]["rayClusterSpec"]
        head = cluster["headGroupSpec"]["template"]["spec"]["containers"][0]
        worker = cluster["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]
        self.assertIn(dict(name="NVIDIA_VISIBLE_DEVICES", value="void"), head["env"])
        self.assertFalse(any(env["name"] == "NVIDIA_VISIBLE_DEVICES" for env in worker["env"]))

    def test_private_config_is_exclusive_and_mode_0600(self):
        """Exercise the actual filesystem API that failed on cloud, without credentials."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config"
            setup.write_private_config(path, "synthetic-config")
            self.assertEqual(path.read_text(), "synthetic-config\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                setup.write_private_config(path, "overwrite")
            self.assertEqual(path.read_text(), "synthetic-config\n")

    def test_partial_setup_requires_matching_pin_and_single_node(self):
        """Continuation skips installation only for the recorded early failure boundary."""
        with tempfile.TemporaryDirectory() as folder:
            previous = Path(folder)
            (previous / "setup_review.json").write_text(json.dumps(dict(success=False, k3s=setup.K3S_VERSION, node=setup.NODE)))
            responses = [setup.K3S_VERSION, json.dumps(dict(items=[dict(metadata=dict(name=setup.NODE))]))]
            with patch.object(setup, "run", side_effect=responses):
                self.assertFalse(setup.verify_partial_setup(previous)["success"])
            (previous / "setup_objects.json").write_text("[]")
            with patch.object(setup, "run") as run:
                with self.assertRaises(ValueError):
                    setup.verify_partial_setup(previous)
                run.assert_not_called()

    def test_verified_gpu_node_matches_chart_affinity(self):
        """Set the NVIDIA chart's required GPU-present label on only our node."""
        with patch.object(setup, "run") as run:
            setup.label_gpu_node(["kubectl", "--context", "default"])
        run.assert_called_once_with(["kubectl", "--context", "default", "label", "node", setup.NODE,
                                     "nvidia.com/gpu.present=true", "--overwrite"])

    def test_registration_race_retries_before_ready_wait(self):
        """Node absence and transient API failures must not abort fresh installation."""
        responses = [RuntimeError("API starting"), '{"items": []}',
                     json.dumps(dict(items=[dict(metadata=dict(name=setup.NODE))])), "Ready"]
        with patch.object(setup, "run", side_effect=responses) as run, patch.object(setup.time, "sleep") as sleep:
            setup.wait_for_node(["kubectl"])
        self.assertEqual(sleep.call_count, 2)
        self.assertIn("wait", run.call_args.args[0])

    def test_resume_reuses_only_matching_private_config(self):
        """Resume may retain a private original kubeconfig, never different credentials."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config"
            setup.write_private_config(path, "synthetic")
            setup.ensure_private_config(path, "synthetic", allow_reuse=True)
            with self.assertRaises(ValueError):
                setup.ensure_private_config(path, "different", allow_reuse=True)
            with self.assertRaises(ValueError):
                setup.ensure_private_config(path, "synthetic")
