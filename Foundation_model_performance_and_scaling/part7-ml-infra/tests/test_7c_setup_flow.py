"""Full installer control-flow tests with simulated host/API commands, not cloud E2E."""
from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
setup = import_module("infrastructure.7c_kuberay.setup_cluster")


class SetupFlowTests(unittest.TestCase):
    def run_install(self, fail_chart=False):
        """Exercise real install orchestration, files and validators with a fake system.

        Optional chart failure verifies receipt and Secret handling. Returns saved
        receipt, command trace, created objects and private file mode; no subprocess
        or privileged filesystem action is executed.
        """
        commands, created = [], []
        reads = dict(nodes=0, deployments=0)
        node = dict(metadata=dict(name=setup.NODE, labels={"kubernetes.io/hostname": setup.NODE}),
                    spec={}, status=dict(conditions=[dict(type="Ready", status="True")],
                    allocatable={"cpu":"52", "memory":"442Gi", "nvidia.com/gpu":"2"}))
        operator = dict(metadata=dict(name="kuberay-operator", generation=1),
                        spec=dict(template=dict(spec=dict(containers=[dict(image="quay.io/kuberay/operator:v1.4.2")]))),
                        status=dict(observedGeneration=1, availableReplicas=1))
        pvc = dict(metadata=dict(name="part7-workspace"),spec=dict(accessModes=["ReadWriteOnce"]),
                   status=dict(phase="Bound",capacity=dict(storage="100Gi")))
        crd = dict(spec=dict(versions=[dict(name="v1",served=True)]),
                   status=dict(conditions=[dict(type="Established",status="True")]))

        def fake_run(command, *, input_text=None, timeout=120):
            """Model external commands; fail on unexpected behavior or requested chart failure."""
            commands.append(command)
            if command[0] == "curl":
                Path(command[-1]).write_text("# simulated installer")
            elif command[:3] == ["sudo", "-n", "env"]:
                self.assertIn("INSTALL_K3S_VERSION=" + setup.K3S_VERSION, command)
            elif command[:3] == ["sudo", "-n", "cat"]:
                return "synthetic-private-kubeconfig"
            elif command[:3] in (["sudo", "-n", "mkdir"], ["sudo", "-n", "chown"]):
                pass
            elif command == ["/usr/local/bin/k3s", "--version"]:
                return setup.K3S_VERSION
            elif command[0] == "/usr/local/bin/kubectl":
                if "create" in command:
                    obj = json.loads(input_text)
                    if fail_chart and obj["kind"] == "HelmChart":
                        raise RuntimeError("simulated chart failure")
                    created.append(obj)
                elif "label" in command or "wait" in command:
                    pass
                else:
                    kind = command[command.index("get") + 1]
                    if kind == "nodes":
                        reads["nodes"] += 1
                        return json.dumps(dict(items=[] if reads["nodes"] == 1 else [node]))
                    if kind == "deployments":
                        reads["deployments"] += 1
                        return json.dumps(dict(items=[] if reads["deployments"] == 1 else [operator]))
                    return json.dumps({"node":node,"pvc":pvc,"deployment":operator,"crd":crd}[kind])
            else:
                raise AssertionError(command)
            return ""

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            home.mkdir()
            output = Path(directory) / "evidence"
            with patch.object(setup, "host_check", return_value=dict(gpus=["synthetic0","synthetic1"])), patch.object(setup.Path, "home", return_value=home), patch.object(setup, "run", side_effect=fake_run), patch.object(setup.time, "sleep"), patch.dict(setup.os.environ, {"HF_TOKEN":"synthetic-token"}):
                if fail_chart:
                    with self.assertRaisesRegex(RuntimeError,"chart failure"):
                        setup.install(output, with_hf_secret=True)
                else:
                    setup.install(output, with_hf_secret=True)
            receipt = json.loads((output / "setup_review.json").read_text())
            self.assertNotIn("synthetic-token", json.dumps(receipt))
            self.assertNotIn("synthetic-private-kubeconfig", json.dumps(receipt))
            mode = (home / ".kube/part7-config").stat().st_mode & 0o777
        return receipt, commands, created, mode

    def test_fresh_install_complete_control_flow(self):
        """Run installer through registration, labels, charts, PVC, validator and Secret."""
        receipt, commands, created, mode = self.run_install()
        self.assertTrue(receipt["success"])
        self.assertTrue(receipt["hub_secret_created"])
        self.assertEqual(mode, 0o600)
        self.assertEqual(receipt["cluster"]["allocatable"]["nvidia.com/gpu"], "2")
        label = next(i for i,c in enumerate(commands) if "label" in c)
        create = next(i for i,c in enumerate(commands) if "create" in c)
        self.assertLess(label, create)
        self.assertEqual([o["kind"] for o in created],
                         ["Namespace","HelmChart","HelmChart","PersistentVolume","PersistentVolumeClaim","Secret"])

    def test_partial_failure_retains_unsuccessful_receipt(self):
        """A chart failure must not create a Secret or report a successful bootstrap."""
        receipt, commands, created, mode = self.run_install(fail_chart=True)
        self.assertFalse(receipt["success"])
        self.assertNotIn("Secret", [o["kind"] for o in created])
