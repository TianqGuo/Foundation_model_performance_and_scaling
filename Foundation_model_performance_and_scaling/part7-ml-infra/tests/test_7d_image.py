"""Metadata and mocked publication gates; no Docker, network or model execution."""
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
module=import_module("infrastructure.7d_serve.image_publish")


class ServingImageTests(unittest.TestCase):
    def test_explicit_core_pins_and_independent_base(self):
        """Prevent silent version downgrades or reuse of the old training GPU ABI."""
        import re
        folder=module.ROOT/"infrastructure/7d_serve/environment"
        lock=(folder/"requirements.lock").read_text()
        for name,expected in module.control.settings.CORE_VERSIONS.items():
            match=re.search(r"^"+name+r"(?:\[.*\])?==([^\n]+)",lock,re.M)
            self.assertIsNotNone(match,name)
            self.assertEqual(match.group(1),expected)
        source=(folder/"requirements.in").read_text()
        self.assertNotIn("gpu-constraints.txt",source)
        self.assertIn("vllm[audio]=="+module.control.settings.VLLM_VERSION,source)
        dockerfile=(module.ROOT/"infrastructure/7d_serve/Dockerfile").read_text()
        self.assertIn(module.control.settings.BASE_IMAGE,dockerfile)
        self.assertNotIn("sclion/part7-training@",dockerfile)

    def test_failed_import_build_never_pushes(self):
        """The build/import subprocess must finish successfully before publication."""
        calls=[]
        def run(args,**kwargs):
            """Fail the synthetic build, recording whether push was attempted."""
            calls.append(args)
            if args[0]=="bash":raise subprocess.CalledProcessError(1,args)
            return SimpleNamespace(returncode=0,stdout="",stderr="")
        with tempfile.TemporaryDirectory() as folder,patch.object(module.control,"require_cloud_host"),patch.object(
                module.shared,"temporary_docker_config",return_value=nullcontext("/tmp/test-config")),patch.dict(
                module.os.environ,{"DOCKERHUB_TOKEN":"synthetic-not-a-real-token"}),patch.object(module.subprocess,"run",side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):module.execute("sclion/part7-training",Path(folder)/"new",True)
        self.assertFalse(any("push" in call for call in calls))

    def test_publish_uses_separate_tag_and_registry_digest(self):
        """A passing build permits upload; receipt uses push digest plus verified pull."""
        calls=[]
        def run(args,**kwargs):
            """Simulate registry commands without contacting Docker or Hub."""
            calls.append((args,kwargs))
            return SimpleNamespace(returncode=0,stdout="digest: sha256:"+"a"*64 if "push" in args else "",stderr="")
        with tempfile.TemporaryDirectory() as folder,patch.object(module.control,"require_cloud_host"),patch.object(
                module.shared,"temporary_docker_config",return_value=nullcontext("/tmp/test-config")),patch.dict(
                module.os.environ,{"DOCKERHUB_TOKEN":"synthetic-not-a-real-token"}),patch.object(module.subprocess,"run",side_effect=run):
            output=Path(folder)/"new"
            receipt=module.execute("sclion/part7-training",output,True)
            self.assertTrue(receipt["published"])
            self.assertIn(":7d-",receipt["tag"])
            self.assertEqual(receipt["image"],"sclion/part7-training@sha256:"+"a"*64)
            self.assertFalse(receipt["e2e_verified"])
            self.assertNotIn("synthetic-not-a-real-token",(output/"image_publish.json").read_text())
        self.assertTrue(any("pull" in call[0] for call in calls))

    def test_build_only_never_authenticates_or_uploads(self):
        """Omitting --publish leaves only a checked local image and requires no token."""
        calls=[]
        def run(args,**kwargs):
            """Record local-only synthetic Docker commands."""
            calls.append(args);return SimpleNamespace(returncode=0,stdout="",stderr="")
        with tempfile.TemporaryDirectory() as folder,patch.object(module.control,"require_cloud_host"),patch.object(
                module.shared,"temporary_docker_config",return_value=nullcontext("/tmp/test-config")),patch.object(
                module.subprocess,"run",side_effect=run),patch.object(module.getpass,"getpass",side_effect=AssertionError("No login needed")):
            receipt=module.execute("sclion/part7-training",Path(folder)/"new",False)
            self.assertFalse(receipt["published"])
        self.assertFalse(any("login" in call or "push" in call for call in calls))


class BuildCommandRegressionTests(unittest.TestCase):
    def test_actual_sync_cli_from_dockerfile(self):
        """Parse the Docker sync invocation using real uv against an empty temporary venv.

        Offline dry-run exercises CLI syntax without installing packages, loading
        checkpoints or executing Docker. Catches install-only flags copied into sync.
        """
        import re
        import os
        import shlex
        import shutil
        import sys
        executable=shutil.which("uv")
        if not executable:self.skipTest("uv unavailable for actual CLI check")
        source=(module.ROOT/"infrastructure/7d_serve/Dockerfile").read_text().replace("\\\n", " ")
        args=shlex.split(re.search(r"uv pip sync (.*?)&&",source,re.S).group(1))
        with tempfile.TemporaryDirectory() as folder:
            env=Path(folder)/"venv"
            requirements=Path(folder)/"empty.txt"
            requirements.write_text("")
            subprocess.run([executable,"venv","--cache-dir",str(Path(folder)/"cache"),"--offline","--python",sys.executable,str(env)],
                           check=True,capture_output=True,text=True,timeout=30)
            args=[str(env/"bin/python") if arg in ("/opt/part7/.venv-7a/bin/python","/usr/bin/python3.12") else
                  str(requirements) if arg=="infrastructure/7d_serve/environment/requirements.lock" else arg for arg in args]
            result=subprocess.run([executable,"pip","sync","--cache-dir",str(Path(folder)/"cache"),"--offline","--dry-run",*args],
                                  env=dict(os.environ,UV_OVERRIDE=""),capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_wsl_rejected_before_gpu_or_docker_commands(self):
        """A systemd-enabled WSL workstation must not pass as the cloud GPU VM."""
        with patch.object(module.control.platform,"release",return_value="6.6.87.2-microsoft-standard-WSL2"),patch.object(
                module.control.subprocess,"run") as run,patch.object(module.control.subprocess,"check_output") as check:
            with self.assertRaisesRegex(ValueError,"WSL"):
                module.control.require_cloud_host()
            run.assert_not_called()
            check.assert_not_called()
