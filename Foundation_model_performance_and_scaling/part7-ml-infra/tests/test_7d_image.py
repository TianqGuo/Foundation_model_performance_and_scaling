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
    def test_gpu_constraints_preserve_training_abi(self):
        """Serving Ray can change while CUDA/PyTorch/tokenizer constraints stay pinned."""
        folder=module.ROOT/"infrastructure/7d_serve/environment"
        training=(module.ROOT/"infrastructure/7a_verl/environment/requirements.lock").read_text().splitlines()
        pins=[line for line in (folder/"gpu-constraints.txt").read_text().splitlines() if line and not line.startswith("#")]
        self.assertEqual(len(pins),8)
        for pin in pins:self.assertIn(pin,training)
        lock=(folder/"requirements.lock").read_text()
        self.assertIn("=="+module.control.settings.RAY_VERSION,lock)
        self.assertNotIn("ray==2.49.2",lock)
        self.assertNotIn("7a_verl/environment/requirements.lock",(folder/"requirements.in").read_text())

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
