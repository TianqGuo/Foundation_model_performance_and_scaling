"""CPU tests for image publication inputs and receipts; no Docker or network calls."""
from importlib import import_module
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
image = import_module("infrastructure.7c_kuberay.image_publish")


class ImageTests(unittest.TestCase):
    def test_repository_scope(self):
        """Accept the agreed Hub namespace and reject registry/option ambiguity."""
        self.assertEqual(image.validate_repository("sclion/part7-training"), "sclion")
        for value in ("--help", "ghcr.io/sclion/image", "sclion/image:latest", "sclion"):
            with self.assertRaises(ValueError):
                image.validate_repository(value)

    def test_published_digest_is_not_local_image_id(self):
        """Require an actual push receipt, with no configuration-ID fallback."""
        digest = "sha256:" + "a" * 64
        self.assertEqual(image.publish_digest("tag: digest: " + digest + " size: 123"), digest)
        for output in (digest, "", "digest: " + digest + "\ndigest: sha256:" + "b" * 64):
            with self.assertRaises(ValueError):
                image.publish_digest(output)

    def test_check_does_not_authenticate_build_or_publish(self):
        """Preview with publication flag still has no Docker/network/token side effects."""
        with patch("sys.argv", ["image", "--check", "--publish"]), patch.object(image, "execute") as execute, contextlib.redirect_stdout(io.StringIO()) as output:
            image.main()
        execute.assert_not_called()
        self.assertIn("sclion/part7-training", output.getvalue())

    def test_unprivileged_auth_cleanup_after_failure(self):
        """Remove temporary credentials even when the wrapped build raises."""
        config = None
        with self.assertRaisesRegex(RuntimeError, "build failed"):
            with image.temporary_docker_config(["docker"]) as config:
                (Path(config) / "config.json").write_text("synthetic")
                raise RuntimeError("build failed")
        self.assertFalse(Path(config).exists())

    def test_sudo_cleanup_uses_only_new_directory(self):
        """Root-owned buildx cleanup must target the allocated auth directory only."""
        with tempfile.TemporaryDirectory() as folder, patch.object(image.tempfile, "mkdtemp", return_value=folder), patch.object(image.subprocess, "run") as run:
            with image.temporary_docker_config(["sudo", "-n", "docker"]) as config:
                self.assertEqual(config, folder)
            run.assert_called_once_with(["sudo", "-n", "rm", "-rf", "--", folder], check=True, timeout=60)

    def test_image_tool_check_runs_before_publish_without_gpu_or_network(self):
        """Check wget in the actual candidate image; missing tools must fail the action."""
        with patch.object(image.subprocess,"run") as run:
            image.verify_image_tools(["docker"], "synthetic:image")
        command = run.call_args.args[0]
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", command)
        self.assertIn("none", command)
        self.assertIn("wget", command[-1])
        self.assertIn("pwd.getpwuid", command[-1])
        self.assertIn("torch._inductor.codecache", command[-1])
        self.assertIn("TemporaryFile", command[-1])
        self.assertTrue(run.call_args.kwargs["check"])
        with patch.object(image.subprocess,"run", side_effect=image.subprocess.CalledProcessError(1, command)):
            with self.assertRaises(image.subprocess.CalledProcessError):
                image.verify_image_tools(["docker"], "synthetic:image")
