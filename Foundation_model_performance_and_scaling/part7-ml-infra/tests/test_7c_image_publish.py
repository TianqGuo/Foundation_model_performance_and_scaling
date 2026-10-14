"""CPU tests for image publication inputs and receipts; no Docker or network calls."""
from importlib import import_module
import contextlib
import io
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
