"""CPU-only selection checks with fake nvcc; no downloads or GPU operations."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "infrastructure/7a_verl/environment/cuda_7a.sh"


class CudaSelectionTests(unittest.TestCase):
    def fake_toolkit(self, root, version):
        (root / "bin").mkdir(parents=True)
        executable = root / "bin/nvcc"
        executable.write_text(f'#!/bin/sh\necho "Cuda compilation tools, release {version}"\n')
        executable.chmod(0o755)

    def select(self, root, existing):
        environment = dict(os.environ, CUDA_HOME=str(existing))
        return subprocess.run(["bash", "-c",
            'set -euo pipefail; source "$1"; select_7a_cuda "$2" reuse; printf "%s\\n" "$CUDA_HOME"; nvcc --version',
            "test", str(HELPER), str(root)], env=environment, capture_output=True, text=True)

    def test_reuses_compatible_toolkit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            self.fake_toolkit(existing, "12.8")
            result = self.select(root, existing)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines()[0], str(existing))
            self.assertFalse((root / ".cuda-7a").exists())

    def test_selects_separate_toolkit_instead_of_image_cuda13(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing, separate = root / "image", root / ".cuda-7a/12.8"
            self.fake_toolkit(existing, "13.0")
            self.fake_toolkit(separate, "12.8")
            (separate / ".setup_complete").touch()
            result = self.select(root, existing)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines()[0], str(separate))
            self.assertIn("release 12.8", result.stdout)
            self.assertIn("13.0", (existing / "bin/nvcc").read_text())
