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
            'set -euo pipefail; source "$1"; select_7a_cuda; printf "%s\\n" "$CUDA_HOME"; nvcc --version',
            "test", str(HELPER), str(root)], env=environment, capture_output=True, text=True)

    def test_reuses_compatible_toolkit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            self.fake_toolkit(existing, "13.0")
            result = self.select(root, existing)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines()[0], str(existing))
            self.assertFalse((root / ".cuda-7a").exists())

    def test_rejects_cuda12_without_downloading_or_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "image"
            self.fake_toolkit(existing, "12.8")
            result = self.select(root, existing)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CUDA 13", result.stderr)
            self.assertFalse((root / ".cuda-7a").exists())
            self.assertIn("12.8", (existing / "bin/nvcc").read_text())
