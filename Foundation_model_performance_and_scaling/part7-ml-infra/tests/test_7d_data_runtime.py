"""Optional native CPU ABI check using a separate, explicitly prepared interpreter."""
import json
import os
import re
from pathlib import Path
import subprocess
import unittest


class DataRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("DATA_COMPAT_PYTHON"),"set DATA_COMPAT_PYTHON to the isolated NumPy/pandas/Arrow interpreter")
    def test_native_roundtrip_without_model_libraries(self):
        """Exercise real filesystem/Parquet conversion while excluding Ray/Torch imports.

        The interpreter is provisioned separately; this test performs no installs,
        network requests, model loads or GPU execution.
        """
        script=("import importlib,json,sys; "
                "m=importlib.import_module('infrastructure.7d_serve.runtime_check'); "
                "result=m.check_data_runtime(); "
                "assert 'torch' not in sys.modules and 'ray' not in sys.modules; "
                "print(json.dumps(result))")
        result=subprocess.run([os.environ["DATA_COMPAT_PYTHON"],"-c",script],
            cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        receipt=json.loads(result.stdout)
        self.assertEqual(receipt["round_trip_rows"],3)
        self.assertTrue(receipt["numpy"].startswith("1.26."))

    @unittest.skipUnless(os.environ.get("DATA_COMPAT_PYTHON"),"set DATA_COMPAT_PYTHON to an isolated interpreter with Ray core/boto3/data libraries")
    def test_ray_filesystem_registry_without_clients_or_models(self):
        """Import actual providers while forbidding cloud clients and network connections."""
        script=("import importlib,json,sys,socket; from unittest.mock import patch; "
                "import boto3; "
                "m=importlib.import_module('infrastructure.7d_serve.runtime_check'); "
                "guards=[patch('boto3.client',side_effect=AssertionError('No clients')), "
                "patch('boto3.session.Session',side_effect=AssertionError('No sessions')), "
                "patch('socket.socket.connect',side_effect=AssertionError('No network'))]; "
                "[guard.start() for guard in guards]; "
                "result=m.check_cloud_filesystems(); "
                "assert 'torch' not in sys.modules; "
                "print(json.dumps(result))")
        result=subprocess.run([os.environ["DATA_COMPAT_PYTHON"],"-c",script],
            cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        receipt=json.loads(result.stdout)
        self.assertEqual(len(receipt["providers"]),4)
        lock=(Path(__file__).resolve().parents[1]/"infrastructure/7d_serve/environment/requirements.lock").read_text()
        for package in ("boto3","botocore"):
            expected=re.search(r"^"+package+r"==([^\n]+)",lock,re.M).group(1)
            self.assertEqual(receipt[package],expected,"CPU fixture must match serving SDK pins")
