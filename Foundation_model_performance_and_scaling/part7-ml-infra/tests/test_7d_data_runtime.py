"""Optional native CPU ABI check using a separate, explicitly prepared interpreter."""
import json
import os
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
