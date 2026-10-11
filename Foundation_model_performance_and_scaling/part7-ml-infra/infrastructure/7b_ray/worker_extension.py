"""Cloud-only model.external_lib entry: reuse 7A evidence and add resume audits."""
import os
from importlib import import_module

if os.environ.get("PART7_7B_AUDIT") != "1":
    raise RuntimeError("7B worker extension requires an explicitly enabled cloud run")
import_module("infrastructure.7a_verl.verl_diagnostics")
from .recovery_audit import install_worker_hooks

install_worker_hooks()
