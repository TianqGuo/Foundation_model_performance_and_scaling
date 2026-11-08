"""Model-free native import/API gate for the isolated 7E cloud image."""
import argparse
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform

CORE_VERSIONS = {"ray": "2.59.0", "torch": "2.13.0", "transformers": "5.20.2",
                 "huggingface_hub": "2.3.0", "numpy": "2.3.5", "pyarrow": "26.0.0", "pandas": "3.1.0"}


def validate_versions(versions):
    """Return version receipt after exact core checks, allowing explicit CUDA suffixes."""
    for name, expected in CORE_VERSIONS.items():
        if versions.get(name, "").split("+", 1)[0] != expected:
            raise ValueError(f"Unexpected 7E package version: {name}")
    return versions


def check_runtime():
    """Import native Train/FSDP2/DCP/data/model APIs without model/runtime creation.

    Cloud image-build gate only; no process group, CUDA tensor, download or model
    instance is created. Checks exact versions, CUDA-13 build, Python 3.12, native
    API keyword binding and a tiny in-memory data round trip. Returns receipt;
    raises on dependency/API incompatibility. This is not GPU training evidence.
    """
    if os.environ.get("RAY_TRAIN_V2_ENABLED", "1") != "1":
        raise ValueError("7E requires the reviewed Ray Train V2 API")
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError("The isolated 7E image requires Python 3.12")
    versions = validate_versions({name: metadata.version(name) for name in CORE_VERSIONS})
    import torch
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.checkpoint import save, load
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from ray.train import ScalingConfig, RunConfig, FailureConfig, report
    from ray.train.torch import TorchTrainer, TorchConfig, get_device
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not torch.version.cuda or torch.version.cuda.split(".")[0] != "13":
        raise ValueError("Expected a CUDA 13 PyTorch build")
    inspect.signature(fully_shard).bind(object(), mesh=None, reshard_after_forward=True,
                                       mp_policy=MixedPrecisionPolicy())
    inspect.signature(get_state_dict).bind(object(), object())
    inspect.signature(set_state_dict).bind(object(), object(), model_state_dict={}, optim_state_dict={})
    inspect.signature(save).bind({}, checkpoint_id="/tmp/unused-7e-checkpoint")
    inspect.signature(load).bind({}, checkpoint_id="/tmp/unused-7e-checkpoint")
    scaling = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"CPU":4,"GPU":1},
                            placement_strategy="STRICT_PACK")
    run = RunConfig(name="native-api-check", storage_path="/tmp/unused-7e-storage",
                    failure_config=FailureConfig(max_failures=0, controller_failure_limit=0, max_preemption_failures=0))
    inspect.signature(TorchTrainer).bind(train_loop_per_worker=object(), train_loop_config={},
        scaling_config=scaling, run_config=run, torch_config=TorchConfig(backend="nccl"))
    inspect.signature(report).bind({"loss":1.0})
    if not callable(get_device) or not hasattr(AutoTokenizer, "from_pretrained") or not hasattr(AutoModelForCausalLM, "from_pretrained"):
        raise ValueError("Native training/tokenizer/model API missing")
    table = pa.Table.from_pandas(pd.DataFrame({"x":np.array([1,2,3])}), preserve_index=False)
    stream = pa.BufferOutputStream()
    pq.write_table(table, stream)
    if pq.read_table(pa.BufferReader(stream.getvalue())).to_pydict() != {"x":[1,2,3]}:
        raise ValueError("7E data-library round trip failed")
    source = Path(__file__).parent
    lock = source/"environment/requirements.lock"
    source_hashes = {name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                    for name in ("config.py","worker.py","run_7e.py","runtime_check.py")}
    return dict(packages=versions, cuda_runtime=torch.version.cuda, python=platform.python_version(),
                dependency_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
                source_hashes=source_hashes,
                round_trip_rows=3, ray_train_api="v2", fsdp2_api=True, distributed_checkpoint_api=True,
                model_loaded=False, gpu_execution_verified=False, recovery_verified=False)


def main():
    """Require explicit cloud-container use before importing the candidate GPU stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", action="store_true", required=True)
    parser.parse_args()
    if not Path("/.dockerenv").exists() and not os.environ.get("POD_UID"):
        raise RuntimeError("Native gate must run in the isolated cloud image")
    print(json.dumps(check_runtime(), indent=2))


if __name__ == "__main__":
    main()
