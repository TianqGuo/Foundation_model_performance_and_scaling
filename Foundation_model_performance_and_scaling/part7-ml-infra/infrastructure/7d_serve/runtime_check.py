"""Cloud-container, model-free compatibility gate for the candidate serving stack."""
import argparse
import ast
import importlib
from importlib.metadata import version, distributions
import inspect
import json
import os
import pwd
import getpass
import hashlib
import shutil
import tempfile
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
settings = importlib.import_module("infrastructure.7d_serve.config")


def validate_constructor_calls(source, constructor, name="AsyncLLM"):
    """Bind explicit constructor calls in source against the installed signature.

    Inputs: adapter Python source, installed constructor and its source alias.
    Recognizes direct calls and explicit single-step aliases. Returns checked
    call count. No calls execute; dynamic **kwargs are rejected
    rather than treated as proof of compatibility. Missing calls or incompatible
    arguments raise ValueError/TypeError. This checks signatures, not semantics.
    """
    signature = inspect.signature(constructor)
    tree = ast.parse(source)
    aliases = {name}
    # Ray 2.59 invokes engine_cls after engine_cls = AsyncLLM. A dynamic name
    # without that explicit assignment is not accepted as evidence.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == name:
            aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in aliases:
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(key.arg is None for key in node.keywords):
            raise ValueError("Dynamic constructor arguments require manual review")
        signature.bind(*[None for arg in node.args], **{key.arg: None for key in node.keywords})
        checked += 1
    if not checked:
        raise ValueError("No constructor calls found; adapter changed, review required")
    return checked


def check_data_runtime():
    """Exercise native NumPy/pandas/Arrow filesystem and Parquet interoperability.

    No inputs. Returns installed versions and a three-row round-trip count; raises
    on import/ABI/conversion failure. Uses an in-memory buffer, not models, datasets,
    credentials, Ray or GPU libraries. Suitable before expensive image export.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.fs as fs
    import pyarrow.parquet as pq
    values=np.arange(3,dtype=np.int64)
    table=pa.Table.from_pandas(pd.DataFrame({"value":values}),preserve_index=False)
    sink=pa.BufferOutputStream()
    pq.write_table(table,sink)
    restored=pq.read_table(pa.BufferReader(sink.getvalue())).to_pandas()["value"].to_numpy()
    if not np.array_equal(restored,values):
        raise ValueError("Native data-library round trip changed values")
    fs.LocalFileSystem()
    return dict(numpy=np.__version__,pandas=pd.__version__,pyarrow=pa.__version__,round_trip_rows=len(values))


def check_cloud_filesystems():
    """Import the pinned Ray cloud filesystem registry without creating clients.

    No inputs; returns SDK versions and provider class names. Ray eagerly imports
    its S3 provider, so boto3/botocore must exist even for a local staged model.
    Does not authenticate, access S3/GCS/Azure, start Ray or import model libraries.
    Missing registry/SDK imports raise directly, before model-framework imports.
    """
    import boto3
    import botocore
    from ray.llm._internal.common.utils.cloud_filesystem import (
        AzureFileSystem, GCSFileSystem, S3FileSystem, PyArrowFileSystem)
    return dict(boto3=boto3.__version__,botocore=botocore.__version__,
                providers=[provider.__name__ for provider in
                    (AzureFileSystem,GCSFileSystem,S3FileSystem,PyArrowFileSystem)])


def validate_serve_config(document):
    """Validate a native Serve deployment dictionary and return its schema instance.

    Uses the public model constructor, shared by Pydantic v1/v2. Earlier Ray
    schemas used v1 compatibility and lacked the v2-only model_validate method;
    constructor validation also works with the reviewed newer schema. No app/model is built;
    invalid configuration raises the native validation exception.
    """
    from ray.serve.schema import ServeDeploySchema
    return ServeDeploySchema(**document)


def validate_core_versions(installed):
    """Require reviewed core versions, allowing an explicit CUDA local-version suffix.

    Input: installed package-version mapping. Returns None or raises on drift,
    including missing vLLM pins. No package imports or installation occurs.
    Full wheel/CUDA identities remain in the caller's evidence receipt.
    """
    for name,expected in settings.CORE_VERSIONS.items():
        if installed.get(name, "").split("+")[0] != expected:
            raise ValueError("Installed "+name+" differs from reviewed serving pin")


def check_hub_api():
    """Check model-staging Hub API signatures without sending requests or loading assets.

    No inputs; returns Hub version. Binds the exact snapshot/model-info arguments
    and checks the token property used by our staging adapter. Fails on API drift;
    the synthetic token is never transmitted or stored.
    """
    from huggingface_hub import HfApi,ModelInfo,snapshot_download
    api=HfApi(token="part7-signature-only")
    if ModelInfo(id="owner/model",sha="a"*40).sha != "a"*40:
        raise ValueError("Hub model revision field changed")
    inspect.signature(api.model_info).bind("owner/model",revision="a"*40)
    inspect.signature(snapshot_download).bind(repo_id="owner/model",revision="a"*40,
        token=api.token,local_dir="/unused",allow_patterns=["*.json"])
    return dict(huggingface_hub=version("huggingface-hub"))


def require_payload_fields(expected, actual, path=""):
    """Require request fields to survive schema validation while allowing added defaults.

    Inputs: expected/actual JSON values and diagnostic field path. Recurses through
    dictionaries; raises on missing/changed values. Additional framework defaults
    (e.g. continuous_usage_stats=False) are valid and must not cause false failures.
    """
    if isinstance(expected,dict):
        if not isinstance(actual,dict):
            raise ValueError("Completion protocol changed request field: "+path)
        for name,value in expected.items():
            require_payload_fields(value,actual.get(name),path+"."+name if path else name)
    elif actual != expected:
        raise ValueError("Completion protocol changed request field: "+path)


def check_completion_protocol():
    """Validate actual benchmark request fields against the pinned Ray completion schema.

    Returns True after constructor validation and round-trip field checks. No HTTP,
    generation or tokenizer load; prevents flags being silently ignored after upgrades.
    """
    from ray.llm._internal.serve.core.configs.openai_api_models import CompletionRequest
    requests=importlib.import_module("infrastructure.7d_serve.benchmark")
    payload=requests.completion_payload("part7-math-grpo","raw prompt",8)
    request=CompletionRequest(**payload)
    restored=request.model_dump() if hasattr(request,"model_dump") else request.dict()
    require_payload_fields(payload,restored)
    return True


def check_runtime():
    """Import native Serve/vLLM APIs and validate configuration without loading weights.

    Runs only in the cloud serving image. Returns package identities/check counts;
    import, pin, schema or signature failures abort the image gate. Does not start
    Ray, build a Serve application, download assets or construct an inference engine.
    Actual HTTP/GPU execution remains a separate E2E requirement.
    """
    if os.getuid() != 1000 or os.getgid() != 1000:
        raise ValueError("Expected nonroot UID/GID 1000")
    pwd.getpwuid(os.getuid())
    getpass.getuser()
    for tool in ("bash", "wget", "tar", "git", "ray", "python"):
        if not shutil.which(tool):
            raise ValueError("Missing container tool: " + tool)
    with tempfile.TemporaryFile(dir=os.environ["HOME"]):
        pass
    data_runtime = check_data_runtime()
    cloud_filesystems = check_cloud_filesystems()
    hub_api = check_hub_api()
    import torch._inductor.codecache
    installed = {name: version(name) for name in ("ray", "vllm", "torch", "transformers")}
    validate_core_versions(installed)
    if str(torch.version.cuda).split(".")[0] != "13":
        raise ValueError("Serving candidate requires CUDA 13 Torch wheels")
    from ray.serve.llm import LLMConfig, build_openai_app
    from vllm.v1.engine.async_llm import AsyncLLM
    adapter = importlib.import_module("ray.llm._internal.serve.engines.vllm.vllm_engine")
    from vllm.v1.executor.abstract import Executor
    from vllm.entrypoints.openai.cli_args import FrontendArgs
    from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger
    config = settings.ServingConfig()
    LLMConfig.model_validate(config.llm_config())
    validate_serve_config(dict(applications=[dict(
        name="part7-llm", route_prefix="/", import_path="ray.serve.llm:build_openai_app",
        args=dict(llm_configs=[config.llm_config()]))]))
    if not callable(build_openai_app):
        raise TypeError("Serve LLM builder is not callable")
    calls = validate_constructor_calls(inspect.getsource(adapter), AsyncLLM)
    completion_protocol = check_completion_protocol()
    lock_hash=hashlib.sha256((Path(__file__).parent/"environment/requirements.lock").read_bytes()).hexdigest()
    package_versions={dist.metadata["Name"]:dist.version for dist in distributions() if dist.metadata.get("Name")}
    return dict(packages=installed, installed_packages=dict(sorted(package_versions.items())),
                dependency_lock_sha256=lock_hash,data_runtime=data_runtime, cloud_filesystems=cloud_filesystems,
                constructor_calls_checked=calls, hub_api=hub_api, completion_protocol=completion_protocol,
                cuda_runtime=torch.version.cuda, python=sys.version.split()[0],
                model_loaded=False, gpu_execution_verified=False, http_verified=False)


def main():
    """Require explicit cloud-container mode; print a receipt only when all gates pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", required=True, action="store_true")
    parser.parse_args()
    if not Path("/.dockerenv").exists() or not Path("/opt/part7").is_dir():
        raise ValueError("Run inside the cloud serving image")
    print(json.dumps(check_runtime(), indent=2))


if __name__ == "__main__":
    main()
