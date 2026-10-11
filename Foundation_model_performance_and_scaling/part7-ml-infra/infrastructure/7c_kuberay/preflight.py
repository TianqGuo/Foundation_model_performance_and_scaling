"""Cloud-only GPU RayJob preflight; no model loading/training occurs.

Probe two GPU actors on KubeRay's existing cluster and persist resource identities
on the mounted PVC. Training/resume integration is a subsequent 7C increment.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def check_placement(nodes, probes):
    """Validate two GPU actors share one live two-GPU worker node.

    Inputs: Ray node records and actor probe dictionaries. Returns None; raises
    ValueError for wrong resource totals, split placement, duplicate GPUs or no
    CUDA access. Logical assignment is checked alongside worker pod identity.
    """
    alive = [node for node in nodes if node["Alive"]]
    if sum(node["Resources"].get("GPU", 0) for node in alive) != 2:
        raise ValueError("Initial 7C topology requires exactly two Ray GPUs")
    if len(probes) != 2 or len({probe["node_id"] for probe in probes}) != 1:
        raise ValueError("Require two probes on the same GPU worker")
    if any(len(probe["gpu_ids"]) != 1 for probe in probes):
        raise ValueError("Each probe must hold exactly one GPU")
    target = [node for node in alive if node["NodeID"] == probes[0]["node_id"]]
    if len(target) != 1 or target[0]["Resources"].get("GPU", 0) != 2:
        raise ValueError("Probes are not placed on the live two-GPU worker")
    if len({probe["pod_uid"] for probe in probes}) != 1:
        raise ValueError("Initial probes must share one GPU worker pod")
    ids = [gpu for probe in probes for gpu in probe["gpu_ids"]]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("GPU probes did not receive distinct whole GPUs")
    if any(not probe["cuda_available"] or probe["cuda_device_count"] != 1 or not probe["pod_uid"] or probe["cuda_probe_value"] != 1.0 for probe in probes):
        raise ValueError("GPU device exposure or pod identity is missing")


def main():
    """Connect to existing Ray, validate Secret/PVC/GPU access and write evidence.

    Input: explicit --cloud and fresh PVC output directory. Returns None. Only
    GPU actors import Torch; the CPU head reads a mounted token to verify Hub
    account access without printing/exporting it. No login, repository creation
    or model download is performed. Failures persist status and propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not os.environ.get("POD_UID") or not os.environ.get("NODE_NAME"):
        raise RuntimeError("Run inside the prepared Kubernetes pod, not a standalone VM")
    if output.exists():
        raise FileExistsError("Use a fresh PVC evidence path")
    if not Path("/run/secrets/huggingface/token").is_file():
        raise FileNotFoundError("Mounted Hub Secret key token is missing")
    output.mkdir(parents=True)
    status = dict(success=False, scope="gpu-preflight-only")
    started = time.monotonic()
    import ray
    actors = []
    try:
        if ray.__version__ != "2.49.2":
            raise RuntimeError("Unexpected Ray version in image")
        # Never start a private Ray cluster inside a KubeRay job.
        ray.init(address="auto")
        from huggingface_hub import HfApi
        token = Path("/run/secrets/huggingface/token").read_text().strip()
        account = HfApi(token=token).whoami()["name"]
        del token

        @ray.remote(num_cpus=1, num_gpus=1)
        class GPUProbe:
            """Hold one GPU reservation while recording actual CUDA/pod identity."""
            def inspect(self):
                """Return GPU placement, dependency identity and memory without models."""
                import torch
                from cutlass.cute.core import ThrMma
                from flash_attn.cute import interface
                from verl.workers.rollout.vllm_rollout import ServerAdapter
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError("GPU actor must expose exactly one working CUDA device")
                probe_value = torch.ones(1, device="cuda").sum().item()
                return dict(cuda_probe_value=probe_value, node_id=ray.get_runtime_context().get_node_id(), gpu_ids=ray.get_gpu_ids(),
                            pod_uid=os.environ.get("POD_UID"), pod_name=os.environ.get("POD_NAME"),
                            physical_node=os.environ.get("NODE_NAME"),
                            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                            cuda_available=torch.cuda.is_available(), cuda_device_count=torch.cuda.device_count(),
                            torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                            driver_inventory=subprocess.check_output(["nvidia-smi", "-L"], text=True),
                            shared_memory_bytes=shutil.disk_usage("/dev/shm").total)

        actors = [GPUProbe.remote(), GPUProbe.remote()]
        probes = ray.get([actor.inspect.remote() for actor in actors], timeout=180)
        nodes = ray.nodes()
        check_placement(nodes, probes)
        report = dict(scope="gpu-preflight-only", nodes=nodes, probes=probes, hub_account=account,
                      head_pod_uid=os.environ["POD_UID"], physical_node=os.environ["NODE_NAME"],
                      packages={dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
                      seconds=time.monotonic() - started)
        (output / "preflight.json").write_text(json.dumps(report, indent=2))
        status["success"] = True
        print("7C GPU access preflight passed; training/recovery verification is still pending.", flush=True)
    finally:
        try:
            for actor in actors:
                try:
                    ray.kill(actor)
                except ray.exceptions.RayError as error:
                    status.update(success=False, cleanup_error_type=type(error).__name__)
        finally:
            ray.shutdown()
            (output / "job_status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
