"""Render a bounded RayService using native Ray Serve LLM, without API contacts."""
import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
training = import_module("infrastructure.7c_kuberay.render")
settings = import_module("infrastructure.7d_serve.config")


def render_service(image, node_name, *, config=None, namespace="part7", name="part7-7d",
                   pvc="part7-workspace", secret="part7-huggingface", pull_secret=None):
    """Return a digest-pinned RayService dictionary; no deployment/model loading.

    Inputs: serving-image digest, selected node, validated workload settings and
    existing storage/Secret names. Reuses tested 7C pod security/mount helpers;
    uses one CPU head plus one single-GPU serving worker and local-only exposure.
    Both pods stage the same immutable model through a locked init-container path.
    """
    config = (config or settings.ServingConfig()).validate()
    cluster = training.render_job(image, namespace, name, pvc, secret, node_name, pull_secret)["spec"]["rayClusterSpec"]
    cluster["rayVersion"] = settings.RAY_VERSION
    worker = cluster["workerGroupSpecs"][0]
    worker["groupName"] = "serving"
    worker["rayStartParams"]["num-cpus"] = "16"
    worker["rayStartParams"]["num-gpus"] = "1"
    for group in (cluster["headGroupSpec"], worker):
        template = group["template"]
        template["metadata"]["labels"]["part7-milestone"] = "7d"
        pod = template["spec"]
        container = pod["containers"][0]
        if group is worker:
            container["resources"] = dict(requests={"cpu":"16","memory":"64Gi","nvidia.com/gpu":1},
                                          limits={"cpu":"16","memory":"64Gi","nvidia.com/gpu":1})
        else:
            container["ports"].append(dict(name="serve", containerPort=8000))
        init_env = [entry for entry in container["env"] if entry["name"] != "NVIDIA_VISIBLE_DEVICES"]
        init_env.append(dict(name="NVIDIA_VISIBLE_DEVICES", value="void"))
        pod["initContainers"] = [dict(name="stage-model", image=image,
            command=["/opt/part7/.venv-7a/bin/python"], args=[
                "/opt/part7/infrastructure/7d_serve/prepare_model.py", "--cloud",
                "--repo", config.model_repo, "--revision", config.model_revision,
                "--output", config.snapshot_path()],
            env=init_env, securityContext=container["securityContext"],
            resources=dict(requests={"cpu":"1","memory":"2Gi"},limits={"cpu":"1","memory":"2Gi"}),
            volumeMounts=[dict(mount) for mount in container["volumeMounts"] if mount["name"] != "dshm"])]
        # After staging, native serving reads a complete local model without Hub calls.
        container["env"].extend([dict(name="HF_HUB_OFFLINE",value="1"),dict(name="TRANSFORMERS_OFFLINE",value="1")])
        for mount in container["volumeMounts"]:
            if mount["name"] == "workspace":
                mount["readOnly"] = True
    serve_config = dict(proxy_location="HeadOnly", http_options=dict(host="0.0.0.0",port=8000),
        applications=[dict(name="part7-llm", route_prefix="/",
                           import_path="ray.serve.llm:build_openai_app",
                           args=dict(llm_configs=[config.llm_config()]))])
    return dict(apiVersion="ray.io/v1", kind="RayService",
                metadata=dict(name=name,namespace=namespace,labels={"part7-milestone":"7d"},
                    annotations={"part7/model-repo":config.model_repo,"part7/model-revision":config.model_revision,
                                 "part7/status":"cpu-prepared-cloud-verification-pending"}),
                spec=dict(serveConfigV2=json.dumps(serve_config), rayClusterConfig=cluster))


def main():
    """Parse render options and exclusively write JSON/YAML-compatible manifest.

    Requires an actual serving-image digest; no placeholder is treated as verified.
    Does not call Kubernetes, Ray, Hugging Face or load model weights.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--name", default="part7-7d")
    parser.add_argument("--namespace", default="part7")
    parser.add_argument("--pvc", default="part7-workspace")
    parser.add_argument("--secret", default="part7-huggingface")
    parser.add_argument("--pull-secret")
    parser.add_argument("--model-repo", default=settings.DEFAULT_REPO)
    parser.add_argument("--model-revision", default=settings.DEFAULT_REVISION)
    parser.add_argument("--model-id", default="part7-math-grpo")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = settings.ServingConfig(args.model_repo,args.model_revision,args.model_id,
                                   args.max_model_len,args.max_num_seqs,args.gpu_memory_utilization)
    document = render_service(args.image,args.node_name,config=config,namespace=args.namespace,
                              name=args.name,pvc=args.pvc,secret=args.secret,pull_secret=args.pull_secret)
    serialized = json.dumps(document,indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(serialized)
    else:
        print(serialized,end="")


if __name__ == "__main__":
    main()
