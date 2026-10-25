"""Compose bounded cloud serving validation; image compatibility is a mandatory gate."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
platform=import_module("infrastructure.7c_kuberay.e2e")
control=import_module("infrastructure.7d_serve.service_control")
workload=import_module("workloads.math_workload")
SOURCE_FILES=("config.py","render.py","prepare_model.py","runtime_check.py","benchmark.py",
              "environment/requirements.lock","Dockerfile")


def prepare_prompts(part5, output):
    """Verify accepted Part 5 inputs and save the same 32 raw evaluation prompts.

    Inputs: cloned Part 5 root and fresh output directory. Uses its hash-verified
    selection helper (seed 12345), existing MATH adapter and R1 template. Returns
    prompt strings plus a provenance receipt. No Parquet/model/GPU dependencies.
    Output contains prompt text and indices, but no solutions or generated text.
    """
    platform.verify_references(part5)
    template=(part5/platform.INPUT_HASHES["prompt"][0]).read_text()
    examples=workload.read_examples(part5/platform.INPUT_HASHES["validation"][0])
    helper=workload.load_reference(str(part5/platform.INPUT_HASHES["selection_helper"][0]))
    indices=helper.evaluation_indices(len(examples),32,12345)
    if len(indices)!=32:
        raise ValueError("Expected the accepted 32 evaluation examples")
    rows=[dict(source_index=index,prompt=workload.make_row(examples[index],index,"validation",template)["prompt"][0]["content"])
          for index in indices]
    output.mkdir(parents=True,exist_ok=False)
    raw="".join(json.dumps(row)+"\n" for row in rows)
    (output/"prompts.jsonl").write_text(raw)
    receipt=dict(selection=dict(seed=12345,indices=indices,count=32),
        inputs={name:dict(relative=relative,sha256=digest) for name,(relative,digest) in platform.INPUT_HASHES.items()},
        prompts_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(workload.__file__).read_bytes()).hexdigest(),
        semantics="raw R1 prompt; no chat template; serving load measurement, not accuracy")
    control.save_json(output/"provenance.json",receipt)
    return [row["prompt"] for row in rows],receipt


def image_gate(image, output):
    """Pull/check an immutable serving image on cloud before cluster mutations.

    Inputs: validated registry digest and owned evidence path. Runs model-free native
    import checks without network/GPU visibility inside the container. Requires Ray
    version and pod-used source hashes to match this checkout; records stdout/stderr.
    Fails on stale source, failed imports or malformed receipt. No image publication.
    """
    control.renderer.training.pinned_image(image)
    docker=["docker"]
    if subprocess.run(["docker","info"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        docker=["sudo","-n","docker"]
    with (output/"image_pull.log").open("w") as stream:
        subprocess.run([*docker,"pull",image],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)
    script=("import importlib,json,hashlib,pathlib; "
        "m=importlib.import_module('infrastructure.7d_serve.runtime_check'); r=m.check_runtime(); "
        "r['source_hashes']={name:hashlib.sha256((pathlib.Path(m.__file__).parent/name).read_bytes()).hexdigest() "
        "for name in "+repr(SOURCE_FILES)+"}; print('PART7_RECEIPT='+json.dumps(r))")
    result=subprocess.run([*docker,"run","--rm","--network","none","--env","NVIDIA_VISIBLE_DEVICES=void",
        "--entrypoint","python",image,"-c",script],text=True,capture_output=True,timeout=180)
    (output/"image_gate.log").write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError("Serving image imports failed; inspect image_gate.log before deployment")
    receipts=[line.removeprefix("PART7_RECEIPT=") for line in result.stdout.splitlines() if line.startswith("PART7_RECEIPT=")]
    if len(receipts)!=1:
        raise ValueError("Expected one native image-check receipt")
    receipt=json.loads(receipts[0])
    expected={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    if receipt.get("packages",{}).get("ray")!=control.settings.RAY_VERSION or receipt.get("source_hashes")!=expected:
        raise ValueError("Image Ray/source differs from checkout; rebuild the serving image")
    control.save_json(output/"image_gate.json",receipt)
    return receipt


def execute(args):
    """Run image gate, native setup/reuse, prompt preparation and lifecycle experiment.

    Inputs: parsed CLI settings. Refuses fresh-mode reuse; preserves existing cluster,
    source and model artifacts. Uses accepted K3s/KubeRay installer and Secret flow.
    Writes an overall failure/success receipt; phase failures retain service evidence.
    Actual serving/recovery claims require subsequent review of the cloud evidence.
    """
    control.require_cloud_host()
    installed=platform.cloud_cluster_present()
    if args.fresh_setup and installed:
        raise ValueError("--fresh-setup requires a clean VM; no automatic reset")
    platform.verify_references(args.part5_root)
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(success=False,image=args.image,setup_mode="reuse" if installed else "fresh",
                 fresh_setup_verified=False,stages=[])
    try:
        print("7D E2E: validating candidate serving image before cluster changes",flush=True)
        image_gate(args.image,args.output)
        receipt["stages"].append("image_import_gate")
        if not installed:
            print("7D E2E: installing Kubernetes/KubeRay through the accepted fresh path",flush=True)
            platform.setup.install(args.output/"setup",with_hf_secret=True)
            receipt["fresh_setup_verified"]=True
        if not os.environ.get("KUBECONFIG"):
            os.environ["KUBECONFIG"]=str(Path.home()/".kube/part7-config")
        report=platform.cluster.inspect(args.context,"part7",platform.setup.NODE,"part7-workspace","default","kuberay-operator")
        control.save_json(args.output/"cluster_check.json",report)
        control.api(args.context,"part7",["get","secret","part7-huggingface","-o","name"])
        control.api(args.context,"part7",["get","rayservices","--chunk-size=1","-o","name"])
        receipt["stages"].append("cluster_ready")
        prompts,provenance=prepare_prompts(args.part5_root,args.output/"data")
        receipt["stages"].append("prompts_ready")
        document=control.renderer.render_service(args.image,platform.setup.NODE,name=args.name,config=getattr(args,"config",None))
        print("7D E2E: service readiness, HTTP load, worker restart and recovery",flush=True)
        control.execute(document,args.context,args.output/"service",prompts,cleanup_on_success=True)
        cleanup=json.loads((args.output/"service/cleanup.json").read_text())
        review=json.loads((args.output/"service/review.json").read_text())
        if (cleanup.get("success") is not True or review.get("success") is not True
                or not review.get("service_uid") or cleanup.get("service_uid")!=review["service_uid"]):
            raise ValueError("Successful service review and verified cleanup required")
        receipt["stages"].extend(["service_review_passed","owned_cleanup_verified"])
        receipt["success"]=True
    except Exception as error:
        receipt["error_type"]=type(error).__name__
        raise
    finally:
        control.save_json(args.output/"e2e_review.json",receipt)
    print("7D E2E finished; retain/review evidence at "+str(args.output),flush=True)


def main():
    """Parse CPU preview/cloud execution; require an explicit serving image digest.

    --check verifies source inputs and renders configuration without network/models.
    Cloud execution must follow agreed hardware/spending and compatible image review.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--cloud",action="store_true")
    parser.add_argument("--image",required=True)
    parser.add_argument("--fresh-setup",action="store_true")
    parser.add_argument("--model-repo",default=control.settings.DEFAULT_REPO)
    parser.add_argument("--model-revision",default=control.settings.DEFAULT_REVISION)
    parser.add_argument("--model-id",default="part7-math-grpo")
    parser.add_argument("--max-model-len",type=int,default=4096)
    parser.add_argument("--max-num-seqs",type=int,default=16)
    parser.add_argument("--gpu-memory-utilization",type=float,default=.5)
    parser.add_argument("--context",default="default")
    parser.add_argument("--part5-root",type=Path,default=ROOT.parent/"part5-alignment")
    parser.add_argument("--name",default=datetime.now(timezone.utc).strftime("part7-7d-%Y%m%d-%H%M%S"))
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    args.output=args.output or ROOT/"results/7d"/args.name
    args.config=control.settings.ServingConfig(args.model_repo,args.model_revision,args.model_id,
        args.max_model_len,args.max_num_seqs,args.gpu_memory_utilization).validate()
    control.renderer.render_service(args.image,platform.setup.NODE,name=args.name,config=args.config)
    platform.verify_references(args.part5_root)
    if args.check:
        print(json.dumps(dict(cloud_verified=False,image=args.image,name=args.name,
            native_runtime_gate_required=True,model_repo=args.config.model_repo,
            model_revision=args.config.model_revision),indent=2))
    else:
        execute(args)


if __name__=="__main__":
    main()
