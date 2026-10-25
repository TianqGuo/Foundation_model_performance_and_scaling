"""Cloud-only serving image build/import gate and optional Docker Hub publication."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
shared=import_module("infrastructure.7c_kuberay.image_publish")
control=import_module("infrastructure.7d_serve.service_control")


def execute(repository, output, publish):
    """Build/check on the cloud VM, optionally publish and verify its immutable digest.

    Inputs: Docker Hub repository, fresh evidence directory and explicit publish flag.
    Uses 7C's isolated credential-directory/digest helpers; authentication is hidden
    and never saved in evidence. Native import failure stops before any upload.
    Writes build/push/identity receipts; no Kubernetes provisioning or model loading.
    """
    owner=shared.validate_repository(repository)
    control.require_cloud_host()
    output.mkdir(parents=True,exist_ok=False)
    docker=["docker"]
    if subprocess.run(["docker","info"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        docker=["sudo","-n","docker"]
    tag=repository+":7d-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with shared.temporary_docker_config(docker) as config:
        command=[*docker,"--config",config]
        if publish:
            token=os.environ.get("DOCKERHUB_TOKEN") or getpass.getpass("Docker Hub write token (hidden): ")
            if not token.strip():raise ValueError("Empty Docker Hub token")
            result=subprocess.run([*command,"login","--username",owner,"--password-stdin"],
                input=token+"\n",text=True,capture_output=True,timeout=120)
            del token
            if result.returncode:raise RuntimeError("Docker Hub login failed")
        env=dict(os.environ,PART7_DOCKER_CONFIG=config,PART7_BUILD_OUTPUT=str((output/"build").resolve()))
        subprocess.run(["bash",str(Path(__file__).with_name("build_image.sh")),"--cloud",tag],env=env,check=True)
        receipt=dict(tag=tag,repository=repository,base_image=control.settings.BASE_IMAGE,
                     expected_core_versions=dict(control.settings.CORE_VERSIONS),
                     dependency_lock_sha256=hashlib.sha256((Path(__file__).parent/"environment/requirements.lock").read_bytes()).hexdigest(),
                     published=False,native_import_gate_passed=True,
                     inference_verified=False,e2e_verified=False)
        if publish:
            result=subprocess.run([*command,"push",tag],text=True,capture_output=True,timeout=3600)
            (output/"push.log").write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError("Docker push failed; inspect push.log")
            image=repository+"@"+shared.publish_digest(result.stdout+result.stderr)
            subprocess.run([*command,"pull",image],check=True,timeout=1800)
            receipt.update(image=image,published=True,registry_pull_verified=True)
            print("Published serving image: "+image,flush=True)
        control.save_json(output/"image_publish.json",receipt)
        return receipt


def main():
    """Select CPU preview or explicit cloud build; --publish separately opts into upload."""
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--cloud",action="store_true")
    parser.add_argument("--publish",action="store_true")
    parser.add_argument("--repository",default="sclion/part7-training")
    parser.add_argument("--output",type=Path,default=ROOT/"results/7d"/datetime.now(timezone.utc).strftime("image_%Y%m%dT%H%M%SZ"))
    args=parser.parse_args()
    shared.validate_repository(args.repository)
    if args.check:
        print(json.dumps(dict(repository=args.repository,publish=args.publish,core_versions=control.settings.CORE_VERSIONS,
                              base_image=control.settings.BASE_IMAGE,
                              cloud_verified=False),indent=2))
    else:
        execute(args.repository,args.output,args.publish)


if __name__=="__main__":
    main()
