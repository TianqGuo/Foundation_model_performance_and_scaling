"""Build and optionally publish the public training image on a cloud VM only."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"


def validate_repository(repository):
    """Validate a Docker Hub user/repository, returning its owner for login.

    Reject other registries, options and ambiguous paths; no Docker/network calls.
    """
    if not re.fullmatch(r"[a-z0-9]{4,30}/[a-z0-9]+(?:[._-][a-z0-9]+)*", repository):
        raise ValueError("Expected Docker Hub username/repository")
    return repository.split("/")[0]


def publish_digest(output):
    """Extract Docker push's final registry digest; reject missing/ambiguous results.

    Input: push stdout. Returns sha256 digest; no local configuration-ID fallback.
    """
    matches = set(re.findall(r"digest: (sha256:[0-9a-f]{64})", output))
    if len(matches) != 1:
        raise ValueError("Push did not report one registry manifest digest")
    return matches.pop()


@contextmanager
def temporary_docker_config(docker):
    """Yield an isolated authentication directory and remove it on success or failure.

    Input: selected Docker command prefix. A sudo Docker client can create root-only
    buildx files, so that branch uses sudo to delete only this newly allocated
    directory. Never touches the user's persistent Docker configuration.
    Cleanup failures remain visible; credentials must not silently remain behind.
    """
    config = tempfile.mkdtemp(prefix="part7-docker-auth-")
    try:
        yield config
    finally:
        if docker[:2] == ["sudo", "-n"]:
            subprocess.run(["sudo", "-n", "rm", "-rf", "--", config], check=True, timeout=60)
        else:
            shutil.rmtree(config)


def verify_image_tools(command, tag):
    """Check image startup and Kubernetes probe tools before publishing.

    Inputs: Docker command prefix and image tag. Runs as image's nonroot user with
    no network or GPU visibility, checks required executables and Python version.
    Missing tools abort publication; this does not prove Ray/GPU runtime health.
    """
    script = "set -eu; for tool in bash wget tar git ray python; do command -v \"$tool\" >/dev/null; done; wget --version >/dev/null; python --version"
    subprocess.run([*command, "run", "--rm", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void",
                    "--entrypoint", "/bin/bash", tag, "-ec", script], check=True, timeout=120)


def execute(repository, base, output, publish):
    """Build on a cloud VM and optionally authenticate/publish the reviewed public image.

    Inputs: Docker Hub repository, base reference, fresh evidence path, publish flag.
    Invokes existing build helper, verifies Python without GPU/model execution,
    and retains build/publish receipts. Token uses stdin and a temporary Docker
    config outside results, removed afterwards. Build/push failures stop the flow.
    """
    owner = validate_repository(repository)
    if Path("/.dockerenv").exists() or subprocess.check_output(["ps", "-p", "1", "-o", "comm="], text=True).strip() != "systemd":
        raise ValueError("Run image preparation on the cloud VM, not inside a container")
    if output.exists():
        raise FileExistsError("Fresh image evidence directory required")
    output.parent.mkdir(parents=True, exist_ok=True)
    docker = ["docker"]
    if subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["sudo", "-n", "docker", "info"], stdout=subprocess.DEVNULL, check=True)
        docker = ["sudo", "-n", "docker"]
    tag = repository + ":7c-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with temporary_docker_config(docker) as config:
        command = [*docker, "--config", config]
        if publish:
            token = os.environ.get("DOCKERHUB_TOKEN") or getpass.getpass("Docker Hub access token with write permission (hidden): ")
            if not token.strip():
                raise ValueError("Empty Docker Hub token")
            login = subprocess.run([*command, "login", "--username", owner, "--password-stdin"],
                                   input=token + "\n", text=True, capture_output=True, timeout=120)
            if login.returncode:
                raise RuntimeError("Docker Hub login failed; verify token and username")
            del token
        env = dict(os.environ, DOCKER_CONFIG=config, PART7_BUILD_OUTPUT=str(output.resolve()))
        subprocess.run(["bash", str(ROOT / "infrastructure/7c_kuberay/build_image.sh"), "--cloud", base, tag], env=env, check=True)
        verify_image_tools(command, tag)
        if publish:
            result = subprocess.run([*command, "push", tag], text=True, capture_output=True, timeout=3600)
            (output / "push.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError("Docker push failed; inspect retained push.log")
            digest = publish_digest(result.stdout + result.stderr)
            image = repository + "@" + digest
            # Confirm the published digest is retrievable, rather than trusting a tag.
            subprocess.run([*command, "pull", "--platform", "linux/amd64", image], check=True, timeout=1800)
            receipt = dict(repository=repository, tag=tag, image=image, published=True,
                           registry_pull_verified=True, gpu_verified=False)
            (output / "image_publish.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print("Published training image: " + image)
        else:
            print("Built only; use --publish to upload. Tag: " + tag)


def main():
    """Parse preview/cloud mode and optional publication; defaults match user destination.

    --check validates/prints settings only. --cloud invokes build; --publish adds
    public upload with hidden authentication. No resources are provisioned.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--cloud", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--repository", default="sclion/part7-training")
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--output", type=Path, default=ROOT / "results/7c" /
                        ("image_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    args = parser.parse_args()
    validate_repository(args.repository)
    if args.check:
        print(json.dumps(dict(repository=args.repository, base=args.base, publish=args.publish,
                              gpu_verified=False), indent=2))
    else:
        execute(args.repository, args.base, args.output, args.publish)


if __name__ == "__main__":
    main()
