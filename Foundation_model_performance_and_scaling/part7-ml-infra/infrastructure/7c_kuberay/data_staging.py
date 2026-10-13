"""Stage only verified portable workload files onto the existing PVC using a CPU pod."""
from importlib import import_module
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
control = import_module("infrastructure.7c_kuberay.job_control")
reference = import_module("infrastructure.7a_verl.run_7a")


def pack_bundle(source, destination):
    """Hash-check prepared data and archive only manifest-recorded regular files.

    Inputs: existing bundle and fresh archive path. Returns data manifest; reject
    path traversal, symlinks, missing files or mismatched hashes. Unrecorded local
    files are excluded, so caches/credentials cannot hitchhike into this transfer.
    """
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    files = ["manifest.json", *manifest["outputs"]]
    for name in files:
        path = source / name
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
            raise ValueError("Unsafe bundle path")
        if not path.resolve().is_relative_to(source) or path.is_symlink() or not path.is_file():
            raise ValueError("Bundle requires regular files inside its root")
    reference.verify_bundle(source)
    with tarfile.open(destination, "x:gz") as archive:
        for name in files:
            archive.add(source / name, arcname=name, recursive=False)
    return manifest


def stage(source, image, context, namespace, name, pvc, node, output, pull_secret=None):
    """Create a bounded CPU staging pod, copy/verify data on PVC, then delete the pod.

    Inputs: bundle, immutable image, explicit cluster/PVC/node and fresh evidence.
    Returns None; writes archive/manifest/receipt. Reuse existing destination only
    if hashes match. Never overwrite data, install operators or create PVC/Secrets.
    Partial incoming data remains on failure. Cleanup checks the created pod UID.
    """
    name = control.renderer.dns_label(name)
    # Reuse the validated head pod template for permissions, image and node/PVC;
    # remove Secret, ports and Ray metadata since this pod only stages CPU bytes.
    job = control.renderer.render_job(image, namespace, name, pvc, "unused-secret", node, pull_secret)
    spec = job["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    spec["restartPolicy"] = "Never"
    spec["activeDeadlineSeconds"] = 600
    spec["volumes"] = [v for v in spec["volumes"] if v["name"] == "workspace"]
    container = spec["containers"][0]
    container.update(name="stage", command=["python", "-c", "import time; time.sleep(600)"],
                     resources=dict(requests=dict(cpu="1", memory="1Gi"), limits=dict(cpu="1", memory="1Gi")))
    container.pop("ports", None)
    container["volumeMounts"] = [v for v in container["volumeMounts"] if v["name"] == "workspace"]
    document = dict(apiVersion="v1", kind="Pod", metadata=dict(name=name, namespace=namespace,
                    labels={"part7-milestone":"7c", "part7-role":"data-staging"}), spec=spec)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "data_bundle.tar.gz"
    identity = pack_bundle(source, archive)
    manifest_path = output / "staging_pod.json"
    manifest_path.write_text(json.dumps(document, indent=2))
    api = control.kubectl
    if api(context, namespace, ["get", "pod", name, "--ignore-not-found", "-o", "json"]).strip():
        raise FileExistsError("Staging pod already exists")
    created = json.loads(api(context, namespace, ["create", "-f", str(manifest_path), "-o", "json"]))
    uid = created["metadata"]["uid"]
    receipt = dict(success=False, pod_uid=uid, pod=name, namespace=namespace, context=context)
    try:
        api(context, namespace, ["wait", "--for=condition=Ready", "pod/" + name, "--timeout=90s"])
        api(context, namespace, ["cp", str(archive), name + ":/tmp/data_bundle.tar.gz", "-c", "stage"])
        script = """import sys,json,tarfile
from pathlib import Path
from importlib import import_module
reference=import_module('infrastructure.7a_verl.run_7a')
destination=Path('/workspace/data/step2_math_final')
incoming=Path('/workspace/data/'+sys.argv[1]+'-incoming')
if incoming.exists(): raise FileExistsError('Incoming data path already exists')
incoming.mkdir(parents=True)
with tarfile.open('/tmp/data_bundle.tar.gz') as archive: archive.extractall(incoming,filter='data')
identity=reference.verify_bundle(incoming)
if destination.exists():
 existing=reference.verify_bundle(destination)
 if existing['outputs'] != identity['outputs']: raise ValueError('Existing PVC bundle differs')
else: incoming.rename(destination)
print(json.dumps({'verified':True,'path':str(destination),'outputs':identity['outputs']}))
"""
        result = json.loads(api(context, namespace, ["exec", name, "-c", "stage", "--", "python", "-c", script, name]))
        if not result.get("verified") or result["outputs"] != identity["outputs"]:
            raise ValueError("Staged data identity differs")
        receipt.update(success=True, data=result)
    finally:
        try:
            current = json.loads(api(context, namespace, ["get", "pod", name, "-o", "json"]))
            if current["metadata"]["uid"] != uid:
                raise RuntimeError("Staging pod UID changed; refuse deletion")
            api(context, namespace, ["delete", "pod", name, "--wait=true", "--timeout=90s"])
            receipt["pod_cleanup_verified"] = True
        finally:
            (output / "staging_receipt.json").write_text(json.dumps(receipt, indent=2))
