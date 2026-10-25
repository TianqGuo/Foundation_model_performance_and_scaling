"""Own one bounded RayService lifecycle; cloud execution is not yet E2E-verified."""
from contextlib import contextmanager
from importlib import import_module
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
renderer = import_module("infrastructure.7d_serve.render")
settings = import_module("infrastructure.7d_serve.config")
benchmark = import_module("infrastructure.7d_serve.benchmark")


def validate_manifest(document):
    """Reconstruct a rendered service and reject modified commands/resources/settings.

    Input: parsed JSON manifest. Returns namespace, name and ServingConfig. This
    pure gate runs before API calls; only the current bounded renderer is accepted.
    """
    metadata, spec = document["metadata"], document["spec"]
    cluster = spec["rayClusterConfig"]
    pod = cluster["headGroupSpec"]["template"]["spec"]
    volumes = {entry["name"]:entry for entry in pod["volumes"]}
    native = json.loads(spec["serveConfigV2"])["applications"][0]["args"]["llm_configs"][0]
    engine = native["engine_kwargs"]
    config = settings.ServingConfig(model_repo=metadata["annotations"]["part7/model-repo"],
        model_revision=metadata["annotations"]["part7/model-revision"],
        model_id=native["model_loading_config"]["model_id"],max_model_len=engine["max_model_len"],
        max_num_seqs=engine["max_num_seqs"],gpu_memory_utilization=engine["gpu_memory_utilization"],
        tensor_parallel_size=engine["tensor_parallel_size"])
    expected = renderer.render_service(pod["containers"][0]["image"],
        pod["nodeSelector"]["kubernetes.io/hostname"],config=config,
        namespace=metadata["namespace"],name=metadata["name"],
        pvc=volumes["workspace"]["persistentVolumeClaim"]["claimName"],
        secret=volumes["hub-token"]["secret"]["secretName"],
        pull_secret=pod.get("imagePullSecrets",[{}])[0].get("name"))
    if document != expected:
        raise ValueError("Use an unmodified manifest from part7D.sh render")
    return metadata["namespace"],metadata["name"],config


def api(context, namespace, arguments, *, body=None):
    """Run bounded kubectl in an explicit context; optionally supply JSON on stdin.

    Returns stdout text; sanitized exceptions omit command output/Secret values.
    Mutations occur only for explicit create/delete calls supplied by this module.
    """
    result = subprocess.run(["kubectl","--context",context,"--namespace",namespace,
        "--request-timeout=30s",*arguments],input=None if body is None else json.dumps(body),
        text=True,capture_output=True,timeout=45)
    if result.returncode:
        raise RuntimeError("kubectl operation failed: "+arguments[0])
    return result.stdout


def require_owner(resource, owner_uid):
    """Return resource after requiring its controller owner UID; reject unrelated objects."""
    if not any(ref.get("uid")==owner_uid and ref.get("controller") is True
               for ref in resource["metadata"].get("ownerReferences",[])):
        raise ValueError("Resource is not controlled by the recorded parent UID")
    return resource


def service_ready(document):
    """Return active cluster when current-generation operator and Serve statuses agree.

    Pure inspection of pinned KubeRay 1.4.2 fields. Missing/stale/empty statuses
    stay pending; deployment failure raises RuntimeError. HTTP readiness is separate.
    """
    status = document.get("status",{})
    for section in ("activeServiceStatus","pendingServiceStatus"):
        apps = status.get(section,{}).get("applicationStatuses",{})
        if any(app.get("status")=="DEPLOY_FAILED" for app in apps.values()):
            raise RuntimeError("Native Serve application deployment failed")
    generation = document["metadata"].get("generation",1)
    if status.get("observedGeneration",0) < generation:
        return None
    if not any(item.get("type")=="Ready" and item.get("status")=="True"
               and item.get("observedGeneration",generation)>=generation
               for item in status.get("conditions",[])):
        return None
    active = status.get("activeServiceStatus",{})
    app = active.get("applicationStatuses",{}).get("part7-llm",{})
    deployments = app.get("serveDeploymentStatuses",{})
    if app.get("status")!="RUNNING" or not deployments or any(
            item.get("status")!="HEALTHY" for item in deployments.values()):
        return None
    return active.get("rayClusterName") if status.get("numServeEndpoints",0)>0 else None


def pod_ready(pod):
    """Return whether a nonterminating pod has Kubernetes Ready=True; no API calls."""
    return not pod["metadata"].get("deletionTimestamp") and any(
        item.get("type")=="Ready" and item.get("status")=="True"
        for item in pod.get("status",{}).get("conditions",[]))


def delete_owned(call, namespace, plural, name, uid):
    """Delete one service or pod using atomic API UID preconditions, avoiding name races.

    Inputs: bound kubectl callable, validated resource identifiers and recorded UID.
    Returns API response text. Never force deletes, deletes storage or uses selectors.
    An object replaced under the same name causes an API conflict instead of deletion.
    """
    renderer.training.dns_label(namespace)
    renderer.training.dns_label(name)
    if plural not in ("rayservices","pods") or not isinstance(uid,str) or not uid:
        raise ValueError("Expected an owned RayService/pod UID")
    prefix = "/apis/ray.io/v1" if plural=="rayservices" else "/api/v1"
    path = f"{prefix}/namespaces/{namespace}/{plural}/{name}"
    return call(["delete","--raw",path,"-f","-"],body=dict(
        apiVersion="v1",kind="DeleteOptions",preconditions=dict(uid=uid),propagationPolicy="Foreground"))


def save_json(path, value):
    """Write JSON evidence under a controller-owned directory; return None."""
    path.write_text(json.dumps(value,indent=2)+"\n")


class Session:
    """Track a newly created service UID and its cluster; never adopt existing services."""
    def __init__(self, document, context, output, *, call=None):
        """Validate manifest/context and bind API transport; performs no cluster operations."""
        self.namespace,self.name,self.config = validate_manifest(document)
        if not context or context.startswith("-"):
            raise ValueError("Explicit Kubernetes context required")
        self.document,self.context,self.output = document,context,output
        self.call = call or (lambda arguments,**kwargs:api(context,self.namespace,arguments,**kwargs))
        self.uid = self.cluster = self.cluster_uid = self.generation = None

    def create(self):
        """Create a fresh service and retain identity; existing names fail instead of apply/adopt."""
        self.output.mkdir(parents=True,exist_ok=False)
        save_json(self.output/"manifest.json",self.document)
        created = json.loads(self.call(["create","-f","-","-o","json"],body=self.document))
        self.uid = created["metadata"]["uid"]
        self.generation = created["metadata"]["generation"]
        save_json(self.output/"ownership.json",dict(namespace=self.namespace,name=self.name,uid=self.uid))
        print("Created RayService "+self.name+"; waiting for native Serve readiness...",flush=True)

    def current(self):
        """Read the service and require the created UID; reject deletion/replacement."""
        document = json.loads(self.call(["get","rayservice",self.name,"-o","json"]))
        if not self.uid or document["metadata"]["uid"] != self.uid:
            raise ValueError("RayService UID changed; refusing to operate")
        if document["metadata"].get("generation") != self.generation:
            raise ValueError("RayService spec changed during the experiment")
        return document

    def bind_cluster(self, cluster):
        """Record and verify the service-controlled cluster UID, refusing later replacements."""
        renderer.training.dns_label(cluster)
        resource = require_owner(json.loads(self.call(["get","raycluster",cluster,"-o","json"])),self.uid)
        uid = resource["metadata"]["uid"]
        if self.cluster and (self.cluster != cluster or self.cluster_uid != uid):
            raise ValueError("Unexpected RayCluster replacement outside restart scope")
        self.cluster,self.cluster_uid = cluster,uid
        save_json(self.output/"cluster_identity.json",dict(name=cluster,uid=uid))

    def wait_ready(self, timeout=1200):
        """Poll bounded operator readiness; record statuses, cluster UID and timeout evidence."""
        deadline = time.monotonic()+timeout
        previous = None
        next_report = 0
        while time.monotonic()<deadline:
            document = self.current()
            save_json(self.output/"service_status.json",document.get("status",{}))
            pending=document.get("status",{}).get("pendingServiceStatus",{}).get("rayClusterName")
            if pending:
                self.bind_cluster(pending)
            cluster = service_ready(document)
            status = document.get("status",{}).get("serviceStatus","initializing")
            if status != previous or time.monotonic() >= next_report:
                print("RayService "+self.name+": "+(status or "not ready")+
                      "; cluster="+(self.cluster or "pending"),flush=True)
                previous = status
                next_report = time.monotonic()+30
            if cluster:
                self.bind_cluster(cluster)
                return cluster
            time.sleep(3)
        raise TimeoutError("RayService readiness timed out; inspect retained status/logs")

    def pods(self):
        """Return pods controlled by the recorded RayCluster; reject label-only impostors."""
        if not self.cluster_uid:
            raise ValueError("No owned RayCluster recorded")
        items = json.loads(self.call(["get","pods","-l","ray.io/cluster="+self.cluster,"-o","json"]))["items"]
        return [require_owner(pod,self.cluster_uid) for pod in items]

    def collect(self, label):
        """Retain placement/replica status and current/previous logs, including model init.

        Omits pod environment, Secret objects and weight files. Log errors are
        recorded explicitly; callers may retain resources rather than clean up.
        """
        directory = self.output/label
        directory.mkdir(exist_ok=False)
        save_json(directory/"service_status.json",self.current().get("status",{}))
        pods = self.pods()
        save_json(directory/"pods.json",[dict(name=p["metadata"]["name"],uid=p["metadata"]["uid"],
            labels=p["metadata"].get("labels",{}),node=p["spec"].get("nodeName"),
            status=p.get("status",{}),containers=[{k:c.get(k) for k in ("name","image","resources")}
            for c in p["spec"]["containers"]]) for p in pods])
        errors=[]
        for pod in pods:
            for container in pod["spec"].get("initContainers",[])+pod["spec"]["containers"]:
                for previous in (False,True):
                    if previous and not any(s.get("name")==container["name"] and s.get("restartCount",0)>0
                        for s in pod.get("status",{}).get("containerStatuses",[])+pod.get("status",{}).get("initContainerStatuses",[])):
                        continue
                    name=pod["metadata"]["name"]
                    try:
                        text=self.call(["logs",name,"-c",container["name"],"--timestamps","--tail=10000"]+
                                       (["--previous"] if previous else []))
                        (directory/(name+"_"+container["name"]+("_previous" if previous else "")+".log")).write_text(text)
                    except RuntimeError:
                        errors.append(dict(pod=name,container=container["name"],previous=previous))
        save_json(directory/"log_errors.json",errors)
        return not errors

    def restart_worker(self):
        """Delete exactly one owned ready worker pod; return its old name/UID.

        Requires the current same-node one-worker topology. Completion/recovery
        probes will be driven by the E2E controller; this method makes no recovery claim.
        """
        if service_ready(self.current()) != self.cluster:
            raise ValueError("Service is no longer ready on the owned cluster")
        workers=[pod for pod in self.pods() if pod["metadata"].get("labels",{}).get("ray.io/node-type")=="worker"]
        if len(workers)!=1 or not pod_ready(workers[0]):
            raise ValueError("Expected exactly one ready owned worker")
        old={key:workers[0]["metadata"][key] for key in ("name","uid")}
        old["node"]=workers[0]["spec"].get("nodeName")
        save_json(self.output/"restart_intent.json",old)
        delete_owned(self.call,self.namespace,"pods",old["name"],old["uid"])
        return old

    def recover_worker(self, old, base_url, prompt, timeout=600):
        """Observe replacement pod readiness and successful generation after intentional deletion.

        Inputs: restart intent, owned-head URL and rendered probe prompt. Returns
        recovery receipt with new UID and measured probe failures/timing. Requires
        old-pod disappearance as well as a new ready pod; HTTP outage may be
        absent and is reported honestly. Same cluster/node only, no VM failover claim.
        """
        started=time.monotonic()
        observations=[]
        pod_loss=False
        while time.monotonic()-started < timeout:
            cluster=service_ready(self.current())
            workers=[pod for pod in self.pods() if pod["metadata"].get("labels",{}).get("ray.io/node-type")=="worker"]
            old_present=any(pod["metadata"]["uid"]==old["uid"] for pod in workers)
            old_ready=any(pod["metadata"]["uid"]==old["uid"] and pod_ready(pod) for pod in workers)
            pod_loss = pod_loss or not old_ready
            replacements=[pod for pod in workers if pod["metadata"]["uid"]!=old["uid"] and pod_ready(pod)]
            # Real completion probes distinguish a live proxy/catalog from working inference.
            result=benchmark.request_completion(base_url,self.config.model_id,prompt,8,10)
            observations.append(dict(seconds=time.monotonic()-started,pod_ready=bool(replacements),
                                     old_worker_present=old_present,
                                     request_success=result["success"],error_type=result.get("error_type")))
            save_json(self.output/"recovery_observations.json",observations)
            # A terminating old worker can still answer HTTP requests. Require its
            # disappearance before accepting inference as replacement recovery.
            if pod_loss and not old_present and len(replacements)==1 and cluster==self.cluster and result["success"]:
                replacement=replacements[0]
                if replacement["spec"].get("nodeName") != old["node"]:
                    raise ValueError("Replacement moved nodes outside accepted recovery scope")
                receipt=dict(success=True,old_worker=old,
                    new_worker=dict(name=replacement["metadata"]["name"],uid=replacement["metadata"]["uid"],
                                    node=replacement["spec"].get("nodeName")),
                    recovery_seconds=time.monotonic()-started,pod_readiness_loss_observed=pod_loss,
                    old_worker_absent=True,
                    http_probe_failures=sum(not row["request_success"] for row in observations),
                    scope="same-node owned worker pod replacement; not VM/bitwise recovery")
                save_json(self.output/"recovery.json",receipt)
                return receipt
            time.sleep(3)
        raise TimeoutError("Owned worker did not recover readiness and inference")

    def cleanup(self, timeout=180):
        """Delete owned service only after a passing review; verify descendant absence.

        Requires review.json success=True with matching service UID. Atomic deletion
        protects replacements; PVCs/Secrets/namespaces are preserved. Timeout/API
        failures prevent a successful cleanup receipt.
        """
        review=json.loads((self.output/"review.json").read_text())
        if review.get("success") is not True or review.get("service_uid") != self.uid:
            raise ValueError("Successful matching review required before cleanup")
        self.current()
        delete_owned(self.call,self.namespace,"rayservices",self.name,self.uid)
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            service=self.call(["get","rayservice",self.name,"--ignore-not-found","-o","json"]).strip()
            cluster=self.call(["get","raycluster",self.cluster,"--ignore-not-found","-o","json"]).strip()
            pods=json.loads(self.call(["get","pods","-l","ray.io/cluster="+self.cluster,"-o","json"]))["items"]
            if not service and not cluster and not pods:
                save_json(self.output/"cleanup.json",dict(success=True,service_uid=self.uid,cluster_uid=self.cluster_uid))
                return
            time.sleep(2)
        raise TimeoutError("Owned service/cluster/pods did not finish cleanup")


@contextmanager
def port_forward(session):
    """Expose the owned ready CPU head HTTP port on VM loopback and always stop transport.

    Input: ready Session. Yields local base URL. Captures kubectl transport logs;
    validates process survival. Does not expose public ingress or delete cluster resources.
    """
    heads=[pod for pod in session.pods() if pod["metadata"].get("labels",{}).get("ray.io/node-type")=="head" and pod_ready(pod)]
    if len(heads)!=1:
        raise ValueError("Expected exactly one owned ready head pod")
    with (session.output/"port_forward.log").open("w") as logs:
        process=subprocess.Popen(["kubectl","--context",session.context,"--namespace",session.namespace,
            "--request-timeout=30s","port-forward","--address=127.0.0.1",
            "pod/"+heads[0]["metadata"]["name"],"0:8000"],stdout=logs,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError("Owned head port-forward exited; inspect transport log")
                match=re.search(r"Forwarding from 127\.0\.0\.1:(\d+)",(session.output/"port_forward.log").read_text())
                if match:
                    yield "http://127.0.0.1:"+match.group(1)
                    if process.poll() is not None:
                        raise RuntimeError("Port-forward died during measurement")
                    return
                time.sleep(.2)
            raise TimeoutError("Port-forward did not start")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def http_ready(base_url, model_id):
    """Check that /v1/models lists the requested ID; actual generation is a separate probe.

    Returns bool on HTTP/JSON failure without recording response bodies. Used only
    through the controller's owned-pod loopback tunnel, with a bounded request.
    """
    try:
        with urlopen(base_url+"/v1/models",timeout=5) as response:
            data=json.load(response)
        return any(item.get("id")==model_id for item in data.get("data",[]))
    except Exception:
        return False


def validate_driver_versions(output):
    """Validate nvidia-smi driver-version rows against the CUDA 13 baseline (R580).

    Input: CSV/no-header driver strings. Returns the parsed rows or raises before
    image/cluster mutations. This is a minimum baseline, not proof that all PTX/JIT
    features work; actual GPU serving remains the acceptance test.
    """
    rows=[line.strip() for line in output.splitlines() if line.strip()]
    if not rows or any(not re.fullmatch(r"\d+(?:\.\d+)+",row) for row in rows):
        raise ValueError("Could not identify GPU driver versions")
    if any(int(row.split(".")[0])<580 for row in rows):
        raise ValueError("CUDA 13 serving candidate requires NVIDIA driver R580 or newer")
    return rows


def require_cloud_host():
    """Reject restricted containers/non-systemd hosts and require visible cloud GPUs.

    No model imports; rejects WSL, then checks PID 1 and the NVIDIA utility.
    Hardware checks alone cannot attest that an arbitrary host is a cloud VM.
    This is an execution guard, not paid-provisioning authorization or a capacity check.
    """
    if any(marker in platform.release().lower() for marker in ("microsoft","wsl")):
        raise ValueError("WSL/local execution is unsupported; run on the cloud GPU VM")
    if platform.machine().lower() not in ("x86_64","amd64"):
        raise ValueError("Serving image profile requires an x86_64 cloud VM")
    if Path("/.dockerenv").exists() or subprocess.check_output(
            ["ps","-p","1","-o","comm="],text=True,timeout=5).strip()!="systemd":
        raise ValueError("Run lifecycle control on the cloud GPU VM")
    result=subprocess.run(["nvidia-smi","-L"],capture_output=True,text=True,check=True,timeout=15)
    if not any(line.startswith("GPU ") for line in result.stdout.splitlines()):
        raise ValueError("No visible GPUs on the cloud host")
    drivers=subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
        capture_output=True,text=True,check=True,timeout=15)
    validate_driver_versions(drivers.stdout)


def execute(document, context, output, prompts, *, cleanup_on_success=False):
    """Run one service/HTTP/benchmark/restart experiment from the future cloud E2E driver.

    Inputs: validated manifest, explicit context, fresh evidence folder and rendered
    prompts. Creates service, retains pre-restart logs, tests owned worker replacement,
    saves review and optionally cleans up only after success. Failures retain resources
    and a sanitized failure receipt. Not exposed as a partial normal cloud CLI yet.
    """
    if not prompts or any(not isinstance(prompt,str) or not prompt.strip() for prompt in prompts):
        raise ValueError("Validated nonempty rendered prompts required")
    require_cloud_host()
    session=Session(document,context,output)
    try:
        session.create()
        session.wait_ready()
        with port_forward(session) as base_url:
            deadline=time.monotonic()+120
            while not http_ready(base_url,session.config.model_id):
                if time.monotonic()>deadline:
                    raise TimeoutError("HTTP model catalog readiness timed out")
                time.sleep(2)
            report=benchmark.run(base_url,session.config.model_id,prompts,requests=16,concurrency=4)
            save_json(output/"benchmark_before.json",report)
            if report["summary"]["errors"]:
                raise RuntimeError("Initial benchmark contained errors")
            if not session.collect("before_restart"):
                raise RuntimeError("Evidence incomplete before worker restart")
            old=session.restart_worker()
            session.recover_worker(old,base_url,prompts[0])
            report=benchmark.run(base_url,session.config.model_id,prompts,requests=4,concurrency=1,warmup=0)
            save_json(output/"benchmark_after.json",report)
            if report["summary"]["errors"]:
                raise RuntimeError("Post-recovery benchmark contained errors")
            if not session.collect("after_restart"):
                raise RuntimeError("Evidence incomplete after worker restart")
        save_json(output/"review.json",dict(success=True,service_uid=session.uid,
            cluster_uid=session.cluster_uid,model_repo=session.config.model_repo,
            model_revision=session.config.model_revision,
            scope="single replica/GPU, same-node pod restart; cloud verification pending review"))
        if cleanup_on_success:
            session.cleanup()
    except Exception as error:
        # Never delete or overwrite evidence belonging to an earlier run if create failed.
        if session.uid:
            save_json(output/"failure.json",dict(success=False,error_type=type(error).__name__,
                                               resources_retained=True,service_uid=session.uid))
            if session.cluster_uid:
                try:
                    session.collect("failure_evidence")
                except Exception:
                    save_json(output/"evidence_capture_failure.json",dict(success=False))
        raise

