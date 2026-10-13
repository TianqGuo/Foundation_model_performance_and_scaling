"""CPU staging/orchestration/evidence tests; all Kubernetes execution is mocked."""
import hashlib
from importlib import import_module
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
staging=import_module("infrastructure.7c_kuberay.data_staging")
experiment=import_module("infrastructure.7c_kuberay.experiment")


class WorkflowTests(unittest.TestCase):
    def bundle(self,root):
        """Create tiny synthetic bytes and hashes, without Parquet/model libraries."""
        root.mkdir();(root/"train.parquet").write_bytes(b"synthetic data")
        manifest=dict(outputs={"train.parquet":dict(sha256=hashlib.sha256(b"synthetic data").hexdigest())})
        (root/"manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_bundle_archive_excludes_unrecorded_files(self):
        """Package only declared workload files, excluding unrelated local content."""
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"bundle";self.bundle(root);(root/"unrelated.txt").write_text("exclude")
            archive=Path(d)/"data.tar.gz";staging.pack_bundle(root,archive)
            with tarfile.open(archive) as a:self.assertEqual(set(a.getnames()),{"manifest.json","train.parquet"})

    def test_bundle_corruption_and_traversal_rejected(self):
        """Reject changed bytes or escaped manifest paths before staging a pod."""
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"bundle";self.bundle(root);(root/"train.parquet").write_bytes(b"changed")
            with self.assertRaises(ValueError):staging.pack_bundle(root,Path(d)/"bad.tar.gz")
            (root/"manifest.json").write_text(json.dumps(dict(outputs={"../outside":dict(sha256="hash")})))
            with self.assertRaisesRegex(ValueError,"Unsafe"):staging.pack_bundle(root,Path(d)/"bad.tar.gz")

    def test_staging_cpu_pod_and_owned_cleanup(self):
        """Mock staging, asserting no GPU/Secret allocation and scoped pod deletion."""
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"bundle";identity=self.bundle(root);calls=[]
            def api(context,namespace,args,**kwargs):
                """Return synthetic pod and verification responses without a cluster."""
                calls.append(args)
                if args[0]=="get":return "" if "--ignore-not-found" in args else '{"metadata":{"uid":"owned"}}'
                if args[0]=="create":return '{"metadata":{"uid":"owned"}}'
                if args[0]=="exec":return json.dumps(dict(verified=True,outputs=identity["outputs"],path="/workspace/data/step2_math_final"))
                return ""
            with patch.object(staging.control,"kubectl",side_effect=api):
                staging.stage(root,"registry/image@sha256:"+"a"*64,"ctx","part7","stage","pvc","node",Path(d)/"out")
            pod=json.loads((Path(d)/"out/staging_pod.json").read_text())
            self.assertEqual(pod["spec"]["containers"][0]["resources"]["requests"],dict(cpu="1",memory="1Gi"))
            self.assertEqual([v["name"] for v in pod["spec"]["volumes"]],["workspace"])
            self.assertEqual(calls[-1][:3],["delete","pod","stage"])
            self.assertTrue(json.loads((Path(d)/"out/staging_receipt.json").read_text())["pod_cleanup_verified"])

    def test_archive_extraction_rejects_links_paths_and_tensors(self):
        """Reject unsafe archive members before creating an evidence directory."""
        with tempfile.TemporaryDirectory() as d:
            for index,name in enumerate(("../escape","/absolute","state.pt")):
                path=Path(d)/f"bad{index}.tar.gz"
                with tarfile.open(path,"w:gz") as archive:
                    info=tarfile.TarInfo(name);info.size=1;archive.addfile(info,io.BytesIO(b"x"))
                with self.assertRaises(ValueError):experiment.unpack_evidence(path,Path(d)/f"out{index}")
            path=Path(d)/"link.tar.gz"
            with tarfile.open(path,"w:gz") as archive:
                info=tarfile.TarInfo("link");info.type=tarfile.SYMTYPE;info.linkname="/etc/passwd";archive.addfile(info)
            with self.assertRaises(ValueError):experiment.unpack_evidence(path,Path(d)/"link-out")

    def test_extracts_regular_evidence_only(self):
        """Extract a tiny valid evidence file without models or checkpoint loading."""
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"valid.tar.gz"
            with tarfile.open(path,"w:gz") as archive:
                info=tarfile.TarInfo("job_status.json");info.size=2;archive.addfile(info,io.BytesIO(b"{}"))
            out=experiment.unpack_evidence(path,Path(d)/"out")
            self.assertEqual((out/"job_status.json").read_text(),"{}")

    def test_four_jobs_use_cleanup_and_pinned_fresh_restore(self):
        """Verify phase order and forward the interruption receipt to fresh resume."""
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(output=Path(d)/"run",data=Path(d)/"bundle",image="registry/image@sha256:"+"a"*64,
                context="ctx",namespace="part7",name="experiment",pvc="pvc",node_name="node",pull_secret=None,secret="hf")
            calls=[]
            def run(manifest,context,output,cleanup_on_success):
                """Record generated jobs without executing Kubernetes or Ray."""
                output.mkdir();calls.append((manifest,cleanup_on_success))
            def unpack(path,destination):
                """Provide the synthetic interrupted Hub receipt to orchestration."""
                destination.mkdir()
                (destination/"hf_checkpoint.json").write_text(json.dumps(dict(repo_id="account/repo",revision="a"*40)))
                return destination
            with patch.object(experiment.staging,"stage"),patch.object(experiment.control,"run",side_effect=run),patch.object(experiment,"unpack_evidence",side_effect=unpack),patch.object(experiment,"review",return_value={"verified":True}):
                experiment.execute(args)
            self.assertEqual([m["metadata"]["name"] for m,_ in calls],["experiment-"+p for p in ("preflight","baseline","interrupt","resume")])
            self.assertTrue(all(cleanup for _,cleanup in calls))
            command=calls[-1][0]["spec"]["entrypoint"]
            self.assertIn("--restore-revision "+"a"*40,command)
            self.assertIn("/workspace/restored/experiment/global_step_2",command)

    def evidence(self,root):
        """Write synthetic pod/cleanup/transfer/update evidence for the 7C layer."""
        for phase in ("preflight","baseline","interrupt","resume"):
            folder=root/phase;folder.mkdir()
            (folder/"cleanup.json").write_text('{"verified":true}')
        for index,(phase,count) in enumerate((("baseline",6),("interrupt",4),("resume",2))):
            run=root/phase/"evidence";run.mkdir();(run/"diagnostics").mkdir()
            worker=dict(pod_uid="pod"+str(index),node_id="ray"+str(index),physical_node="node")
            (run/"identities.json").write_text(json.dumps(dict(platform=dict(worker=worker,image="same"))))
            records=[dict(event="first_optimizer_update",rank=rank,world_size=2,sharded_tensors=3) for rank in (0,1)]
            records += [dict(event="optimizer_update",rank=rank) for rank in (0,1) for _ in range(count)]
            (run/"diagnostics/events.jsonl").write_text("\n".join(json.dumps(x) for x in records))
        receipt=dict(repo_id="account/repo",revision="a"*40,full_checkpoint=True)
        (root/"interrupt/evidence/hf_checkpoint.json").write_text(json.dumps(receipt))
        (root/"resume/evidence/hf_restore.json").write_text(json.dumps(dict(**receipt,saved_step=2,file_hashes_verified=True,destination="/workspace/restored/run/global_step_2")))
        (root/"resume/evidence/resolved_config.yaml").write_text("trainer:\n  resume_mode: resume_path\n  resume_from_path: /workspace/restored/run/global_step_2\n")

    def test_review_requires_fresh_pods_and_correct_updates(self):
        """Reuse 7B state review while enforcing fresh pod/runtime and update evidence."""
        with tempfile.TemporaryDirectory() as d,patch.object(experiment.previous,"review_recovery",return_value={"state_restore_verified":True}):
            root=Path(d);self.evidence(root)
            report=experiment.review(root)
            self.assertEqual(report["optimizer_calls_per_rank"]["resume"],{"0":2,"1":2})
            path=root/"resume/evidence/identities.json";identity=json.loads(path.read_text());identity["platform"]["worker"]["pod_uid"]="pod1";path.write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError,"fresh"):experiment.review(root)

    def test_review_rejects_unverified_cleanup_or_different_commit(self):
        """Prevent completion claims without cleanup or matching pinned transfer receipts."""
        with tempfile.TemporaryDirectory() as d,patch.object(experiment.previous,"review_recovery",return_value={}):
            root=Path(d);self.evidence(root)
            (root/"preflight/cleanup.json").write_text('{"verified":false}')
            with self.assertRaisesRegex(ValueError,"cleanup"):experiment.review(root)
            (root/"preflight/cleanup.json").write_text('{"verified":true}')
            path=root/"resume/evidence/hf_restore.json";value=json.loads(path.read_text());value["revision"]="b"*40;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"receipts"):experiment.review(root)
