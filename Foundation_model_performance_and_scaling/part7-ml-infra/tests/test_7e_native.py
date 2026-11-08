"""Token math, integrity and cloud guards; no GPU libraries or model loading."""
from importlib import import_module
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

runner = import_module("infrastructure.7e_sft.run_7e")
prepare = import_module("infrastructure.7e_sft.prepare")
config = import_module("infrastructure.7e_sft.config")
workload = import_module("workloads.sft_workload")
gate = import_module("infrastructure.7e_sft.runtime_check")


class FakeTokenizer:
    """Map whitespace-separated synthetic text to IDs without external libraries."""
    eos_token_id = 0

    def encode(self, text, *, add_special_tokens):
        """Reject implicit special tokens and return deterministic synthetic IDs."""
        if add_special_tokens:
            raise AssertionError("No implicit tokenizer special tokens")
        return [len(piece) for piece in text.split()]


class NativePreparationTests(unittest.TestCase):
    def make_bundle(self, root):
        """Write a small synthetic prepared bundle with explicit source provenance."""
        source = root/"source.jsonl"
        raw = "".join(json.dumps(dict(prompt=f"p{i}",response=f"r{i}"))+"\n" for i in range(18)).encode()
        source.write_bytes(raw)
        bundle = root/"bundle"
        prepare.prepare(source,bundle,config.SFTConfig(train_count=16,eval_count=2),
                        expected_sha256=hashlib.sha256(raw).hexdigest())
        return bundle

    def test_accumulation_consumes_disjoint_contiguous_global_batches(self):
        """Ranks/microbatches cover each update once, with an explicit next-step cursor."""
        for step in range(3):
            positions = []
            for rank in range(2):
                batches = workload.update_indices(step,rank,world_size=2,micro_batch_size=2,accumulation_steps=3)
                self.assertEqual(len(batches),3)
                positions.extend(index for batch in batches for index in batch)
            self.assertEqual(sorted(positions),list(range(step*12,(step+1)*12)))
        with self.assertRaises(ValueError):workload.update_indices(0,2,world_size=2,micro_batch_size=1,accumulation_steps=1)

    def test_uneven_evaluation_has_equal_collectives_without_duplicate_weight(self):
        """Cover each held-out row once while keeping forward counts equal on ranks."""
        for count in (1, 3, 16, 17):
            slots = [workload.evaluation_slots(count, rank, 2) for rank in range(2)]
            self.assertEqual(len(slots[0]),len(slots[1]))
            included = [index for rank_slots in slots for index,include in rank_slots if include]
            self.assertEqual(sorted(included),list(range(count)))

    def test_global_response_token_weighting_with_unequal_lengths(self):
        """FSDP-averaged accumulated derivatives equal the concatenated token objective."""
        counts = [[1,9],[4,2]]
        gradients = [[2.0,-1.0],[4.0,3.0]]
        total = sum(sum(row) for row in counts)
        observed = sum(sum(gradient*workload.loss_scale(count,total,2)
            for count,gradient in zip(rank_counts,rank_gradients))
            for rank_counts,rank_gradients in zip(counts,gradients))/2
        expected = sum(count*gradient for c,g in zip(counts,gradients) for count,gradient in zip(c,g))/total
        self.assertAlmostEqual(observed,expected)
        naive = sum(sum(row) for row in gradients)/4
        self.assertNotAlmostEqual(observed,naive)
        with self.assertRaises(ValueError):workload.loss_scale(1,0,2)

    def test_cloud_token_adapter_records_exclusions_and_content_hash(self):
        """Exclude full overlength examples; preserve source identities and tokenizer settings."""
        rows = [dict(source_index=4,prompt="short p",response="answer"),
                dict(source_index=7,prompt="p "*10,response="answer")]
        examples, receipt = workload.tokenize_pairs(rows,FakeTokenizer(),max_length=6)
        self.assertEqual([row["source_index"] for row in examples],[4])
        self.assertEqual(receipt["excluded"],[dict(source_index=7,tokens=12)])
        other,_ = workload.tokenize_pairs(rows[:1],FakeTokenizer(),max_length=6)
        self.assertEqual(examples,other)
        rows[0]["response"] = "another longer answer"
        _, changed = workload.tokenize_pairs(rows,FakeTokenizer(),max_length=6)
        self.assertNotEqual(receipt["tokens_sha256"],changed["tokens_sha256"])

    def test_padding_is_never_supervised_even_when_pad_is_eos(self):
        """Pad both attention and target masks without masking the actual terminal EOS."""
        first = workload.response_example([2],[4],eos_token_id=0,max_length=8)
        second = workload.response_example([2,3],[4,5],eos_token_id=0,max_length=8)
        batch = workload.collate_lists([first,second],pad_token_id=0)
        self.assertEqual(batch["input_ids"][0],[2,4,0,0,0])
        self.assertEqual(batch["labels"][0],[-100,4,0,-100,-100])
        self.assertEqual(batch["attention_mask"][0],[1,1,1,0,0])
        with self.assertRaises(ValueError):workload.collate_lists([],pad_token_id=0)

    def test_read_bundle_rejects_changed_files_and_adapter(self):
        """Actual reads enforce file hashes and adapter provenance before native imports."""
        with tempfile.TemporaryDirectory() as folder:
            bundle = self.make_bundle(Path(folder))
            data, cfg, identity = runner.read_bundle(bundle)
            self.assertEqual(len(data["train"]),16)
            self.assertEqual(len(identity),64)
            path = bundle/"manifest.json"
            manifest = json.loads(path.read_text())
            manifest["adapter_sha256"] = "0"*64
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):runner.read_bundle(bundle)
            manifest["adapter_sha256"] = hashlib.sha256(Path(workload.__file__).read_bytes()).hexdigest()
            path.write_text(json.dumps(manifest))
            (bundle/"train.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):runner.read_bundle(bundle)

    def test_manifest_cannot_hide_split_overlap(self):
        """Even self-consistent file hashes do not permit duplicated evaluation prompts."""
        with tempfile.TemporaryDirectory() as folder:
            bundle = self.make_bundle(Path(folder))
            row = json.loads((bundle/"train.jsonl").read_text().splitlines()[0])
            path = bundle/"evaluation.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["prompt"] = row["prompt"]
            raw = "".join(json.dumps(item)+"\n" for item in rows).encode()
            path.write_bytes(raw)
            manifest = json.loads((bundle/"manifest.json").read_text())
            manifest["files"][path.name]["sha256"] = hashlib.sha256(raw).hexdigest()
            (bundle/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):runner.read_bundle(bundle)

    def test_snapshot_staging_and_integrity_without_loading_tensors(self):
        """Exercise immutable download arguments and detect changed synthetic asset bytes."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"model"
            cfg = config.SFTConfig(model_revision="a"*40)
            calls = []
            def download(**kwargs):
                """Write inert fixture assets, never a real checkpoint or tokenizer."""
                calls.append(kwargs)
                partial = Path(kwargs["local_dir"])
                for name in ("config.json","tokenizer_config.json","tokenizer.json"):
                    (partial/name).write_text("{}")
                (partial/"model.safetensors").write_bytes(b"inert fixture; not model weights")
            api = SimpleNamespace(token="synthetic-unused-token",model_info=lambda *a,**k:SimpleNamespace(sha="a"*40))
            with patch.dict(sys.modules,{"huggingface_hub":SimpleNamespace(snapshot_download=download)}):
                receipt = runner.stage_model(root,cfg,api)
                self.assertEqual(runner.verify_snapshot(root,cfg),receipt)
                self.assertEqual(runner.stage_model(root,cfg,api),receipt)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]["revision"],"a"*40)
            (root/"config.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):runner.verify_snapshot(root,cfg)

    def test_incomplete_or_escaping_shard_index_rejected(self):
        """A synthetic index cannot refer to an absent or parent-directory shard."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("config.json","tokenizer_config.json","tokenizer.json"):
                (root/name).write_text("{}")
            (root/"model-1.safetensors").write_bytes(b"inert")
            (root/"model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"x":"../elsewhere.safetensors"})))
            with self.assertRaises(ValueError):runner.snapshot_inventory(root)

    def test_native_worker_refuses_local_execution_before_heavy_imports(self):
        """Import and call the cloud-only worker with no environment or native libraries."""
        worker = import_module("infrastructure.7e_sft.worker")
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):worker.train_loop_per_worker({})
        script = "import importlib,sys; importlib.import_module('infrastructure.7e_sft.run_7e'); importlib.import_module('infrastructure.7e_sft.worker'); assert not any(x in sys.modules for x in ['torch','ray','transformers','huggingface_hub'])"
        subprocess.run([sys.executable,"-c",script],cwd=runner.ROOT,check=True)

    def test_cloud_guard_rejects_legacy_train_api(self):
        """Fail before native imports if an external setting switches Train to V1."""
        env = dict(POD_UID="synthetic", NODE_NAME="node", PART7_IMAGE="registry/model@sha256:"+"a"*64)
        with patch.dict(os.environ,env,clear=True):
            self.assertEqual(runner.require_cloud_pod(),env["PART7_IMAGE"])
            os.environ["RAY_TRAIN_V2_ENABLED"] = "0"
            with self.assertRaises(ValueError):runner.require_cloud_pod()

    def test_version_gate_rejects_drift_and_keeps_cuda_suffix(self):
        """Require the reviewed core while retaining the actual Torch build identity."""
        versions = dict(gate.CORE_VERSIONS,torch="2.13.0+cu130")
        self.assertEqual(gate.validate_versions(versions)["torch"],"2.13.0+cu130")
        versions["ray"] = "2.49.2"
        with self.assertRaises(ValueError):gate.validate_versions(versions)


if __name__ == "__main__":
    unittest.main()
