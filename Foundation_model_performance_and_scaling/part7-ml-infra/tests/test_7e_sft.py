"""CPU-only SFT boundary/data tests; no tokenizers, models, Ray or Torch imports."""
import hashlib
from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest

config = import_module("infrastructure.7e_sft.config")
prepare = import_module("infrastructure.7e_sft.prepare")
workload = import_module("workloads.sft_workload")


class SFTPreparationTests(unittest.TestCase):
    def source(self, root, rows):
        """Write synthetic input pairs and return path/digest for explicit provenance."""
        path = root/"source.jsonl"
        raw = "".join(json.dumps(row)+"\n" for row in rows).encode()
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_prompt_group_isolation_and_exact_text(self):
        """Duplicate prompt variants never leak across splits or rewrite response text."""
        rows = [dict(prompt=" question "+str(i), response=" answer "+str(i)) for i in range(5)]
        rows.append(dict(prompt=rows[0]["prompt"], response="different answer"))
        with tempfile.TemporaryDirectory() as folder:
            path, digest = self.source(Path(folder), rows)
            a,b,receipt = workload.select_pairs(path, expected_sha256=digest, train_count=3, eval_count=2)
            a2,b2,_ = workload.select_pairs(path, expected_sha256=digest, train_count=3, eval_count=2)
            self.assertEqual((a,b), (a2,b2))
            self.assertFalse({x["prompt"] for x in a} & {x["prompt"] for x in b})
            self.assertEqual(receipt["duplicate_prompt_rows"], 1)
            for row in a+b:
                self.assertEqual(row["response"], rows[row["source_index"]]["response"])
                if row["prompt"] == rows[0]["prompt"]:
                    self.assertEqual(row["response"], rows[0]["response"])

    def test_invalid_or_changed_source_rejected(self):
        """Fail closed on modified sources, empty responses and too few prompt groups."""
        with tempfile.TemporaryDirectory() as folder:
            path,digest = self.source(Path(folder), [dict(prompt="p",response="r")]*3)
            with self.assertRaises(ValueError):workload.select_pairs(path, expected_sha256=digest, train_count=1,eval_count=1)
            with self.assertRaises(ValueError):workload.select_pairs(path, expected_sha256="0"*64)
            path,digest = self.source(Path(folder), [dict(prompt="p",response=" ")])
            with self.assertRaises(ValueError):workload.select_pairs(path, expected_sha256=digest)

    def test_causal_response_mask_and_eos(self):
        """First response token is supervised after the masked prompt; EOS is not doubled."""
        value = workload.response_example([11,12], [21,22], eos_token_id=0, max_length=5)
        self.assertEqual(value["input_ids"], [11,12,21,22,0])
        self.assertEqual(value["labels"], [-100,-100,21,22,0])
        self.assertEqual(value["attention_mask"], [1]*5)
        self.assertEqual(workload.response_example([11,12],[21,22,0], eos_token_id=0,max_length=5),value)

    def test_no_silent_loss_of_response_targets(self):
        """Reject overlength/empty/noninteger token inputs instead of empty-target loss."""
        for p,a,limit in [([1],[2,3],2), ([],[2],4), ([1],[],4), ([True],[2],4)]:
            with self.subTest(p=p,a=a):
                with self.assertRaises(ValueError):workload.response_example(p,a,eos_token_id=0,max_length=limit)

    def test_cloud_identity_topology_and_budget_bounds(self):
        """Unresolved identity is preview-only; topology and smoke bounds are explicit."""
        self.assertEqual(config.SFTConfig().preview()["global_batch_size"],4)
        with self.assertRaises(ValueError):config.SFTConfig().validate(cloud=True)
        config.SFTConfig(model_revision="a"*40).validate(cloud=True)
        for changes in [dict(world_size=1),dict(world_size=True),dict(train_count=1),
                        dict(checkpoint_step=4),dict(learning_rate=float("nan")),dict(total_steps=True)]:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):config.SFTConfig(**changes).validate()

    def test_bundle_hashes_and_overwrite_guard(self):
        """Prepare provenance for future checkpoint matching; refuse stale output reuse."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path,digest=self.source(root,[dict(prompt=f"p{i}",response=f"r{i}") for i in range(18)])
            out=root/"prepared"
            receipt=prepare.prepare(path,out,config.SFTConfig(train_count=16,eval_count=2), expected_sha256=digest)
            for name,identity in receipt["files"].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),identity["sha256"])
            self.assertEqual(json.loads((out/"manifest.json").read_text()),receipt)
            with self.assertRaises(FileExistsError):prepare.prepare(path,out,config.SFTConfig(train_count=16,eval_count=2),expected_sha256=digest)

    def test_invalid_source_leaves_no_output(self):
        """Check source integrity before creating a misleading partial bundle."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path,_=self.source(root,[dict(prompt="p",response="r")])
            with self.assertRaises(ValueError):prepare.prepare(path,root/"output",config.SFTConfig())
            self.assertFalse((root/"output").exists())


if __name__ == "__main__":
    unittest.main()
