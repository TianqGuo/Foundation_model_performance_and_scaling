"""Preview/prepare the bounded SFT workload; no runtime/model dependencies.

Usage: part7E.sh check | part7E.sh prepare --output results/7e/prepared_sft
"""
import argparse
from dataclasses import asdict
from importlib import import_module
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
settings = import_module("infrastructure.7e_sft.config")
workload = import_module("workloads.sft_workload")


def prepare(source, output, config, *, expected_sha256=workload.PART5_SFT_SHA256):
    """Write fresh plain JSONL splits, configuration and a provenance manifest.

    Validates source/config before creating output. Refuses existing output or
    writing inside preserved Part 5. Returns receipt; writes no tokenizer/model
    cache. Manifest records exact file and adapter hashes for future recovery.
    """
    config.validate()
    output = Path(output).resolve()
    source = Path(source).resolve()
    part5 = (ROOT.parent / "part5-alignment").resolve()
    if output == part5 or part5 in output.parents or output == source or output in source.parents:
        raise ValueError("Output must not overwrite the source or preserved Part 5")
    train, evaluation, receipt = workload.select_pairs(source, expected_sha256=expected_sha256,
        train_count=config.train_count, eval_count=config.eval_count, seed=config.seed)
    output.mkdir(parents=True, exist_ok=False)
    receipt.update(schema_version=1, config=asdict(config),
        adapter_sha256=hashlib.sha256(Path(workload.__file__).read_bytes()).hexdigest(), files={})
    for name, rows in (("train.jsonl", train), ("evaluation.jsonl", evaluation)):
        raw = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows).encode()
        (output/name).write_bytes(raw)
        receipt["files"][name] = dict(rows=len(rows), sha256=hashlib.sha256(raw).hexdigest())
    (output/"manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    (output/"resolved_config.json").write_text(json.dumps(config.preview(), indent=2)+"\n")
    return receipt


def main():
    """Parse preview/preparation inputs; print settings or write a new CPU data bundle.

    Defaults discover the sibling Part 5 source and Part 7 results. No cloud action
    is exposed in this increment; the runtime and environment remain pending.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare"))
    parser.add_argument("--source", type=Path, default=ROOT.parent/"part5-alignment/data/math/sft_filtered_repo.jsonl")
    parser.add_argument("--expected-sha256", default=workload.PART5_SFT_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT/"results/7e/prepared_sft")
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument("--model-repo", default=settings.SFTConfig.model_repo)
    parser.add_argument("--model-revision")
    args = parser.parse_args()
    config = settings.SFTConfig(model_repo=args.model_repo, model_revision=args.model_revision,
        train_count=args.train_count, eval_count=args.eval_count)
    receipt = config.preview() if args.action == "check" else prepare(
        args.source, args.output, config, expected_sha256=args.expected_sha256)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
