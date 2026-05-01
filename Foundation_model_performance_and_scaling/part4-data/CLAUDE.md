# CLAUDE.md — CS336 Assignment 4 (Data)

Extends global `~/.claude/CLAUDE.md` with project-specific conventions.

## Module Layout

```
cs336_data/
├── assets/                  # Downloaded model files (lid.176.bin, dolma models, quality_classifier.bin)
├── filtering_cc/            # Part 2 — each section in its own subfolder + part_2_N.sh
├── deduplication/           # Part 3 — each section in its own subfolder + part_3_N.sh
└── leaderboard/             # Part 4 — each step in its own subfolder + part_4_*.sh
    ├── download_wet/        # download CC WET files
    ├── filter_data/         # parallel WET filtering
    ├── inspect_filtered_data/
    ├── tokenize_data/       # tokenize to .bin
    ├── train_quality_classifier/
    ├── download_paloma/     # download Paloma validation set
    └── train_model/         # launch training
```

## Results

```
results/
├── filtering_cc/            # Part 2 written analysis and evaluation outputs
├── leaderboard/             # Part 4 inspection outputs
└── screenshots/             # Training curves (wandb PNG exports)
```

## Assets

Downloaded via `get_assets.sh` at the repo root. On the Together cluster, files are at `/data/classifiers/` and `/data/wiki/` — use symlinks.

## Tests

```bash
uv run pytest -v                          # all tests
uv run pytest -k <test_name> -v          # single test
```

Adapters: `tests/adapters.py` — one function per problem, lazy inline imports.

## Shell Script Naming

- Part 2: `part_2_1.sh` through `part_2_7.sh`
- Part 3: `part_3_1.sh`, `part_3_2.sh`
- Part 4: `part_4_*.sh` (download, filter, tokenize, train, etc.)
- `ROOT="$(cd ../../.. && pwd)"` from a section subfolder (3 levels deep)

## Cloud VM Deployment (vast.ai)

`setup_vm.sh` at the repo root handles full VM setup: installs uv, syncs deps, downloads assets, trains quality classifier.

Full pipeline after setup:
```bash
bash cs336_data/leaderboard/download_wet/part_4_download.sh 600
bash cs336_data/leaderboard/filter_data/part_4_filter.sh
bash cs336_data/leaderboard/tokenize_data/part_4_tokenize.sh
export WANDB_ENTITY=<username>
bash cs336_data/leaderboard/train_model/part_4_train.sh
```

**wandb issue on cloud VMs:** Entity `gtqscat` fails with `upsertBucket: entity not found` even after successful login. Workaround: `export WANDB_MODE=disabled` before running the train script.

## README Style

Write READMEs as project showcases for external visitors — not academic writeups. Never use words like "assignment", "deliverable", or "problem statement". Frame everything as outcomes and decisions.
