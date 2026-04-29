# CLAUDE.md — CS336 Assignment 4 (Data)

Extends global `~/.claude/CLAUDE.md` with project-specific conventions.

## Module Layout

```
cs336_data/
├── assets/                  # Downloaded model files (lid.176.bin, dolma models, quality_classifier.bin)
├── filtering_cc/            # Part 2 — each section in its own subfolder + part_2_N.sh
└── deduplication/           # Part 3 — each section in its own subfolder + part_3_N.sh
```

## Results

```
results/
└── filtering_cc/            # Part 2 written answers and evaluation outputs
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
- `ROOT="$(cd ../../.. && pwd)"` from a section subfolder (3 levels deep)
