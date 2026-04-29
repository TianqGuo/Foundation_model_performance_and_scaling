# CS336 Assignment 4 (Data) — Implementation

Implementation of Parts 2, 3, and 4 of Assignment 4.

## Structure

```
cs336_data/
├── assets/                     # Downloaded model files (lid.176.bin, dolma models, quality_classifier.bin)
├── filtering_cc/               # Part 2 — Filtering Common Crawl
│   ├── look_at_data/           # 2.1 – Data exploration
│   ├── html_extraction/        # 2.2 – extract_text_from_html_bytes
│   ├── language_id/            # 2.3 – identify_language
│   ├── pii/                    # 2.4 – mask_emails, mask_phone_numbers, mask_ips
│   ├── harmful_content/        # 2.5 – classify_nsfw, classify_toxic_speech
│   ├── quality_rules/          # 2.6 – gopher_quality_filter
│   └── quality_classifier/     # 2.7 – classify_quality (train + inference)
├── deduplication/              # Part 3 — Deduplication
│   ├── exact_line/             # 3.1 – exact_line_deduplication
│   └── minhash/                # 3.2 – minhash_deduplication
└── leaderboard/                # Part 4 — Leaderboard pipeline
    └── filter_data/            # 4.filter – parallel WET filtering
```

## Setup

```bash
# Create virtualenv and install dependencies
uv sync

# Download model assets (lid.176.bin, Dolma NSFW/hate-speech models)
./get_assets.sh

# Train quality classifier (required before Part 4)
cd cs336_data/filtering_cc/quality_classifier && ./part_2_7.sh
```

## Part 2: Filtering Common Crawl

Each section has its own subfolder with an implementation file and a `part_2_N.sh` runner.

```bash
cd cs336_data/filtering_cc/look_at_data    && ./part_2_1.sh  # → results/filtering_cc/look_at_cc_observations.txt
cd cs336_data/filtering_cc/html_extraction && ./part_2_2.sh  # → results/filtering_cc/wet_comparison.txt
cd cs336_data/filtering_cc/language_id     && ./part_2_3.sh
cd cs336_data/filtering_cc/pii             && ./part_2_4.sh
cd cs336_data/filtering_cc/harmful_content && ./part_2_5.sh
cd cs336_data/filtering_cc/quality_rules   && ./part_2_6.sh
cd cs336_data/filtering_cc/quality_classifier && ./part_2_7.sh  # → cs336_data/assets/quality_classifier.bin
```

## Part 3: Deduplication

```bash
cd cs336_data/deduplication/exact_line && ./part_3_1.sh  # or: uv run pytest -k test_exact_line_deduplication
cd cs336_data/deduplication/minhash    && ./part_3_2.sh  # or: uv run pytest -k test_minhash_deduplication
```

## Part 4: Leaderboard

```bash
# Quick smoke test (5 files)
cd cs336_data/leaderboard/filter_data && ./part_4_filter.sh --limit 5

# Full run (5000 files, cluster)
cd cs336_data/leaderboard/filter_data && ./part_4_filter.sh
# → data/filtered/*.txt  +  data/filtered/filter_stats.json
```

## Results

All written-answer outputs:

| File | Section |
|------|---------|
| `results/filtering_cc/look_at_cc_observations.txt` | 2.1 |
| `results/filtering_cc/extract_text_answers.md` | 2.2 |
| `results/filtering_cc/language_id_answers.md` | 2.3 |
| `results/filtering_cc/pii_answers.md` | 2.4 |
| `results/filtering_cc/harmful_content_answers.md` | 2.5 |
| `results/filtering_cc/gopher_answers.md` | 2.6 |
| `results/filtering_cc/quality_classifier_answers.md` | 2.7 |
| `data/filtered/filter_stats.json` | 4 (filter stats) |

## Running All Tests

```bash
uv run pytest -v
```