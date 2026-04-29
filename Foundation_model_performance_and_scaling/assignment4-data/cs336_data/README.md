# CS336 Assignment 4 (Data) — Implementation

Implementation of Parts 2 and 3 of Assignment 4.

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
└── deduplication/              # Part 3 — Deduplication
    └── exact_line.py           # 3.1 – exact_line_deduplication
```

## Setup

```bash
# Create virtualenv and install dependencies
uv sync

# Download model assets (lid.176.bin, Dolma NSFW/hate-speech models)
./get_assets.sh
```

## Part 2: Filtering Common Crawl

Each section has its own subfolder with an implementation file and a `part_2_N.sh` runner.

```bash
# 2.1 — Download CC sample files and generate observations
cd cs336_data/filtering_cc/look_at_data && ./part_2_1.sh
# → results/filtering_cc/look_at_cc_observations.txt

# 2.2 — HTML extraction + WET comparison
cd cs336_data/filtering_cc/html_extraction && ./part_2_2.sh
# → results/filtering_cc/wet_comparison.txt

# 2.3 — Language identification
cd cs336_data/filtering_cc/language_id && ./part_2_3.sh

# 2.4 — PII masking
cd cs336_data/filtering_cc/pii && ./part_2_4.sh

# 2.5 — Harmful content classification
cd cs336_data/filtering_cc/harmful_content && ./part_2_5.sh

# 2.6 — Gopher quality filters
cd cs336_data/filtering_cc/quality_rules && ./part_2_6.sh

# 2.7 — Quality classifier (train + evaluate)
cd cs336_data/filtering_cc/quality_classifier && ./part_2_7.sh
# → cs336_data/assets/quality_classifier.bin
```

## Part 3: Deduplication

```bash
# 3.1 — Exact line deduplication
uv run pytest -k test_exact_line_deduplication -v

# 3.2 — MinHash + LSH deduplication (not yet implemented)
uv run pytest -k test_minhash_deduplication -v
```

## Results

All written-answer outputs go to `results/filtering_cc/`:

| File | Section |
|------|---------|
| `look_at_cc_observations.txt` | 2.1 |
| `extract_text_answers.md` | 2.2 |
| `language_id_answers.md` | 2.3 |
| `pii_answers.md` | 2.4 |
| `harmful_content_answers.md` | 2.5 |
| `gopher_answers.md` | 2.6 |
| `quality_classifier_answers.md` | 2.7 |

## Running All Tests

```bash
uv run pytest -v
```