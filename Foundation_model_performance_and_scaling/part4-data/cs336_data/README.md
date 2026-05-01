# LM Data Pipeline — Implementation

Implementation of the filtering, deduplication, and training pipeline. See the [root README](../README.md) for results and an overview.

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
└── leaderboard/                # Part 4 — Full training pipeline
    ├── download_wet/           # 4.1 – download CC WET files
    ├── filter_data/            # 4.2 – parallel WET filtering
    ├── inspect_filtered_data/  # 4.3 – inspect filter output
    ├── tokenize_data/          # 4.4 – tokenize to .bin
    ├── train_quality_classifier/ # 4.5 – train classifier on VM
    ├── download_paloma/        # 4.6 – download Paloma validation set
    └── train_model/            # 4.7 – launch training
```

## Setup

```bash
uv sync          # install dependencies
./get_assets.sh  # download lid.176.bin, Dolma models

# Train quality classifier (required before running the pipeline)
cd cs336_data/filtering_cc/quality_classifier && ./part_2_7.sh
```

### Cloud VM Deployment

```bash
# One-shot setup: installs deps, downloads assets, trains quality classifier
bash setup_vm.sh

# Then run the full pipeline:
bash cs336_data/leaderboard/download_wet/part_4_download.sh 600
bash cs336_data/leaderboard/filter_data/part_4_filter.sh
bash cs336_data/leaderboard/tokenize_data/part_4_tokenize.sh
export WANDB_ENTITY=<your-wandb-username>
bash cs336_data/leaderboard/train_model/part_4_train.sh
```

---

## Part 2: Text Extraction & Quality Filtering

Each section has its own subfolder with an implementation file and a `part_2_N.sh` runner.

```bash
cd cs336_data/filtering_cc/look_at_data      && ./part_2_1.sh  # → results/filtering_cc/look_at_cc_observations.txt
cd cs336_data/filtering_cc/html_extraction   && ./part_2_2.sh  # → results/filtering_cc/wet_comparison.txt
cd cs336_data/filtering_cc/language_id       && ./part_2_3.sh
cd cs336_data/filtering_cc/pii               && ./part_2_4.sh
cd cs336_data/filtering_cc/harmful_content   && ./part_2_5.sh
cd cs336_data/filtering_cc/quality_rules     && ./part_2_6.sh
cd cs336_data/filtering_cc/quality_classifier && ./part_2_7.sh  # → cs336_data/assets/quality_classifier.bin
```

---

## Part 3: Deduplication

```bash
cd cs336_data/deduplication/exact_line && ./part_3_1.sh
cd cs336_data/deduplication/minhash    && ./part_3_2.sh
```

---

## Part 4: Full Training Pipeline

End-to-end pipeline from raw Common Crawl to a trained language model evaluated on Paloma.

### Pipeline Steps

#### Step 1 — Download CC WET files

```bash
# Download N WET files from CC-MAIN-2025-18 (default 100)
bash cs336_data/leaderboard/download_wet/part_4_download.sh 600
# → data/CC/*.warc.wet.gz
```

#### Step 2 — Filter

```bash
# Smoke test (5 files)
bash cs336_data/leaderboard/filter_data/part_4_filter.sh --limit 5

# Full run
bash cs336_data/leaderboard/filter_data/part_4_filter.sh
# → data/filtered/*.txt  +  data/filtered/filter_stats.json
```

Filter pipeline (applied in order, first rejection wins):
1. Minimum length — skip records < 100 chars
2. Language ID — keep English (score ≥ 0.65)
3. Gopher rules — word count, mean word length, ellipsis ratio, alpha ratio
4. Quality classifier — keep pages with wiki-probability ≥ 0.3
5. NSFW filter — discard NSFW pages (confidence ≥ 0.8)
6. PII masking — mask emails, phones, IPs (always applied)

**Filter results (600 WET files, CC-MAIN-2025-18):**

| Stage | Removed | % of total |
|---|---|---|
| Non-English | 10,603,105 | 64.8% |
| Low quality | 3,723,914 | 22.8% |
| Gopher fail | 481,112 | 2.9% |
| Too short | 262,401 | 1.6% |
| NSFW | 2,810 | 0.02% |
| **Kept** | **1,292,650** | **7.9%** |
| Total records | 16,365,992 | — |

Processing time: 899s (1.5s/file) across 16 workers.

#### Step 3 — Tokenize

```bash
bash cs336_data/leaderboard/tokenize_data/part_4_tokenize.sh
# → data/tokenized/train.bin  (GPT-2 tokenizer, np.uint16)
```

#### Step 4 — Train

```bash
export WANDB_ENTITY=<your-wandb-username>
bash cs336_data/leaderboard/train_model/part_4_train.sh
# → cs336-basics/output/your_data/model.pt
```

The script auto-detects GPU count, patches `cs336-basics/configs/experiment/your_data.yaml`,
and downloads the Paloma validation set if not already present.

### Model Architecture

| Hyperparameter | Value |
|---|---|
| Parameters (non-embedding) | 84.95M |
| Vocabulary size | 50,257 |
| Context length | 512 |
| d_model | 768 |
| Layers | 12 |
| Attention heads | 12 |
| d_ff | 2,048 |
| dtype | bfloat16 |

### Training Results

Trained for **100,000 steps** on 2× A100 GPUs. Total tokens per step: 131,072.

| Metric | Value |
|---|---|
| Train loss (final) | ~2.5 |
| Best eval loss (Paloma) | ~4.3 (step ~60k) |
| Final eval loss (Paloma) | ~4.5 |
| Peak learning rate | 1e-3 (cosine decay) |

Training curves: `results/screenshots/losses_and_lr.png`

---

## Results

All written-answer outputs:

| File | Section |
|------|---------|
| `results/filtering_cc/look_at_cc_observations.txt` | 2.1 – CC data observations |
| `results/filtering_cc/look_at_cc_answers.md` | 2.1 – written answers |
| `results/filtering_cc/wet_comparison.txt` | 2.2 – HTML vs WET comparison |
| `results/filtering_cc/extract_text_answers.md` | 2.2 – written answers |
| `results/filtering_cc/language_id_answers.md` | 2.3 |
| `results/filtering_cc/pii_answers.md` | 2.4 |
| `results/filtering_cc/harmful_content_answers.md` | 2.5 |
| `results/filtering_cc/gopher_answers.md` | 2.6 |
| `results/filtering_cc/quality_classifier_answers.md` | 2.7 |
| `data/filtered/filter_stats.json` | 4 – filter statistics |
| `results/screenshots/losses_and_lr.png` | 4 – training curves |
| `cs336-basics/output/your_data/model.pt` | 4 – final model checkpoint |

## Running All Tests

```bash
uv run pytest -v
```