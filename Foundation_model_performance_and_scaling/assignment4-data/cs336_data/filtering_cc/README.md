# Part 2: Filtering Common Crawl

Implementation for Assignment 4 Part 2 — converting raw CC data into usable LM training data.

## Structure

Each section of Part 2 has its own subfolder:

```
filtering_cc/
├── look_at_data/           # 2.1 – Data exploration (no test adapter)
│   ├── explore_cc.py       # Parses WARC/WET, produces written-answer observations
│   └── part_2_1.sh         # Downloads CC files and runs explorer
│
├── html_extraction/        # 2.2 – extract_text_from_html_bytes
│   ├── extract.py          # Implementation (UTF-8 + encoding detection fallback)
│   ├── compare_wet.py      # Compares our output vs CC WET file side-by-side
│   └── part_2_2.sh         # Runs test + comparison
│
├── language_id/            # 2.3 – identify_language
│   ├── identify.py         # Implementation (fastText lid.176.bin, strips __label__ prefix)
│   └── part_2_3.sh         # Downloads lid.176.bin and runs tests
│
├── pii/                    # 2.4 – mask_emails, mask_phone_numbers, mask_ips
│   ├── mask.py             # Implementation (regex-based email, phone, IPv4 masking)
│   └── part_2_4.sh         # Runs all PII masking tests
│
├── harmful_content/        # 2.5 – classify_nsfw, classify_toxic_speech
│   ├── classify.py         # Implementation (Dolma fastText NSFW + hate-speech models)
│   └── part_2_5.sh         # Downloads models and runs tests
│
├── quality_rules/          # 2.6 – gopher_quality_filter
├── quality_rules/          # 2.6 – gopher_quality_filter
│   ├── gopher.py           # Implementation (word count, mean len, ellipsis, alpha ratio)
│   └── part_2_6.sh         # Runs all Gopher filter tests
│
└── quality_classifier/     # 2.7 – classify_quality (train + inference)
    ├── train.py            # Data prep + fastText training (wiki positives vs CC negatives)
    ├── classify.py         # Inference (loads quality_classifier.bin from assets/)
    └── part_2_7.sh         # Downloads wiki URLs, trains, runs test
```

## Quick Start

```bash
# 2.1 — Download CC sample files and generate observations
cd cs336_data/filtering_cc/look_at_data
./part_2_1.sh
# → results/filtering_cc/look_at_cc_observations.txt

# 2.2 — Run extract_text test and compare vs WET
cd cs336_data/filtering_cc/html_extraction
./part_2_2.sh
# → results/filtering_cc/wet_comparison.txt

# 2.3 — Download lid.176.bin and run language identification tests
cd cs336_data/filtering_cc/language_id
./part_2_3.sh
```

## Results

All outputs go to `results/filtering_cc/`.