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
├── pii/                    # 2.4 – mask_emails, mask_phone_numbers, mask_ips
├── harmful_content/        # 2.5 – classify_nsfw, classify_toxic_speech
├── quality_rules/          # 2.6 – gopher_quality_filter
└── quality_classifier/     # 2.7 – classify_quality (train + inference)
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
```

## Results

All outputs go to `results/filtering_cc/`.