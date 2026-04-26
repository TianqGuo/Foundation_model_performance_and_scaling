# Part 2: Filtering Common Crawl

Implementation for Assignment 4 Part 2 — converting raw CC data into usable LM training data.

## Files

| File | Description |
|------|-------------|
| `extract.py` | `extract_text_from_html_bytes()` — converts raw HTML bytes to plain text using resiliparse |
| `explore_cc.py` | Analyzes WARC/WET sample files; produces observations for written questions (a)–(d) |
| `part_2_1.sh` | Downloads CC sample files and runs the explorer; entry point for Section 2.1 |

## Quick Start

```bash
cd cs336_data/filtering_cc
./part_2_1.sh
# Output: ../../results/filtering_cc/look_at_cc_observations.txt
```

## Results

All outputs go to `results/filtering_cc/`.