# CS336 Assignment 4 (Data): Filtering Language Modeling Data
**Version 1.0.4 | CS336 Staff | Spring 2025**

---

## Part 1: Assignment Overview

### What You Will Implement
1. Convert Common Crawl HTML to text.
2. Filter the extracted text with various methods (e.g., harmful content, personal identifiable information, etc.).
3. Deduplicate the training data.

### What You Will Run
1. Train language models on different datasets to better understand the impact of particular processing decisions on performance.

### Code Structure

| Path | Description |
|------|-------------|
| `cs336-basics/*` | Code for training the model from assignment 1 (slightly optimized — PyTorch-native cross-entropy kernel, multi-GPU DDP training script). Used to train on filtered data and make leaderboard submission. |
| `cs336_data/*` | Where you write your code for assignment 4. Empty module — implement from scratch. |
| `tests/*.py` | All tests that must pass. Tests invoke hooks defined in `tests/adapters.py`. Implement adapters to connect your code to the tests. |
| `README.md` | Details on expected directory structure and environment setup. |

### How to Submit
- **Gradescope**: Submit `writeup.pdf` (typeset responses to all written questions) and `code.zip` (all written code).
- **Leaderboard**: Submit a PR to `github.com/stanford-cs336/assignment4-data-leaderboard` (see leaderboard README for detailed instructions).

---

## Part 2: Filtering Common Crawl

Most researchers sourcing training data don't build their own web crawler — they use publicly available crawls. The most popular is **Common Crawl**, a non-profit providing a free corpus of web pages ("over 250 billion pages spanning 17 years").

Turning raw CC dumps into usable LM training data requires significant work: extracting text from HTML, filtering low-quality/duplicate/harmful/PII-containing pages, etc. This assignment sets up such a pipeline.

### 2.1 Looking at the Data

CC data is available in three formats:

| Format | Contents |
|--------|----------|
| **WARC** (Web ARChive) | Raw CC data: page IDs, URLs, metadata, HTTP request details, raw page content (HTML) |
| **WAT** (Web Archive Transformation) | Higher-level metadata extracted from WARC as JSON (e.g., links, page title) |
| **WET** (Web Extracted Text) | Extracted plain text from raw HTML pages |

**Sample files (April 2018 crawl):**
```bash
# Download sample WARC file
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz

# Download corresponding WET file
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/wet/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz
```

On the Together cluster:
- `/data/CC/example.warc.gz`
- `/data/CC/example.warc.wet.gz`

> **WARNING:** These files contain completely unfiltered Internet pages, which may include a large volume of potentially harmful content.

**Browsing the files:**
```bash
zcat /data/CC/example.warc.gz | less      # WARC file
zcat /data/CC/example.warc.wet.gz | less  # WET file
# Navigate with arrow keys / Page Up / Page Down. Press 'q' to exit.
```

---

### Problem: `look_at_cc` (4 points)

**(a)** Download the WARC file (or use the cluster copy). Look at the first page in the file.
- What is its URL? Is it still accessible?
- Can you tell what the page is about from the raw HTML?

**Deliverable:** 2–3 sentence response.

---

**(b)** Look at the corresponding WET file. Note that WET files include HTTP headers (e.g., `Content-Length`) that are not part of the extracted text.
- Are there parts of the extracted text that should have been filtered out by the extractor?
- What might go wrong training a model on text like this? What useful information could a model extract?

**Deliverable:** 3–4 sentence response.

---

**(c)** What makes a good training example is highly contextual.
- Describe one application domain where this example would be **useful** in training data.
- Describe one domain where it would **not** be useful.

**Deliverable:** 1–2 sentence response.

---

**(d)** Look through **25 more WET records**. For each, briefly note: language (if identifiable), domain name, page type, and any other relevant observations. How many examples does it take until you see what you'd consider a "high-quality" webpage?

**Deliverable:**
- Brief annotations for all 25 documents (language, domain, page type, misc. notes).
- The number of examples until you encounter a high-quality example.

---

*More parts to be added.*
