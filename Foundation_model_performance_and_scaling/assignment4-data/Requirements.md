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

### 2.2 HTML to Text Conversion

Extracting text from HTML is non-trivial. Visible content (e.g., `<p>` tags) still includes navigation menus, footers, and boilerplate that a reader would not consider "main content". This assignment uses the **Resiliparse** library for extraction. Resiliparse also handles a more basic problem: detecting the encoding of raw HTML bytes. Although ~98% of web pages use UTF-8, the pipeline must be robust to other encodings.

> Use `fastwarc` to iterate over WARC records:
> ```python
> from fastwarc.warc import ArchiveIterator, WarcRecordType
> ```

---

#### Problem: `extract_text` (3 points)

**(a)** Write a function that extracts plain text from raw HTML bytes.
- Use `resiliparse.extract.html2text.extract_plain_text` for extraction.
- Decode bytes to Unicode first. Input may not be UTF-8 — use `resiliparse.parse.encoding.detect_encoding()` as a fallback.

**Deliverable:** Function taking HTML `bytes` → extracted `str`. Implement adapter `run_extract_text_from_html_bytes`. Must pass:
```bash
uv run pytest -k test_extract_text_from_html_bytes
```

---

**(b)** Run your extraction function on a single WARC file and compare its output to the corresponding WET file. What differences and similarities do you notice? Which extraction seems better?

**Deliverable:** 2–3 sentence response.

---

### 2.3 Language Identification

The web contains pages in thousands of languages. Most LM training sets limit to a small set of languages (typically English-centric), since training truly multilingual models at standard compute budgets is challenging. This section uses **fastText** for language identification.

- **Model:** `lid.176.bin` — download from [fasttext.cc](https://fasttext.cc/docs/en/language-identification.html), or find it at `/data/classifiers/lid.176.bin` on the Together cluster.

---

#### Problem: `language_identification` (6 points)

**(a)** Write a function that takes a Unicode string and returns the top predicted language and a confidence score in `[0, 1]`.

**Deliverable:** Function returning `(language_id, score)`. Implement adapter `run_identify_language`. Note: tests expect `"en"` for English and `"zh"` for Chinese — remap fastText labels if needed. Must pass:
```bash
uv run pytest -k test_identify_language
```

---

**(b)** What issues could arise downstream in a language model from errors in language identification? In a higher-stakes deployment scenario, how would you mitigate them?

**Deliverable:** 2–5 sentence response.

---

**(c)** Run your language identification system on WARC-extracted text (using your extraction function from 2.2). Manually label 20 random examples and compare against classifier predictions. Report errors, the fraction of English documents, and a suitable confidence threshold for filtering.

**Deliverable:** 2–5 sentence response.

---

### 2.4 Personal Identifiable Information (PII)

Web data contains large quantities of PII — email addresses, phone numbers, IP addresses — that we don't want a user-facing LM to reproduce. A standard step is to **mask** these out in the training data before use.

---

#### Problem: `mask_pii` (3 points)

**(1) Mask emails.** Replace all email addresses in a string with `"|||EMAIL_ADDRESS|||"`.

**Deliverable:** Function returning `(masked_str, count)`. Implement adapter `run_mask_emails`. Must pass:
```bash
uv run pytest -k test_mask_emails
```

---

**(2) Mask phone numbers.** Replace all phone numbers with `"|||PHONE_NUMBER|||"`. Must handle at minimum common US formats and be robust to minor syntactic variation.

**Deliverable:** Function returning `(masked_str, count)`. Implement adapter `run_mask_phone_numbers`. Must pass:
```bash
uv run pytest -k test_mask_phones
```

---

**(3) Mask IPv4 addresses.** Replace all IPv4 addresses (4 octets ≤ 255 separated by dots) with `"|||IP_ADDRESS|||"`.

**Deliverable:** Function returning `(masked_str, count)`. Implement adapter `run_mask_ips`. Must pass:
```bash
uv run pytest -k test_mask_ips
```

---

**(4)** What downstream problems might arise in a language model when these filters are naïvely applied to the training set? How might you mitigate them?

**Deliverable:** 2–5 sentence response.

---

**(5)** Run your PII masking functions on WARC-extracted text. Look through 20 random examples where a replacement was made. Report examples of false positives and false negatives.

**Deliverable:** 2–5 sentence response.

---

*More parts to be added.*
