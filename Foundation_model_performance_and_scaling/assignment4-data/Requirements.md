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

### 2.5 Harmful Content

Unfiltered web dumps contain large volumes of text a model should not reproduce at inference time — including content from otherwise harmless sites (e.g., toxic Wikipedia talk-page comments). This section identifies two categories using **fastText classifiers from the Dolma project** trained on the Jigsaw Toxic Comments dataset:

- **NSFW** — pornography, profanity, or otherwise disturbing content
- **Toxic speech** — "rude, disrespectful, or unreasonable language that is likely to make someone leave a discussion"

**Classifier files:**

| Classifier | Download URL | Cluster path |
|------------|-------------|--------------|
| NSFW | `dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin` | `/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin` |
| Hate speech | `dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin` | `/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin` |

> Also available via `get_assets.sh` in the repo root.

---

#### Problem: `harmful_content` (6 points)

**(1) NSFW classifier.** Write a function that labels a string as NSFW or not, returning `(label, confidence_score)`.

**Deliverable:** Implement adapter `run_classify_nsfw`. Must pass (sanity check only — validate accuracy separately):
```bash
uv run pytest -k test_classify_nsfw
```

---

**(2) Toxic speech classifier.** Write a function that labels a string as toxic or not, returning `(label, confidence_score)`.

**Deliverable:** Implement adapter `run_classify_toxic_speech`. Must pass:
```bash
uv run pytest -k test_classify_toxic_speech
```

---

**(3)** What downstream problems might arise in a language model when these filters are applied to create the training set? How might you mitigate them?

**Deliverable:** 2–5 sentence response.

---

**(4)** Run your harmful content filters on WARC-extracted text. Look through 20 random examples and compare classifier predictions to your own judgment. Report errors, the fraction of harmful documents, and suitable confidence threshold(s).

**Deliverable:** 2–5 sentence response.

---

### 2.6 Quality Rules (Gopher Filters)

Even after language and harmful-content filtering, many pages are still low-quality for LM training: paywalled content, broken-link placeholders, login/signup forms, pages whose content is mostly images or video. The **Gopher paper** [Rae et al., 2021] defines a set of heuristic rules for exactly these cases. You will implement the following subset:

| Rule | Threshold |
|------|-----------|
| Word count | Must be between 50 and 100,000 words |
| Mean word length | Must be between 3 and 10 characters |
| Lines ending with ellipsis (`"..."`) | Must be ≤ 30% of all lines |
| Words containing ≥ 1 alphabetic character | Must be ≥ 80% of all words |

A document **fails** (returns `False`) if it violates any rule. See Appendix A of the Gopher paper for the full set of filters.

> For tokenizing into words, `nltk.word_tokenize` is recommended but not required.

---

#### Problem: `gopher_quality_filters` (3 points)

**(a)** Implement the Gopher quality filters described above.

**Deliverable:** Function `(text: str) -> bool` — `True` if the document passes all filters. Implement adapter `run_gopher_quality_filter`. Must pass:
```bash
uv run pytest -k test_gopher
```

---

**(b)** Run your filter on WARC-extracted text. Look through 20 random examples and compare filter decisions to your own judgment. Comment on any disagreements.

**Deliverable:** 2–5 sentence response.

---

### 2.7 Quality Classifier

Heuristic rules capture only syntactic quality signals. A complementary approach is a learned classifier. The key insight: **high-quality pages tend to be linked from high-quality sources**. OpenAI used Reddit karma-filtered links for WebText/GPT-2; an alternative is Wikipedia, whose external links tend to point to trusted pages [Touvron et al., 2023].

**Approach:** Use Wikipedia-linked URLs as positive examples, random CC pages as negatives, and train a **fastText classifier**. The resulting score can then filter pages across the full Common Crawl — the threshold trades off precision vs. recall.

**Wikipedia URL file:**
- Cluster: `/data/wiki/enwiki-20240420-extracted_urls.txt.gz` — 43.5M external links from the April 2024 English Wikipedia dump
- Download: `https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz`
- Subsample these URLs to get positive training examples. Apply earlier filters (language ID, Gopher rules, etc.) to improve positive example quality.

**Scraping URLs to WARC format:**
```bash
wget --timeout=5 \
     -i subsampled_positive_urls.txt \
     --warc-file=subsampled_positive_urls.warc \
     -O /dev/null
```

---

#### Problem: `quality_classifier` (15 points)

**(a)** Train a fastText quality classifier that takes text and returns a numeric quality score.

**Deliverable:** A trained quality classifier for use in part (b).

---

**(b)** Write a function that labels a page as high- or low-quality with a confidence score.

**Deliverable:** Function `(text: str) -> (label, confidence_score)`. Implement adapter `run_classify_quality`. Must pass:
```bash
uv run pytest -k test_classify_quality
```

---

## Proposed Code Structure for Part 2

As Part 2 has many subparts, each with its own implementation, `cs336_data/filtering_cc/` should be organized into subfolders:

```
cs336_data/
└── filtering_cc/
    ├── __init__.py
    ├── html_extraction/        # 2.2 — extract_text_from_html_bytes
    ├── language_id/            # 2.3 — identify_language
    ├── pii/                    # 2.4 — mask_emails, mask_phones, mask_ips
    ├── harmful_content/        # 2.5 — classify_nsfw, classify_toxic_speech
    ├── quality_rules/          # 2.6 — gopher_quality_filter
    └── quality_classifier/     # 2.7 — classify_quality (train + inference)
```

Exploration scripts (`explore_cc.py`, `compare_wet.py`) and shell scripts (`part_2_1.sh`, etc.) live in their respective subfolder alongside the implementation they support.

---

## Part 3: Deduplication

The web contains a significant amount of duplicate content. Some pages are exact duplicates of each other — think about archives, or default pages generated by standard tools, like 404 pages from popular Web servers. But most of the duplication happens at a more granular level. For example, consider all the question pages on Stack Overflow. While each page has unique content (e.g., the question, comments, the answers themselves), all pages also have a substantial amount of redundancy, such as the header, menu options and footer, that will be repeated in exact form when all such pages are rendered. In the first part of this section, we will deal with this kind of exact duplication, and later see how to handle approximate duplicates.

---

### 3.1 Exact Line Deduplication

A simple approach for deduplicating exact repetition is to only keep the lines in a document that are unique in the corpus. This turns out to be sufficient to eliminate a large portion of redundancy, such as the header and menu options mentioned above. In the simpler cases, when removing lines that are exactly repeated elsewhere, we are often left with the unique main content of each page (such as the question and answers on StackOverflow).

To do this, we can make one pass through the corpus to count how many occurrences of each line we observe. Then, in a second pass, we can rewrite each document by preserving only its unique lines.

Naïvely, the data structure to keep the counters can use as much space as it takes to store all the unique lines in the corpus. One simple memory efficiency trick is to instead use a hash of the line as the key, making the keys have a fixed size (instead of depending on the line's length).

---

#### Problem: `exact_deduplication` (3 points)

Write a function that takes a list of paths to input files and performs exact line deduplication on them. It should first count the frequency of each line in the corpus, using a hash to reduce memory, and then rewrite each file by only keeping its unique lines.

**Deliverable:** A function that performs exact line deduplication. Your function should take two arguments: (a) a list of paths to its input files, and (b) an output directory. It should rewrite each input file to the output directory with the same name, but deduplicate the content by removing lines that occur more than once in the set of input files. For example, if the input paths are `a/1.txt` and `a/2.txt`, and the output directory is `b/`, your function should write the files `b/1.txt` and `b/2.txt`.

Implement adapter `run_exact_line_deduplication`. Must pass:
```bash
uv run pytest -k test_exact_line_deduplication
```

---

### 3.2 MinHash + LSH Document Deduplication

Exact deduplication is useful for removing content that is repeated verbatim across multiple webpages, but does not handle cases where the document content slightly differs. For example, consider software license documents — the license document is often generated from a template that requires the year and the software author's name. As a result, the license file for one MIT-licensed project contains largely the same content as another MIT-licensed project, but they aren't exact duplicates. To remove this type of repeated, mostly-templated content, we need fuzzy deduplication. To efficiently perform fuzzy document-level deduplication, we will use **minhash with locality-sensitive hashing (LSH)**.

> For a more in-depth treatment of LSH and minhashing, see Chapter 3 of Leskovec et al. [2014], available at infolab.stanford.edu/ullman/mmds/ch3n.pdf.

#### MinHashing

To address the memory concern, we replace our set of n-grams document representation with a signature. We construct signatures such that comparing the signatures of two documents yields an approximation of the Jaccard similarity between the documents' respective sets of n-grams. Minhash signatures fulfill these properties.

To compute the minhash signature for a set of document n-grams `S = {s1, ..., sn}`, we need `k` distinct hash functions `h1, ..., hk`. Each hash function maps an n-gram to an integer. Given a hash function `hi`, the minhash of the set of document n-grams `S` is:

```
minhash(hi, S) = min(hi(s1), hi(s2), ..., hi(sn))
```

The signature of the document n-grams `S` is a vector in `R^k`, where each element `i` contains the minhash of `S` under the random hash function `hi`:

```
[minhash(h1, S), minhash(h2, S), ..., minhash(hk, S)]
```

For two document n-gram sets `S1` and `S2`, the Jaccard similarity between the sets is approximated by the proportion of columns with the same minhash value. For example, given signatures `[1, 2, 3, 2]` and `[5, 2, 3, 4]`, the Jaccard similarity is approximated as `2/4` (second and third columns match).

> These `k` distinct hash functions could be of the same family, but with different seeds. For example, MurmurHash3 is a family of hash functions where a particular seed instantiates a specific function within that family.

#### Locality-Sensitive Hashing (LSH)

Although minhashing gives us a memory-efficient document representation that preserves expected similarity between any document pair, we're still left with the need to compare all pairs of documents. LSH provides a way to efficiently bucket documents that are likely to have high similarity.

To apply LSH to document signatures (a vector in `R^k`), we divide the signature into `b` bands containing `r` minhashes each, with `k = b * r`. For example, with 100-element signatures we can use 2 bands of 50, 4 bands of 25, or 50 bands of 2. If two documents have the same hash values for a particular band, they will be clustered into the same bucket and considered candidate duplicates. Increasing the number of bands increases recall and decreases precision.

**Concrete example:** Document D1 with signature `[1, 2, 3, 4, 5, 6]` and D2 with signature `[1, 2, 3, 5, 1, 2]`. With 3 bands of 2:
- Band 1: D1=`[1,2]`, D2=`[1,2]` → **match** → candidate duplicates
- Band 2: D1=`[3,4]`, D2=`[3,5]` → no match
- Band 3: D1=`[5,6]`, D2=`[1,2]` → no match

Since D1 and D2 match in at least one band, they are candidate duplicates.

Once candidate duplicates are identified, compute the true Jaccard similarity between all candidate duplicate pairs and label those exceeding a set threshold as duplicates. Then cluster duplicate documents across buckets using union-find (transitive closure): if A≡B in one bucket and B≡C in another, treat A, B, C as a single cluster. Randomly remove all but one document from each cluster.

---

#### Problem: `minhash_deduplication` (8 points)

Write a function that takes a list of paths to input files and performs fuzzy document deduplication with minhash and LSH. In particular, your function should:
1. Compute minhash signatures for each document
2. Use LSH with the provided number of bands to identify candidate duplicates
3. Compute the true n-gram Jaccard similarity between candidate duplicates
4. Remove documents that exceed the given Jaccard threshold

To improve recall (following Penedo et al., 2023), normalize the text before computing minhash signatures and comparing Jaccard similarity by:
- Lowercasing
- Removing punctuation
- Normalizing whitespace
- Removing accents
- Applying NFD unicode normalization

**Deliverable:** A function that performs fuzzy document deduplication. Arguments:
- (a) A list of paths to input files
- (b) Number of hashes for computing minhash signatures
- (c) Number of bands for LSH
- (d) N-gram length (in words) for computing minhash signatures
- (e) An output directory

You may assume the number of hashes is evenly divisible by the number of bands.

Your function should rewrite each input file to the output directory with the same name, writing only documents that are (a) not candidate duplicates, or (b) randomly selected to be retained from clustered buckets. For example, if the input paths are `a/1.txt` and `a/2.txt`, and the output directory is `b/`, your function should write `b/1.txt` and `b/2.txt`.

Implement adapter `run_minhash_deduplication`. Must pass:
```bash
uv run pytest -k test_minhash_deduplication
```

---

*More parts to be added.*
