# Section 2.7 — Written Answers

## Problem: `quality_classifier`

---

### Training

A fastText binary classifier was trained to distinguish Wikipedia-linked pages (`wiki`) from raw Common Crawl pages (`cc`).

**Positive examples (wiki):** 1000 pages collected by subsampling 4000 URLs from
`enwiki-20240420-extracted_urls.txt.gz` (43.5M external links from the April 2024
English Wikipedia dump), fetching each with a 5s timeout (30 parallel workers), and
filtering with language ID (`en`, score ≥ 0.7) + Gopher quality rules. ~27% of
attempted URLs yielded a usable positive example.

**Negative examples (cc):** 1000 pages extracted from
`CC-MAIN-20250417135010-20250417165010-00065.warc.gz` without any language or quality
filtering — raw CC pages as they come off the crawl (mostly Chinese, multilingual,
navigation boilerplate, spam).

**Model:** fastText with bigrams (`wordNgrams=2`), 10 epochs, lr=0.5, dim=100,
softmax loss. Final training loss: **0.170** (vs. log(2) = 0.693 for a random
classifier), indicating strong learning signal.

**Note on CC negatives:** An earlier attempt filtered CC negatives to English-only
(`lang == "en"`, score ≥ 0.7), which produced a model with loss ≈ 0.694 (random).
After removing the language filter, the two classes became clearly separable: high-quality
English academic text vs. raw multilingual web noise.

---

### Formal test (`test_classify_quality`)

| Fixture | Expected | Predicted | Score | Result |
|---------|----------|-----------|-------|--------|
| `low_quality_cc.txt` (ESL forum boilerplate) | `cc` | `cc` | — | PASS |
| `high_quality_wiki_reference.txt` (Stanford Encyclopedia of Philosophy) | `wiki` | `wiki` | — | PASS |

---

### Empirical evaluation on 20 WARC records

| # | Label | Score | Human judgment | Agree? | URL (abbreviated) |
|---|-------|-------|----------------|--------|-------------------|
| 1 | cc | 1.000 | cc | ✓ | 0371rykj.com — Chinese adult spam |
| 2 | cc | 0.994 | cc | ✓ | chinatikfans.com — Chinese fan forum |
| 3 | **wiki** | 0.528 | wiki | ✓ | 13.usnccm.org — English conference site |
| 4 | cc | 0.973 | cc | ✓ | utchat888.com — Chinese adult chat |
| 5 | cc | 1.000 | cc | ✓ | 176766.cn — Chinese industrial spam |
| 6 | cc | 0.972 | cc | ✓ | tgtg97.com — Chinese adult chat |
| 7 | cc | 0.998 | cc | ✓ | 18sex.v340.info — Chinese adult chat |
| 8 | cc | 1.000 | high | ✗ | klimtoren.be — Dutch school blog |
| 9 | cc | 0.980 | high | ✗ | mysch.gr — Greek helpdesk forum |
| 10 | cc | 0.975 | high | ✗ | mysch.gr — Greek helpdesk login |
| 11 | cc | 0.985 | cc | ✓ | yhxzseo.com — Chinese news spam |
| 12 | cc | 0.982 | high | ✗ | 20com20.fr — Turkish Apache docs |
| 13 | cc | 0.656 | high | ✗ | 24ktcasino.net — English casino blog |
| 14 | cc | 1.000 | cc | ✓ | 2kgames.eu — 404 Not Found page |
| 15 | cc | 0.982 | cc | ✓ | yizhangting.com — Chinese health spam |
| 16 | cc | 1.000 | cc | ✓ | 303323.com — Chinese (HTML leakage) |
| 17 | cc | 0.967 | cc | ✓ | 30bad.com — Chinese video site |
| 18 | cc | 0.974 | high | ✗ | 312001.net — Chinese hospital site |
| 19 | cc | 0.963 | cc | ✓ | 354577.mwe075.com — Chinese adult chat |
| 20 | cc | 0.995 | cc | ✓ | schoollibrary.edu.pe.ca — English library |

**wiki: 1/20 (5%)   cc: 19/20 (95%)**
**Agreement with human judgment: 14/20 (70%)**

---

### Discussion

**What the classifier learned:** The model strongly separates non-English CC pages
from English Wikipedia-linked content. For Chinese, Greek, Turkish, and Dutch pages it
predicts `cc` with near-perfect confidence (≥0.97), because the training negatives were
raw multilingual CC pages. Record 3 (`13.usnccm.org`, English academic conference) is the
only `wiki` prediction — and correctly so, with moderate confidence (0.528), reflecting
that it is a real academic/professional site of the kind Wikipedia would link to.

**Disagreements (6/20):** Records 8–10, 12, 13, 18 are judged high-quality by a human
but predicted `cc`. The root cause is the same in all cases: the classifier was trained
on *English* Wikipedia-linked pages as positives, so non-English pages — even
high-quality Dutch, Greek, or Turkish content — have no learned positive signal. The
model cannot distinguish "good Dutch blog" from "Chinese spam" because neither looks
like the English training positives. This is a fundamental limitation of the
Wikipedia-link training approach when applied to multilingual corpora.

**Confidence threshold:** A threshold of `score(wiki) ≥ 0.5` would retain Record 3 and
reject all others. For a more permissive filter one could lower the threshold to 0.3,
but given this segment is mostly non-English, few additional pages would be recovered.