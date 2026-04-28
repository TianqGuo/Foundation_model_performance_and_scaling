# Section 2.5 — Written Answers

## Problem: `harmful_content`

---

### Part (3) — Downstream problems from applying harmful content filters

**False negatives (under-filtering):** The Dolma classifiers are trained on English Wikipedia talk-page comments (Jigsaw dataset). They are therefore essentially blind to harmful content in other languages. Chinese, Arabic, Russian, or Spanish NSFW/toxic text will pass through with near-perfect non-harmful confidence. For a multilingual training corpus like Common Crawl this is a severe gap — a large fraction of harmful content will never be caught.

**False positives (over-filtering):** Content that superficially resembles toxic Wikipedia comments may be removed even if it is harmless in context — e.g., academic papers discussing violence or hate speech as a subject, fiction with explicit dialogue, legal/medical text, or news reporting on harmful events. This can create systematic gaps in domain coverage and introduce bias against certain genres, fields, or cultures.

**Downstream model effects:** Training on a corpus where the filter has high false-negative rate for non-English harmful content means the model will learn to generate harmful content in those languages. Meanwhile, over-filtering benign English text in domains like medicine or law degrades the model's utility precisely where factual accuracy matters most.

**Mitigations:** (1) train or fine-tune classifiers on multilingual corpora; (2) apply a language-identification step first and use language-specific classifiers; (3) combine keyword blocklists for common non-English harmful patterns; (4) use a higher-capacity multilingual model (e.g., mBERT or multilingual fastText trained on diverse harmful-content data).

---

### Part (4) — Running classifiers on WARC-extracted text (20 records, 300 total)

**Results:** Running both classifiers on 300 WARC-extracted records, **0 pages were classified as NSFW and 0 as toxic** despite several pages being clearly harmful:

| # | URL | Manual label | NSFW pred | Confidence | Notes |
|---|-----|-------------|-----------|------------|-------|
| 1 | 0371rykj.com/ipfhsb/34.html | NSFW | non-nsfw | 1.000 | Chinese adult spam with explicit sexual keywords |
| 4 | 176.utchat888.com | NSFW | non-nsfw | 0.986 | Traditional Chinese adult video chat platform |
| 6 | 1796370.tgtg97.com | NSFW | non-nsfw | 0.990 | Traditional Chinese adult chat, explicit profile info |
| 7 | 18sex.v340.info | NSFW | non-nsfw | 1.000 | Adult video chat (URL contains "18sex") |
| 19 | 354577.mwe075.com | NSFW | non-nsfw | 0.947 | Traditional Chinese adult chat with escort profiles |

**Root cause:** The entire first WARC segment sampled is dominated by Chinese-language pages (14/20). Because the classifier is trained exclusively on English Wikipedia comments, it has no bigram signal for Chinese characters and defaults to non-harmful with high confidence. This is a fundamental language mismatch, not a threshold problem.

**Fraction of harmful documents:** ~0% detected by the classifier, but manual inspection identifies at least 5/20 (25%) of the first 20 records as clearly NSFW. The classifier severely underestimates the true harmful-content rate in this segment.

**Confidence threshold:** Since the classifier produces near-identical high-confidence scores for both genuinely safe and genuinely harmful non-English pages, adjusting the threshold does not improve recall for this corpus. A useful threshold only exists for English-language content, where scores below ~0.7 indicate borderline cases worth closer review.