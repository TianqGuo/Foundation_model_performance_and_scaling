# Section 2.3 — Written Answers

## Problem: `language_identification`

---

### Part (b) — Downstream issues from language identification errors

Errors in language identification will silently contaminate the training corpus: false negatives (non-English pages passing the filter) inject foreign-language text into an English LM's training set, while false positives (English pages being rejected) reduce data diversity and coverage. For a model trained on the resulting noisy corpus, perplexity on clean English test data will be degraded, and the model may learn spurious cross-lingual associations (e.g., mixing Chinese and English in generation). Short pages, error pages, and code-heavy pages are especially prone to misclassification because the signal is sparse, so even a classifier with >99% accuracy on long text will have much higher error rates on the tail of the distribution.

In a higher-stakes deployment scenario, mitigation strategies include: (1) using a higher confidence threshold and discarding borderline predictions rather than forcing a binary keep/reject decision; (2) running a second, independent language identifier and only keeping documents where both agree; (3) post-hoc auditing a sample of filtered-out documents to estimate false-positive rate; and (4) imposing a minimum text-length requirement before running the classifier, since very short pages produce unreliable predictions.

---

### Part (c) — Language ID on WARC-extracted text (20 records)

**Results table:**

| # | Predicted | Confidence | Manual Label | Correct? | URL (abbreviated) |
|---|-----------|------------|--------------|----------|-------------------|
| 1 | zh | 0.7374 | zh | ✓ | 0371rykj.com — Chinese industrial spam |
| 2 | zh | 0.9228 | zh | ✓ | chinatikfans.com — Chinese fan forum |
| 3 | en | 0.8342 | en | ✓ | 13.usnccm.org — English conference site |
| 4 | zh | 0.9952 | zh | ✓ | utchat888.com — Traditional Chinese chat |
| 5 | zh | 0.9239 | zh | ✓ | 176766.cn — Chinese industrial site |
| 6 | zh | 0.9060 | zh | ✓ | 178mh.com — Chinese error page (3 words) |
| 7 | zh | 0.9582 | zh | ✓ | tgtg97.com — Traditional Chinese chat |
| 8 | zh | 0.9808 | zh | ✓ | 18sex.v340.info — Traditional Chinese |
| 9 | nl | 0.9234 | nl | ✓ | klimtoren.be — Dutch school blog |
| 10 | el | 0.9999 | el | ✓ | mysch.gr — Greek helpdesk forum |
| 11 | el | 0.9999 | el | ✓ | mysch.gr — Greek helpdesk (login page) |
| 12 | zh | 0.9812 | zh | ✓ | yhxzseo.com — Chinese news/spam |
| 13 | tr | 0.8746 | tr | ✓ | 20com20.fr — Apache docs in Turkish |
| 14 | en | 0.9301 | en | ✓ | 24ktcasino.net — English casino blog |
| 15 | **da** | **0.2814** | **en** | **✗** | 2kgames.eu — 404 Not Found (nginx) |
| 16 | zh | 0.9615 | zh | ✓ | yizhangting.com — Chinese health site |
| 17 | zh | 0.9707 | zh | ✓ | 303323.com — Chinese site (HTML leak) |
| 18 | zh | 0.9298 | zh | ✓ | 30bad.com — Chinese video site |
| 19 | zh | 0.9965 | zh | ✓ | 312001.net — Chinese hospital site |
| 20 | zh | 0.9559 | zh | ✓ | 354577.mwe075.com — Traditional Chinese |

**Errors:** 1 out of 20 (Record 15). The page contained only "404 Not Found\nnginx" — 4 tokens with no real language signal. The classifier assigned Danish (da) with very low confidence (0.28), which is essentially a random guess on near-empty content.

**Fraction of English documents:** 2/20 = 10%. Note this is not representative of CC overall — this WARC segment happens to be heavily Chinese-dominated (14/20 Chinese, 2/20 Greek, 1/20 Dutch, 1/20 Turkish, 2/20 English).

**Confidence threshold:** A threshold of **0.7** would correctly reject the only misclassification (confidence 0.28) while retaining all true positives (next lowest confidence is 0.74 for Record 1). In practice, this threshold also discards pages with insufficient text to be reliably classified, which is desirable — such pages are unlikely to be high-quality training examples regardless.