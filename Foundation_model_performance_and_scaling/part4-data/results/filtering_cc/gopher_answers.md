# Section 2.6 — Written Answers

## Problem: `gopher_quality_filters`

---

### Part (b) — Running Gopher filter on WARC-extracted text (20 records)

**Results table:**

| # | Filter | Human | Agree? | Words | MeanLen | Alpha% | Failure reason | URL (abbreviated) |
|---|--------|-------|--------|-------|---------|--------|----------------|-------------------|
| 1 | FAIL | low | ✓ | 312 | 11.1 | 58.7% | alpha < 80% | 0371rykj.com — Chinese adult spam |
| 2 | FAIL | low | ✓ | 191 | 9.6 | 74.9% | alpha < 80% | chinatikfans.com — Chinese fan forum |
| 3 | PASS | high | ✓ | 333 | 5.8 | 85.6% | — | 13.usnccm.org — English conference |
| 4 | PASS | low | ✗ | 236 | 5.6 | 90.3% | — | utchat888.com — Chinese adult chat (NSFW) |
| 5 | FAIL | low | ✓ | 274 | 7.9 | 55.1% | alpha < 80% | 176766.cn — Chinese industrial site |
| 6 | FAIL | low | ✗ | 328 | 14.6 | 87.8% | mean len > 10 | tgtg97.com — Chinese adult chat (wrong reason) |
| 7 | FAIL | low | ✓ | 487 | 3.8 | 73.1% | alpha < 80% | 18sex.v340.info — Chinese chat |
| 8 | FAIL | high | ✗ | 254 | 3.8 | 47.6% | alpha < 80% | klimtoren.be — Dutch school blog |
| 9 | PASS | high | ✓ | 159 | 5.6 | 88.1% | — | mysch.gr — Greek helpdesk |
| 10 | PASS | high | ✓ | 149 | 5.8 | 89.3% | — | mysch.gr — Greek helpdesk login |
| 11 | FAIL | low | ✗ | 95 | 19.3 | 96.8% | mean len > 10 | yhxzseo.com — Chinese spam (wrong reason) |
| 12 | FAIL | high | ✗ | 1254 | 6.0 | 77.5% | alpha < 80% | 20com20.fr — Turkish Apache docs |
| 13 | PASS | high | ✓ | 491 | 4.8 | 94.3% | — | 24ktcasino.net — English casino blog |
| 14 | FAIL | low | ✓ | 4 | 4.0 | 75.0% | words < 50 | 2kgames.eu — 404 Not Found page |
| 15 | FAIL | low | ✓ | 303 | 13.9 | 68.3% | mean len > 10, alpha < 80% | yizhangting.com — Chinese health spam |
| 16 | FAIL | low | ✓ | 121 | 10.4 | 71.9% | mean len > 10 | 303323.com — Chinese (HTML leakage) |
| 17 | FAIL | low | ✓ | 243 | 15.4 | 71.6% | mean len > 10 | 30bad.com — Chinese video site |
| 18 | PASS | high | ✓ | 98 | 4.8 | 88.8% | — | 312001.net — Chinese hospital |
| 19 | FAIL | low | ✓ | 690 | 8.5 | 79.3% | alpha < 80% | 354577.mwe075.com — Chinese chat |
| 20 | PASS | high | ✓ | 93 | 7.5 | 98.9% | — | schoollibrary.edu.pe.ca — English |

**Pass rate: 7/20 (35%) — Agreement with human judgment: 15/20 (75%)**

---

**Agreements (15/20):**

The filter correctly rejects obvious junk: Record 14 (404 error page, 4 words), Records 1/2/5/7/15/16/17/19 (Chinese pages dominated by navigation menus, product codes, or HTML leakage). It correctly passes Records 3, 9, 10, 13, 18, 20 as readable text with real content.

**Disagreements (5/20):**

1. **Record 8 (klimtoren.be, Dutch school blog) — false rejection.** This is the highest-quality page in the sample by human standards: genuine human-written blog posts in fluent Dutch. However, the blog archive sidebar (entries like "► 2018 (40)", "► juni (10)") contributes many non-alphabetic tokens, pushing the alpha word ratio to 47.6% and triggering a false rejection. The Gopher filter was designed for English and does not account for pages where structural navigation is mixed with high-quality body text.

2. **Record 4 (utchat888.com, Chinese adult chat) — false pass.** This is a clearly NSFW site with escort profiles and explicit content, yet it passes all four rules. The text is clean enough structurally (236 words, mean length 5.6, 90.3% alphabetic) to satisfy Gopher's heuristics. The filter correctly identifies *textual* quality but has no concept of *content* appropriateness — a separate harmful-content filter is needed.

3. **Chinese text systemically triggers mean-word-length failures.** Records 6, 11, 15, 16, 17 all fail due to mean word length > 10. Chinese text split on whitespace produces long tokens (multi-character words, mixed Chinese-pinyin strings, or Chinese characters followed by annotations like "恒溫恒濕試驗(yàn)箱") that far exceed 10 characters per token. The Gopher thresholds were calibrated on English and do not transfer well to CJK languages.