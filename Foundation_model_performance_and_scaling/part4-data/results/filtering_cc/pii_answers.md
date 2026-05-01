# Section 2.4 — Written Answers

## Problem: `mask_pii`

---

### Part (4) — Downstream problems from naïve PII masking

Naïve regex masking creates two categories of problem in the training corpus. First, **false positives** (over-masking): numeric identifiers that happen to match phone-number patterns — product codes, order IDs, QQ numbers, tracking numbers — get replaced with `|||PHONE_NUMBER|||`, corrupting otherwise meaningful text. The model then learns to generate these placeholder tokens in contexts where they do not belong, or learns incorrect associations between certain number formats and the placeholder string. Second, **false negatives** (under-masking): phone numbers in international format (`+44 7911 123456`, `+1-800-555-1234`), obfuscated emails (`user [at] domain [dot] com`), and IPv6 addresses all pass through unmasked, leaving real PII in the training data.

A subtler problem is that masking changes the marginal distribution of the training text — placeholders like `|||PHONE_NUMBER|||` appear far more often than in natural text, so the model assigns non-trivial probability to generating them. In a deployed model, this can cause the model to hallucinate these tokens in generations.

Mitigations: (1) use an NER model rather than regex for higher precision; (2) validate phone numbers against known-valid area codes; (3) set a minimum context length before applying phone-number patterns to reduce false matches on short numeric strings; (4) mask with a distribution of plausible replacements rather than a fixed token to avoid the model memorizing the token.

---

### Part (5) — False positives and false negatives on WARC-extracted text

Running the three maskers on 200 WARC-extracted records, PII was detected in 20. Notable examples:

**False positives (over-masking):**

- **QQ numbers as phone numbers** (`51mwtx.com`, `8mm8mm.com`): Chinese QQ instant-messaging IDs like `1723913320` and `3320069488` are 10-digit numbers that our phone regex matches. They appear both as email local-parts (`1723913320@qq.com`) and standalone — the phone masker fires on the standalone digit string, producing a spurious `|||PHONE_NUMBER|||` replacement. The email masker correctly handles the email form, but the same digits in plain text get double-categorised.
- **Template placeholder email** (`3rte.com.br`): `contact@domain.com` is a HTML template default, not a real person's address. Our regex correctly replaces it, but it is not genuine PII — over-masking a non-personal string.
- **Sequential/counter strings** (`adishori.ryujin.shop`): `2627282930` looks like a page counter or ID sequence; matched as a 10-digit phone number.

**False negatives (under-masking):**

- **International phone numbers**: numbers like `061 784 8975` (South Africa, `abo-green.com`) match our pattern incidentally because the digit-group widths are compatible, but numbers in formats like `+44 7911 123456` or `+1-800-555-1234` (country-code prefix) are not matched at all.
- **Emails with uncommon TLDs or subdomains**: the regex handles most cases, but obfuscated forms (`user [at] domain dot com`) are not captured.

Overall, the phone-number masker has the highest false-positive rate due to the prevalence of 10-digit numeric identifiers in web data that are not phone numbers.