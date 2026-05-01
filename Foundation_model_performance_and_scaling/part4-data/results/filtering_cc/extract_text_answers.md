# Section 2.2 — Written Answers

## Problem: `extract_text` — Part (b)

**Comparison of our extraction vs. CC WET (20 records)**

**Similarities:**
Both outputs capture the same core body text — for simple, low-boilerplate pages (Records 6, 15, 19) the outputs are nearly identical in character count and content. Both handle non-ASCII encodings correctly across Chinese, Greek, Turkish, and Dutch pages. On structurally simple pages the two extractors converge.

**Differences:**
Our extraction is consistently larger — often by 30–200%:

- **List bullets**: `list_bullets=True` in our call to `extract_plain_text` causes navigation menus to be rendered with `•` markers, which WET omits. This inflates our output with boilerplate navigation text.
- **Whitespace / blank lines**: `preserve_formatting=True` retains blank lines and indentation; WET removes them, producing denser output.
- **Page title**: WET prepends the HTML `<title>` element as the first line of extracted text (e.g., Record 3: "Welcome to USNCCM13! | USNCCM 13"; Record 7: "live173影音live秀 - 主播 :: 神仙表姐"). Our extractor omits the `<title>`, which can be useful metadata for quality filtering.
- **HTML tag leakage (Record 17)**: Our extractor emitted raw HTML tag strings (`<form id="rwjdv"></form>`, `<em id="rwjdv">`, etc.) from a page that contained obfuscated/invalid markup. WET correctly ignored them and produced clean output (1027 chars vs. our 3878 chars). This is a clear failure mode of resiliparse on malformed pages.
- **Main-content aggressiveness**: WET is notably shorter on content-heavy pages (Record 1: 10121 vs. 3496; Record 5: 6691 vs. 2103), suggesting CC's extractor applies stronger main-content filtering than our `main_content=False` setting.

**Which is better?**
For LM training purposes, WET is generally preferable: it is more compact, suppresses more navigation/footer boilerplate, and handles malformed markup more robustly (Record 17). Our extraction with `preserve_formatting=True, list_bullets=True, main_content=False` retains structural information (useful for document-understanding tasks) but at the cost of more noise. Switching to `main_content=True` in resiliparse would bring our output closer to WET in compactness, but risks dropping legitimate body text on pages with unconventional structure.