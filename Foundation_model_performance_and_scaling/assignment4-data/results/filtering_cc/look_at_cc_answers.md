# Problem: look_at_cc — Written Answers (Section 2.1)

---

## Part (a)

**URL:** `http://0371rykj.com/ipfhsb/34.html`

The page is almost certainly no longer accessible — the domain is a Chinese adult-content SEO spam site, and the auto-generated path (`/ipfhsb/34.html`) is typical of link-farm doorway pages. From the raw HTML it is immediately clear that the `<title>` and `<h1>` contain explicit adult keywords in Chinese (pornographic clickbait terms), while the actual body content is scraped product specifications from a completely unrelated industrial equipment company (Shanghai Linpin Instrument). This is a classic doorway-page tactic: embed high-traffic adult keywords in meta/head elements to attract search-engine traffic, while serving different content in the body.

---

## Part (b)

The WET extracted text opens with a few lines of adult-content SEO keywords (decoded from the HTML entities in the `<title>`, `<meta>`, and `<h1>` tags), then abruptly transitions into product documentation for a temperature/humidity test chamber. Several things should have been filtered:

- **The first 2–4 lines** — adult SEO keyword spam; harmful and contextually incoherent.
- **Navigation boilerplate** — menu items (首頁, 林頻產品, 試驗箱系列), breadcrumbs ("您所在的位置："), and footer link lists make up a large fraction of the extracted text but carry no informational content.
- **Structural table headers** rendered as raw text (e.g., repeated 設備型號 / 工作室尺寸 headers without coherent tabular context).

Training on text like this risks teaching the model to produce navigation menus, SEO keyword stuffing, and incoherent topic switches mid-document. On the upside, the technical specifications in the body (temperature ranges, humidity tolerances, component materials) are well-formed Chinese technical prose that could be useful for learning product description language or Chinese technical vocabulary.

---

## Part (c)

This page might be **useful** for a Chinese-language industrial product search assistant, where understanding temperature/humidity testing equipment specifications and supplier terminology is valuable. It would be **harmful** for a general-purpose writing or educational assistant, where the adult keyword contamination in the first lines would introduce toxic associations and the navigation noise would degrade the coherence of generated text.

---

## Part (d) — 25-record WET Survey

| # | Domain | Language | Page Type | Notes |
|---|--------|----------|-----------|-------|
| 1 | 0371rykj.com | Chinese | Adult/spam doorway page | SEO spam keywords + scraped industrial content |
| 2 | chinatikfans.com | Chinese | Fan forum | HTML entities unescaped in title; navigation-heavy |
| 3 | usnccm.org | English | Academic conference site | USNCCM13 — decent structure but mostly navigation |
| 4 | utchat888.com | Chinese (Traditional) | Adult video chat | Age-gate warning page; no real content |
| 5 | 176766.cn | Chinese | Adult content spam | Explicit keyword lists; no body content |
| 6 | 178mh.com | Chinese | Broken/template error | "模板不存在" (template not found); empty |
| 7 | tgtg97.com | Chinese (Traditional) | Live streaming site | Leaderboard + promo text; no coherent prose |
| 8 | v340.info | Chinese (Traditional) | Adult content | Navigation and membership prompts only |
| 9 | klimtoren.be | Dutch | **Elementary school blog ★** | First high-quality page — teacher's assignment post with coherent prose |
| 10 | mysch.gr | Greek | IT helpdesk | Government school IT support portal; navigation-heavy |
| 11 | mysch.gr | Greek | IT helpdesk (duplicate) | Same domain, member control panel page |
| 12 | yhxzseo.com | Chinese | App download spam | Fake sports betting app download page |
| 13 | 20com20.fr | Turkish/French | Apache sitemap | Auto-generated server documentation |
| 14 | 24ktcasino.net | English | Casino affiliate | RSS/nav skeleton; no real content |
| 15 | 2kgames.eu | English | 404 error | nginx 404 page |
| 16 | yizhangting.com | Chinese | Lottery/gaming spam | Navigation columns; no coherent text |
| 17 | 303323.com | Chinese (Traditional) | Medical device company | Product page for electrosurgical cutter; moderate quality |
| 18 | 30bad.com | Chinese | Video streaming | Movie listing page; boilerplate-heavy |
| 19 | 312001.net | Chinese | Community health center | Government/institutional page; structured but navigation-heavy |
| 20 | mwe075.com | Chinese (Traditional) | Adult video chat | Login/membership prompts only |
| 21 | edu.pe.ca | English | School library catalog | PEI school system search results; structured but low prose content |
| 22 | haaxz.com | Chinese (Traditional) | Adult content | Membership/payment prompts only |
| 23 | haaxz.com | Chinese (Traditional) | Adult content (duplicate) | Same domain, different page — same boilerplate |
| 24 | 387tel.com | Chinese (Traditional) | Video chat / dating | Leaderboard stats; no prose |
| 25 | blogspot.com | Spanish | Political blog | Coherent narrative prose in Spanish |

**First high-quality example: record #9** (klimtoren.be — a Dutch elementary school teacher's blog with original, coherent writing). It takes 9 records before encountering anything resembling quality training data, illustrating why raw Common Crawl requires aggressive filtering before use.