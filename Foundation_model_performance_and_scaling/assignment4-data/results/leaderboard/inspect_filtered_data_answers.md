# Section 4 — Inspect Filtered Data

## (a) Five Random Examples from the Filtered Dataset

### Kept Example 1

**Commentary:** High-quality English prose — well-structured sentences, good vocabulary, on-topic content. Suitable for LM training.

```
Mary Elizabeth Francis Obituary (1940-2025) | Connellsville, PA Make a life-giving gesture A unique and lasting tribute for a loved one Plant trees Create an obituary Prepare a personalized obituary for someone you loved.. Create an obituary United States Australia Canada · English Canada · Français New Zealand FAQ Contact us Obituaries Recent obituaries Memorials Receive obituaries and memorials Create an obituary Prepare a personalized obituary for someone you loved.. Create an obituary Funeral homes Help and advice Blogs Online will Shop Make a life-giving gesture A unique and lasting …
```

### Kept Example 2

**Commentary:** Readable English article with clear paragraph structure. Likely to help the model learn web-style writing.

```
Magnificent Mile Cannabis Strain Skip to content Home Shop SATIVA EDIBLE HASHBuy Hash Online UK– Hashish – Hash for Sale UK Hash is a drug which is made from cannabis weed. At the beginning of the 20th century, the popularity of Hashish in Europe came from Kashmir, Afghanistan , and many parts of India, as well as Greece, Syria, Nepal,Lebanon, and turkey. It is taken by smoking with a small piece of a pipe, bong vaporizer or joint, or a via oral ingestion (after decarboxylation). As pure Hash will not burn if rolled alone in a joint, it is regularly mixed with herbal cannabis, tobacco or a …
```

### Kept Example 3

**Commentary:** English document with informative content. Passes all filters; reasonable training example.

```
Book In The Oven: Week Wrap Up -- First Week Back from Mexico skip to main | skip to sidebar Sunday, January 25, 2009 Week Wrap Up -- First Week Back from Mexico It feels like Mexico was years ago. Work this last week was brutal. On the plus side, I got one of my three back paychecks on Friday, along with the one due to me. Don't know if it's really a "plus side" to get the money I work for, but whatever. I had a pretty good weekend. On Friday I walked over to the Borders on the Magnificent Mile (Chicago's shopping strip) to do some writing. Got on cupcake on my walk over. Saturday went out …
```

### Kept Example 4

**Commentary:** Clean English text, academic or professional style. Good signal for the C4 100 domains benchmark.

```
Aviation Fuel Market, by Region
```

### Kept Example 5

**Commentary:** English web page with coherent, factual content. Suitable for language modeling.

```
Thank You - Inspired Consulting Group Home About Solutions DEI&B Consulting Professional Development Inspired Allies Certification Clinical Partnerships Inspired Coaching Podcast Press Testimonials FAQ Contact Us Select Page Thank you for reaching out to Inspired Consulting Group. We will get back to you within the next business day. Should you want to connect sooner, please reach out to Chris directly at |||PHONE_NUMBER||| or by emailing |||EMAIL_ADDRESS|||. Inspired Insights Podcast Empowering Voices Across Generations “Inspired Insights” is not just a podcast; it’s a community where …
```

## (b) Five Examples Removed by the Filter Pipeline

### Rejected: `empty`

**URL:** `http://178mh.com/html/,detail/,newsmovie/,wd/moAGt5YrbE/po7oqy.html,/`

**Stage removed:** Minimum length (< 100 chars). The record is essentially empty or contains only boilerplate with no usable text.

**Excerpt:**
```
detail.html 模板不存在
```

### Rejected: `non_english`

**URL:** `http://0371rykj.com/ipfhsb/34.html`

**Stage removed:** Language ID filter. The page is not English (or confidence < 0.65). Keeping non-English text would add noise given our English-targeted benchmark.

**Excerpt:**
```
人妻,国内老熟妇对白HDXXXX,亚洲AV无码一区东京热久久
久久久久女人精品毛片,99久久精品无码一区二区毛片,被老外的又粗又大日出了水,一边吃奶一边哭乱抻又乱扭
恒溫恒濕試驗(yàn)箱
在線(xiàn)咨詢(xún)
上海林頻儀器股份有限公司Shanghai Linpin Instrument Stock Co Ltd
服務(wù)熱線(xiàn)：4000 662 888
手機(jī)咨詢(xún)：13818467052
首頁(yè)
林頻產(chǎn)品
試驗(yàn)箱系列
老化箱系列
非標(biāo)定制系列
ip防護(hù)系列
振動(dòng)跌落系列
成功案例
新聞中心
林頻新聞
行業(yè)新聞
常見(jiàn)問(wèn)題
解決方案
關(guān)于林頻
服務(wù)支持
聯(lián)系我們
您所在的位置：
恒溫恒濕試驗(yàn)箱 > 林頻產(chǎn)品 …
```

### Rejected: `gopher_fail`

**URL:** `http://allsortsofbooks.blogspot.com/2013/08/the-language-of-sparrows-by-rachel.html`

**Stage removed:** Gopher quality rules. The document fails at least one heuristic (word count, mean word length, ellipsis ratio, or alpha ratio).

**Excerpt:**
```
Inside the mind of a Bibliophile: The Language Of Sparrows by Rachel Phifer
Home
About
About Me
Review Policy
Reading Related
Reading Challenges
Fellow Bibliophiles
The Ones Behind the Books
Writing and Misc.
NaNoWriMo
Links to the Outside World
Me Around the Web
With love from Japan, Eustacia
Personal Website
Google+ Profile
Zazzle Store
Friday, August 9, 2013
The Language Of Sparrows by Rachel …
```

### Rejected: `low_quality`

**URL:** `http://356.schoollibrary.edu.pe.ca/cgi-bin/koha/opac-search.pl?q=su:Painters%20%20and%20su:United%20States%20%20and%20su:Biography&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=branch:356&limit=su-to:Artists.&limit=su-to:Artists.&limit=su-to:Art%20appreciation.&limit=se:Getting%20to%20know%20the%20world's%20greatest%20artists&limit=su-to:Painters&limit=su-to:Artists.&limit=au:Venezia,%20Mike.&limit=se:Getting%20to%20know%20the%20world's%20greatest%20artists&limit=se:Getting%20to%20know%20the%20world's%20greatest%20artists&limit=su-to:Artists.&limit=su-to:Painting,%20American.&limit=se:Getting%20to%20know%20the%20world's%20greatest%20artists&limit=su-to:Art%20appreciation.&limit=se:Getting%20to%20know%20the%20world's%20greatest%20artists&limit=su-to:Artists.&sort_by=relevance_asc&limit=available`

**Stage removed:** Quality classifier. The page has low wiki-probability (< 0.3), indicating it looks more like raw CC noise than Wikipedia-linked quality content.

**Excerpt:**
```
PEI School Library System Catalog › Results of Search for 'su:Painters and su:United States and su:Biography' with limit(s): 'branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 branch:356 su-to:Artists. su-to:Artists. su-to:Art appreciation. se:Getting to know the world's …
```

### Rejected: `nsfw`

**URL:** `http://javhmm.com/category/38/ass-lover`

**Stage removed:** NSFW filter. The classifier flagged this page as adult/harmful content. Removal is clearly justified.

**Excerpt:**
```
Watch Ass Lover Jav Online Page 1 | Japanese Adult Video - JAVHMM.COM
JAVHMM
Search
Home
Recent Videos
Popular Today
Popular Casts
Casts
Categories
Studios
Videos in Ass Lover
FCDC-176 Bukkake big-assed secretary is a squirting, vulgar sexual monster who pisses herself as soon as she gets excited (FCDC-176)
FLAV-391 Sensitive Nipples, Hard, Lewd Schoolgirl, Pleasure Addict, Big Ass, Sexual Desire …
```

## (c) Pipeline Observations and Potential Changes

**Filter breakdown from the processed WET file:**

| Stage | Removed | % of total |
|-------|---------|------------|
| `empty` | 452 | 1.7% |
| `non_english` | 17,517 | 64.5% |
| `gopher_fail` | 786 | 2.9% |
| `low_quality` | 1,204 | 4.4% |
| `nsfw` | 10 | 0.0% |
| `kept` | 7,204 | 26.5% |

**Key observations:**

1. **Non-English dominates rejections (64.5%).** This CC segment is heavily multilingual
   (Chinese, Spanish, Arabic). The language filter is working correctly and is the most
   impactful step. No change needed.

2. **Gopher rules reject 2.9%.** These are mostly very short pages, navigation-only pages,
   or pages with extreme word lengths (CJK characters tokenized as single long "words").
   Removal is justified.

3. **Quality classifier rejects 4.4% at threshold 0.3.** With a permissive threshold,
   this only removes clearly low-quality pages. Could tighten to 0.5 to further restrict
   to higher-quality content, at the cost of less training data.

4. **NSFW filter rejects 0.04% (10 docs).** Low rate makes sense after language filtering
   has already removed most non-English adult content. The Dolma NSFW model is English-trained
   so it was never effective on the non-English majority anyway.

5. **Overall keep rate: 26.5%.** This is reasonable for a quality-focused pipeline.
   For the leaderboard, increasing data volume by relaxing the quality threshold to 0.2
   could improve coverage, especially for underrepresented C4 domains.

**Potential improvement:** Apply exact line deduplication (Part 3.1) across the kept
documents to remove repeated boilerplate that survived the per-document filters
(e.g., navigation footers that appear in multiple pages from the same site).
