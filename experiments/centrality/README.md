# Centrality

Which articles matter most in the link graph? PageRank, in-degree and "gateway" (reverse PageRank)
for all 7.2M articles, with a leaderboard, a per-article lookup and a "most over/under-rated" view.

**Needs:** the core database, title index and the *Compute centrality* stage; for prose PageRank also
the *Build prose link graph* stage (needs the text dumps). **Machine:** CPU only. The build takes about 25 minutes (10 without reverse PageRank) and about 4 GB of RAM; the experiment itself
memory-maps `data/centrality/` and the graph, so it opens instantly.

**Run:** from the app (sidebar → Centrality), or

```bash
python -m experiments.centrality top 50 [--metric prose_pagerank|pagerank|indegree|reverse_pagerank] [--skip N]
python -m experiments.centrality "Albert Einstein"
python -m experiments.centrality surprises [--top 100000] [-n 25]
```

## How it works

- `pipeline/build_centrality.py` — power iteration over the memory-mapped link arrays. Each step
  gathers `score / out-degree` along every link in chunks of ~32M edges and sums per article with
  `np.add.reduceat`, so no sparse matrix is built and RAM stays small. Dangling articles (no
  outgoing links) spread their score uniformly; damping is a setting (default 0.85). It stops when
  the L1 change falls below 1e-9 (logging the residual of every step) and writes `data/centrality/`.
  Reverse PageRank is the same computation on the reversed graph: links count the other way, so it
  is high for articles that link *out* to many important ones (lists, overviews, hubs).
- `wikiexp/centrality.py` — `Centrality(data_dir)`: `.pagerank`, `.rank(idx, metric)`,
  `.percentile(idx, metric)`, `.top(n, metric)`, `.score`, `.at_rank`. Accepts a single idx or an
  array; memory-mapped. Other experiments use it to rank articles by notability.
- `explorer.py` — UI-free: leaderboard rows, an article's profile (ranks, percentiles, scores,
  degrees, most notable neighbours), and the surprises.
- `screen.py` / `__main__.py` — the app screen (Leaderboard, Look up, Surprises tabs) and the CLI.

**Surprises** compare the PageRank rank with the in-degree rank among the top N articles by either
measure, as a ratio (rank 50 vs 5,000 is as surprising as 5,000 vs 500,000). "Punching above their
links" have few incoming links but from important articles; "many links, little weight" are linked
from many places that themselves matter little (years, dab-style lists, navigation-heavy pages).

## Prose PageRank

The pagelinks dump counts every link in the *rendered* page, so a citation template that links ISBN
counts like a link in a sentence (see the first Findings table). `wikiexp/prose_links.py` re-reads
the wikitext and keeps only `[[links]]` in running text (not in templates, `<ref>`, tables, galleries,
file captions, comments; see `pipeline/build_prose_graph.py`), and `build_centrality --graph
data/prose_graph` runs the same PageRank on that graph, saving `prose_pagerank.npy`. When it exists it
is the default metric everywhere (`Centrality.default_metric`, the leaderboard, the lookup's
neighbours); the other metrics stay available. Re-running with `--prose-only` adds it to a finished
build in about 3 minutes without touching the other files. Surprises still compare plain PageRank with in-degree.

Built on the same dump: 159.4M prose links, 22% of the 709.9M in pagelinks (over 99.9% of the sampled
ones are also in pagelinks, as they should be). 66 iterations, residual 9.5e-10, 2.8 minutes.

**Top 30 by prose PageRank:**

| # | Article | Prose PageRank | Text links in | PageRank # (all links) |
|---:|---|---:|---:|---:|
| 1 | Association football | 0.119% | 260,784 | 17 |
| 2 | World War II | 0.0991% | 208,949 | 21 |
| 3 | United States | 0.0869% | 148,128 | 9 |
| 4 | France | 0.0597% | 114,659 | 28 |
| 5 | World War I | 0.0531% | 111,560 | 55 |
| 6 | United Kingdom | 0.0504% | 73,811 | 22 |
| 7 | India | 0.0491% | 91,018 | 35 |
| 8 | Iran | 0.0485% | 79,181 | 116 |
| 9 | Germany | 0.0467% | 85,702 | 30 |
| 10 | China | 0.0446% | 73,228 | 49 |
| 11 | Village | 0.0435% | 93,790 | 84 |
| 12 | Catholic Church | 0.0416% | 72,375 | 58 |
| 13 | New York City | 0.0403% | 109,368 | 38 |
| 14 | Australia | 0.0392% | 72,193 | 37 |
| 15 | Moth | 0.0382% | 81,442 | 182 |
| 16 | Latin | 0.0371% | 32,028 | 64 |
| 17 | National Register of Historic Places | 0.0365% | 85,147 | 163 |
| 18 | London | 0.0353% | 85,831 | 50 |
| 19 | Italy | 0.0341% | 64,616 | 42 |
| 20 | Japan | 0.034% | 67,160 | 48 |
| 21 | Soviet Union | 0.0339% | 54,945 | 70 |
| 22 | Russia | 0.0338% | 61,554 | 47 |
| 23 | Canada | 0.033% | 55,136 | 43 |
| 24 | Beetle | 0.0327% | 45,703 | 192 |
| 25 | COVID-19 pandemic | 0.0303% | 55,816 | 120 |
| 26 | The New York Times | 0.0297% | 73,129 | 19 |
| 27 | Species | 0.0295% | 71,059 | 88 |
| 28 | England | 0.0293% | 62,697 | 54 |
| 29 | Spain | 0.0292% | 51,845 | 53 |
| 30 | Genus | 0.029% | 60,774 | 141 |

Compared with plain PageRank: ISBN, coordinates, DOI, Wayback Machine, ISSN and the other citation
plumbing are gone (ISBN falls from #1 to #19,282; it has 305 text links against 1.66M template ones),
and the list is countries, wars, big cities and broad topics. **There is still no person in the top 30**:
people are linked from few places each, while countries and wars are linked from everywhere. Albert
Einstein moves from #2,623 to #1,246 (top 0.017%). Residue of the same effect: *Village*, *Moth*,
*Beetle*, *Genus* and *National Register of Historic Places* are mass-produced stub families whose
prose links the same generic article thousands of times. To rank people by fame, compare prose
PageRank percentiles among people only (or add pageviews).

## Findings

Built from the 2026-09-01 dump: 7,235,024 articles, 709,918,564 links, damping 0.85.

**Build:** forward PageRank 71 iterations (residual 9.3e-10) in 9.4 minutes, reverse PageRank 94
iterations (9.6e-10) in 15.3 minutes, 24.7 minutes in all on a busy 16-core machine, peak RSS 3.9 GB
(most of it memory-mapped graph pages). `--no-reverse` makes it a 10 minute build.

**Top 30 by PageRank:**

| # | Article | PageRank (share of total) | Incoming links |
|---:|---|---:|---:|
| 1 | ISBN | 0.314% | 1,663,972 |
| 2 | Geographic coordinate system | 0.176% | 1,248,090 |
| 3 | Digital object identifier | 0.167% | 729,227 |
| 4 | Wayback Machine | 0.136% | 680,476 |
| 5 | ISSN | 0.128% | 525,096 |
| 6 | Semantic Scholar | 0.066% | 252,847 |
| 7 | PubMed | 0.063% | 234,449 |
| 8 | Wikidata | 0.059% | 516,190 |
| 9 | United States | 0.059% | 407,249 |
| 10 | OCLC | 0.058% | 254,719 |
| 11 | IMDb | 0.057% | 410,600 |
| 12 | Time zone | 0.052% | 456,852 |
| 13 | Taxonomy (biology) | 0.051% | 502,894 |
| 14 | Bibcode | 0.048% | 178,286 |
| 15 | PubMed Central | 0.044% | 151,233 |
| 16 | Global Biodiversity Information Facility | 0.041% | 469,081 |
| 17 | Association football | 0.041% | 276,119 |
| 18 | JSTOR | 0.041% | 159,327 |
| 19 | The New York Times | 0.038% | 203,730 |
| 20 | Animal | 0.036% | 380,874 |
| 21 | World War II | 0.036% | 231,332 |
| 22 | United Kingdom | 0.033% | 214,781 |
| 23 | Catalogue of Life | 0.033% | 373,031 |
| 24 | Open Tree of Life | 0.033% | 403,361 |
| 25 | Binomial nomenclature | 0.033% | 397,572 |
| 26 | Surname | 0.033% | 175,821 |
| 27 | Wikispecies | 0.032% | 273,435 |
| 28 | France | 0.031% | 267,858 |
| 29 | Political party | 0.031% | 200,034 |
| 30 | Germany | 0.029% | 216,013 |

What to take from it:

- Raw PageRank on Wikipedia measures *how many pages link to you through templates*, not fame. The
  top five are citation and infobox plumbing (ISBN, coordinates, DOI, Wayback Machine, ISSN), and
  PageRank agrees with in-degree on exactly those five. Countries, identifiers and taxonomy hubs fill
  the rest; there is no person in the top 30 (Albert Einstein is #2,623 by PageRank, #4,251 by links).
  A notability ranker should skip pure-infrastructure articles, or use PageRank percentile only among
  articles of the type it cares about (people, places).
- Outbound links matter too: Semantic Scholar has 252k incoming links but ranks #6 (in-degree #24)
  because ISBN, DOI and PubMed link to it and they hold a lot of score.
- **Gateway** (reverse PageRank) is a completely different list: "List of lists of lists", "Lists of
  Lepidoptera by region", "Unisex name", "List of most popular given names", and the long "List of
  moths/fishes/plants of ..." pages. They link out to thousands of articles and almost nothing links
  to them.
- **Surprises** (top 100,000 by either measure): the most under-linked articles with high PageRank
  are a closed biodiversity-informatics cluster (Agronomy Journal, Darwin Core, Ebbe Nielsen Prize,
  Supertree), a few dozen links each but from the ISBN/GBIF-like hubs. The most over-linked are
  2026 California election pages and "Mayoral elections in ..." articles: linked thousands of times
  from navigation templates between each other, with little PageRank.

