# Wikipedia project ideas

## Data on hand

| Source | Location | What it's good for |
|---|---|---|
| Kiwix ZIM (English, no pictures, 2026-06, ~53 GB) | `dumps/wikipedia_en_all_nopic_2026-06.zim` | Rendered article HTML, offline reading (`kiwix-serve`), text corpus. Python access: `pip install libzim` |
| Article text (wikitext) | `dumps/enwiki-20260901-pages-articles-multistream.xml.bz2` + `-index.txt.bz2` | Raw wikitext of every article; the index allows random access to a single article |
| Pages | `dumps/…-page.sql.gz` | page ID ↔ title ↔ namespace (every other table joins on this) |
| Link graph | `dumps/…-pagelinks.sql.gz` + `…-linktarget.sql.gz` | Who links to whom (pagelinks stores a link-target ID; linktarget maps it to a title) |
| Redirects | `dumps/…-redirect.sql.gz` | Resolve "USA" → "United States" |
| Categories | `dumps/…-categorylinks.sql.gz`, `…-category.sql.gz` | Category tree |
| Coordinates | `dumps/…-geo_tags.sql.gz` | Lat/lon for geotagged articles |
| Page properties | `dumps/…-page_props.sql.gz` | Includes each article's Wikidata ID (`wikibase_item`) |
| Language links | `dumps/…-langlinks.sql.gz` | Which other language editions have each article |

Tip: don't import the `.sql.gz` files into MySQL (pagelinks alone can take a day to import). Instead, stream-parse
the `INSERT` tuples with a script and write the results into SQLite or Parquet.

## Projects

### Link graph (start here — several other ideas build on it)
- **Six Degrees solver / Wikipedia Game.** Find the shortest link path between any two articles,
  e.g. "Kevin Bacon → Mitochondria in 4 clicks". Could become a playable offline game that grades
  your path against the best one.
- **Getting to Philosophy.** Follow the first body link in each article until you reach Philosophy
  or get stuck in a loop. Run it for all ~7M articles: what fraction reach Philosophy, and what are
  the strangest cycles? (Needs "first link" parsing from the article text, not just pagelinks.)
- **PageRank / centrality.** Which is the most central article on Wikipedia?
- **Community detection + a giant map of knowledge.** Cluster the graph and visualize it with
  Gephi or a WebGL graph viewer.
- **Category tree explorer.** How deep does the tree go, where are the loops, and what's the
  weirdest category?

### Search & Q&A
- **Offline semantic search.** Embed the lead paragraphs (or whole articles) into a vector database
  for "that battle where the weather decided everything"-style searches.
- **Offline RAG assistant.** Use that index with a local LLM (or Claude) to answer questions with
  citations from your copy.
- **Home Wikipedia server.** Run `kiwix-serve` on the ZIM to give every device on the LAN
  offline Wikipedia.

### Fun generators
- "Today in history" feed.
- "Rabbit hole of the day": a chain of related articles.
- Daily trivia quiz built from article lead sentences.
- **Higher/Lower** with pageview counts (needs the separate pageview dumps).

### Text analysis
- Longest articles, most common first words, how many articles claim something is "the largest".
- Readability rankings.
- Markov-chain or small-model "fake Wikipedia" article generator.

### Geography (geo_tags)
- Map every geotagged article; "what's notable within 5 km of me?"
- Density maps — where is the world over/under-documented?

### Wikidata (needs the separate Wikidata dump; `page_props` provides the join key)
- Birth/death timelines; "people who died the year you were born".
- Giant royal family trees.
- Structured queries like "all rivers longer than X".

### History (needs extra downloads)
- `stub-meta-history.xml.gz` (115 GB): metadata for every edit, with no text. Enables edit-war
  detection, most-reverted articles, and activity over time.
- To animate how one article grew over 20 years, fetch that article's history from the API
  (no need for the multi-TB full-history dump).
- `pages-logging.xml.gz` (6.4 GB): deletions, moves and blocks, e.g. "most-deleted page titles".
- Pageview dumps: what the world was curious about on any day; spikes around events.

### Global notability (langlinks)
- Score each article by how many language editions have it; find topics famous everywhere vs.
  only in English.

## Suggested order
1. Parse `page` + `linktarget` + `pagelinks` + `redirect` into SQLite → Six Degrees solver.
2. Getting to Philosophy (adds first-link parsing from the article text).
3. Offline semantic search / Q&A.
4. geo_tags map; later, Wikidata timelines.
