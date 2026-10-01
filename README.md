# wikipedia-experiments

Fun things to do with a full offline copy of English Wikipedia: link-graph games (Six Degrees,
Getting to Philosophy), offline semantic search, maps, timelines and more. See [IDEAS.md](IDEAS.md)
for the project list.

The repo holds code only. The data (~90 GB of Wikimedia dumps and a Kiwix `.zim`, plus whatever
gets built from them) lives outside git, and each machine keeps its own copy. This makes it easy to
run heavy jobs on one computer (e.g. a GPU box) and analyse the results on another.

## The app

```bash
python app.py
```

A terminal app (built with [Textual](https://textual.textualize.io/)) that drives everything. It
runs in any terminal: Linux, macOS, Windows Terminal (no WSL needed), or over SSH.

- **Left:** one card per pipeline stage (grey = not run, blue = running, green = done, red =
  failed) with **Run** and **View** buttons, plus **Run all** / **Stop**. Status comes from what's
  on disk, so stages built earlier (or from the command line) show as done, and a stage already
  running in another terminal is detected and won't be started twice. Below the stages, the
  **experiments**; each unlocks once the stages it needs are done.
- **Right:** settings in tabs (folders, dump date, which optional files to download, build
  options, machines), with the output log underneath.

| Key | Action |
| --- | --- |
| `r` / `ctrl+r` | Run every stage that isn't done yet, in order |
| `s` | Stop the running stage (and everything it started) |
| `ctrl+s` | Save settings to `config.toml` |
| `l` | Maximise / restore the log; `ctrl+↑` / `ctrl+↓` resize it, or drag the bar above it |
| `t` | Cycle colour themes (Campbell, One Half Dark, ...) |
| `q` | Quit (asks first if a stage is running) |

`r`, `s`, `l`, `t` and `q` are ignored while you're typing in a field. A run uses the settings as
they were when it started; unsaved edits apply to runs started from the app, and `ctrl+s` makes
them the default for the command-line scripts too.

Every run started from the app is recorded in `data/runs/`: `pipeline.log` (every line,
timestamped) and `runs.json` (settings, machine, git commit, and each stage's timing, result and
output sizes).

### Six Degrees

Opens from the sidebar once the core database and title index are built. Type part of a title and
pick from the suggestions; only real articles are accepted (the Find button stays disabled until
both ends are picked). Suggestions include exact matches in any case, redirects ("USA" → United
States), prefixes, substrings and typos ("Mitocondrion"), most-linked first. `ctrl+r` picks a
random pair of well-known articles. Also available on the command line:

```bash
python -m experiments.six_degrees "Kevin Bacon" "Mitochondrion"
python -m experiments.six_degrees --random
```

### Nearby

"What's notable near here?" Opens once the places index is built. Type a place name and pick a
suggestion (only articles with coordinates are offered), or type coordinates (`48.8584, 2.2945`,
`48.86N 2.29E`, `48°51′29″N 2°17′40″E`) and press Enter. Set the radius (`500 m`, `2 km`, `3 mi`; a bare
number is km), the sort (`ctrl+o`: most notable / nearest) and a place-type filter; the table lists
each place with distance, compass direction, type and a notability score (PageRank if
`data/centrality/pagerank.npy` exists, else incoming links, blended with language editions). Enter on a
result searches around it, `ctrl+b` goes back. On the command line:

```bash
python -m experiments.nearby "Eiffel Tower" --radius 2km [--sort distance] [--limit 30] [--type landmark]
python -m experiments.nearby "48.8584, 2.2945" -r 500m
```

### Centrality

Opens once the core database, title index and centrality stage are built. Three tabs: a
**Leaderboard** of the top articles by PageRank, in-degree or "gateway" (reverse PageRank), a
**Look up** tab with one article's rank, percentile, score and degrees in every metric (plus its most
notable neighbours), and **Surprises**: articles whose PageRank rank and in-degree rank disagree most.
Enter on a table row opens that article in Look up. Also on the command line:

```bash
python -m experiments.centrality top 50 [--metric indegree]
python -m experiments.centrality "Albert Einstein"
python -m experiments.centrality surprises
```

Other experiments can rank articles by notability with `wikiexp.centrality.Centrality`.

### Semantic search

Opens from the sidebar once the `text` and `keyword` stages are done (the `embed` stage is optional).
Type a description in your own words ("that battle where the weather decided everything") and get
articles back by **hybrid** search: meaning (embeddings, for the articles embedded so far), the words
you used (keyword index, all 7.2M articles) and exact article names in the question, fused into one
ranking. `ctrl+t` switches to semantic-only or keyword-only. Arrow keys browse the results, `ctrl+o`
reads the whole article (cleaned from the dump), `ctrl+l` starts a new query. Also on the command line:

```bash
python -m experiments.semantic_search "that battle where the weather decided everything" -k 10
python -m experiments.semantic_search "Tacoma Narrows Bridge collapse" --mode keyword   # semantic | keyword | hybrid
```

### Ask Wikipedia

Opens once the `core`, `titles`, `text` and `keyword` stages are done. Ask a question in plain English;
a language model answers **only from Wikipedia passages** retrieved from your local copy and cites them
as [1], [2]; the numbered sources are on the right (Enter shows the whole passage, `ctrl+o` the whole
article). Follow-up questions work (`ctrl+n` starts a new conversation), the model runs on this machine
through [ollama](https://ollama.com) (default `llama3.2:3b`; `ollama pull llama3.2:3b`) or on another
one (Ollama server URL in the *Ask* settings tab), or switch to Claude with `ctrl+b` (needs
`ANTHROPIC_API_KEY` in the environment). On the command line:

```bash
python -m experiments.ask "Who designed the Eiffel Tower?" --show-passages
python -m experiments.ask -i                      # a conversation
python -m experiments.ask "..." --backend claude --effort high
```

See [experiments/ask](experiments/ask/README.md) for how it works, measured quality and speed.

## Layout

```
app.py        starts the terminal app
pipeline/     UI-free core: stages, settings registry, runner, progress parsers, run records,
              and the build scripts themselves (download.sh, build_core.py, build_titles.py)
tui/          the Textual app: stage cards, settings form, log, dialogs, title picker
wikiexp/      shared library: paths, dump readers, database/graph access, title search,
              article wikitext + cleaning (wikitext.py, wikitemplates.py), semantic / keyword / hybrid
              search (semantic.py, keyword.py, retrieval.py), RAG + LLM backends (rag.py, llm.py)
experiments/  one folder per experiment (see experiments/README.md)
tools/        sync.py: move data between machines over SSH
tests/        pytest suite; uses a tiny hand-made Wikipedia, no dumps needed
dumps/        raw downloads            - not in git
data/         generated datasets        - not in git
config.toml   this machine's settings   - not in git (see config.example.toml)
```

Everything in `pipeline/` also works without the app, e.g. `python -m pipeline.build_core`.

## Setup

Requires Python 3.11+. Run everything from the repo root.

```bash
git clone https://github.com/LuckyStrix/wikipedia-experiments
cd wikipedia-experiments
python -m venv .venv                 # Windows: py -3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip ...
cp config.example.toml config.toml   # optional: custom data paths, other machines
.venv/bin/python app.py
```

Tests: `pip install -r requirements-dev.txt`, then `python -m pytest tests`.

Then either get the data and build it (below), or copy the built data from a machine that already
has it: `python -m tools.sync pull <machine> wiki.sqlite graph`.

## Getting the data

Nothing large is stored in this repo. Everything comes from two free public sources and goes in
`dumps/` (or wherever `config.toml` points), with the original file names unchanged.

### Wikimedia database dumps

From <https://dumps.wikimedia.org/enwiki/20260901/>. Each file is named `enwiki-20260901-<name>`.
Sizes are compressed.

| file (`<name>`) | size | used for |
|---|---|---|
| `sha1sums.txt` | 0.2 MB | checksums; the pipeline refuses to build from incomplete files |
| `page.sql.gz` | 2.2 GB | **required**: every page's ID, title, redirect flag |
| `redirect.sql.gz` | 178 MB | **required**: redirect targets |
| `linktarget.sql.gz` | 1.3 GB | **required**: link target IDs → titles |
| `pagelinks.sql.gz` | 6.6 GB | **required**: the link graph |
| `categorylinks.sql.gz`, `category.sql.gz` | 2.4 GB | category tree projects |
| `geo_tags.sql.gz` | 51 MB | map projects |
| `page_props.sql.gz` | 449 MB | Wikidata IDs |
| `langlinks.sql.gz` | 556 MB | "how many languages have this article" |
| `pages-articles-multistream.xml.bz2` | 25.0 GB | full article wikitext (text projects) |
| `pages-articles-multistream-index.txt.bz2` | 271 MB | lets you pull out one article from the file above |

On Linux/macOS, `pipeline/download.sh` fetches all of these (about 39 GB, a few hours at
Wikimedia's ~5 MB/s per-connection limit; re-run it to resume). On Windows, download them in a
browser or with `curl.exe -C - -O <url>`.

**Newer dumps:** Wikimedia publishes a new dump roughly twice a month and keeps only the last few
months, so the 20260901 files will eventually disappear. Pick a newer date from
<https://dumps.wikimedia.org/enwiki/>, then set `dump_date = "YYYYMMDD"` in `config.toml` (and
`DUMP_DATE=YYYYMMDD pipeline/download.sh`). Mirrors that keep older dumps are listed at
<https://dumps.wikimedia.org/mirrors.html>.

**Other datasets mentioned in [IDEAS.md](IDEAS.md)** (not needed yet): full edit-history metadata
(`stub-meta-history.xml.gz`, 115 GB, same directory), Wikidata
(<https://dumps.wikimedia.org/wikidatawiki/entities/>) and pageview counts
(<https://dumps.wikimedia.org/other/pageview_complete/>).

### Kiwix offline Wikipedia (.zim)

`wikipedia_en_all_nopic_2026-06.zim` (~53 GB): all of English Wikipedia as rendered HTML, without
images. Download from <https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_2026-06.zim>,
or pick a newer `wikipedia_en_all_nopic_*.zim` from <https://download.kiwix.org/zim/wikipedia/>. The
code uses the newest `.zim` in `dumps/`. Kiwix also offers each file as a torrent (add `.torrent`
to the URL), which is usually faster.

## Building the data

Use the app's stage cards, or run the steps directly:

```bash
python -m pipeline.build_core     # needs the 4 required dumps; ~1 hour, ~16 GB RAM peak
python -m pipeline.build_titles   # needs wiki.sqlite; title search index (~15 minutes)
python -m pipeline.build_geo      # needs wiki.sqlite + geo_tags dump (langlinks optional); places index (~1 minute)
python -m pipeline.build_centrality   # needs data/graph; PageRank etc. (~25 minutes, ~4 GB RAM)
python -m pipeline.build_text     # needs the text dumps + core; clean intro of every article (~20 minutes, 6 workers)
python -m pipeline.build_keyword  # needs text; FTS5 (bm25) index of every title + intro (~4 minutes, 2.2 GB)
python -m pipeline.build_embed    # needs text; sentence embeddings + faiss index. Run on a GPU machine
```

The last stage is **manual**: the app's *Run all* skips it (embedding 7.2M articles takes days on a CPU
and about an hour on a GPU), and its card says so; start it from its own **Run** button or the command
line. See [experiments/semantic_search](experiments/semantic_search/README.md) for how to run it on a
GPU machine and bring the result back.

Parsing the dumps is spread over several processes on Linux/macOS (all but two cores, up to 8; set
`WIKI_WORKERS=n` to change it). It roughly halves the core build on a laptop; machines that hold
their clock speed with every core busy gain more. Writing the database is single-threaded either way.

### data/wiki.sqlite

| table | columns | notes |
|---|---|---|
| `articles` | `page_id`, `idx`, `title`, `length` | main-namespace, non-redirect pages; titles use spaces; `length` is wikitext bytes |
| `redirects` | `page_id`, `title`, `target` | `target` is the final article's `page_id` (chains already followed) |
| `links` | `src`, `dst` | article → article by `page_id`; links via redirects resolved; no duplicates or self-links |
| `meta` | `key`, `value` | dump date, build time, row counts |

### data/titles.sqlite

Every article and redirect title in popularity order (most incoming links first), with a trigram
full-text index. Used through `wikiexp.titles.TitleIndex` (`search`, `resolve`, `random_article`).

### data/geo.sqlite

Every article with a primary Earth coordinate (1.24M): `places` (`page_id`, `idx`, `title`, `lat`,
`lon`, `type`, `pop`, `dim` in metres, `country`, `region`, `langs` = other-language editions, `links` =
incoming links), an R*Tree `places_rt` over lat/lon for box queries, and `meta`. Articles with only
secondary coordinates are skipped (counts are in `meta`). Used by the Nearby project.
### data/centrality/

PageRank (`pagerank.npy`, float32, sums to 1), reverse PageRank (`reverse_pagerank.npy`) and, for
PageRank, in-degree and reverse PageRank, `*rank.npy` (int32, 1 = best) and `*order.npy` (idx best
first), all indexed by graph `idx`, plus `meta.json`. Used through `wikiexp.centrality.Centrality`
(`rank`, `percentile`, `top`, `score`), which memory-maps them.

### data/graph/

The same link graph as compressed-sparse-row numpy arrays indexed by `idx` (matching `articles.idx`):
`out_indptr`/`out_indices` (outgoing links), `in_indptr`/`in_indices` (incoming links) and
`idx_to_page_id`.

```python
from wikiexp import core
db, g = core.connect(), core.Graph()
i = g.idx(core.resolve(db, "USA"))       # redirects resolve to "United States"
print(len(g.out_links(i)), len(g.in_links(i)))
```

### data/text/

`leads.sqlite` (from `pipeline.build_text`, about 7 GB): one row per article in popularity order.

| table | columns | notes |
|---|---|---|
| `leads` | `rank`, `page_id`, `idx`, `title`, `lead`, `disambig` | `rank` 0 is the most-linked article, so `ORDER BY rank LIMIT n` reads the n best-known first; `lead` is the plain-text intro (no templates, references, images or tables), cut at a sentence end to the *Lead length* setting (default 1200 characters); `disambig` = 1 for disambiguation and set-index pages, which are kept but flagged |
| `meta` | `key`, `value` | dump date, lead length, counts |

Also here: `index_*.npy`, a compact cache of the 284 MB dump index so one article's wikitext can be
fetched in about 40 ms with `wikiexp.wikitext.WikiText().fetch(page_id)` (see below).

```python
from wikiexp import wikitext as W
wt = W.WikiText()                       # opens the cached index instantly
raw = wt.fetch(core.resolve(db, "Battle of Waterloo"))     # wikitext of one article
print(W.to_text(raw)[:500])             # clean plain text; W.sections(raw) -> [(heading, text), ...]
for page_id, title, lead in wt.iter_pages(fn=lambda p: W.lead(p.text)):   # all articles, in parallel
    ...
```

`fts.sqlite` (from `pipeline.build_keyword`, 2.2 GB): an FTS5 index (porter stemmer, accents folded) over
title and lead of every article except disambiguation pages, with `rowid` = `leads.rank`. Searched
through `wikiexp.keyword.KeywordIndex` (bm25, title weighted 8x the lead) and, fused with embeddings and
title matches, `wikiexp.retrieval.HybridSearch`:

```python
from wikiexp.retrieval import HybridSearch
for r in HybridSearch().search("who designed the Tacoma Narrows Bridge?", k=5):
    print(r.title, r.via)            # via: which of semantic / keyword / title found it
```

`wikiexp.rag.RAG` builds the question-answering pipeline on top (used by Ask Wikipedia).

### data/search/

`index.faiss` (inner-product faiss index of the normalised embeddings of `title: lead`), `page_ids.npy`
(the page_id of each vector) and `meta.json` (model, dimensions, count, article limit, index type),
built by `pipeline.build_embed` and used through `wikiexp.semantic.SemanticSearch`:

```python
from wikiexp.semantic import SemanticSearch
ss = SemanticSearch()
for hit in ss.search("that battle where the weather decided everything", k=5):
    print(hit.score, hit.title, hit.snippet)
```

While it runs, `shards/` holds float16 embeddings and `progress.json` records which are finished, so a
stopped run resumes where it left off.

## Working across machines

Code moves through git; data moves with `tools/sync.py` (plain `ssh`/`scp`, so it works with
Windows' built-in OpenSSH server, no WSL needed). Describe your other machines in `config.toml`, then:

```bash
python -m tools.sync push gpu wiki.sqlite graph            # send inputs to the GPU machine
python -m tools.sync run  gpu "python -m experiments.x.y"  # git pull + run there
python -m tools.sync pull gpu x                            # fetch data/x back
```

On a Windows machine the command runs in PowerShell: use the venv's Python
(`.venv\Scripts\python -m ...`) so the right version is picked, and single quotes inside the command.

For long jobs, start them in a terminal on the remote machine (or with `tools.sync run` inside
`tmux`/`screen` locally), since an SSH disconnect stops the remote command.

## License

Code: MIT. Wikipedia content is © Wikipedia contributors, licensed CC BY-SA 4.0; none of it is
stored in this repo.
