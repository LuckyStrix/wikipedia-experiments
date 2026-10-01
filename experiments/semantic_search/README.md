# Semantic search

Find articles by meaning, not keywords: "that battle where the weather decided everything" returns
Battle of Kursk, Stalingrad, Atlanta, ... even though none of those words appear in the question.
Every article's introduction is turned into a 384-number vector by a small language model
(`BAAI/bge-small-en-v1.5`); a question is embedded the same way and the nearest vectors win.

**Needs:** the `text` stage (CPU, ~21 minutes) and the `embed` stage (**GPU machine**; ~2.7 days on
this CPU for all 7.2M articles). **Searching** needs no GPU: a query takes ~40 ms on a CPU once the
model is loaded (the first one loads it, ~4 s), and the index is ~2.8 GB in RAM for all articles.

**Run:** from the app (sidebar -> Semantic search; arrow keys browse, `ctrl+o` reads the whole
article, `ctrl+l` asks again), or

```bash
python -m experiments.semantic_search "that battle where the weather decided everything" -k 10
python -m experiments.semantic_search          # asks for queries one after another
```

From Python (this is what the RAG assistant should call):

```python
from wikiexp.semantic import SemanticSearch
from wikiexp import wikitext as W

ss = SemanticSearch()                                   # instant; model + index load on first use
hits = ss.search("first woman to win a Nobel prize", k=5)
hits[0].page_id, hits[0].title, hits[0].score, hits[0].snippet, hits[0].lead
text = W.to_text(W.WikiText().fetch(hits[0].page_id))   # the whole article as clean text (~40 ms)
```

`SemanticSearch(data_dir, embedder=...)` takes any object with `encode(texts, query=False) -> float32
array`, which is how the tests run without a model. `search_vector()`, `leads()` and `encode_query()`
are there for callers that do their own batching. Searching is safe from several threads.

## How it works

- `wikiexp/wikitext.py` reads `pages-articles-multistream.xml.bz2` through its index. The 284 MB
  index is parsed once (~45 s) into three numpy arrays under `data/text/` and memory-mapped afterwards.
  `fetch(page_id)` decompresses just that page's stream (~100 pages, ~40 ms; the last 16 streams are
  cached). `iter_pages(fn=...)` streams every main-namespace non-redirect page over forked worker
  processes (a plain loop on Windows). `to_text`/`sections`/`lead` clean wikitext: templates, references,
  files, categories, tables, comments, galleries and math are dropped, link text is kept, and a few
  templates that carry prose are rendered (`{{lang|fr|...}}`, `{{convert|5|km|mi}}`, birth dates, ...).
- **Stage `text`** (`pipeline/build_text.py`) -> `data/text/leads.sqlite`: the cleaned intro (cut at a
  sentence end, default 1200 characters; setting *Lead length*) of every article, in popularity order
  (in-links), with disambiguation pages kept but flagged (`disambig`).
- **Stage `embed`** (`pipeline/build_embed.py`) -> `data/search/`: embeds `"title: lead"` for the N most
  popular articles (setting *Most popular articles*, 0 = all), skipping disambiguation pages and empty
  leads, in shards of 50,000 articles. Each finished shard is written as float16 and recorded in
  `progress.json`, so a stopped run (Ctrl+C, reboot, power cut) continues with the next shard, and a later
  run with a bigger limit reuses the finished shards. Then it builds the faiss index:
  - up to 250,000 vectors: a flat exact index (fastest and exact at that size);
  - above that `IVF{4*sqrt(N)},SQ8` (inverted lists, 8-bit quantised; 7.2M articles = 10,733 lists,
    ~2.8 GB, 32 lists probed per query), trained on a 50-points-per-list sample.
  `meta.json` (model, dim, count, limit, index type) is written last, so its presence means done.
- bge models are given their query instruction ("Represent this sentence for searching relevant
  passages: ") on questions but not on documents; e5 models get `query: ` / `passage: ` (see
  `wikiexp.semantic.prefixes`). Documents are truncated at 256 tokens; on CUDA the model runs in fp16.
- The `embed` stage is **manual**: *Run all* skips it, and its card says "run on a GPU machine". Start
  it from its own Run button or the command line.

## Running the full embedding on the GPU machine

Do the heavy part once; the result is ~2.8 GB.

1. **On the GPU machine**, get the code and a venv (Windows: `py -3.14 -m venv .venv`, then
   `.venv\Scripts\...`). Install a **CUDA build of torch first**, because on Windows plain
   `pip install torch` gives a CPU-only build; pick the right line at
   <https://pytorch.org/get-started/locally/>, e.g.

   ```
   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   It must print `True` and your card. (The first run also downloads the model, ~130 MB; after that
   it works offline.)
2. **From this machine**, send the leads (3.8 GB; the embed stage only needs this one file):

   ```bash
   python -m tools.sync push gpu text/leads.sqlite
   ```
3. **Try a small run first** on the GPU machine, in a terminal there (or `tools.sync run gpu "..."`
   inside `tmux`/`screen`, so an SSH drop doesn't stop it):

   ```
   .venv\Scripts\python -m pipeline.build_embed --limit 50000
   ```
   It prints articles/s; multiply 7.2M by it to see the full time. Then the full run (it reuses the
   shards of the small run, since model and leads are the same):

   ```
   .venv\Scripts\python -m pipeline.build_embed
   ```
   Options: `--batch-size 256` (raise it for a big GPU, lower it on CUDA out-of-memory errors),
   `--device cuda:1`, `--model <other sentence-transformers model>`, `--restart` (ignore finished
   shards). Stop it any time; running the same command again continues. The settings in the app
   (*Build* tab) are the same options.
4. **Bring the result back**: only the three finished files, not the shards (5 GB):

   ```bash
   python -m tools.sync pull gpu search/index.faiss search/page_ids.npy search/meta.json
   ```
   (`tools.sync pull gpu search` also works but copies the shards too.) Then the *Embed articles*
   card shows done and Semantic search opens. (The card goes green when the three files exist; its
   **View** notes if they were built with a different model or limit than the current settings.)

Rough time on the GPU machine: not measured here (this machine has no GPU). A model this small
(33M parameters, ~100-token leads on average, fp16) typically runs at a few thousand articles per
second on a mid-range NVIDIA card, i.e. roughly 30-90 minutes for all 6.9M embeddable articles; the
`--limit 50000` trial gives the real number.

## Findings (real data)

Timings on this 16-core CPU box, which was also running other jobs (load average 8-11):

| step | result |
|---|---|
| index parse (once) | 45 s -> 200 MB of `.npy`; opens instantly afterwards |
| `fetch(page_id)` | ~40 ms (random article, cold stream) |
| `text` stage, all 7.2M articles, 6 workers | **21 minutes** (5,850 articles/s; cleaning is ~0.3 ms per lead, the bz2 decompression of the 26 GB dump dominates); 3.8 GB, 365,618 disambiguation pages flagged, 6,892 articles with an empty lead (mostly tables-only lists) |
| `embed`, 50,000 most popular, CPU (all cores) | **29 minutes** = 29 articles/s, 49,888 vectors (112 disambiguation/empty skipped), flat index 77 MB |
| extrapolated, all 6.86M embeddable articles | ~66 hours (~2.7 days) on this CPU; a few hours less on an idle machine; roughly 30-90 minutes on a GPU (estimate) |
| query | ~40 ms (CPU, model warm); ~4 s to load the model on the first query |

Real queries over the 50,000 most popular articles (score = cosine similarity):

| query | top results |
|---|---|
| that battle where the weather decided everything | Battle of Kursk, Volgograd, Battle of Stalingrad, ..., Battle of Atlanta, Battle of Chickamauga, Battle of Attu |
| first woman to win a Nobel prize | Marie Curie (0.75), Nobel Prize, Nobel Peace Prize, Jane Addams |
| why did the dinosaurs go extinct | Dinosaur, Paleocene, Extinction, Mesozoic, Extinct comet |
| how plants turn sunlight into energy | Photosynthesis (0.77), Plant, Chlorophyll, Sunlight, Solar power |
| disease that killed a third of Europe in the 14th century | Black Death (0.78), Late Middle Ages, High Middle Ages |
| the Spanish fleet that was scattered by storms | Spanish Navy, Spanish Armada, Mediterranean Fleet |
| mountain climbing disaster in the Himalayas | Mount Everest (0.78), Himalayas, Tibetan Plateau |
| scientist who discovered that continents move | History of Antarctica, Alexander von Humboldt, New World, Plate tectonics |
| programming language invented at Bell Labs | Bell Labs, Programming language, BASIC, Ruby, C (programming language) |
| bridge that collapsed in the wind | Wind wave, Johnson Sea Link accident, Golden Gate Bridge, Bridge (the Tacoma Narrows Bridge is outside the top 50,000) |

What this shows: broad "which article is about X" questions work well, and the weather question
finds battles known for winter, mud and storms (the Battle of the Bulge, rank 7,425 by in-links and so
embedded, does not make the top 10). Weak spots are the expected ones: with only the 50,000
most-linked articles indexed, the specific article you mean is often missing (the Tacoma Narrows
Bridge is rank ~2.2M by in-links, Alfred Wegener ~633k), so the full embed matters; a small model
embeds *topic* better than *relationships* ("collapsed in the wind" matches wind and bridges
separately); and date-list pages ("July 7") surface for battle questions because their leads are lists
of events. Only the intro is embedded; later sections are not searchable.

Cleaning check: ~20 diverse real articles were read by eye and fixed, ~50,000 random leads were
scanned for leftover markup while developing, and in the finished table every 50th lead (144,693) was
checked: none had braces, brackets, `||` table debris or `thumb`, and 6 had a stray HTML tag (the one inspected was `<nowiki>`, which is fixed
now). Known leftovers: templates that compute text (currency conversions, team and sport templates,
population densities) leave a gap because unknown templates are dropped rather than rendered. The common
inline ones (`convert`/`cvt` with the conversion, dates, `lang`, `nihongo`, `chem`, ship names, ...) are
now rendered by `wikiexp/wikitemplates.py`; the `leads.sqlite` built before that change doesn't have this
yet (re-run the `text` stage to get it, ~21 minutes; the numbers are in experiments/ask/README.md).

## Hybrid search

`--mode hybrid` (the default; also in the screen, `ctrl+t` cycles modes) fuses the embeddings with a
keyword index over every article (`keyword` stage) and exact article names found in the query, so the
50,000-article embedding limit no longer hides the article you mean ("Tacoma Narrows Bridge" is found
by its words even though it is not embedded). See `wikiexp/retrieval.py` and experiments/ask/README.md.

## Ideas

- Embed more of each article (sections as separate vectors) for the RAG assistant: `wikitext.sections()`
  already splits articles, and the shard/index code works for any list of texts
- A bigger model (`BAAI/bge-base-en-v1.5`, 768-d, ~2x the index) or a re-ranker on the top 50
- "More like this" from any article: `search_vector` with that article's stored vector
