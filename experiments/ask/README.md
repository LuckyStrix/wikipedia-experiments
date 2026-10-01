# Ask Wikipedia

Ask a question in plain English; a language model answers **only from Wikipedia passages** found in
your local copy and cites them as [1], [2]. When the sources don't contain the answer it says so.
Everything but the language model runs offline on this machine; the model can be a local
[ollama](https://ollama.com) model, ollama on another machine (the GPU box), or Claude.

**Needs:** the `core`, `titles`, `text` and `keyword` stages (CPU). The `embed` stage is optional: when
`data/search/` exists it is used as a third retriever. Plus a language model: ollama running with a
model pulled (default `llama3.2:3b`) or `ANTHROPIC_API_KEY` for Claude.

## Run

From the app: sidebar -> *Ask Wikipedia*. Type a question and press Enter; the answer streams in as
markdown on the left, the numbered sources are on the right.

| key | action |
|---|---|
| Enter | ask (in the question box); on a source: show the whole passage |
| `ctrl+o` | show the whole article of the highlighted source (cleaned from the dump) |
| `ctrl+n` | new conversation (follow-up questions use the earlier turns) |
| `ctrl+b` / `ctrl+e` | switch backend (ollama / claude) / cycle the model (ollama: the models the server has pulled) |
| `ctrl+x` | stop generating |
| `ctrl+l` | focus the question box |

The status line shows retrieval time and generation speed ("retrieval 6,868 ms (search 199, read 544,
rerank 6121 ms over 114 passages), generation 58 tokens at 14.1 tok/s, first token 0.1 s") and which
sources the answer cited. Command line:

```bash
python -m experiments.ask "Who designed the Eiffel Tower, and how tall is it?" --show-passages
python -m experiments.ask -i                         # conversation; /new starts a fresh one
python -m experiments.ask "..." --backend claude [--model claude-sonnet-5-5] [--effort high]
python -m experiments.ask "..." --ollama-url http://gpu-pc:11434 --model llama3.1:8b
```

### Using the GPU machine's ollama

On the GPU machine install ollama, `ollama pull llama3.1:8b` (or any bigger model), and start it so it
listens on the network: `OLLAMA_HOST=0.0.0.0 ollama serve` (Windows: set the `OLLAMA_HOST` environment
variable, restart ollama, and allow port 11434 through the firewall). Here, set *Ask -> Ollama ->
Server URL* to `http://<gpu-pc>:11434` and *Model* to the pulled name (app settings tab *Ask*, saved
with `ctrl+s` into `config.toml` as `[ask] ollama_url / ollama_model`), or pass `--ollama-url` / `--model`.
Only the question and the retrieved passages travel over the network; the Wikipedia data stays here.
A bigger model on a GPU answers in seconds and follows the citation rules better than a 3B model.

### Using Claude

`pip install anthropic` (it is in `requirements.txt`), put `ANTHROPIC_API_KEY` in the environment of the
terminal that starts the app, and choose *Backend: claude* (`ctrl+b`, or *Ask -> Backend* in settings, or
`--backend claude`). The key is only ever read from the environment, never written to `config.toml`.
Defaults: model `claude-opus-5-5`, adaptive thinking, effort `medium` (*Ask -> Claude* settings:
`low`, `medium`, `high`, `xhigh`, `max`), streaming through `client.beta.messages.stream` with
server-side refusal fallbacks (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`). A
refusal, a cut-off answer (`max_tokens`), a bad key, rate limiting and network errors each give a
specific message. The prompt budget is larger for Claude (8,000 tokens of sources instead of 2,400).
**This was implemented from the API docs and tested against a mocked client only: this machine has no
API key, so it has never talked to the real API.**

## How it works (`wikiexp/rag.py`)

1. **Standalone query.** A follow-up like "When was it completed?" is detected (pronoun, very short, "and
   ...") and rewritten by the model into "When was the Eiffel Tower completed?" (heuristic fallback: the
   previous question is prepended). The user's own words still go to the answering prompt.
2. **Candidates.** `HybridSearch` (`wikiexp/retrieval.py`) returns ~15 articles by fusing three ranked
   lists with reciprocal rank fusion: semantic (embeddings, when built), keyword (FTS5 bm25 over title +
   intro of all 6.87M non-disambiguation articles) and exact title matches for names in the question
   (longest phrase first, via the title index), plus a mild popularity prior (up to +15% for the most
   linked-to articles). Hybrid helps most while only 50,000 articles are embedded: "Tacoma Narrows
   Bridge" is not among them, but its words and its name find it.
3. **Read.** The top 4 articles are fetched in full from the dump (~40 ms each) and cut into sections
   with `wikitext.sections` (references, external links, ... dropped).
4. **Passages.** Each section becomes passages of ~120-220 words that never cross a section and carry
   "Article — Section". Articles with many passages are pre-filtered by word overlap (30 + the lead).
5. **Rerank.** The bge embedder (the one semantic search uses, loaded once) scores each passage against
   the standalone query (cosine + a small word-overlap bonus), on CPU: ~100 passages take 4-6 s on this
   loaded machine, and this is the slowest retrieval step.
6. **Pack.** The best passages go in, numbered in score order, until the token budget is spent: at most 3
   per article, only passages within 0.10 of the best score, at most 8.
7. **Answer.** The system prompt says to use only the numbered sources, cite each claim like [1], say
   plainly when the sources don't contain the answer, and not to list the sources. The last 3 turns are
   sent as chat history (with their old citation numbers removed, since the numbering restarts).
8. **Check.** Every `[n]` (also `[1, 2]`, `[1-3]`) is validated against the sources actually given;
   numbers that don't exist are removed and reported ("removed citations to sources that don't exist: [9]"),
   and an answer that cites nothing is flagged.

## Keyword index (`keyword` stage)

`python -m pipeline.build_keyword` -> `data/text/fts.sqlite`: FTS5 (porter stemmer, unicode61 with
accents folded) over title and lead as two columns, so bm25 weights a title hit 8x. Measured on this
machine (16 cores, loaded): **4.0 minutes** (3.7 min indexing at ~33,000 articles/s, 18 s optimising),
**2.16 GB**, 6,869,040 articles (the 365,618 disambiguation pages are excluded; their titles are still
found by the title index). The build is single-threaded and reads
`leads.sqlite` once; it uses about the 1 GB SQLite page cache plus the in-memory index segment (peak RAM
was not measured, but nothing else is held in memory).

Querying turns the question into words, drops stop words and words that occur in more than 5% of
articles, looks each word up in the index vocabulary (stemmed exactly as the index stems it) and ANDs the
rarest six; if that finds fewer than k articles the rarest four are ORed in. A query takes 4-300 ms over
all 6.87M articles (examples below).

## Real runs (this machine, ollama `llama3.2:3b` on CPU, 16 cores shared with other jobs)

Hybrid retrieval over the real data (50,000 articles embedded):

| question | top results (via) |
|---|---|
| why did the Tacoma Narrows Bridge collapse | Tacoma Narrows Bridge (1940) (keyword+name), Tacoma, Washington, Johnson Sea Link accident, Puget Sound, Tacoma Narrows Bridge (1950), ... |
| how tall is Mount Kilimanjaro | Mount Kilimanjaro (semantic+keyword+name), Mount Everest, Summit, ... |
| what is the capital of Burkina Faso | Ouagadougou (semantic+name), Burkina Faso, ... |
| who wrote the novel Middlemarch | Middlemarch (keyword+name), Novelist, Joseph Conrad, George Eliot (keyword), ... |

(Keyword alone for "what is the capital of Burkina Faso" returns villages of Burkina Faso; the title match
and the popularity prior are what put the real answer first. Semantic alone misses the Tacoma Narrows
Bridge, which isn't embedded.) Retrieval takes 130-300 ms; the first question loads the embedding model (~4.5 s).

Thirteen questions through the whole pipeline (a script calling `RAG.answer`; follow-ups share a
conversation; answers are the model's text after citation validation):

| # | question | answer | verdict |
|---|---|---|---|
| 1 | Who designed the Eiffel Tower, and how tall is it? | "Designed by Gustave Eiffel, and it is 330 m (1,083 ft) tall. [1]" | correct, cited; the `{{convert}}` in the source text rendered as "330 m (1,083 ft)" |
| 2 | (follow-up) When was it completed? | "The main structural work was completed at the end of March 1889." | correct; rewritten to "When was the Eiffel Tower completed?"; **no citation** (flagged) |
| 3 | (follow-up) Why did people object to it? | artistic grounds, quoting "a truly tragic street lamp" [1] | correct and cited (rewritten to "What were the objections to the Eiffel Tower's construction?") |
| 4 | Why did the Tacoma Narrows Bridge collapse? | aeroelastic flutter, self-exciting, wind above ~35 mph [1] | correct (the bridge is not embedded; found by words+name); a little garbled, copied from the passage |
| 5 | What is the capital of Burkina Faso? | "Ouagadougou [1]." | correct |
| 6 | Who wrote the novel Middlemarch? | George Eliot, pen name of Mary Ann Evans [1][3] | correct, two sources |
| 7 | How do plants turn sunlight into energy? | photosynthesis, chlorophyll, CO2 + water -> glucose and oxygen [3][6] | correct; 10 sources made this the slowest prompt (2,153 tokens, 46 s), which led to tightening the pack margin from 0.12 to 0.10 and max 8 sources |
| 8 | When was the Battle of Waterloo fought and who won? | 18 June 1815; Wellington's British-led force and Blücher's Prussians beat Napoleon [1] | correct |
| 9 | (follow-up) Who commanded the losing side? | "Napoleon I commanded the French Imperial Army." | correct; rewritten to name the battle; **no citation** (flagged) |
| 10 | Who was the first woman to win a Nobel Prize? | Marie Curie [1] | correct |
| 11 | **What will the weather be like in Paris tomorrow?** (not in Wikipedia) | "The sources don't say. [1]" | declined correctly (sources were Paris — Climate); the stray [1] is a small model's habit |
| 12 | What is the boiling point of water on top of Mount Everest? | "71 °C (160 °F) [1]" | matches the source passage (Boiling point article) |
| 13 | Which scientist proposed that continents drift, and why was he not believed at first? | Alfred Wegener; no known mechanism could move the continents [1] | correct |

Quality in one sentence: with good retrieval a 3B model answers faithfully and mostly cites, but it
sometimes omits the citation (2 of 13, both follow-ups) and writes a citation after a refusal; it is not
asked to reason across many sources, and multi-article "compare X and Y" questions were not tested.
The retrieved sources were right for all 13 questions; the one-article-per-name limit means an answer
spread over more than 4 articles will miss things.

Speed (CPU, `llama3.2:3b`, 16 cores shared with other jobs): retrieval 4.4-9.6 s (of which reranking
~100 passages 3.7-6.6 s, searching 0.1-0.6 s, reading 4 articles 0.4-1.0 s); the prompt is 540-2,150
tokens; generation **12-18 tokens/s**; the time before the first token is dominated by reading the prompt
(11-42 s for 800-2,150 tokens, i.e. ~50-90 tokens/s; ollama caches an identical prompt prefix, so
repeating a question is instant). A whole answer takes **14-52 s** end to end. On a GPU machine's ollama
this should drop to a few seconds (not measured: no GPU here). The ollama context is set to 8,192 tokens (ollama's default 4,096 silently
truncates long prompts).

## Better article text (`wikiexp/wikitemplates.py`)

RAG reads whole articles, so gaps left by dropped templates matter more than in a lead. The cleaner
used to drop every template it didn't know; now the common inline ones are rendered as the reader
sees them: `convert`/`cvt` (value, unit **and** the conversion: `{{convert|300|m}}` -> "300 m (980 ft)",
ranges, "5 ft 6 in", `ftin`, scaled units, precision, `disp=`), `lang`/`lang-xx`/`langx`/`transl`,
`nihongo` ("Tokyo (東京, Tōkyō)"), `zh`, birth/death/start/end dates (`df=y` day-first), `date`,
`as of`, `circa`, `reign`, `BCE`/`CE`, `marriage`, `frac`, `ordinal`, `val`, `formatnum:`, `#expr:` (a
safe arithmetic-only evaluator), `chem`/`chem2`/`nuclide`, ship names (`USS`, `HMS`, `sclass`, `ship`),
`angbr`, `gloss`, `lit`, currency, ISBN, and the typography ones (`nowrap`, `abbr`, `sic`, `small`, `sup`,
dashes, ...). Pronunciation templates (`IPA`, `IPAc-*`, `respell`, `audio`) and citations are still
dropped, and the punctuation they leave in "(IPA; born 1944)" or "(Chinese: x; pinyin: y)" is cleaned
properly now (a run of `;,:` after "(" and before ")", "(or)", ...).

Measured on 799 real articles (400 of the 20,000 most-linked, 399 random; 13.2 MB of plain text), counting
dangling punctuation in the cleaned text, before -> after:

| pattern | whole articles | leads (what the `text` stage stores) |
|---|---|---|
| `(;` / `(,` / `;)` right after or before a parenthesis | 8 -> 0 | 5 -> 0 |
| `,,` (a list whose items were templates) | 12 -> 0 | 0 -> 0 |
| `:` followed by `,;.` ("include:,, and") | 20 -> 11 (the rest are quotes that start with "...") | 1 -> 1 |
| empty `()` / `[]`, space before comma or period | 0 -> 0 | 0 -> 0 |

and 1,496 `convert`/`cvt` calls in that sample (plus ~190 `chem`, ~200 `sclass`, ~230 ship-name
templates and ~400 date templates) now produce text instead of nothing or only the first number. Cleaning speed is
unchanged (lead extraction 300 articles: 0.49 s -> 0.55 s; whole articles 200: 3.13 s -> 3.17 s). Tests are in
`tests/test_wikitemplates.py`.

**The `text` stage has not been re-run** (your decision): `leads.sqlite` on disk predates this change.
Re-running it (~21 minutes) would improve the leads, the keyword index (re-run `keyword` afterwards, ~4
minutes) and the embeddings; the RAG assistant itself already benefits, because it reads articles fresh
from the dump through the new cleaner. `fts.sqlite` records which `leads.sqlite` it was built from and
refuses to run against a different one ("rebuild the keyword index").

## Limits and ideas

- Retrieval reads whole articles but only 4 per question; a question about a *list* or comparison across many
  articles is answered from whatever 4 were best.
- Section hierarchy is flat ("Article — Subsection"): `sections()` doesn't keep heading levels.
- Reranking is bge-small cosine; a cross-encoder re-ranker would be more accurate (and slower on CPU).
- The same retrieval would also serve a "chat with a category/list" mode or return the passages alone as an
  offline passage search.
