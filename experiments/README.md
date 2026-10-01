# Experiments

One folder per experiment, e.g. `experiments/six_degrees/`. Use underscores in the name so it can be
run as a module: `python -m experiments.six_degrees "Kevin Bacon" "Mitochondrion"`.

To show an experiment in the app, add it to `EXPERIMENTS` in `experiments/__init__.py` with the
pipeline stages it needs (`requires`) and its Textual screen (`"module:Class"`; the screen is given
the data folder). Keep the logic UI-free (see `six_degrees/solver.py`) so it also works from the
command line and in tests.

Each experiment folder contains:

- `README.md` — what it does, how to run it, which machine it needs (CPU / GPU / RAM), and findings
- code
- `results/` *(optional)* — small outputs worth keeping in git: charts, CSVs, write-ups.
  Binary data formats here (`.parquet`, `.npy`, `.npz`, `.sqlite`, …) are stored with Git LFS
  automatically (see `.gitattributes`). Keep this under ~100 MB.

Large outputs (embeddings, full tables) go in `data/<experiment_name>/`, which is not in git.
Move them between machines with `python -m tools.sync push|pull <machine> <experiment_name>`.

Shared code goes in `wikiexp/`:

- `wikiexp.paths` — where `DATA`, `DUMPS`, `DB`, `GRAPH` live on this machine
- `wikiexp.core` — `connect()` to wiki.sqlite, `resolve(db, title)`, and `Graph()` for the link graph
- `wikiexp.titles` — `TitleIndex`: title search with autocomplete-style suggestions
- `wikiexp.progress` — `progress(pct, text)` so a long script drives the app's progress bar
- `wikiexp.sqldump` — streaming readers for Wikimedia `*.sql.gz` dumps
- `wikiexp.wikitext` — `WikiText().fetch(page_id)` (random access to one article's wikitext),
  `iter_pages()` (all articles, in parallel), `to_text()` / `sections()` / `lead()` (clean plain text)
- `wikiexp.semantic` — `SemanticSearch().search(query, k)`: articles closest in meaning (the RAG building block)
- `tui.widgets.TitlePicker` — a title input for screens that only accepts real articles
