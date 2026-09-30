# Experiments

One folder per experiment, e.g. `experiments/six_degrees/`. Use underscores in the name so it can be
run as a module: `python -m experiments.six_degrees.solve "Kevin Bacon" "Mitochondria"`.

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
- `wikiexp.sqldump` — streaming readers for Wikimedia `*.sql.gz` dumps
