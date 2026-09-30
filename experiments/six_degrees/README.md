# Six Degrees of Wikipedia

The shortest chain of links from one article to another, e.g. Kevin Bacon → … → Mitochondrion.

**Needs:** the core database and title index (pipeline stages 2 and 3). **Machine:** any; CPU
only. The link graph is memory-mapped, so it loads instantly; the first searches warm the OS cache
(~6 GB of graph arrays for enwiki) and later ones take well under a second.

**Run:** from the app (sidebar → Six Degrees), or

```bash
python -m experiments.six_degrees "Kevin Bacon" "Mitochondrion"
python -m experiments.six_degrees --random
```

## How it works

- `solver.py` — bidirectional breadth-first search: forward along outgoing links from the start,
  backward along incoming links from the goal, always expanding the side with fewer links to follow,
  until the two meet. Each level is expanded with vectorised numpy over the CSR arrays. Returns one
  shortest path (there are usually many).
- `finder.py` — loads the title index, graph and database once and maps titles ↔ graph indices.
- `screen.py` — the app screen: two `TitlePicker`s (only real articles can be picked), Find /
  Swap / Random pair, and a results log.
- `__main__.py` — command line; if a title isn't an article it lists suggestions to pick from.

Links are directed (A links to B doesn't mean B links to A), so A → B and B → A can differ.

## Ideas

- All shortest paths, not just one (count them, or show the most "surprising" one)
- Distribution of distances between random pairs; the average degree of separation on Wikipedia
- The most "central" articles: those on the most shortest paths
