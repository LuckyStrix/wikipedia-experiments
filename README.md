# wikipedia-experiments

Fun things to do with a full offline copy of English Wikipedia: link-graph games (Six Degrees,
Getting to Philosophy), offline semantic search, maps, timelines and more. See [IDEAS.md](IDEAS.md)
for the project list.

The repo holds code only. The data (~90 GB of Wikimedia dumps and a Kiwix `.zim`, plus whatever
gets built from them) lives outside git, and each machine keeps its own copy. This makes it easy to
run heavy jobs on one computer (e.g. a GPU box) and analyse the results on another.

## Layout

```
pipeline/     turns raw dumps into shared datasets (download.sh, build_core.py, ...)
wikiexp/      shared Python library used by the pipeline and experiments
experiments/  one folder per experiment (see experiments/README.md)
tools/        sync.py: move data between machines over SSH
dumps/        raw downloads            - not in git
data/         generated datasets        - not in git
config.toml   this machine's settings   - not in git (see config.example.toml)
```

## Setup

Requires Python 3.11+. Run everything from the repo root.

```bash
git clone https://github.com/LuckyStrix/wikipedia-experiments
cd wikipedia-experiments
pip install -r requirements.txt
cp config.example.toml config.toml   # optional: custom data paths, other machines
```

Then either build the data (below), or copy it from a machine that already has it:
`python -m tools.sync pull <machine> wiki.sqlite graph`.

## Building the data

```bash
pipeline/download.sh          # ~39 GB from dumps.wikimedia.org, a few hours; resumable
python -m pipeline.build_core # ~1 hour, ~15 GB RAM peak
```

`pipeline/download.sh` needs bash and wget (Linux/macOS). On Windows, download the files it lists by
hand, or build on Linux and sync the results over. The Kiwix `.zim` comes from
<https://download.kiwix.org/zim/wikipedia/> and goes in `dumps/` too.

### data/wiki.sqlite

| table | columns | notes |
|---|---|---|
| `articles` | `page_id`, `idx`, `title`, `length` | main-namespace, non-redirect pages; titles use spaces; `length` is wikitext bytes |
| `redirects` | `page_id`, `title`, `target` | `target` is the final article's `page_id` (chains already followed) |
| `links` | `src`, `dst` | article → article by `page_id`; links via redirects resolved; no duplicates or self-links |
| `meta` | `key`, `value` | dump date, build time, row counts |

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

## Working across machines

Code moves through git; data moves with `tools/sync.py` (plain `ssh`/`scp`, so it works with
Windows' built-in OpenSSH server, no WSL needed). Describe your other machines in `config.toml`, then:

```bash
python -m tools.sync push gpu wiki.sqlite graph            # send inputs to the GPU machine
python -m tools.sync run  gpu "python -m experiments.x.y"  # git pull + run there
python -m tools.sync pull gpu x                            # fetch data/x back
```

For long jobs, start them in a terminal on the remote machine (or with `tools.sync run` inside
`tmux`/`screen` locally), since an SSH disconnect stops the remote command.

## License

Code: MIT. Wikipedia content is © Wikipedia contributors, licensed CC BY-SA 4.0; none of it is
stored in this repo.
