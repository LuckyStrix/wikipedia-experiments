"""How notable is an article? Ranks from data/centrality/ (pipeline/build_centrality.py).

    from wikiexp.centrality import Centrality
    c = Centrality()                       # or Centrality(data_dir)
    c.rank(idx)                            # 1 = the most notable article (see below)
    c.percentile(idx)                      # 99.9 = in the top 0.1%
    c.top(10, "indegree")                  # idx array of the 10 most linked-to articles
    c.notability(idx)                      # percentile by the default metric

Metrics: ``pagerank`` and ``indegree`` count every link in the pagelinks dump, including the ones
templates generate (citations, infoboxes, navboxes), so ISBN and "Geographic coordinate system" top
them. ``prose_pagerank`` counts only links written in article text (see wikiexp/prose_links.py) and
ranks what people expect; it exists once the prose graph is built and centrality has run with
``--graph``. ``c.default_metric`` is ``prose_pagerank`` when present, else ``pagerank``, and is what
every method uses when you don't name a metric.

Everything is indexed by graph idx (articles.idx) and memory-mapped, so loading is instant and a
lookup touches a few pages. Functions accept a single idx or a numpy array of them, so a ranker can
score thousands of candidates in one call: ``c.percentile(candidate_idxs)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths

# metric -> what it is
METRICS = {
    "pagerank": "PageRank (importance flowing along links)",
    "indegree": "In-degree (number of incoming links)",
    "reverse_pagerank": "Reverse PageRank (hubs: links out to important articles)",
    "prose_pagerank": "Prose PageRank (importance flowing along links written in article text)",
}
# metric -> file prefix of its rank/order arrays
_PREFIX = {"pagerank": "", "indegree": "indegree_", "reverse_pagerank": "reverse_", "prose_pagerank": "prose_"}


class Centrality:
    def __init__(self, data_dir: Path | str | None = None):
        data_dir = Path(data_dir or paths.DATA)
        self.dir = data_dir / "centrality"
        try:
            self.meta = json.loads((self.dir / "meta.json").read_text())
        except OSError:
            raise FileNotFoundError(f"{self.dir} not found - run: python -m pipeline.build_centrality") from None
        self._graph_dir = data_dir / "graph"
        self._cache: dict = {}
        self.pagerank = self._load("pagerank")
        self.n = len(self.pagerank)
        try:
            self.prose_meta = json.loads((self.dir / "prose_meta.json").read_text())
        except (OSError, ValueError):
            self.prose_meta = {}
        graph = Path(self.prose_meta.get("graph", "prose_graph"))
        self._prose_graph_dir = graph if graph.is_absolute() else data_dir / graph

    def _load(self, name):
        if name not in self._cache:
            self._cache[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")
        return self._cache[name]

    def _graph(self, name):
        if name not in self._cache:
            self._cache[name] = np.load(self._graph_dir / f"{name}.npy", mmap_mode="r")
        return self._cache[name]

    def _prose(self, name):
        key = f"prose/{name}"
        if key not in self._cache:
            self._cache[key] = np.load(self._prose_graph_dir / f"{name}.npy", mmap_mode="r")
        return self._cache[key]

    def __len__(self):
        return self.n

    @property
    def metrics(self) -> list[str]:
        """The metrics available in this build (reverse and prose PageRank are optional)."""
        if "metrics" not in self._cache:
            have = [m for m in METRICS
                    if m in ("pagerank", "indegree") or (self.dir / f"{_PREFIX[m]}rank.npy").exists()]
            if "prose_pagerank" in have and self.prose_meta.get("articles") != self.n:
                have.remove("prose_pagerank")     # no meta file, or computed for a different graph
            self._cache["metrics"] = have
        return self._cache["metrics"]

    @property
    def default_metric(self) -> str:
        """What "notable" means unless you say otherwise: prose PageRank if built, else PageRank."""
        return "prose_pagerank" if "prose_pagerank" in self.metrics else "pagerank"

    def _check(self, metric: str | None) -> str:
        metric = metric or self.default_metric
        if metric not in self.metrics:
            raise ValueError(f"unknown or unbuilt metric {metric!r}; choose from {', '.join(self.metrics)}")
        return metric

    @property
    def reverse_pagerank(self):
        return self._load("reverse_pagerank")

    @property
    def prose_pagerank(self):
        return self._load("prose_pagerank")

    @property
    def indegree(self):
        """Incoming links per idx (from the link graph's in_indptr)."""
        if "indegree" not in self._cache:
            self._cache["indegree"] = np.diff(self._graph("in_indptr"))
        return self._cache["indegree"]

    @property
    def outdegree(self):
        if "outdegree" not in self._cache:
            self._cache["outdegree"] = np.diff(self._graph("out_indptr"))
        return self._cache["outdegree"]

    @property
    def has_prose_graph(self) -> bool:
        return (self._prose_graph_dir / "in_indptr.npy").exists()

    @property
    def prose_indegree(self):
        """Incoming links per idx counting only links written in article text."""
        if "prose_indegree" not in self._cache:
            self._cache["prose_indegree"] = np.diff(self._prose("in_indptr"))
        return self._cache["prose_indegree"]

    @property
    def prose_outdegree(self):
        if "prose_outdegree" not in self._cache:
            self._cache["prose_outdegree"] = np.diff(self._prose("out_indptr"))
        return self._cache["prose_outdegree"]

    def score(self, idx, metric: str | None = None):
        """The raw value of a metric: PageRank score (a share of 1), or the in-degree."""
        metric = self._check(metric)
        arr = {"pagerank": lambda: self.pagerank, "indegree": lambda: self.indegree,
               "reverse_pagerank": lambda: self.reverse_pagerank,
               "prose_pagerank": lambda: self.prose_pagerank}[metric]()
        return arr[idx]

    def rank(self, idx, metric: str | None = None):
        """1 = best. An int for an int idx, an int32 array for an array."""
        metric = self._check(metric)
        r = self._load(_PREFIX[metric] + "rank")[idx]
        return int(r) if np.ndim(r) == 0 else np.asarray(r)

    def percentile(self, idx, metric: str | None = None):
        """0-100, higher is more notable: the share of articles this one outranks (the top
        article is 100, the median is 50)."""
        r = np.asarray(self.rank(idx, metric), dtype=np.float64)
        p = 100.0 * (self.n - r) / max(self.n - 1, 1)
        return float(p) if p.ndim == 0 else p

    def notability(self, idx):
        """Percentile (0-100) by the default metric: how notable an article is, as people mean it."""
        return self.percentile(idx, self.default_metric)

    def top(self, n: int, metric: str | None = None, skip: int = 0) -> np.ndarray:
        """idx of the articles ranked skip+1 .. skip+n, best first."""
        metric = self._check(metric)
        return np.array(self._load(_PREFIX[metric] + "order")[skip:skip + max(n, 0)])

    def at_rank(self, rank: int, metric: str | None = None) -> int:
        """idx of the article with this rank (1-based)."""
        if not 1 <= rank <= self.n:
            raise IndexError(f"rank {rank} outside 1..{self.n}")
        metric = self._check(metric)
        return int(self._load(_PREFIX[metric] + "order")[rank - 1])
