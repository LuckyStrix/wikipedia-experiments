"""How notable is an article? Ranks from data/centrality/ (pipeline/build_centrality.py).

    from wikiexp.centrality import Centrality
    c = Centrality()                       # or Centrality(data_dir)
    c.rank(idx)                            # 1 = the article with the highest PageRank
    c.percentile(idx)                      # 99.9 = in the top 0.1%
    c.top(10, "indegree")                  # idx array of the 10 most linked-to articles

Everything is indexed by graph idx (articles.idx) and memory-mapped, so loading is instant and a
lookup touches a few pages. Functions accept a single idx or a numpy array of them, so a ranker can
score thousands of candidates in one call: ``c.percentile(candidate_idxs)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths

# metric -> (what it is, file prefix of its rank/order arrays)
METRICS = {
    "pagerank": "PageRank (importance flowing along links)",
    "indegree": "In-degree (number of incoming links)",
    "reverse_pagerank": "Reverse PageRank (hubs: links out to important articles)",
}
_PREFIX = {"pagerank": "", "indegree": "indegree_", "reverse_pagerank": "reverse_"}


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

    def _load(self, name):
        if name not in self._cache:
            self._cache[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")
        return self._cache[name]

    def _graph(self, name):
        if name not in self._cache:
            self._cache[name] = np.load(self._graph_dir / f"{name}.npy", mmap_mode="r")
        return self._cache[name]

    def __len__(self):
        return self.n

    @property
    def metrics(self) -> list[str]:
        """The metrics available in this build (reverse PageRank is optional)."""
        return [m for m in METRICS if m != "reverse_pagerank" or (self.dir / "reverse_rank.npy").exists()]

    def _check(self, metric: str) -> None:
        if metric not in self.metrics:
            raise ValueError(f"unknown or unbuilt metric {metric!r}; choose from {', '.join(self.metrics)}")

    @property
    def reverse_pagerank(self):
        return self._load("reverse_pagerank")

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

    def score(self, idx, metric: str = "pagerank"):
        """The raw value of a metric: PageRank/reverse PageRank score, or the in-degree."""
        self._check(metric)
        arr = {"pagerank": lambda: self.pagerank, "indegree": lambda: self.indegree,
               "reverse_pagerank": lambda: self.reverse_pagerank}[metric]()
        return arr[idx]

    def rank(self, idx, metric: str = "pagerank"):
        """1 = best. An int for an int idx, an int32 array for an array."""
        self._check(metric)
        r = self._load(_PREFIX[metric] + "rank")[idx]
        return int(r) if np.ndim(r) == 0 else np.asarray(r)

    def percentile(self, idx, metric: str = "pagerank"):
        """0-100, higher is more notable: the share of articles this one outranks (the top
        article is 100, the median is 50)."""
        r = np.asarray(self.rank(idx, metric), dtype=np.float64)
        p = 100.0 * (self.n - r) / max(self.n - 1, 1)
        return float(p) if p.ndim == 0 else p

    def top(self, n: int, metric: str = "pagerank", skip: int = 0) -> np.ndarray:
        """idx of the articles ranked skip+1 .. skip+n, best first."""
        self._check(metric)
        return np.array(self._load(_PREFIX[metric] + "order")[skip:skip + max(n, 0)])

    def at_rank(self, rank: int, metric: str = "pagerank") -> int:
        """idx of the article with this rank (1-based)."""
        if not 1 <= rank <= self.n:
            raise IndexError(f"rank {rank} outside 1..{self.n}")
        self._check(metric)
        return int(self._load(_PREFIX[metric] + "order")[rank - 1])
