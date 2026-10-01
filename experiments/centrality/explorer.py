"""UI-free logic for the Centrality experiment: leaderboards, one article's profile, and the
"surprises" (articles whose PageRank rank and in-degree rank disagree most).

When the prose link graph is built (links written in article text, no templates), prose PageRank is
the default metric: the full-graph PageRank is topped by ISBN, coordinates and other template targets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wikiexp import core, paths
from wikiexp.centrality import METRICS, Centrality
from wikiexp.titles import Match, TitleIndex

SHORT = {"pagerank": "PageRank", "indegree": "In-links", "reverse_pagerank": "Gateway",
         "prose_pagerank": "Prose PR"}


@dataclass
class Row:
    """One article with its standing in every metric that was built."""
    idx: int
    title: str
    score: float                    # of the metric asked for
    in_links: int                   # of the metric's graph: text-only links for prose PageRank, else all links
    out_links: int
    ranks: dict[str, int]           # metric -> rank


@dataclass
class Profile(Row):
    percentiles: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    linked_from: list[tuple[str, int]] = field(default_factory=list)   # most notable inbound, (title, rank by `metric`)
    links_to: list[tuple[str, int]] = field(default_factory=list)
    metric: str = "pagerank"        # which metric ranks the neighbours (the default one)
    prose_in: int | None = None     # links written in article text, when the prose graph is built
    prose_out: int | None = None


@dataclass
class Surprise:
    idx: int
    title: str
    pagerank_rank: int
    indegree_rank: int
    in_links: int

    @property
    def factor(self) -> float:
        """How many times better the better rank is than the worse one."""
        a, b = self.pagerank_rank, self.indegree_rank
        return max(a, b) / min(a, b)


def format_score(metric: str, value: float) -> str:
    if metric == "indegree":
        return f"{int(value):,}"
    return f"{100 * value:.3g}%"     # a share of all the PageRank in Wikipedia


class Explorer:
    def __init__(self, data_dir: Path | None = None):
        data_dir = Path(data_dir or paths.DATA)
        self.c = Centrality(data_dir)
        self.titles = TitleIndex(data_dir / "titles.sqlite")
        self.graph = core.Graph(data_dir / "graph")
        self.prose_graph = core.Graph(self.c._prose_graph_dir) if self.c.has_prose_graph else None
        self.db = core.connect(data_dir / "wiki.sqlite", check_same_thread=False)

    @property
    def metrics(self) -> list[str]:
        return self.c.metrics

    @property
    def default_metric(self) -> str:
        return self.c.default_metric

    def _degrees(self, idxs, metric: str):
        """(in, out) link counts per idx in the graph a metric was computed on."""
        if metric == "prose_pagerank" and self.prose_graph is not None:
            return self.c.prose_indegree[idxs], self.c.prose_outdegree[idxs]
        return self.c.indegree[idxs], self.c.outdegree[idxs]

    def titles_of(self, idxs) -> list[str]:
        """Article titles for graph indices, in the order given."""
        idxs = [int(i) for i in idxs]
        found: dict[int, str] = {}
        for k in range(0, len(idxs), 500):
            chunk = idxs[k:k + 500]
            q = ",".join("?" * len(chunk))
            found.update(self.db.execute(f"SELECT idx, title FROM articles WHERE idx IN ({q})", chunk))
        return [found.get(i, "?") for i in idxs]

    def idx_of(self, match: Match) -> int:
        i = self.graph.idx(match.page_id)
        if i is None:
            raise ValueError("article not in the link graph - were titles and graph built from the same dump?")
        return i

    def _rows(self, idxs: np.ndarray, metric: str) -> list[Row]:
        idxs = np.asarray(idxs)
        titles = self.titles_of(idxs)
        scores = np.asarray(self.c.score(idxs, metric))
        ins, outs = self._degrees(idxs, metric)
        ranks = {m: self.c.rank(idxs, m) for m in self.metrics}
        return [Row(int(i), titles[k], float(scores[k]), int(ins[k]), int(outs[k]),
                    {m: int(r[k]) for m, r in ranks.items()}) for k, i in enumerate(idxs)]

    def leaderboard(self, metric: str | None = None, n: int = 50, skip: int = 0) -> list[Row]:
        metric = metric or self.default_metric
        return self._rows(self.c.top(n, metric, skip), metric)

    def profile(self, idx: int, neighbours: int = 5) -> Profile:
        """An article's standing in every metric. Its degrees are those of the full link graph (with
        the prose-graph degrees alongside), its neighbours the most notable ones by the default
        metric, taken from the graph that metric was computed on."""
        row = self._rows(np.array([idx]), "pagerank")[0]
        metric = self.default_metric
        p = Profile(**row.__dict__, metric=metric)
        for m in self.metrics:
            p.percentiles[m] = self.c.percentile(idx, m)
            p.scores[m] = float(self.c.score(idx, m))
        if self.prose_graph is not None:
            p.prose_in, p.prose_out = (int(x[0]) for x in self._degrees(np.array([idx]), "prose_pagerank"))
        graph = self.prose_graph if metric == "prose_pagerank" and self.prose_graph is not None else self.graph
        rank = self.c.rank
        for attr, links in (("linked_from", graph.in_links(idx)), ("links_to", graph.out_links(idx))):
            links = np.asarray(links)
            if len(links):
                best = links[np.argsort(rank(links))[:neighbours]]   # the most notable neighbours
                setattr(p, attr, list(zip(self.titles_of(best), (int(r) for r in rank(best)))))
        return p

    def surprises(self, top: int = 100_000, n: int = 25) -> tuple[list[Surprise], list[Surprise]]:
        """Among the `top` articles by either measure, the n whose PageRank rank is furthest
        better than their in-degree rank ("punch above their links": a few links, but from
        important places) and the n furthest worse ("many links, little weight").

        Ranks are compared as ratios (rank 50 vs 5,000 is as surprising as 5,000 vs 500,000)."""
        top = min(top, len(self.c))
        cand = np.union1d(self.c.top(top, "pagerank"), self.c.top(top, "indegree"))
        pr, ind = self.c.rank(cand, "pagerank").astype(np.float64), self.c.rank(cand, "indegree").astype(np.float64)
        ratio = np.log(ind / pr)          # > 0: PageRank rank is better
        up = np.argsort(-ratio, kind="stable")[:n]
        down = np.argsort(ratio, kind="stable")[:n]
        ids = [int(cand[k]) for k in np.union1d(up, down)]
        titles = dict(zip(ids, self.titles_of(ids)))

        def make(sel):
            return [Surprise(int(cand[k]), titles[int(cand[k])], int(pr[k]), int(ind[k]),
                             int(self.c.indegree[cand[k]])) for k in sel]
        return make(up), make(down)


def describe(metric: str) -> str:
    return METRICS[metric]
