"""Glue between titles and the solver: load the data once, then answer "path from A to B"."""
from __future__ import annotations

from pathlib import Path

from wikiexp import core, paths
from wikiexp.titles import Match, TitleIndex

from .solver import Result, shortest_path


class PathFinder:
    def __init__(self, data_dir: Path | None = None):
        data_dir = Path(data_dir or paths.DATA)
        self.titles = TitleIndex(data_dir / "titles.sqlite")
        self.graph = core.Graph(data_dir / "graph")
        self.db = core.connect(data_dir / "wiki.sqlite", check_same_thread=False)

    def title_of(self, idx: int) -> str:
        return self.db.execute("SELECT title FROM articles WHERE idx = ?", (int(idx),)).fetchone()[0]

    def find(self, start: Match, goal: Match) -> tuple[Result, list[str]]:
        a, b = self.graph.idx(start.page_id), self.graph.idx(goal.page_id)
        if a is None or b is None:
            raise ValueError("article not in the link graph - were titles and graph built from the same dump?")
        result = shortest_path(self.graph, a, b)
        return result, [self.title_of(i) for i in result.path or []]
