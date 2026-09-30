"""Shared fixtures: a tiny hand-made Wikipedia (articles, redirects, links) built with the real
pipeline code, so tests don't need the multi-GB dumps."""
import sqlite3
import sys

import numpy as np
import pytest

from pipeline import build_core
from wikiexp import paths

ARTICLES = ["Kevin Bacon", "Footloose", "Film", "Biology", "Cell (biology)", "Mitochondrion",
            "Isolated article", "Bacon"]
REDIRECTS = {"Mitochondria": "Mitochondrion", "Bacon, Kevin": "Kevin Bacon", "Movie": "Film"}
LINKS = [("Kevin Bacon", "Footloose"), ("Footloose", "Film"), ("Film", "Biology"),
         ("Biology", "Cell (biology)"), ("Cell (biology)", "Mitochondrion"), ("Biology", "Mitochondrion"),
         ("Film", "Kevin Bacon"), ("Bacon", "Kevin Bacon"), ("Footloose", "Kevin Bacon"),
         ("Mitochondrion", "Biology")]


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    """data dir with wiki.sqlite, graph/ and titles.sqlite for the articles above."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(paths, "DATA", data)
    monkeypatch.setattr(paths, "DB", data / "wiki.sqlite")
    monkeypatch.setattr(paths, "GRAPH", data / "graph")
    monkeypatch.setattr(build_core, "log", lambda *a, **k: None)

    art_ids = np.array([100 + i for i in range(len(ARTICLES))], dtype=np.int64)
    idx = {t: i for i, t in enumerate(ARTICLES)}
    keys = np.array(sorted((idx[a] << 32) | idx[b] for a, b in LINKS), dtype=np.int64)
    src, dst = build_core.split_keys(keys)
    build_core.write_graph(src, dst, art_ids)
    redirect_titles = [(900 + i, t.encode(), idx[target]) for i, (t, target) in enumerate(REDIRECTS.items())]
    build_core.write_sqlite(art_ids, [t.encode() for t in ARTICLES], np.full(len(ARTICLES), 50),
                            redirect_titles, src, dst, {"articles": len(ARTICLES)})

    from pipeline import build_titles
    monkeypatch.setattr(build_titles, "log", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["build_titles"])
    build_titles.main()
    return data
