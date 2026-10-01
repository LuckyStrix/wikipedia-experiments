"""Prose PageRank: build_centrality --graph, the Centrality loader's default metric, the Centrality
experiment (explorer, CLI, screen) and the stage wiring. Tiny data only."""
import json
import sys

import numpy as np
import pytest

from pipeline import build_centrality as bc
from pipeline import build_core
from pipeline import settings as S
from pipeline import stages as ST
from tests.conftest import ARTICLES
from tests.test_centrality import built, run_cli          # noqa: F401  (fixture reused)
from tests.test_tui import make_app, wait_for
from wikiexp.centrality import Centrality

# links written in article text: a different shape from the template-polluted full graph
PROSE_LINKS = [("Film", "Kevin Bacon"), ("Footloose", "Kevin Bacon"), ("Bacon", "Kevin Bacon"),
               ("Kevin Bacon", "Footloose"), ("Cell (biology)", "Mitochondrion"), ("Biology", "Mitochondrion")]


def write_prose_graph(data, links=PROSE_LINKS):
    idx = {t: i for i, t in enumerate(ARTICLES)}
    keys = np.array(sorted((idx[a] << 32) | idx[b] for a, b in links), dtype=np.int64)
    src, dst = build_core.split_keys(keys)
    build_core.write_graph(src, dst, np.load(data / "graph" / "idx_to_page_id.npy"), data / "prose_graph",
                           lambda *a: None)
    return data / "prose_graph"


def run_centrality(monkeypatch, *argv):
    monkeypatch.setattr(bc, "log", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["build_centrality", *argv])
    bc.main()


@pytest.fixture
def prose_built(built, monkeypatch):
    run_centrality(monkeypatch, "--graph", str(write_prose_graph(built)))
    return built


# ── the build ────────────────────────────────────────────────────────────────

def test_build_with_graph_adds_prose_pagerank_and_keeps_the_rest(prose_built):
    d = prose_built / "centrality"
    pr = np.load(d / "prose_pagerank.npy")
    assert pr.dtype == np.float32 and pr.sum() == pytest.approx(1.0, abs=1e-5)
    rank, order = np.load(d / "prose_rank.npy"), np.load(d / "prose_order.npy")
    assert np.array_equal(rank[order], np.arange(1, len(pr) + 1))
    assert ARTICLES[order[0]] in ("Kevin Bacon", "Mitochondrion")            # the text-link favourites
    meta = json.loads((d / "prose_meta.json").read_text())
    assert meta["articles"] == len(ARTICLES) and meta["links"] == len(PROSE_LINKS) and meta["residual"] < 1e-9
    assert meta["graph"] == "prose_graph"                                     # relative to the data folder
    assert json.loads((d / "meta.json").read_text())["articles"] == len(ARTICLES)
    assert not list(d.glob("*.tmp*"))
    # it is the PageRank of the prose graph, not of the full one
    g = prose_built / "prose_graph"
    expected, *_ = bc.pagerank(np.load(g / "in_indptr.npy"), np.load(g / "in_indices.npy"),
                               np.diff(np.load(g / "out_indptr.npy")))
    assert np.allclose(pr, expected, atol=1e-6) and not np.allclose(pr, np.load(d / "pagerank.npy"), atol=1e-3)


def test_prose_only_leaves_existing_files_alone(built, monkeypatch):
    d = built / "centrality"
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()}
    run_centrality(monkeypatch, "--graph", str(write_prose_graph(built)), "--prose-only")
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()}
    assert {k: v for k, v in after.items() if not k.startswith("prose_")} == before
    assert {"prose_pagerank.npy", "prose_rank.npy", "prose_order.npy", "prose_meta.json"} <= set(after)


def test_graph_option_validation(built, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_centrality", "--prose-only"])
    with pytest.raises(SystemExit):
        bc.main()
    other = write_prose_graph(built)
    np.save(other / "out_indptr.npy", np.load(other / "out_indptr.npy")[:-1])    # one article short
    monkeypatch.setattr(sys, "argv", ["build_centrality", "--graph", str(other)])
    with pytest.raises(SystemExit):
        bc.main()


# ── the loader ───────────────────────────────────────────────────────────────

def test_loader_defaults_to_prose_pagerank(prose_built):
    c = Centrality(prose_built)
    assert c.metrics == ["pagerank", "indegree", "reverse_pagerank", "prose_pagerank"]
    assert c.default_metric == "prose_pagerank"
    top = c.top(2)
    assert top.tolist() == c.top(2, "prose_pagerank").tolist() and c.rank(int(top[0])) == 1
    kb = ARTICLES.index("Kevin Bacon")
    plain_rank = int(np.load(prose_built / "centrality" / "rank.npy")[kb])
    assert c.rank(kb) == c.rank(kb, "prose_pagerank") and c.rank(kb, "pagerank") == plain_rank
    assert c.score(kb) == pytest.approx(c.prose_pagerank[kb])
    assert c.score(kb, "pagerank") == pytest.approx(c.pagerank[kb])
    assert c.percentile(kb) == c.percentile(kb, "prose_pagerank") == c.notability(kb)
    assert c.at_rank(1) == int(top[0])
    assert c.has_prose_graph and c.prose_indegree[kb] == 3 and c.prose_outdegree[kb] == 1
    assert c.indegree[kb] == 3 and c.outdegree[kb] == 1                         # the full graph's, untouched


def test_loader_without_prose_falls_back_to_pagerank(built):
    c = Centrality(built)
    assert c.default_metric == "pagerank" and "prose_pagerank" not in c.metrics and not c.has_prose_graph
    assert c.notability(0) == c.percentile(0, "pagerank")
    with pytest.raises(ValueError):
        c.rank(0, "prose_pagerank")


def test_loader_ignores_prose_results_for_another_graph(prose_built):
    meta = prose_built / "centrality" / "prose_meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "articles": 5}))
    c = Centrality(prose_built)
    assert "prose_pagerank" not in c.metrics and c.default_metric == "pagerank"
    meta.unlink()                                                               # files without their meta file
    assert "prose_pagerank" not in Centrality(prose_built).metrics


# ── the experiment ───────────────────────────────────────────────────────────

def test_explorer_uses_prose_by_default(prose_built):
    from experiments.centrality.explorer import Explorer
    ex = Explorer(prose_built)
    assert ex.default_metric == "prose_pagerank"
    rows = ex.leaderboard(n=3)
    assert [r.ranks["prose_pagerank"] for r in rows] == [1, 2, 3]
    assert rows[0].title in ("Kevin Bacon", "Mitochondrion")
    kb = next(r for r in ex.leaderboard("prose_pagerank", 8) if r.title == "Kevin Bacon")
    assert kb.in_links == 3 and kb.out_links == 1                               # of the prose graph
    p = ex.profile(ARTICLES.index("Footloose"))
    assert p.metric == "prose_pagerank" and (p.in_links, p.out_links) == (1, 2)     # full-graph degrees
    assert (p.prose_in, p.prose_out) == (1, 1)
    assert [t for t, _ in p.links_to] == ["Kevin Bacon"]                        # neighbours: the prose graph
    assert p.percentiles["prose_pagerank"] == ex.c.percentile(ARTICLES.index("Footloose"))


def test_cli_defaults_to_prose_pagerank(prose_built, monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, "top", "3")
    assert out.startswith("Prose PageRank") and "Prose PR" in out and "PageRank #" in out
    out = run_cli(monkeypatch, capsys, "top", "3", "--metric", "pagerank")
    assert out.startswith("PageRank (importance flowing") and "Prose PR #" in out
    out = run_cli(monkeypatch, capsys, "Mitochondrion")
    assert "in running text only: 2 incoming, 0 outgoing" in out and "(by Prose PR rank)" in out


def test_cli_without_prose_still_defaults_to_pagerank(built, monkeypatch, capsys):
    assert run_cli(monkeypatch, capsys, "top", "2").startswith("PageRank (importance")
    assert "running text" not in run_cli(monkeypatch, capsys, "Film")
    with pytest.raises(SystemExit, match="Build prose link graph"):
        run_cli(monkeypatch, capsys, "top", "--metric", "prose_pagerank")


@pytest.mark.asyncio
async def test_centrality_screen_defaults_to_prose(prose_built, tmp_path):
    from experiments.centrality.screen import CentralityScreen
    from textual.widgets import DataTable, Select
    app = make_app(tmp_path, [], data_dir=prose_built)
    async with app.run_test(size=(150, 50)) as pilot:
        screen = CentralityScreen(prose_built)
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.ex is not None)
        board = screen.query_one("#ce-table", DataTable)

        def labels():
            return [str(c.label) for c in board.columns.values()]
        assert await wait_for(pilot, lambda: board.row_count == len(ARTICLES) and "Text links in" in labels())
        assert screen.query_one("#ce-metric", Select).value == "prose_pagerank"
        assert labels()[2] == "Prose PR" and str(board.get_cell_at((0, 1))) in ("Kevin Bacon", "Mitochondrion")
        screen.query_one("#ce-metric", Select).value = "pagerank"
        assert await wait_for(pilot, lambda: labels()[2] == "PageRank" and "Prose PR #" in labels())


# ── stage and setting ────────────────────────────────────────────────────────

def mark_prose_graph_built(pg):
    for f in ("out_order.npy", "lead_count.npy", "first_link.npy", "meta.json"):
        (pg / f).write_bytes(b"x")


def test_stage_uses_the_prose_graph_when_built_and_enabled(built, monkeypatch):
    st, s = ST.BY_KEY["centrality"], S.Settings({"data_dir": str(built)})
    assert "--graph" not in st.command(s) and st.is_done(s)                      # no prose graph yet
    pg = write_prose_graph(built)
    mark_prose_graph_built(pg)
    assert ST.BY_KEY["prose_graph"].is_done(s)
    cmd = st.command(s)
    assert cmd[cmd.index("--graph") + 1] == str(pg)
    assert cmd[-1] == "--prose-only"                                            # the rest is built: add prose
    assert not st.is_done(s)                                                    # the prose metric is missing
    off = S.Settings({"data_dir": str(built), "centrality_prose": False})
    assert "--graph" not in st.command(off) and st.is_done(off)
    run_centrality(monkeypatch, "--graph", str(pg), "--prose-only")
    assert st.is_done(s) and "--prose-only" not in st.command(s)                # a rerun recomputes everything
    lines = st.summary(s)
    assert any("prose_pagerank.npy" in l for l in lines) and any("prose graph:" in l for l in lines)


def test_prose_setting_is_registered():
    st = S.BY_KEY["centrality_prose"]
    assert st.type == S.BOOL and st.default is True and st.toml == ("stages", "centrality", "prose")
    assert S.Settings()["centrality_prose"] is True
