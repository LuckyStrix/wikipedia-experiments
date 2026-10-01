"""PageRank build, the Centrality loader, the experiment's logic, CLI and screen (tiny data only)."""
import json
import sys

import numpy as np
import pytest

from pipeline import build_centrality as bc
from pipeline import settings as S
from pipeline import stages as ST
from tests.conftest import ARTICLES
from tests.test_tui import log_text, make_app, wait_for
from wikiexp.centrality import Centrality


def csr_in(adj):
    """In-link CSR (who points at v) and out-degrees of a dense 0/1 adjacency matrix."""
    src, dst = np.nonzero(adj)
    order = np.argsort(dst, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(dst, minlength=len(adj)))])
    return indptr, src[order].astype(np.int32), adj.sum(1)


def dense_pagerank(adj, d):
    n = len(adj)
    out = adj.sum(1)
    p = np.where(out[:, None] > 0, adj / np.maximum(out, 1)[:, None], 1.0 / n)   # dangling rows jump anywhere
    g = d * p + (1 - d) / n
    x = np.full(n, 1.0 / n)
    for _ in range(3000):
        x = x @ g
    return x


@pytest.mark.parametrize("chunk", [5, 1_000_000])
def test_pagerank_matches_dense_solution(monkeypatch, chunk):
    monkeypatch.setattr(bc, "CHUNK_EDGES", chunk)   # tiny chunks exercise the chunk boundaries
    rng = np.random.default_rng(3)
    adj = rng.random((40, 40)) < 0.08
    np.fill_diagonal(adj, 0)
    adj[5, :] = 0                                   # a dangling node
    indptr, indices, out = csr_in(adj)
    pr, iters, residual = bc.pagerank(indptr, indices, out, damping=0.85, tol=1e-12, max_iter=500)
    assert residual < 1e-12 and iters < 500
    assert pr.sum() == pytest.approx(1.0)
    assert np.abs(pr - dense_pagerank(adj, 0.85)).max() < 1e-10


def test_pagerank_reports_residual_each_iteration():
    adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    seen = []
    bc.pagerank(*csr_in(adj), damping=0.85, tol=1e-9, max_iter=3, report=lambda i, r: seen.append((i, r)))
    assert [i for i, _ in seen] == [1, 2, 3]


def test_ranking_orders_best_first_with_stable_ties():
    rank, order = bc.ranking(np.array([0.1, 0.5, 0.1, 0.9], dtype=np.float32))
    assert order.tolist() == [3, 1, 0, 2]
    assert rank.tolist() == [3, 2, 4, 1]
    assert rank.dtype == order.dtype == np.int32


@pytest.fixture
def built(tiny_data, monkeypatch):
    monkeypatch.setattr(bc, "log", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["build_centrality"])
    bc.main()
    return tiny_data


def test_build_writes_consistent_files(built):
    d = built / "centrality"
    pr = np.load(d / "pagerank.npy")
    assert pr.dtype == np.float32 and pr.sum() == pytest.approx(1.0, abs=1e-5)
    for prefix in ("", "indegree_", "reverse_"):
        rank, order = np.load(d / f"{prefix}rank.npy"), np.load(d / f"{prefix}order.npy")
        assert np.array_equal(rank[order], np.arange(1, len(pr) + 1))
    meta = json.loads((d / "meta.json").read_text())
    assert meta["damping"] == 0.85 and meta["residual"] < 1e-9 and meta["articles"] == len(ARTICLES)
    assert not list(d.glob("*.tmp*"))
    # Kevin Bacon is linked from 3 articles and sits in the cycle; the isolated article is last-ish
    assert ARTICLES[np.load(d / "order.npy")[0]] in ("Kevin Bacon", "Film", "Biology")


def test_build_without_reverse(tiny_data, monkeypatch):
    monkeypatch.setattr(bc, "log", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["build_centrality", "--no-reverse", "--damping", "0.5"])
    bc.main()
    c = Centrality(tiny_data)
    assert c.metrics == ["pagerank", "indegree"] and c.meta["damping"] == 0.5
    with pytest.raises(ValueError):
        c.rank(0, "reverse_pagerank")


def test_loader(built):
    c = Centrality(built)
    assert len(c) == len(ARTICLES) and c.metrics == ["pagerank", "indegree", "reverse_pagerank"]
    top = c.top(3)
    assert len(top) == 3 and c.rank(int(top[0])) == 1
    assert c.rank(top).tolist() == [1, 2, 3]                       # arrays work too
    assert c.at_rank(2) == int(top[1])
    assert c.percentile(int(top[0])) == 100.0
    assert c.percentile(int(c.top(len(c))[-1])) == 0.0
    assert c.top(2, skip=1).tolist() == top[1:3].tolist()
    kb = ARTICLES.index("Kevin Bacon")
    assert c.indegree[kb] == 3 and c.outdegree[kb] == 1
    assert c.score(kb, "indegree") == 3 and c.score(kb) == pytest.approx(c.pagerank[kb])
    assert c.top(1, "indegree")[0] == kb                           # most linked-to
    assert c.top(0).tolist() == []
    with pytest.raises(IndexError):
        c.at_rank(0)
    with pytest.raises(ValueError):
        c.top(1, "bogus")


def test_loader_missing_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_centrality"):
        Centrality(tmp_path)


# ── experiment logic ─────────────────────────────────────────────────────────

@pytest.fixture
def explorer(built):
    from experiments.centrality.explorer import Explorer
    return Explorer(built)


def test_leaderboard_and_profile(explorer):
    rows = explorer.leaderboard("indegree", 3)
    assert rows[0].title == "Kevin Bacon" and rows[0].score == 3 and rows[0].ranks["indegree"] == 1
    assert [r.ranks["pagerank"] for r in explorer.leaderboard("pagerank", 4)] == [1, 2, 3, 4]
    assert [r.title for r in explorer.leaderboard("pagerank", 2, skip=1)] == \
        [r.title for r in explorer.leaderboard("pagerank", 3)[1:]]
    p = explorer.profile(ARTICLES.index("Biology"))
    assert p.title == "Biology" and p.in_links == 2 and p.out_links == 2
    assert {t for t, _ in p.links_to} == {"Cell (biology)", "Mitochondrion"}
    assert p.percentiles["pagerank"] == explorer.c.percentile(ARTICLES.index("Biology"))
    iso = explorer.profile(ARTICLES.index("Isolated article"))
    assert iso.linked_from == [] and iso.links_to == []


def test_surprises_are_sorted_and_consistent(explorer):
    up, down = explorer.surprises(top=8, n=3)
    assert len(up) == len(down) == 3
    assert all(s.pagerank_rank <= s.indegree_rank for s in up[:1])
    assert [s.factor for s in up] == sorted((s.factor for s in up), reverse=True)
    assert [s.indegree_rank / s.pagerank_rank for s in up] == \
        sorted((s.indegree_rank / s.pagerank_rank for s in up), reverse=True)
    assert [s.pagerank_rank / s.indegree_rank for s in down] == \
        sorted((s.pagerank_rank / s.indegree_rank for s in down), reverse=True)
    assert explorer.surprises(top=10_000_000, n=2)   # pool larger than the data is fine


def test_idx_of_and_titles_of(explorer):
    m = explorer.titles.resolve("Mitochondria")
    assert explorer.titles_of([explorer.idx_of(m)]) == ["Mitochondrion"]
    assert explorer.titles_of([]) == []


# ── CLI ──────────────────────────────────────────────────────────────────────

def run_cli(monkeypatch, capsys, *argv):
    from experiments.centrality import __main__ as cli
    monkeypatch.setattr(sys, "argv", ["centrality", *argv])
    cli.main()
    return capsys.readouterr().out


def test_cli_top(built, monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, "top", "3")
    assert "PageRank" in out and out.count("\n") == 5            # heading, header, 3 rows
    out = run_cli(monkeypatch, capsys, "top", "2", "--metric", "indegree")
    assert "Kevin Bacon" in out and "In-links" in out
    assert "Gateway #" in run_cli(monkeypatch, capsys, "top")     # default 25 rows, other metrics' ranks


def test_cli_lookup_and_redirect(built, monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, "kevin", "bacon")
    assert out.startswith("Kevin Bacon") and "3 incoming links" in out and "Reverse" not in out
    assert "Gateway" in out and "Most notable articles linking here" in out
    out = run_cli(monkeypatch, capsys, "Mitochondria")
    assert "(via Mitochondria)" in out and out.startswith("Mitochondrion")


def test_cli_unknown_title_suggests_and_exits(built, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        run_cli(monkeypatch, capsys, "Kevin Bac")
    assert e.value.code == 2
    assert "Did you mean" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="No article matches"):
        run_cli(monkeypatch, capsys, "Zzzzqqqq Xxyyzz")


def test_cli_surprises(built, monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, "surprises", "--top", "8", "-n", "2")
    assert "Punching above" in out and "Many links, little weight" in out


def test_cli_without_data(tmp_path, monkeypatch):
    from wikiexp import paths
    monkeypatch.setattr(paths, "DATA", tmp_path)
    with pytest.raises(SystemExit, match="build_centrality"):
        from experiments.centrality import __main__ as cli
        monkeypatch.setattr(sys, "argv", ["centrality", "top"])
        cli.main()


# ── stage and settings ───────────────────────────────────────────────────────

def test_stage_command_follows_settings():
    st = ST.BY_KEY["centrality"]
    cmd = st.command(S.Settings({"centrality_damping": "0.9", "centrality_reverse": False}))
    assert cmd[-3:] == ["--damping", "0.9", "--no-reverse"]
    assert "--no-reverse" not in st.command(S.Settings())
    assert S.Settings({"centrality_damping": "1.5"}).invalid() == ["PageRank damping"]
    assert S.Settings({"centrality_damping": "0.7"}).invalid() == []


def test_stage_done_and_summary(built):
    s = S.Settings({"data_dir": str(built)})
    st = ST.BY_KEY["centrality"]
    assert st.is_done(s) and not st.missing_inputs(s)
    assert any("iterations" in line for line in st.summary(s))
    assert not st.is_done(S.Settings({"data_dir": str(built / "nowhere")}))


# ── screen ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_centrality_screen(built, tmp_path):
    from experiments.centrality.screen import CentralityScreen
    from textual.widgets import DataTable, RichLog, Select, TabbedContent
    from tui.widgets import TitlePicker
    app = make_app(tmp_path, [], data_dir=built)
    async with app.run_test(size=(150, 50)) as pilot:
        screen = CentralityScreen(built)
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.ex is not None)

        # leaderboard: filled for PageRank, then switched to in-degree
        board = screen.query_one("#ce-table", DataTable)
        assert await wait_for(pilot, lambda: board.row_count == len(ARTICLES))   # 100 asked, 8 exist
        screen.query_one("#ce-metric", Select).value = "indegree"
        assert await wait_for(pilot, lambda: board.row_count and
                              str(board.get_cell_at((0, 1))) == "Kevin Bacon")

        # Enter on a row opens that article in Look up
        board.focus()
        await pilot.press("enter")
        detail = screen.query_one("#ce-detail", RichLog)
        assert await wait_for(pilot, lambda: "Kevin Bacon" in log_text(detail))
        assert screen.query_one(TabbedContent).active == "ce-lookup"
        assert "PageRank" in log_text(detail) and "incoming links" in log_text(detail)

        # typing a title in the picker looks it up too (and un-picking clears the panel)
        picker = screen.query_one("#ce-picker", TitlePicker)
        picker.input.focus()
        await pilot.press("ctrl+a", *"mitochondria", "enter")
        assert await wait_for(pilot, lambda: "Mitochondrion" in log_text(detail))
        await pilot.press("end", "backspace")
        assert await wait_for(pilot, lambda: not log_text(detail).strip())

        # surprises tab fills, and the direction can be switched
        surp = screen.query_one("#ce-surp-table", DataTable)
        assert await wait_for(pilot, lambda: surp.row_count > 0)
        first = str(surp.get_cell_at((0, 0)))
        screen.query_one("#ce-surp-kind", Select).value = "down"
        assert await wait_for(pilot, lambda: str(surp.get_cell_at((0, 0))) != first)


@pytest.mark.asyncio
async def test_centrality_screen_without_data(tmp_path):
    from experiments.centrality.screen import CentralityScreen
    from textual.widgets import Label
    app = make_app(tmp_path, [], data_dir=tmp_path)
    async with app.run_test(size=(150, 50)) as pilot:
        screen = CentralityScreen(tmp_path)
        app.push_screen(screen)
        def failed():
            try:
                return "Can't load data" in str(screen.query_one("#ce-status", Label).render())
            except Exception:   # not composed yet
                return False
        assert await wait_for(pilot, failed)
        await pilot.pause(0.3)   # let the header finish mounting before the app shuts down
