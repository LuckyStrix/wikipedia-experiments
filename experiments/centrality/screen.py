"""Centrality screen for the app: a leaderboard, a one-article lookup, and "surprises"."""
from __future__ import annotations

from pathlib import Path

from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, RichLog, Select, TabbedContent, TabPane

from tui.widgets import TitlePicker
from wikiexp.centrality import METRICS

from .explorer import SHORT, Explorer, format_score

COUNTS = [25, 50, 100, 250, 500, 1000]
POOLS = [10_000, 100_000, 1_000_000]
SURPRISE_ROWS = 50
SURPRISE_KINDS = [("Punching above their links (PageRank rank >> in-degree rank)", "up"),
                  ("Many links, little weight (in-degree rank >> PageRank rank)", "down")]


def num(n, bold=False) -> Text:
    return Text(f"{n:,}", justify="right", style="bold" if bold else "")


class CentralityScreen(Screen):
    TITLE = "Centrality: which articles matter most?"
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]
    DEFAULT_CSS = """
    CentralityScreen #ce-body { padding: 1 2; height: 1fr; }
    CentralityScreen .ce-intro { color: $text-muted; width: 100%; height: auto; margin-bottom: 1; }
    CentralityScreen .ce-controls { height: 3; margin-bottom: 1; }
    CentralityScreen .ce-controls Label { margin: 1 1 0 0; color: $secondary; text-style: bold; }
    CentralityScreen .ce-controls Select { width: 34; margin-right: 2; }
    CentralityScreen .ce-controls .ce-narrow { width: 16; }
    CentralityScreen #ce-surp-kind { width: 66; }
    CentralityScreen DataTable { height: 1fr; }
    CentralityScreen #ce-detail { height: 1fr; background: $surface; padding: 0 1; margin-top: 1; }
    CentralityScreen #ce-status { color: $text-muted; }
    CentralityScreen TabPane { padding: 1 0 0 0; }
    """

    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = data_dir
        self.ex: Explorer | None = None
        self._surprises: dict[int, tuple] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="ce-body"):
            yield Label("PageRank scores an article by how many important articles link to it. Compare "
                        "it with a plain count of incoming links, or with 'gateway' (reverse PageRank), "
                        "which favours articles that link out to many important ones.", classes="ce-intro")
            with TabbedContent(id="ce-tabs"):
                with TabPane("Leaderboard", id="ce-board"):
                    with Horizontal(classes="ce-controls"):
                        yield Label("Rank by")
                        yield Select([(METRICS[m], m) for m in METRICS], value="pagerank", allow_blank=False,
                                     id="ce-metric", compact=True)
                        yield Label("Show")
                        yield Select([(f"top {n:,}", n) for n in COUNTS], value=100, allow_blank=False,
                                     id="ce-count", classes="ce-narrow", compact=True)
                    yield DataTable(id="ce-table", cursor_type="row", zebra_stripes=True)
                with TabPane("Look up", id="ce-lookup"):
                    yield TitlePicker("Type an article title, e.g. Albert Einstein", self._search, self._resolve,
                                      id="ce-picker")
                    yield RichLog(id="ce-detail", wrap=True, markup=False, highlight=False)
                with TabPane("Surprises", id="ce-surp"):
                    with Horizontal(classes="ce-controls"):
                        yield Select(SURPRISE_KINDS, value="up", allow_blank=False, id="ce-surp-kind", compact=True)
                        yield Label("Among the")
                        yield Select([(f"top {n:,}", n) for n in POOLS], value=100_000, allow_blank=False,
                                     id="ce-pool", classes="ce-narrow", compact=True)
                    yield DataTable(id="ce-surp-table", cursor_type="row", zebra_stripes=True)
            yield Label("Loading centrality data…", id="ce-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#ce-picker", TitlePicker).disabled = True
        self._load()

    # ── loading ───────────────────────────────────────────────────────────────
    @work(thread=True)
    def _load(self) -> None:
        try:
            ex = Explorer(self.data_dir)
        except Exception as e:
            self.app.call_from_thread(self._status, f"Can't load data: {e}")
            return
        self.app.call_from_thread(self._loaded, ex)

    def _loaded(self, ex: Explorer) -> None:
        self.ex = ex
        built = ex.c.meta
        self._status(f"{len(ex.c):,} articles · PageRank damping {built.get('damping')}, "
                     f"{built.get('iterations')} iterations · dump {built.get('dump_date')}")
        if "reverse_pagerank" not in ex.metrics:
            self.query_one("#ce-metric", Select).set_options([(METRICS[m], m) for m in ex.metrics])
        self.query_one("#ce-picker", TitlePicker).disabled = False
        self._refresh_board()
        self._refresh_surprises()

    def _status(self, text: str) -> None:
        self.query_one("#ce-status", Label).update(text)

    # ── leaderboard ───────────────────────────────────────────────────────────
    @property
    def metric(self) -> str:
        return self.query_one("#ce-metric", Select).value

    @on(Select.Changed, "#ce-metric, #ce-count")
    def _board_choice_changed(self) -> None:
        self._refresh_board()

    def _refresh_board(self) -> None:
        if self.ex:
            self._board(self.metric, self.query_one("#ce-count", Select).value)

    @work(thread=True, exclusive=True, group="board")
    def _board(self, metric: str, n: int) -> None:
        rows = self.ex.leaderboard(metric, n)
        self.app.call_from_thread(self._show_board, metric, rows)

    def _show_board(self, metric: str, rows) -> None:
        if metric != self.metric:
            return   # a newer choice is on its way
        t = self.query_one("#ce-table", DataTable)
        t.clear(columns=True)
        others = [m for m in self.ex.metrics if m != metric]
        t.add_column(Text("#", justify="right"))
        t.add_column("Article", width=44)
        t.add_column(Text(SHORT[metric], justify="right"))
        t.add_column(Text("Links in", justify="right"))
        t.add_column(Text("Links out", justify="right"))
        for m in others:
            t.add_column(Text(f"{SHORT[m]} #", justify="right"))
        for i, r in enumerate(rows, 1):
            t.add_row(num(i, True), r.title, Text(format_score(metric, r.score), justify="right"),
                      num(r.in_links), num(r.out_links), *(num(r.ranks[m]) for m in others), key=str(r.idx))
        self._status(f"{METRICS[metric]}. Enter on a row opens it in Look up.")

    @on(DataTable.RowSelected)
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        self._open(int(event.row_key.value))

    def _open(self, idx: int) -> None:
        """Jump to Look up for an article (by graph idx)."""
        if self.ex:
            self._to_lookup(idx)

    @work(thread=True, group="open")
    def _to_lookup(self, idx: int) -> None:
        title = self.ex.titles_of([idx])[0]
        m = self.ex.titles.resolve(title)
        if m:
            self.app.call_from_thread(self._show_lookup_tab, m)

    def _show_lookup_tab(self, m) -> None:
        self.query_one(TabbedContent).active = "ce-lookup"
        self.query_one("#ce-picker", TitlePicker).set_match(m)

    # ── look up ───────────────────────────────────────────────────────────────
    def _search(self, q: str):
        return self.ex.titles.search(q, limit=8) if self.ex else []

    def _resolve(self, q: str):
        return self.ex.titles.resolve(q) if self.ex else None

    @on(TitlePicker.Picked)
    def _picked(self, event: TitlePicker.Picked) -> None:
        m = event.picker.match
        if m is None:
            self.query_one("#ce-detail", RichLog).clear()
        else:
            self._profile(m)

    @work(thread=True, exclusive=True, group="profile")
    def _profile(self, match) -> None:
        p = self.ex.profile(self.ex.idx_of(match))
        self.app.call_from_thread(self._show_profile, match, p)

    def _show_profile(self, match, p) -> None:
        log = self.query_one("#ce-detail", RichLog)
        theme = self.app.current_theme
        log.clear()
        log.write(Text(p.title, style=f"bold {theme.primary}"))
        if match.title != match.article:
            log.write(Text(f"(from the redirect {match.title})", style="dim"))
        log.write(Text(f"{p.in_links:,} incoming links · {p.out_links:,} outgoing links · "
                       f"{len(self.ex.c):,} articles in all", style="dim"))
        table = Table(box=None, pad_edge=False, header_style=f"bold {theme.secondary}")
        table.add_column("Metric")
        table.add_column("Rank", justify="right")
        table.add_column("Top %", justify="right")
        table.add_column("Score", justify="right")
        for m in self.ex.metrics:
            table.add_row(SHORT[m], f"{p.ranks[m]:,}", f"{100 - p.percentiles[m]:.3f}%",
                          format_score(m, p.scores[m]))
        log.write(Text(""))
        log.write(table)
        for label, rows in (("Most notable articles linking here", p.linked_from),
                            ("Most notable articles it links to", p.links_to)):
            if rows:
                log.write(Text(""))
                log.write(Text(label, style=f"bold {theme.secondary}"))
                for title, rank in rows:
                    log.write(Text.assemble("  ", title, (f"  PageRank #{rank:,}", "dim")))

    # ── surprises ─────────────────────────────────────────────────────────────
    @on(Select.Changed, "#ce-surp-kind, #ce-pool")
    def _surprise_choice_changed(self) -> None:
        self._refresh_surprises()

    def _refresh_surprises(self) -> None:
        if self.ex:
            self._status("Comparing PageRank rank with in-degree rank…")
            self._compute_surprises(self.query_one("#ce-pool", Select).value)

    @work(thread=True, exclusive=True, group="surprises")
    def _compute_surprises(self, pool: int) -> None:
        if pool not in self._surprises:
            self._surprises[pool] = self.ex.surprises(pool, SURPRISE_ROWS)
        self.app.call_from_thread(self._show_surprises, pool)

    def _show_surprises(self, pool: int) -> None:
        if pool != self.query_one("#ce-pool", Select).value:
            return
        kind = self.query_one("#ce-surp-kind", Select).value
        rows = self._surprises[pool][0 if kind == "up" else 1]
        t = self.query_one("#ce-surp-table", DataTable)
        t.clear(columns=True)
        t.add_column("Article", width=44)
        for h in ("PageRank #", "In-links #", "Links in", "Gap"):
            t.add_column(Text(h, justify="right"))
        for s in rows:
            t.add_row(s.title, num(s.pagerank_rank), num(s.indegree_rank), num(s.in_links),
                      Text(f"{s.factor:,.1f}x", justify="right", style="bold"), key=str(s.idx))
        self._status(f"Among the {pool:,} top articles by either measure. Enter on a row opens it in Look up.")
