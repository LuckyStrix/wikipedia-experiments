"""Six Degrees screen for the app: two title pickers (only real articles are accepted), then the path."""
from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, RichLog

from tui.widgets import TitlePicker

from .finder import PathFinder


class SixDegreesScreen(Screen):
    TITLE = "Six Degrees of Wikipedia"
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("ctrl+f", "find", "Find path", priority=True),
        Binding("ctrl+r", "random", "Random pair", priority=True),
        Binding("ctrl+w", "swap", "Swap", priority=True),
    ]

    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = data_dir
        self.finder: PathFinder | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="sd-body"):
            yield Label("Find the shortest chain of links from one article to another. Start typing a "
                        "title and pick from the suggestions; only real articles are accepted.",
                        classes="sd-intro")
            with Horizontal(classes="sd-row"):
                yield Label("From", classes="sd-lbl")
                yield TitlePicker("e.g. Kevin Bacon", self._search, self._resolve, id="sd-from")
            with Horizontal(classes="sd-row"):
                yield Label("To", classes="sd-lbl")
                yield TitlePicker("e.g. Mitochondrion", self._search, self._resolve, id="sd-to")
            with Horizontal(id="sd-btns"):
                yield Button("Find path", id="sd-find", variant="primary", compact=True, disabled=True)
                yield Button("⇅ Swap", id="sd-swap", compact=True)
                yield Button("Random pair", id="sd-random", compact=True)
            yield Label("Loading title index and link graph…", id="sd-status")
            yield RichLog(id="sd-results", wrap=True, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#sd-from", TitlePicker).disabled = True
        self.query_one("#sd-to", TitlePicker).disabled = True
        self._load()

    @work(thread=True)
    def _load(self) -> None:
        try:
            finder = PathFinder(self.data_dir)
        except Exception as e:
            self.app.call_from_thread(self._status, f"Can't load data: {e}")
            return
        self.app.call_from_thread(self._loaded, finder)

    def _loaded(self, finder: PathFinder) -> None:
        self.finder = finder
        for p in self.query(TitlePicker):
            p.disabled = False
        self.query_one("#sd-from", TitlePicker).input.focus()
        self._status(f"Ready: {len(finder.graph):,} articles.")

    def _status(self, text: str) -> None:
        self.query_one("#sd-status", Label).update(text)

    # called from TitlePicker worker threads
    def _search(self, q: str):
        return self.finder.titles.search(q, limit=8) if self.finder else []

    def _resolve(self, q: str):
        return self.finder.titles.resolve(q) if self.finder else None

    @property
    def pickers(self) -> tuple[TitlePicker, TitlePicker]:
        return self.query_one("#sd-from", TitlePicker), self.query_one("#sd-to", TitlePicker)

    @on(TitlePicker.Picked)
    def _picked(self, event: TitlePicker.Picked) -> None:
        a, b = self.pickers
        self.query_one("#sd-find", Button).disabled = not (a.match and b.match)
        if event.picker is a and a.match and not b.match:
            b.input.focus()
        elif a.match and b.match:
            self.query_one("#sd-find", Button).focus()

    @on(Button.Pressed, "#sd-find")
    def action_find(self) -> None:
        a, b = self.pickers
        if not (self.finder and a.match and b.match):
            self._status("Pick both articles from the suggestions first.")
            return
        self._status(f"Searching {a.match.article} → {b.match.article}…")
        self._find(a.match, b.match)

    @work(thread=True, exclusive=True, group="find")
    def _find(self, start, goal) -> None:
        result, titles = self.finder.find(start, goal)
        self.app.call_from_thread(self._show, start, goal, result, titles)

    def _show(self, start, goal, result, titles) -> None:
        log = self.query_one("#sd-results", RichLog)
        theme = self.app.current_theme
        log.write(Text(""))
        if result.path is None:
            log.write(Text(f"{start.article} → {goal.article}: no path", style=f"bold {theme.error}"))
            self._status(f"No path (searched {result.visited:,} articles in {result.seconds:.2f}s).")
            return
        d = result.degrees
        log.write(Text(f"{start.article} → {goal.article}: {d} click{'s' * (d != 1)}",
                       style=f"bold {theme.primary}"))
        for i, t in enumerate(titles):
            log.write(Text.assemble(("   " if i else "", ""), ("→ " if i else "  ", "dim"), t))
        self._status(f"Searched {result.visited:,} article{'s' * (result.visited != 1)} in {result.seconds:.2f}s.")

    @on(Button.Pressed, "#sd-swap")
    def action_swap(self) -> None:
        a, b = self.pickers
        ma, mb = a.match, b.match
        a.set_match(mb)
        b.set_match(ma)

    @on(Button.Pressed, "#sd-random")
    def action_random(self) -> None:
        if self.finder:
            self._random()

    @work(thread=True, group="random")
    def _random(self) -> None:
        t = self.finder.titles
        a = t.random_article()
        b = t.random_article()
        for _ in range(20):   # avoid picking the same article twice
            if b.page_id != a.page_id:
                break
            b = t.random_article()
        pair = a, b
        self.app.call_from_thread(self._set_pair, *pair)

    def _set_pair(self, a, b) -> None:
        pa, pb = self.pickers
        pa.set_match(a)
        pb.set_match(b)
        self.action_find()
