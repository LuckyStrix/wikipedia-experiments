"""Semantic search screen: describe what you're looking for, get the articles closest in meaning."""
from __future__ import annotations

import time
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from wikiexp.semantic import Hit, SemanticSearch

K = 15
EXAMPLES = ("that battle where the weather decided everything · first woman to win a Nobel prize · "
            "why the dinosaurs died out · a bridge that collapsed in a windstorm")


class SemanticSearchScreen(Screen):
    TITLE = "Semantic search"
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("ctrl+o", "full_article", "Read article", priority=True),
        Binding("ctrl+l", "focus_query", "New query", priority=True),
    ]
    DEFAULT_CSS = """
    SemanticSearchScreen #ss-body { padding: 1 2; height: 1fr; }
    SemanticSearchScreen .ss-intro { color: $text-muted; width: 100%; height: auto; margin-bottom: 1; }
    SemanticSearchScreen #ss-status { color: $text-muted; margin: 1 0 0 0; height: auto; }
    SemanticSearchScreen #ss-main { height: 1fr; margin-top: 1; }
    SemanticSearchScreen #ss-results { width: 48%; border: round $panel-lighten-2; }
    SemanticSearchScreen #ss-results:focus { border: round $primary; }
    SemanticSearchScreen #ss-detail-wrap { width: 1fr; border: round $panel-lighten-2; padding: 0 1;
                                           margin-left: 1; }
    SemanticSearchScreen #ss-detail { height: auto; width: 100%; }
    """

    def __init__(self, data_dir: Path, search: SemanticSearch | None = None):
        super().__init__()
        self.data_dir = data_dir
        self.search: SemanticSearch | None = search
        self.hits: list[Hit] = []
        self._generation = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="ss-body"):
            yield Label("Describe what you're looking for in your own words; the closest articles "
                        "by meaning come back, even if they share no words with your question.",
                        classes="ss-intro")
            yield Input(placeholder="e.g. that battle where the weather decided everything",
                        id="ss-query", compact=True, disabled=True)
            yield Label("Loading the search index and model…", id="ss-status")
            with Horizontal(id="ss-main"):
                yield OptionList(id="ss-results")
                with VerticalScroll(id="ss-detail-wrap"):
                    yield Static(Text("Examples: " + EXAMPLES, style="dim"), id="ss-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._load()

    @work(thread=True)
    def _load(self) -> None:
        t0 = time.time()
        try:
            ss = self.search or SemanticSearch(self.data_dir)
            ss.load()
            ss.search("warm up", 1)      # loads the model now, not on the first real query
        except Exception as e:
            self.app.call_from_thread(self._status, f"Can't load search data: {e}")
            return
        self.app.call_from_thread(self._loaded, ss, time.time() - t0)

    def _loaded(self, ss: SemanticSearch, seconds: float) -> None:
        self.search = ss
        q = self.query_one("#ss-query", Input)
        q.disabled = False
        q.focus()
        self._status(f"Ready: {len(ss):,} articles searchable ({ss.meta.get('model', '?')}, "
                     f"loaded in {seconds:.1f}s). Enter to search.")

    def _status(self, text: str) -> None:
        self.query_one("#ss-status", Label).update(text)

    def action_focus_query(self) -> None:
        q = self.query_one("#ss-query", Input)
        q.focus()
        q.select_all()

    # ── searching ──────────────────────────────────────────────────────────────
    @on(Input.Submitted, "#ss-query")
    def _submit(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query or self.search is None:
            return
        self._generation += 1
        self._status(f"Searching for “{query}”…")
        self._run(query, self._generation)

    @work(thread=True, exclusive=True, group="search")
    def _run(self, query: str, gen: int) -> None:
        t0 = time.time()
        try:
            hits = self.search.search(query, K)
        except Exception as e:
            self.app.call_from_thread(self._status, f"Search failed: {e}")
            return
        self.app.call_from_thread(self._show, query, hits, time.time() - t0, gen)

    def _show(self, query: str, hits: list[Hit], seconds: float, gen: int) -> None:
        if gen != self._generation:
            return
        self.hits = hits
        ol = self.query_one("#ss-results", OptionList)
        ol.clear_options()
        theme = self.app.current_theme
        for h in hits:
            ol.add_option(Option(Text.assemble((f"{h.score:.3f} ", theme.accent or "bold"),
                                               (h.title, "bold"), "\n",
                                               (h.snippet[:110] + ("…" if len(h.snippet) > 110 else ""), "dim"))))
        self._status(f"{len(hits)} results in {seconds * 1000:.0f} ms. Arrow keys to browse, "
                     f"ctrl+o reads the whole article, ctrl+l asks again.")
        if hits:
            ol.focus()
            ol.highlighted = 0
        else:
            self.query_one("#ss-detail", Static).update(Text("No results.", style="dim"))

    @on(OptionList.OptionHighlighted, "#ss-results")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index is not None and event.option_index < len(self.hits):
            self._detail(self.hits[event.option_index])

    def _detail(self, h: Hit, article: str | None = None) -> None:
        theme = self.app.current_theme
        t = Text.assemble((h.title, f"bold {theme.primary}"),
                          (f"   score {h.score:.3f} · page {h.page_id}\n\n", "dim"))
        t.append(article if article is not None else h.lead or "(no lead stored)")
        self.query_one("#ss-detail", Static).update(t)
        self.query_one("#ss-detail-wrap").scroll_home(animate=False)

    # ── whole article ──────────────────────────────────────────────────────────
    def action_full_article(self) -> None:
        ol = self.query_one("#ss-results", OptionList)
        if ol.highlighted is None or not self.hits:
            return
        self._fetch(self.hits[ol.highlighted])

    @work(thread=True, exclusive=True, group="article")
    def _fetch(self, h: Hit) -> None:
        try:
            from wikiexp import wikitext as W
            wt = W.WikiText(cache_dir=self.data_dir / "text")
            raw = wt.fetch(h.page_id)
            text = W.to_text(raw) if raw else None
        except Exception as e:
            self.app.call_from_thread(self._status, f"Can't read the article text: {e}")
            return
        if text:
            self.app.call_from_thread(self._detail, h, text[:20000])
