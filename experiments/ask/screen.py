"""Ask Wikipedia screen: a question box, a streaming markdown answer, and the numbered sources it cites."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from wikiexp import wikitext as W
from wikiexp.llm import BACKENDS, LLMError, have_claude_credentials
from wikiexp.rag import RAG, Answer, Conversation, Source

from . import session

CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"]
FLUSH_SECONDS = 0.12          # how often streamed text is pushed to the screen
EXAMPLES = ("Who designed the Eiffel Tower? · Why did the Tacoma Narrows Bridge collapse? · "
            "What is the capital of Burkina Faso? · How do plants turn sunlight into energy?")


class AskScreen(Screen):
    TITLE = "Ask Wikipedia"
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("ctrl+n", "new_conversation", "New conversation", priority=True),
        Binding("ctrl+b", "switch_backend", "Backend", priority=True),
        Binding("ctrl+e", "switch_model", "Model", priority=True),
        Binding("ctrl+o", "full_article", "Read article", priority=True),
        Binding("ctrl+x", "stop", "Stop", priority=True),
        Binding("ctrl+l", "focus_query", "Ask", priority=True),
    ]
    DEFAULT_CSS = """
    AskScreen #ask-body { padding: 1 2; height: 1fr; }
    AskScreen #ask-backend { color: $secondary; height: auto; margin-bottom: 1; }
    AskScreen #ask-status { color: $text-muted; height: auto; margin-top: 1; }
    AskScreen #ask-main { height: 1fr; margin-top: 1; }
    AskScreen #ask-convo { width: 3fr; border: round $panel-lighten-2; padding: 0 1; }
    AskScreen #ask-right { width: 2fr; margin-left: 1; }
    AskScreen #ask-sources { height: 45%; border: round $panel-lighten-2; }
    AskScreen #ask-sources:focus { border: round $primary; }
    AskScreen #ask-detail-wrap { height: 1fr; border: round $panel-lighten-2; padding: 0 1; margin-top: 1; }
    AskScreen #ask-detail { height: auto; width: 100%; }
    AskScreen #ask-answer { margin: 0; padding: 0; }
    """

    def __init__(self, data_dir: Path, rag: RAG | None = None, settings=None, backend=None):
        super().__init__()
        self.data_dir = data_dir
        self.rag: RAG | None = rag
        self._settings = settings
        self._backend = backend               # injected (tests); else built from settings
        self.conv = Conversation()
        self.sources: list[Source] = []
        self.last: Answer | None = None
        self._transcript = ""                  # finished turns as markdown
        self._busy = False
        self._cancel = threading.Event()
        self._gen = 0
        self._models: dict[str, list[str]] = {}

    @property
    def settings(self):
        if self._settings is None:
            self._settings = getattr(self.app, "settings", None) or session.load_settings()
        return self._settings

    # ── layout ─────────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="ask-body"):
            yield Label("", id="ask-backend")
            yield Input(placeholder="Ask a question about anything in Wikipedia", id="ask-query",
                        compact=True, disabled=True)
            yield Label("Loading the search indexes and models…", id="ask-status")
            with Horizontal(id="ask-main"):
                with VerticalScroll(id="ask-convo"):
                    yield Markdown(f"*Examples: {EXAMPLES}*", id="ask-answer")
                with Vertical(id="ask-right"):
                    yield OptionList(id="ask-sources")
                    with VerticalScroll(id="ask-detail-wrap"):
                        yield Static(Text("Sources appear here. Enter shows the whole passage, ctrl+o the "
                                          "whole article.", style="dim"), id="ask-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._show_backend()
        self.query_one("#ask-sources", OptionList).border_title = "Sources"
        self._load()

    # ── loading ────────────────────────────────────────────────────────────────
    @work(thread=True)
    def _load(self) -> None:
        t0 = time.time()
        try:
            if self.rag is None:
                backend = self._backend or session.backend_from(self.settings)
                self.rag = session.build_rag(self.data_dir, backend)
            elif self._backend is not None:
                self.rag.backend = self._backend
            if not self.rag.hybrid.modes():
                raise FileNotFoundError("neither the keyword index nor the embeddings exist; run the 'keyword' stage")
            self.rag.warm_up()
        except Exception as e:
            self.app.call_from_thread(self._status, f"Can't load Ask Wikipedia: {e}")
            return
        self.app.call_from_thread(self._loaded, time.time() - t0)

    def _loaded(self, seconds: float) -> None:
        q = self.query_one("#ask-query", Input)
        q.disabled = False
        q.focus()
        modes = "+".join(self.rag.hybrid.modes()) or "no retrievers"
        self._show_backend()
        self._status(f"Ready (retrieval: {modes}; loaded in {seconds:.1f}s). Enter to ask. "
                     f"ctrl+n new conversation, ctrl+b backend, ctrl+e model.")

    def _status(self, text: str) -> None:
        self.query_one("#ask-status", Label).update(text)

    def _show_backend(self) -> None:
        b = self.rag.backend if self.rag else None
        if b is None:
            name = self._backend.name if self._backend else str(self.settings["ask_backend"])
            model = self._backend.model if self._backend else ""
        else:
            name, model = b.name, b.model
        extra = f" at {b.url}" if getattr(b, "url", "") else ""
        self.query_one("#ask-backend", Label).update(
            f"Model: {name} · {model}{extra}   (ctrl+b switches backend, ctrl+e model)")

    # ── switching backend / model ──────────────────────────────────────────────
    def action_switch_backend(self) -> None:
        if self._busy or self.rag is None:
            return
        cur = self.rag.backend.name
        name = BACKENDS[(BACKENDS.index(cur) + 1) % len(BACKENDS)] if cur in BACKENDS else BACKENDS[0]
        try:
            backend = session.backend_from(self.settings, name)
        except LLMError as e:
            self._status(str(e))
            return
        self._set_backend(backend)
        if name == "claude" and not have_claude_credentials():
            self._status("Switched to claude, but ANTHROPIC_API_KEY isn't set in this environment; "
                         "set it and restart the app before asking.")

    def _set_backend(self, backend) -> None:
        self.rag.backend = backend
        self.rag.budget_tokens = session.BUDGET.get(backend.name, self.rag.budget_tokens)
        self.rag.max_answer_tokens = session.ANSWER_TOKENS.get(backend.name, self.rag.max_answer_tokens)
        self._show_backend()

    def action_switch_model(self) -> None:
        if self._busy or self.rag is None:
            return
        b = self.rag.backend
        if b.name == "claude":
            self._cycle_model(CLAUDE_MODELS)
        elif b.name in self._models:
            self._cycle_model(self._models[b.name])
        else:
            self._status("Asking ollama which models it has…")
            self._fetch_models(b)

    @work(thread=True, exclusive=True, group="models")
    def _fetch_models(self, backend) -> None:
        try:
            models = backend.models()
        except LLMError as e:
            self.app.call_from_thread(self._status, str(e))
            return
        self.app.call_from_thread(self._got_models, backend.name, models)

    def _got_models(self, name: str, models: list[str]) -> None:
        self._models[name] = models
        if models:
            self._cycle_model(models)
        else:
            self._status(f"ollama has no models yet. Pull one with: ollama pull {self.rag.backend.model}")

    def _cycle_model(self, models: list[str]) -> None:
        b = self.rag.backend
        options = models if b.model in models else [b.model] + models
        b.model = options[(options.index(b.model) + 1) % len(options)]
        self._show_backend()
        self._status(f"Using {b.name} model {b.model}.")

    # ── conversation ───────────────────────────────────────────────────────────
    def action_new_conversation(self) -> None:
        if self._busy:
            return
        self.conv.reset()
        self._transcript = ""
        self.sources = []
        self.last = None
        self.query_one("#ask-answer", Markdown).update(f"*New conversation. Examples: {EXAMPLES}*")
        self.query_one("#ask-sources", OptionList).clear_options()
        self.query_one("#ask-detail", Static).update(Text("Sources appear here.", style="dim"))
        self._status("New conversation. Ask anything.")
        self.action_focus_query()

    def action_focus_query(self) -> None:
        q = self.query_one("#ask-query", Input)
        q.focus()
        q.select_all()

    def action_stop(self) -> None:
        if self._busy:
            self._cancel.set()
            self._status("Stopping…")

    @on(Input.Submitted, "#ask-query")
    def _submit(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question or self.rag is None or self._busy:
            return
        self._busy = True
        self._cancel.clear()
        self._gen += 1
        event.input.value = ""
        self._show_convo(f"**You:** {question}\n\n*Searching Wikipedia…*")
        self.query_one("#ask-sources", OptionList).clear_options()
        self._run(question, self._gen)

    def _show_convo(self, current: str) -> None:
        md = self.query_one("#ask-answer", Markdown)
        md.update(self._transcript + current)
        self.query_one("#ask-convo", VerticalScroll).scroll_end(animate=False)

    @work(thread=True, exclusive=True, group="ask")
    def _run(self, question: str, gen: int) -> None:
        call = self.app.call_from_thread
        parts: list[str] = []
        finished = None
        last_flush = 0.0
        t0 = time.time()
        try:
            for ev in self.rag.ask(question, self.conv):
                if self._cancel.is_set():
                    break
                if ev.kind == "status":
                    call(self._status, str(ev.data))
                elif ev.kind == "sources":
                    call(self._show_sources, ev.data)
                elif ev.kind == "token":
                    parts.append(str(ev.data))
                    if time.time() - last_flush > FLUSH_SECONDS:
                        last_flush = time.time()
                        call(self._show_convo, f"**You:** {question}\n\n{''.join(parts)}")
                elif ev.kind == "done":
                    finished = ev.data       # keep iterating: the generator records the turn after "done"
            if finished is not None:
                call(self._finish, question, finished, gen)
            else:
                call(self._stopped, question, "".join(parts))
        except Exception as e:           # a bug or an unexpected failure must not leave the screen stuck
            call(self._failed, question, f"{e.__class__.__name__}: {e}", "".join(parts))

    def _finish(self, question: str, answer: Answer, gen: int) -> None:
        self.last = answer
        body = answer.text or "*(no answer)*"
        notes = [f"⚠ {answer.error}"] if answer.error else []
        notes += [f"note: {n}" for n in answer.notes]
        tail = "".join(f"\n\n> {n}" for n in notes)
        self._transcript += f"**You:** {question}\n\n{body}{tail}\n\n---\n\n"
        self._show_convo("")
        searched = f" Searched for: “{answer.query}”." if answer.query != answer.question else ""
        self._status(f"{session.describe(answer)}.{searched}  Cited: "
                     f"{', '.join(f'[{n}]' for n in answer.cited) or 'none'}.")
        self._idle()

    def _stopped(self, question: str, partial: str) -> None:
        self._transcript += f"**You:** {question}\n\n{partial}\n\n> stopped\n\n---\n\n"
        self._show_convo("")
        self._status("Stopped.")
        self._idle()

    def _failed(self, question: str, error: str, partial: str) -> None:
        self._transcript += f"**You:** {question}\n\n{partial}\n\n> ⚠ {error}\n\n---\n\n"
        self._show_convo("")
        self._status(error)
        self._idle()

    def _idle(self) -> None:
        self._busy = False
        self.action_focus_query()

    # ── sources ────────────────────────────────────────────────────────────────
    def _show_sources(self, sources: list[Source]) -> None:
        self.sources = list(sources)
        ol = self.query_one("#ask-sources", OptionList)
        ol.clear_options()
        theme = self.app.current_theme
        for s in sources:
            ol.add_option(Option(Text.assemble((f"[{s.n}] ", theme.accent or "bold"), (s.label, "bold"), "\n",
                                               (s.passage.text[:90].replace("\n", " ") + "…", "dim"))))
        if sources:
            ol.highlighted = 0

    def _source_at(self, index: int | None) -> Source | None:
        return self.sources[index] if index is not None and 0 <= index < len(self.sources) else None

    def _detail(self, s: Source, body: str, hint: str = "") -> None:
        theme = self.app.current_theme
        t = Text.assemble((f"[{s.n}] {s.label}", f"bold {theme.primary}"),
                          (f"   score {s.passage.score:.2f} · page {s.page_id}{hint}\n\n", "dim"))
        t.append(body)
        self.query_one("#ask-detail", Static).update(t)
        self.query_one("#ask-detail-wrap").scroll_home(animate=False)

    @on(OptionList.OptionHighlighted, "#ask-sources")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        s = self._source_at(event.option_index)
        if s:
            text = s.passage.text
            self._detail(s, text[:400] + ("…" if len(text) > 400 else ""),
                         "\nEnter: whole passage · ctrl+o: whole article" if len(text) > 400 else
                         "\nctrl+o: whole article")

    @on(OptionList.OptionSelected, "#ask-sources")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        s = self._source_at(event.option_index)
        if s:
            self._detail(s, s.passage.text, "\nctrl+o: whole article")

    def action_full_article(self) -> None:
        ol = self.query_one("#ask-sources", OptionList)
        s = self._source_at(ol.highlighted)
        if s and self.rag is not None:
            self._status(f"Reading “{s.title}”…")
            self._fetch_article(s)

    @work(thread=True, exclusive=True, group="article")
    def _fetch_article(self, s: Source) -> None:
        try:
            raw = self.rag.wikitext.fetch(s.page_id)
            text = W.to_text(raw)[:30000] if raw else ""
        except Exception as e:
            self.app.call_from_thread(self._status, f"Can't read the article: {e}")
            return
        self.app.call_from_thread(self._article, s, text)

    def _article(self, s: Source, text: str) -> None:
        self._detail(s, text or "(article text not found)", "   · whole article")
        self._status(f"Showing the whole article “{s.title}”.")
