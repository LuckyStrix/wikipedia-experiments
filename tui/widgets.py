"""Widgets: stage cards, the log pane, the settings form generated from pipeline.settings.REGISTRY,
and TitlePicker (an input that only accepts real article titles, with live suggestions)."""
from __future__ import annotations

import time
from itertools import groupby
from typing import Any, Callable

from rich.text import Text
from textual import events, on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.validation import Function
from textual.widgets import (
    Button, Checkbox, Input, Label, OptionList, ProgressBar, RichLog, Static, TabbedContent, TabPane,
)
from textual.widgets.option_list import Option

from pipeline import settings as S

# ── stage card ────────────────────────────────────────────────────────────────

ICONS = {"waiting": "○", "running": "◐", "done": "●", "failed": "✕"}
DEFAULT_STATUS = {"waiting": "Not run yet", "running": "Running…", "done": "Done", "failed": "Failed"}


class StageCard(Vertical):
    """One pipeline stage: state, progress, Run + View buttons."""

    class RunPressed(Message):
        def __init__(self, key: str):
            super().__init__()
            self.key = key

    class ViewPressed(Message):
        def __init__(self, key: str):
            super().__init__()
            self.key = key

    def __init__(self, key: str, number: int, name: str, description: str):
        super().__init__(classes="stage waiting", id=f"stage-{key}")
        self.key, self.stage_name = key, f"{number} {name}"
        self.description = description
        self.state = "waiting"
        self.status = DEFAULT_STATUS["waiting"]
        self.started = 0.0
        self.external = False     # running, but started outside this app
        self.border_title = f"{ICONS['waiting']} {self.stage_name}"
        self.tooltip = description

    def compose(self) -> ComposeResult:
        yield ProgressBar(total=100, show_eta=False, show_percentage=True)
        yield Label(self.status, classes="detail")
        with Horizontal(classes="stage-btns"):
            yield Button("Run", compact=True, classes="run-stage")
            yield Button("View", compact=True, classes="view-stage")

    def set_state(self, state: str, status: str = "", external: bool = False) -> None:
        was_external, self.external = self.external, external
        if state == self.state and external and was_external:
            self.status = status or self.status   # still running elsewhere: keep the bar as it is
            self.refresh_detail()
            return
        if state != self.state:
            self.remove_class(self.state, "-has-progress")
            self.add_class(state)
            self.state = state
            self.border_title = f"{ICONS[state]} {self.stage_name}"
        self.status = status or DEFAULT_STATUS[state]
        bar = self.query_one(ProgressBar)
        if state == "running":
            self.started = 0.0 if external else time.monotonic()
            bar.update(total=None, progress=0)     # indeterminate until progress arrives
        elif state == "done":
            bar.update(total=100, progress=100)
        else:
            bar.update(total=100, progress=0)
        self.query_one(".run-stage", Button).disabled = state == "running"
        self.refresh_detail()

    def set_progress(self, pct: int, text: str = "") -> None:
        if self.state != "running":
            return
        self.add_class("-has-progress")
        self.query_one(ProgressBar).update(total=100, progress=pct)
        if text:
            self.status = text
        self.refresh_detail()

    def refresh_detail(self) -> None:
        t = Text(self.status)
        if self.state == "running" and self.started:
            s = int(time.monotonic() - self.started)
            t.append(f"  {s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}", style="dim")
        self.query_one(".detail", Label).update(t)

    @on(Button.Pressed, ".run-stage")
    def _run(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.RunPressed(self.key))

    @on(Button.Pressed, ".view-stage")
    def _view(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.ViewPressed(self.key))


# ── log ───────────────────────────────────────────────────────────────────────

LOG_MAX_LINES = 5000
LOG_MIN_HEIGHT = 3


class LogHandle(Static):
    """1-row bar above the log; drag it to resize the log."""

    class Resized(Message):
        def __init__(self, height: int, final: bool):
            super().__init__()
            self.height, self.final = height, final

    def on_mount(self) -> None:
        self.update(Text.assemble(("─── Output log ", "bold"),
                                  ("  drag to resize · ctrl+↑/↓ · l maximise", "dim")))
        self._dragging = False

    def _height_at(self, screen_y: int) -> int:
        return self.app.query_one("#log-wrap").region.bottom - screen_y - 1

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._dragging = True
        self.capture_mouse()
        self.add_class("-dragging")

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging:
            self.post_message(self.Resized(self._height_at(event.screen_y), final=False))

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()
            self.remove_class("-dragging")
            self.post_message(self.Resized(self._height_at(event.screen_y), final=True))


class LogPane(RichLog):
    """Timestamped app messages, styled command headers, raw subprocess output."""

    def __init__(self, **kw):
        super().__init__(max_lines=LOG_MAX_LINES, wrap=True, markup=False, highlight=False,
                         auto_scroll=True, **kw)

    def add(self, text: str, kind: str = "output") -> None:
        theme = self.app.current_theme
        if kind == "header":
            self.write(Text(""))
            self.write(Text("── $ " + text, style=f"bold {theme.primary}"))
        elif kind == "info":
            t = Text(time.strftime("%H:%M:%S "), style="dim")
            bad = any(w in text.lower() for w in ("failed", "can't", "error", "[warn]"))
            t.append(text, style=theme.error if bad else (theme.secondary or ""))
            self.write(t)
        else:
            self.write(Text(text, style=theme.error if text.startswith(("Traceback", "FAILED")) else ""))


# ── settings form ─────────────────────────────────────────────────────────────

class SettingsForm(Vertical):
    """Tabs of fields generated from REGISTRY, plus a read-only Machines tab."""

    class SettingChanged(Message):
        def __init__(self, key: str, value: Any):
            super().__init__()
            self.key, self.value = key, value

    def __init__(self, settings: S.Settings, machines: dict, **kw):
        super().__init__(**kw)
        self.settings, self.machines = settings, machines

    def compose(self) -> ComposeResult:
        with TabbedContent(id="tabs"):
            for tab, in_tab in groupby(S.REGISTRY, key=lambda st: st.tab):
                with TabPane(tab):
                    with VerticalScroll():
                        for section, fields in groupby(in_tab, key=lambda st: st.section):
                            yield Label(section, classes="section")
                            for st in fields:
                                yield from self._field(st)
            with TabPane("Machines"):
                with VerticalScroll():
                    yield Label("Other machines (for tools/sync.py)", classes="section")
                    if self.machines:
                        for name, m in self.machines.items():
                            yield Label(Text.assemble((f"{name:<10}", "bold"),
                                                      f"{m.get('host', '?')}  {m.get('repo', '')}  "
                                                      f"({m.get('os', 'posix')})"), classes="machine")
                    else:
                        yield Label("None configured.", classes="help")
                    yield Label("Add or edit machines in config.toml ([machines.<name>] with host, repo, "
                                "os); see config.example.toml. Saving settings here keeps them.",
                                classes="help")

    def _field(self, st: S.Setting):
        value = self.settings[st.key]
        if st.type == S.BOOL:
            yield Checkbox(st.label, value=bool(value), id=f"set-{st.key}", classes="chk", compact=True)
        else:
            with Horizontal(classes="row"):
                yield Label(st.label, classes="lbl")
                validators = [Function(lambda v, st=st: S.valid(st, v), "Invalid")] if st.pattern else []
                yield Input(str(value), id=f"set-{st.key}", compact=True, validators=validators,
                            placeholder=st.hint, classes="path" if st.type == S.PATH else "text")
                if st.hint and st.type != S.PATH:
                    yield Label(st.hint, classes="hint")
        if st.help:
            yield Label(st.help, classes="help")

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        key = (event.input.id or "").removeprefix("set-")
        if key in S.BY_KEY:
            event.stop()
            self.settings[key] = event.value
            self.post_message(self.SettingChanged(key, event.value))

    @on(Checkbox.Changed)
    def _checkbox_changed(self, event: Checkbox.Changed) -> None:
        key = (event.checkbox.id or "").removeprefix("set-")
        if key in S.BY_KEY:
            event.stop()
            self.settings[key] = event.value
            self.post_message(self.SettingChanged(key, event.value))


# ── title picker ──────────────────────────────────────────────────────────────

class TitlePicker(Vertical):
    """Type part of a title; pick from live suggestions. ``match`` is set only when the text is a
    real article (picked from the list, or typed exactly), so callers can refuse anything else.

    ``search`` is called from a worker thread: (query) -> list of wikiexp.titles.Match.
    ``resolve`` likewise: (query) -> Match | None.
    """

    class Picked(Message):
        def __init__(self, picker: "TitlePicker"):
            super().__init__()
            self.picker = picker

        @property
        def control(self):
            return self.picker

    DEBOUNCE = 0.12

    def __init__(self, placeholder: str, search: Callable, resolve: Callable, **kw):
        super().__init__(classes="title-picker", **kw)
        self.placeholder, self._search, self._resolve = placeholder, search, resolve
        self.match = None
        self._timer = None
        self._set_value = None   # value we put in the Input ourselves (its Changed arrives later)
        self._results, self._shown_query = [], None
        self._generation = 0

    def compose(self) -> ComposeResult:
        yield Input(placeholder=self.placeholder, compact=True)
        yield OptionList(classes="suggestions")
        yield Label("", classes="picker-status")

    @property
    def input(self) -> Input:
        return self.query_one(Input)

    def set_match(self, m) -> None:
        self.match = m
        self._set_value = m.article if m else ""
        self.input.value = self._set_value
        self.input.cursor_position = len(self._set_value)
        self._hide()
        self._status(f"✓ {m.article}  ·  {m.links:,} incoming link{'s' * (m.links != 1)}" if m else "")
        self.input.remove_class("-bad")
        self.post_message(self.Picked(self))

    def _status(self, text: str, bad: bool = False) -> None:
        lbl = self.query_one(".picker-status", Label)
        lbl.update(text)
        lbl.set_class(bad, "-bad")

    def _hide(self) -> None:
        self.query_one(OptionList).display = False

    @on(Input.Changed)
    def _changed(self, event: Input.Changed) -> None:
        event.stop()
        if event.value == self._set_value:
            self._set_value = None
            return
        self._set_value = None
        if self.match is not None:
            self.match = None
            self.post_message(self.Picked(self))
        if self._timer:
            self._timer.stop()
        q = event.value
        if not q.strip():
            self._hide()
            self._status("")
            return
        self._generation += 1
        gen = self._generation
        self._timer = self.set_timer(self.DEBOUNCE, lambda: self._lookup(q, gen))

    @work(thread=True, exclusive=True, group="title-search")
    def _lookup(self, q: str, gen: int) -> None:
        results = self._search(q)
        self.app.call_from_thread(self._show, q, results, gen)

    def _show(self, q: str, results: list, gen: int) -> None:
        if gen != self._generation or q != self.input.value:
            return   # the user kept typing; a newer lookup is on its way
        ol = self.query_one(OptionList)
        ol.clear_options()
        self._results, self._shown_query = results, q
        if not results:
            self._hide()
            self._status("No article matches that. Keep typing, or check the spelling.", bad=True)
            self.input.add_class("-bad")
            return
        ol.add_options([Option(Text.assemble(r.label, (f"  {r.links:,} link{'s' * (r.links != 1)}", "dim"),
                                             (f"  {r.how}" if r.how == "similar" else "", "italic dim")))
                        for r in results])
        ol.highlighted = 0
        ol.display = True
        exact = results[0].how == "exact"
        self._status("Enter or click to choose" + ("" if exact else " · not an exact title yet"))
        self.input.set_class(not exact, "-bad")

    @on(Input.Submitted)
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        ol = self.query_one(OptionList)
        if ol.display and ol.highlighted is not None and self._shown_query == event.value:
            self.set_match(self._results[ol.highlighted])
        elif event.value.strip():
            # suggestions are missing or out of date: use the exact title, else the best suggestion
            self._resolve_now(event.value)

    @work(thread=True, exclusive=True, group="title-resolve")
    def _resolve_now(self, q: str) -> None:
        m = self._resolve(q) or next(iter(self._search(q)), None)
        self.app.call_from_thread(self._resolved, q, m)

    def _resolved(self, q: str, m) -> None:
        if q != self.input.value:
            return
        if m:
            self.set_match(m)
        else:
            self._status("No article matches that. Keep typing, or check the spelling.", bad=True)

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.set_match(self._results[event.option_index])
        self.input.focus()

    def on_key(self, event: events.Key) -> None:
        ol = self.query_one(OptionList)
        if not ol.display or not self.input.has_focus:
            return
        if event.key in ("down", "up"):
            event.stop()
            event.prevent_default()
            n = ol.option_count
            cur = ol.highlighted or 0
            ol.highlighted = (cur + (1 if event.key == "down" else -1)) % n
        elif event.key == "escape":
            event.stop()
            self._hide()
