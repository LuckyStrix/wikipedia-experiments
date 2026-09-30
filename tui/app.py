"""Textual app: layout, bindings, themes, and the glue between the settings form, the stage cards,
the experiments list, the log and the pipeline runner."""
from __future__ import annotations

import importlib
import queue

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widgets import Button, Footer, Header, Label

from experiments import EXPERIMENTS
from pipeline import settings as S
from pipeline import stages as ST
from pipeline.runner import PipelineRunner

from .screens import ConfirmScreen, ViewScreen
from .uistate import UI_STATE_PATH, UiState
from .widgets import LOG_MIN_HEIGHT, LogHandle, LogPane, SettingsForm, StageCard

# ── Windows Terminal colour schemes (same as the 3D modeling app) ─────────────
CAMPBELL = Theme(
    name="campbell", primary="#3B78FF", secondary="#61D6D6", accent="#B4009E",
    foreground="#CCCCCC", background="#0C0C0C", surface="#141414", panel="#1F1F1F",
    success="#16C60C", warning="#F9F1A5", error="#E74856", dark=True,
)
ONE_HALF_DARK = Theme(
    name="one-half-dark", primary="#61AFEF", secondary="#56B6C2", accent="#C678DD",
    foreground="#DCDFE4", background="#282C34", surface="#2F333D", panel="#3A3F4B",
    success="#98C379", warning="#E5C07B", error="#E06C75", dark=True,
)
THEMES = ["campbell", "one-half-dark", "tokyo-night", "textual-dark", "textual-light"]

EVENT_POLL = 0.05


def _short(p) -> str:
    """Path with the home folder shown as ~."""
    from pathlib import Path
    try:
        return "~/" + str(Path(p).resolve().relative_to(Path.home()))
    except ValueError:
        return str(p)


class WikiApp(App):
    TITLE = "Wikipedia Experiments"
    SUB_TITLE = "offline English Wikipedia"
    CSS_PATH = "app.tcss"

    # Bare-letter keys are ignored while typing in the settings form, so a stray "r" in a path
    # field can't start a run. ctrl+r always works.
    FORM_GUARDED = {"key_run_all", "key_stop", "key_quit", "toggle_log", "cycle_theme"}

    BINDINGS = [
        Binding("r", "key_run_all", "Run all"),
        Binding("ctrl+r", "run_all", "Run all", show=False),
        Binding("s", "key_stop", "Stop"),
        Binding("ctrl+s", "save_settings", "Save settings"),
        Binding("l", "toggle_log", "Log ↕"),
        Binding("ctrl+up", "log_resize(2)", "Log +", show=False),
        Binding("ctrl+down", "log_resize(-2)", "Log −", show=False),
        Binding("t", "cycle_theme", "Theme"),
        Binding("q", "key_quit", "Quit"),
    ]

    def __init__(self, settings: S.Settings | None = None, config_path=S.CONFIG_PATH,
                 ui_state_path=UI_STATE_PATH, runner_factory=PipelineRunner, stages=None):
        super().__init__()
        self.config_path = config_path
        self.ui_state_path = ui_state_path
        self.settings = settings if settings is not None else S.Settings.load(config_path)
        self.machines = S.read_config(config_path).get("machines", {})
        self.stages = stages if stages is not None else ST.STAGES
        self.ui = UiState.load(ui_state_path)
        self.register_theme(CAMPBELL)
        self.register_theme(ONE_HALF_DARK)
        # runner callbacks arrive on its worker thread; queue them and apply them in order
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self.runner = runner_factory(
            self.settings, stages=self.stages,
            on_log=lambda text, kind: self._events.put(("log", text, kind)),
            on_state=lambda key, state, status: self._events.put(("state", key, state, status)),
            on_progress=lambda key, pct, text: self._events.put(("progress", key, pct, text)),
        )

    # ── key guards ────────────────────────────────────────────────────────────
    def check_action(self, action: str, parameters) -> bool | None:
        if action in self.FORM_GUARDED and self._typing():
            return None
        return True

    def _typing(self) -> bool:
        f = self.focused
        return f is not None and any(a.id in ("form",) or "title-picker" in a.classes
                                     for a in f.ancestors_with_self)

    async def action_key_run_all(self) -> None:
        self.action_run_all()

    async def action_key_stop(self) -> None:
        self.action_stop()

    async def action_key_quit(self) -> None:
        await self.action_request_quit()

    # ── layout ────────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with VerticalScroll(id="side"):
                yield Label("DATA PIPELINE", classes="side-title")
                yield Label("", id="data-path")
                self.cards = {st.key: StageCard(st.key, i, st.name, st.description)
                              for i, st in enumerate(self.stages, 1)}
                yield from self.cards.values()
                with Horizontal(id="actions"):
                    yield Button("▶ Run all", variant="primary", id="run-all", compact=True)
                    yield Button("■ Stop", variant="error", id="stop", compact=True, disabled=True)
                yield Label("EXPERIMENTS", classes="side-title")
                for ex in EXPERIMENTS:
                    yield Button(ex.name, id=f"ex-{ex.key}", classes="experiment", compact=True)
                    yield Label(ex.description, classes="ex-desc", id=f"ex-desc-{ex.key}")
            with Vertical(id="main"):
                yield SettingsForm(self.settings, self.machines, id="form")
                yield LogHandle(id="log-handle")
                with Vertical(id="log-wrap"):
                    yield LogPane(id="log")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = self.ui.theme if self.ui.theme in self.available_themes else "campbell"
        # keep references: these are queried from timers while other screens may be on top
        self.log_pane = self.query_one(LogPane)
        self.stop_btn = self.query_one("#stop", Button)
        self.run_all_btn = self.query_one("#run-all", Button)
        self.data_path_lbl = self.query_one("#data-path", Label)
        self.ex_widgets = {ex.key: (self.query_one(f"#ex-{ex.key}", Button),
                                    self.query_one(f"#ex-desc-{ex.key}", Label)) for ex in EXPERIMENTS}
        self.main = self.query_one("#main")
        self.log_wrap = self.query_one("#log-wrap")
        self._apply_log_height(self.ui.log_height)
        self.call_after_refresh(self._fit_log)
        self.set_interval(1, self._tick)
        self.set_interval(EVENT_POLL, self._drain_events)
        self.set_interval(15, self._periodic_refresh)
        self.refresh_status()
        self.log_pane.add("Ready. Stages already done are shown in green. Press r to run the "
                          "remaining stages, or Run on a card.", "info")

    # ── status from disk ──────────────────────────────────────────────────────
    def refresh_status(self) -> None:
        """Mark stages done whose outputs exist, and enable experiments whose inputs are ready."""
        self.data_path_lbl.update(_short(ST.data_dir(self.settings)))
        done = set()
        for st in self.stages:
            card = self.cards[st.key]
            try:
                ok = st.is_done(self.settings)
            except OSError:
                ok = False
            if ok:
                done.add(st.key)
            if card.state == "running":
                continue
            if ok:
                card.set_state("done", "Done")
            elif card.state == "done":
                card.set_state("waiting")
            elif card.state in ("waiting", "failed"):
                other = st.running_elsewhere()
                missing = st.missing_inputs(self.settings)
                hint = st.hint(self.settings) if st.hint else ""
                if other:
                    status = f"Running outside the app (pid {other})"
                elif missing:
                    status = "Needs " + ", ".join(missing[:2]) + ("…" if len(missing) > 2 else "")
                else:
                    status = hint or "Ready to run"
                if card.state == "waiting" or other:
                    card.set_state("waiting", status)
        names = {st.key: st.name for st in self.stages}
        for ex in EXPERIMENTS:
            need = [names.get(k, k) for k in ex.requires if k not in done]
            btn, desc = self.ex_widgets[ex.key]
            btn.disabled = bool(need)
            desc.update(
                ex.description if not need else f"Needs: {', '.join(need)}")

    def _periodic_refresh(self) -> None:
        """Pick up runs started or finished outside the app."""
        if not self.runner.is_running() and self.screen is self.screen_stack[0]:
            self.refresh_status()

    def _tick(self) -> None:
        for c in self.cards.values():
            if c.state == "running":
                c.refresh_detail()

    # ── runner events ─────────────────────────────────────────────────────────
    def _drain_events(self) -> None:
        changed = False
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "log":
                self.log_pane.add(ev[1], ev[2])
            elif ev[0] == "state":
                self.cards[ev[1]].set_state(ev[2], ev[3])
                changed = True
            elif ev[0] == "progress":
                self.cards[ev[1]].set_progress(ev[2], ev[3])
        running = self.runner.is_running()
        self.stop_btn.disabled = not running
        self.run_all_btn.disabled = running
        if changed and not running:
            self.refresh_status()

    # ── actions ───────────────────────────────────────────────────────────────
    def _can_start(self) -> bool:
        bad = self.settings.invalid()
        if bad:
            self.log_pane.add(f"Can't run: invalid {', '.join(bad)} (shown in red).", "info")
            return False
        return True

    def action_run_all(self) -> None:
        if self._can_start():
            self.runner.run_all()

    def action_stop(self) -> None:
        self.runner.stop()

    @on(Button.Pressed, "#run-all")
    def _run_all_btn(self) -> None:
        self.action_run_all()

    @on(Button.Pressed, "#stop")
    def _stop_btn(self) -> None:
        self.action_stop()

    @on(StageCard.RunPressed)
    def _run_stage(self, event: StageCard.RunPressed) -> None:
        st = ST.BY_KEY.get(event.key) or next(s for s in self.stages if s.key == event.key)
        if not self._can_start():
            return
        if st.is_done(self.settings):
            self.push_screen(ConfirmScreen(f"Run {st.name} again?",
                                           "Its outputs already exist and will be rebuilt.", "Run again"),
                             lambda yes: yes and self.runner.run_stage(event.key))
        else:
            self.runner.run_stage(event.key)

    @on(StageCard.ViewPressed)
    def _view_stage(self, event: StageCard.ViewPressed) -> None:
        st = next(s for s in self.stages if s.key == event.key)
        self.push_screen(ViewScreen(st, self.settings))

    @on(Button.Pressed, ".experiment")
    def _open_experiment(self, event: Button.Pressed) -> None:
        key = (event.button.id or "").removeprefix("ex-")
        ex = next(e for e in EXPERIMENTS if e.key == key)
        module, cls = ex.screen.split(":")
        screen_cls = getattr(importlib.import_module(module), cls)
        self.push_screen(screen_cls(ST.data_dir(self.settings)))

    @on(SettingsForm.SettingChanged)
    def _setting_changed(self, event: SettingsForm.SettingChanged) -> None:
        if event.key in ("data_dir", "dumps_dir", "dump_date", "dl_extras", "dl_text"):
            self.refresh_status()

    def action_save_settings(self) -> None:
        bad = self.settings.invalid()
        if bad:
            self.log_pane.add(f"Not saved: invalid {', '.join(bad)}.", "info")
            return
        self.settings.save(self.config_path)
        self.log_pane.add(f"Settings saved to {self.config_path.name}.", "info")

    async def action_request_quit(self) -> None:
        if not self.runner.is_running():
            self.exit()
            return

        def answer(yes: bool) -> None:
            if yes:
                self.runner.stop()
                self.exit()
        self.push_screen(ConfirmScreen("Quit?", "A stage is running. Quitting stops it.", "Stop and quit"),
                         answer)

    # ── log pane ──────────────────────────────────────────────────────────────
    def _max_log_height(self) -> int:
        return max(LOG_MIN_HEIGHT, self.main.size.height - 6)

    def _apply_log_height(self, h: int) -> int:
        h = max(LOG_MIN_HEIGHT, int(h))
        if self.main.size.height:
            h = min(h, self._max_log_height())
        self.log_wrap.styles.height = h
        return h

    def _fit_log(self) -> None:
        if not self.screen.has_class("-log-max"):
            self._apply_log_height(self.ui.log_height)

    def on_resize(self) -> None:
        self.call_after_refresh(self._fit_log)

    @on(LogHandle.Resized)
    def _log_dragged(self, event: LogHandle.Resized) -> None:
        self.screen.remove_class("-log-max")
        h = self._apply_log_height(event.height)
        if event.final:
            self.ui.log_height = h
            self.ui.save(self.ui_state_path)

    def action_log_resize(self, delta: int) -> None:
        self.screen.remove_class("-log-max")
        self.ui.log_height = self._apply_log_height(self.ui.log_height + delta)
        self.ui.save(self.ui_state_path)

    def action_toggle_log(self) -> None:
        self.screen.toggle_class("-log-max")
        if self.screen.has_class("-log-max"):
            self.log_wrap.styles.height = "1fr"
        else:
            self._apply_log_height(self.ui.log_height)

    def action_cycle_theme(self) -> None:
        i = THEMES.index(self.theme) if self.theme in THEMES else -1
        self.theme = THEMES[(i + 1) % len(THEMES)]
        self.ui.theme = self.theme
        self.ui.save(self.ui_state_path)
        self.notify(f"Theme: {self.theme}", timeout=1.5)

    def on_unmount(self) -> None:
        self.runner.stop()
