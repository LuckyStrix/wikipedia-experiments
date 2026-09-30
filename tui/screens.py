"""Dialogs: confirm, and the View screen that shows a stage's outputs."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel"), Binding("y", "dismiss(True)", "Yes")]

    def __init__(self, title: str, body: str, yes: str = "Yes"):
        super().__init__()
        self.title_text, self.body, self.yes = title, body, yes

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Label(self.body, classes="dialog-body")
            with Horizontal(classes="dialog-btns"):
                yield Button("Cancel", id="no", compact=True)
                yield Button(self.yes, id="yes", variant="error", compact=True)

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss(False)


class ViewScreen(ModalScreen[None]):
    """A stage's outputs: files, sizes and whatever summary the stage provides."""
    BINDINGS = [Binding("escape", "dismiss", "Close"), Binding("q", "dismiss", "Close")]

    def __init__(self, stage, settings):
        super().__init__()
        self.stage, self.settings = stage, dict(settings)

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog", id="view"):
            yield Label(self.stage.name, classes="dialog-title")
            yield Label(self.stage.description, classes="dialog-sub")
            with VerticalScroll():
                yield Static("Loading…", id="view-body")
            with Horizontal(classes="dialog-btns"):
                yield Button("Close", id="close", compact=True)

    def on_mount(self) -> None:
        self._load()

    @work(thread=True)
    def _load(self) -> None:
        try:
            lines = self.stage.summary(self.settings)
            missing = self.stage.missing_inputs(self.settings)
            if missing:
                lines = lines + ["", "Missing inputs: " + ", ".join(missing)]
            text = "\n".join(lines)
        except Exception as e:  # a half-written database, etc.
            text = f"Couldn't read the outputs: {e}"
        self.app.call_from_thread(self.query_one("#view-body", Static).update, text)

    @on(Button.Pressed, "#close")
    def _close(self) -> None:
        self.dismiss()
