"""Per-viewer UI conveniences (theme, log height), kept in .tui_state.json (not in git)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from wikiexp.paths import ROOT

UI_STATE_PATH = ROOT / ".tui_state.json"


@dataclass
class UiState:
    theme: str = "campbell"
    log_height: int = 12

    @classmethod
    def load(cls, path: Path = UI_STATE_PATH) -> "UiState":
        try:
            data = json.loads(Path(path).read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path = UI_STATE_PATH) -> None:
        try:
            Path(path).write_text(json.dumps(asdict(self), indent=1))
        except OSError:
            pass
