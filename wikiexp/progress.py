"""Machine-readable progress lines for the app's stage cards.

Scripts call progress(pct, text). When run from the app (which sets WIKI_PROGRESS=1) this prints
"@progress <pct> <text>", which the runner turns into the card's progress bar and hides from the log.
Run from a terminal, it prints nothing, so command-line output stays clean.
"""
import os

ENABLED = os.environ.get("WIKI_PROGRESS") == "1"
PREFIX = "@progress "


def progress(pct, text=""):
    if ENABLED:
        print(f"{PREFIX}{max(0, min(100, int(pct)))} {text}", flush=True)
