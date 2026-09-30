"""Progress reporting for long scripts, visible to the app however the script was started.

A script calls start("core") once, then progress(pct, text) as it goes.

- Run from the app (which sets WIKI_PROGRESS=1), each call also prints "@progress <pct> <text>",
  which the runner turns into the card's progress bar and hides from the log.
- Always, the latest progress is written to <data>/runs/progress/<stage>.json with the script's
  PID, so the app can show progress for a run started from a terminal too.
"""
import json
import os
import time

from . import paths

ENABLED = os.environ.get("WIKI_PROGRESS") == "1"
PREFIX = "@progress "
_file = None


def progress_file(stage, data_dir=None):
    return (data_dir or paths.DATA) / "runs" / "progress" / f"{stage}.json"


def start(stage):
    global _file
    _file = progress_file(stage)
    progress(0, "starting")


def progress(pct, text=""):
    pct = max(0, min(100, int(pct)))
    if ENABLED:
        print(f"{PREFIX}{pct} {text}", flush=True)
    if _file is not None:
        try:
            _file.parent.mkdir(parents=True, exist_ok=True)
            tmp = _file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"pid": os.getpid(), "pct": pct, "text": text, "time": time.time()}))
            tmp.replace(_file)
        except OSError:
            pass   # progress is a nicety; never fail a build over it


def read(stage, data_dir=None):
    """(pct, text, pid) from a stage's progress file, or None."""
    try:
        d = json.loads(progress_file(stage, data_dir).read_text())
        return d["pct"], d["text"], d["pid"]
    except (OSError, ValueError, KeyError):
        return None
