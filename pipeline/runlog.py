"""Run records: every run started from the app is written to <data>/runs/.

- ``pipeline.log``: every log line with a timestamp, appended under a "===== run started =====" header.
- ``runs.json``: ``{"runs": [...]}``, one entry per run with the settings, machine and versions,
  and for each stage its start/end, duration, result and output sizes. Rewritten after each stage,
  so a crash keeps the stages that finished.
"""
from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Mapping

from wikiexp.paths import ROOT


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def environment() -> dict:
    def quiet(cmd):
        try:
            return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": quiet(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(quiet(["git", "status", "--porcelain"])),
    }


class RunRecorder:
    def __init__(self, runs_dir: Path, settings: Mapping, mode: str):
        self.dir = Path(runs_dir)
        self.lock = threading.Lock()
        self.run = {"started": _now(), "mode": mode, "result": None, "settings": dict(settings),
                    "environment": environment(), "stages": []}
        self._log = None

    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.dir / "pipeline.log", "a", encoding="utf-8")
        self._write(f"===== run started {self.run['started']} ({self.run['mode']}) =====")
        self._save()

    def log(self, text: str, kind: str) -> None:
        if self._log:
            self._write(("$ " if kind == "header" else "") + text)

    def stage_started(self, key: str, command: list[str]) -> None:
        self.run["stages"].append({"stage": key, "command": command, "started": _now(),
                                   "ended": None, "duration_s": None, "result": "running"})
        self._save()

    def stage_finished(self, key: str, result: str, outputs: list[Path]) -> None:
        st = next((x for x in reversed(self.run["stages"]) if x["stage"] == key), None)
        if st is None:
            return
        st["ended"] = _now()
        st["duration_s"] = round((datetime.fromisoformat(st["ended"]) -
                                  datetime.fromisoformat(st["started"])).total_seconds(), 1)
        st["result"] = result
        st["outputs"] = {str(p): p.stat().st_size for p in outputs if p.exists() and p.is_file()}
        self._save()

    def finish(self, result: str) -> None:
        self.run["result"] = result
        self.run["ended"] = _now()
        self._save()
        if self._log:
            self._write(f"===== run {result} =====")
            self._log.close()
            self._log = None

    def _write(self, line: str) -> None:
        with self.lock:
            self._log.write(f"{datetime.now():%H:%M:%S.%f}"[:-3] + " " + line + "\n")
            self._log.flush()

    def _save(self) -> None:
        with self.lock:
            path = self.dir / "runs.json"
            try:
                runs = json.loads(path.read_text())["runs"]
            except (OSError, ValueError, KeyError):
                runs = []
            if runs and runs[-1].get("started") == self.run["started"]:
                runs[-1] = self.run
            else:
                runs.append(self.run)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"runs": runs}, indent=1, default=str))
            tmp.replace(path)
