"""Run stages as subprocesses and report what they're doing.

``PipelineRunner`` is UI-independent. It reports through three callbacks, all called from its worker
thread (a UI must marshal them onto its own thread):

    on_log(text, kind)              kind: "info"   – app message
                                          "header" – the command being run
                                          "output" – subprocess output
    on_state(key, state, status)    state: waiting / running / done / failed
    on_progress(key, pct, text)

Every run is recorded in <data>/runs/ (see runlog).
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Callable, Mapping

from wikiexp.paths import ROOT

from .runlog import RunRecorder
from .settings import env_for
from .stages import BY_KEY, STAGES, Stage, runs_dir

STOP_GRACE_SECONDS = 5.0

LogFn = Callable[[str, str], None]
StateFn = Callable[[str, str, str], None]
ProgressFn = Callable[[str, int, str], None]


def _nothing(*a):
    pass


class PipelineRunner:
    def __init__(self, settings: Mapping, on_log: LogFn = _nothing, on_state: StateFn = _nothing,
                 on_progress: ProgressFn = _nothing, stages: list[Stage] | None = None):
        self.settings = settings
        self.on_log, self.on_state, self.on_progress = on_log, on_state, on_progress
        self.stages = stages if stages is not None else STAGES
        self.by_key = {st.key: st for st in self.stages} if stages is not None else BY_KEY
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()

    # ── public API (call from the UI thread) ──────────────────────────────────
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_stage(self, key: str) -> bool:
        return self._start([self.by_key[key]], mode=f"stage {key}")

    def run_all(self) -> bool:
        """Run every stage that isn't done yet, in order, stopping at the first failure.
        Manual stages (e.g. needing a GPU) are skipped; they run from their own card."""
        pending = [st for st in self.stages if not st.is_done(self.settings)]
        todo = [st for st in pending if not st.manual]
        if len(todo) < len(pending):
            names = ", ".join(st.name for st in pending if st.manual)
            self.on_log(f"Skipping manual stage(s): {names}. Start them with Run on their card.", "info")
        if not todo:
            self.on_log("Every automatic stage is already done.", "info")
            return False
        return self._start(todo, mode="all")

    def stop(self) -> None:
        if not self.is_running():
            return
        self._stop.set()
        self.on_log("Stopping…", "info")
        proc = self._proc
        if proc and proc.poll() is None:
            threading.Thread(target=self._terminate, args=(proc,), daemon=True).start()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ── worker ────────────────────────────────────────────────────────────────
    def _start(self, stages: list[Stage], mode: str) -> bool:
        if self.is_running():
            self.on_log("A stage is already running.", "info")
            return False
        snapshot = dict(self.settings)  # a run uses the settings as they were when it started
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(stages, snapshot, mode), daemon=True)
        self._thread.start()
        return True

    def _run(self, stages: list[Stage], settings: dict, mode: str) -> None:
        rec = RunRecorder(runs_dir(settings), settings, mode)
        try:
            rec.start()
        except OSError as e:
            self.on_log(f"[warn] can't write run records: {e}", "info")
            rec = None
        result = "done"
        for st in stages:
            ok = self._run_one(st, settings, rec)
            if not ok:
                result = "stopped" if self._stop.is_set() else "failed"
                break
        if rec:
            rec.finish(result)

    def _run_one(self, st: Stage, s: dict, rec: RunRecorder | None) -> bool:
        def log(text, kind="info"):
            self.on_log(text, kind)
            if rec:
                rec.log(text, kind)

        other = st.running_elsewhere()
        if other:
            log(f"{st.name}: already running outside the app (process {other}); not starting another copy")
            self.on_state(st.key, "failed", f"Already running elsewhere (pid {other})")
            return False
        missing = st.missing_inputs(s)
        if missing:
            log(f"{st.name}: can't start, missing {', '.join(missing)}")
            self.on_state(st.key, "failed", "Needs " + ", ".join(missing))
            return False
        cmd = st.command(s)
        env = {**os.environ, **env_for(s), **st.env(s),
               "PYTHONUNBUFFERED": "1", "WIKI_PROGRESS": "1", "PYTHONIOENCODING": "utf-8"}
        self.on_state(st.key, "running", "Running…")
        log(" ".join(cmd), "header")
        if rec:
            rec.stage_started(st.key, cmd)
        parser = st.parser(s)
        t0 = time.monotonic()
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                **({"start_new_session": True} if os.name == "posix"
                   else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}))
        except OSError as e:
            log(f"{st.name} failed to start: {e}")
            self.on_state(st.key, "failed", "Failed to start")
            if rec:
                rec.stage_finished(st.key, "failed", [])
            return False
        for line in self._proc.stdout:
            line = line.rstrip("\r\n")
            got = parser.feed(line)
            if got:
                self.on_progress(st.key, *got)
            if not parser.hide(line):
                log(line, "output")
        code = self._proc.wait()
        self._proc = None
        mins = (time.monotonic() - t0) / 60
        if self._stop.is_set():
            result, status = "stopped", "Stopped"
        elif code == 0:
            result, status = "done", f"Done in {mins:.1f} min"
        else:
            result, status = "failed", f"Failed (exit code {code})"
        log(f"{st.name}: {status.lower()}")
        self.on_state(st.key, "done" if result == "done" else "failed", status)
        if rec:
            rec.stage_finished(st.key, result, st.outputs(s))
        return result == "done"

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        """Stop the stage and everything it started (e.g. pigz, wget)."""
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass

