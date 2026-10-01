"""The data pipeline's stages: what each one runs, needs and produces.

UI-free. Each ``Stage`` builds its command line from a ``Settings`` mapping, says which inputs are
missing, whether its outputs already exist (so the app can show it as done after a restart, or
after a run started from the command line), and summarises its outputs for the View button.

Adding a stage: append a ``Stage`` to ``STAGES``. Experiments that need it list its key in their
``requires``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from wikiexp import progress as prog
from wikiexp.paths import ROOT

from .parsers import ProgressParser, WgetParser

CORE_DUMPS = ["page.sql.gz", "redirect.sql.gz", "linktarget.sql.gz", "pagelinks.sql.gz"]
EXTRA_DUMPS = ["categorylinks.sql.gz", "category.sql.gz", "geo_tags.sql.gz", "page_props.sql.gz",
               "langlinks.sql.gz"]
TEXT_DUMPS = ["pages-articles-multistream-index.txt.bz2", "pages-articles-multistream.xml.bz2"]
# approximate compressed sizes in MB (2026-09 enwiki), to weight the download's progress bar
DUMP_MB = {"page.sql.gz": 2362, "redirect.sql.gz": 187, "linktarget.sql.gz": 1396,
           "pagelinks.sql.gz": 7087, "categorylinks.sql.gz": 2577, "category.sql.gz": 35,
           "geo_tags.sql.gz": 53, "page_props.sql.gz": 471, "langlinks.sql.gz": 582,
           "pages-articles-multistream-index.txt.bz2": 284, "pages-articles-multistream.xml.bz2": 26843}
GRAPH_FILES = ["idx_to_page_id.npy", "out_indptr.npy", "out_indices.npy", "in_indptr.npy",
               "in_indices.npy"]


# ── paths derived from settings (mirrors wikiexp.paths, but for the app's live settings) ──

def data_dir(s: Mapping) -> Path:
    return Path(str(s["data_dir"]).strip() or ROOT / "data")


def dumps_dir(s: Mapping) -> Path:
    return Path(str(s["dumps_dir"]).strip() or ROOT / "dumps")


def dump_file(s: Mapping, name: str) -> Path:
    return dumps_dir(s) / f"enwiki-{str(s['dump_date']).strip()}-{name}"


def db_path(s: Mapping) -> Path:
    return data_dir(s) / "wiki.sqlite"


def titles_path(s: Mapping) -> Path:
    return data_dir(s) / "titles.sqlite"


def runs_dir(s: Mapping) -> Path:
    return data_dir(s) / "runs"


def human_size(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return ""


def _size(p: Path) -> str:
    return human_size(p.stat().st_size) if p.exists() else "missing"


# ── stage definition ──────────────────────────────────────────────────────────

@dataclass
class Stage:
    key: str
    name: str
    description: str
    command: Callable[[Mapping], list[str]]
    inputs: Callable[[Mapping], list[Path]]
    outputs: Callable[[Mapping], list[Path]]
    summary: Callable[[Mapping], list[str]]
    parser: Callable[[Mapping], object] = ProgressParser   # called with the settings
    env: Callable[[Mapping], dict] = field(default=lambda s: {})
    done_check: Callable[[Mapping], bool] | None = None   # default: all outputs exist
    tools: tuple[str, ...] = ()                            # programs that must be on PATH
    posix_only: bool = False
    signature: str = ""       # text in the command line of a running copy (to detect one started elsewhere)
    hint: Callable[[Mapping], str] | None = None            # extra status while not done
    live_progress: Callable[[Mapping, int], tuple[int, str] | None] | None = None  # for a copy running elsewhere

    def missing_inputs(self, s: Mapping) -> list[str]:
        if self.posix_only and os.name != "posix":
            return ["Linux or macOS (on Windows, download by hand; see README)"]
        missing = [p.name for p in self.inputs(s) if not p.exists()]
        missing += [f"{t} (program)" for t in self.tools if shutil.which(t) is None]
        return missing

    def running_elsewhere(self, exclude_pids: set[int] = frozenset()) -> int | None:
        """PID of a copy of this stage running outside this app (e.g. started from a terminal)."""
        return find_process(self.signature, exclude_pids) if self.signature else None

    def progress_of(self, s: Mapping, pid: int) -> tuple[int, str] | None:
        """(pct, text) of a copy running elsewhere with this PID, if it reports any."""
        if self.live_progress is not None:
            return self.live_progress(s, pid)
        got = prog.read(self.key, data_dir(s))
        if got and got[2] == pid:
            return got[0], got[1]
        return None

    def is_done(self, s: Mapping) -> bool:
        if self.done_check is not None:
            return self.done_check(s)
        outs = self.outputs(s)
        return bool(outs) and all(p.exists() for p in outs)


def find_process(signature: str, exclude_pids: set[int] = frozenset()) -> int | None:
    """PID of a running process whose command line contains `signature` (Linux only; else None)."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    me = os.getpid()
    for d in proc.iterdir():
        if not d.name.isdigit() or int(d.name) in exclude_pids or int(d.name) == me:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if signature in cmd and "pytest" not in cmd:
            return int(d.name)
    return None


# ── download ──────────────────────────────────────────────────────────────────

def download_files(s: Mapping) -> list[str]:
    files = ["sha1sums.txt"] + CORE_DUMPS
    if s["dl_extras"]:
        files += EXTRA_DUMPS
    if s["dl_text"]:
        files += TEXT_DUMPS
    return files


def verified_downloads(s: Mapping) -> dict[str, bool]:
    """{file name: passed checksum}, from the latest sha1sum result per file in download.log."""
    log = dumps_dir(s) / "download.log"
    status: dict[str, bool] = {}
    if log.exists():
        for m in re.finditer(r"^(enwiki-\S+): (OK|FAILED)", log.read_text(errors="replace"), re.M):
            status[m.group(1)] = m.group(2) == "OK"
    return status


def download_done(s: Mapping) -> bool:
    ok = verified_downloads(s)
    return all(ok.get(dump_file(s, f).name, False) for f in download_files(s) if f != "sha1sums.txt")


def download_hint(s: Mapping) -> str:
    ok = verified_downloads(s)
    files = [f for f in download_files(s) if f != "sha1sums.txt"]
    n = sum(ok.get(dump_file(s, f).name, False) for f in files)
    return f"{n} of {len(files)} files verified" if n else ""


def download_parser(s: Mapping) -> WgetParser:
    return WgetParser(DUMP_MB, [f for f in download_files(s) if f != "sha1sums.txt"])


def download_live_progress(s: Mapping, pid: int) -> tuple[int, str] | None:
    """Progress of a download running elsewhere, from the end of download.log."""
    log = dumps_dir(s) / "download.log"
    try:
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - 256_000))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    files = [f for f in download_files(s) if f != "sha1sums.txt"]
    parser = download_parser(s)
    got = None
    for line in lines:
        m = re.search(r"downloading (\S+)", line)
        if m and "(" not in line.split("downloading")[0][-8:] and m.group(1) in files:
            # older log lines have no "(i/N)": number them from the file list
            line = f"({files.index(m.group(1)) + 1}/{len(files)}) downloading {m.group(1)}"
        got = parser.feed(line) or got
    return got


def download_summary(s: Mapping) -> list[str]:
    ok = verified_downloads(s)
    lines = [f"From https://dumps.wikimedia.org/enwiki/{s['dump_date']}/ into {dumps_dir(s)}", ""]
    for f in download_files(s):
        p = dump_file(s, f)
        state = "verified" if ok.get(p.name) else ("not verified yet" if p.exists() else "")
        lines.append(f"{p.name:<58} {_size(p):>10}  {state}")
    zims = sorted(dumps_dir(s).glob("*.zim"))
    lines += ["", f"Kiwix .zim: {zims[-1].name} ({_size(zims[-1])})" if zims else
              "Kiwix .zim: none (optional; see README for where to get it)"]
    return lines


# ── core database ─────────────────────────────────────────────────────────────

def core_outputs(s: Mapping) -> list[Path]:
    return [db_path(s)] + [data_dir(s) / "graph" / f for f in GRAPH_FILES]


def core_summary(s: Mapping) -> list[str]:
    lines = [f"{p.relative_to(data_dir(s))}: {_size(p)}" for p in core_outputs(s)]
    if db_path(s).exists():
        with sqlite3.connect(f"file:{db_path(s)}?mode=ro", uri=True) as db:
            lines += ["", *(f"{k}: {v}" for k, v in db.execute("SELECT key, value FROM meta"))]
    return lines


def titles_summary(s: Mapping) -> list[str]:
    p = titles_path(s)
    lines = [f"{p.name}: {_size(p)}"]
    if p.exists():
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as db:
            lines += [f"{k}: {v}" for k, v in db.execute("SELECT key, value FROM meta")]
    return lines


def geo_path(s: Mapping) -> Path:
    return data_dir(s) / "geo.sqlite"


def geo_summary(s: Mapping) -> list[str]:
    p = geo_path(s)
    lines = [f"{p.name}: {_size(p)}"]
    if p.exists():
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as db:
            lines += [f"{k}: {v}" for k, v in db.execute("SELECT key, value FROM meta")]
    return lines
def centrality_dir(s: Mapping) -> Path:
    return data_dir(s) / "centrality"


def centrality_outputs(s: Mapping) -> list[Path]:
    return [centrality_dir(s) / "meta.json"]    # written last, so its presence means complete


def centrality_summary(s: Mapping) -> list[str]:
    d = centrality_dir(s)
    lines = [f"{p.relative_to(data_dir(s))}: {_size(p)}" for p in sorted(d.glob("*.*"))]
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        return lines
    return lines + ["", *(f"{k}: {v}" for k, v in meta.items())]


def centrality_command(s: Mapping) -> list[str]:
    return [PY, "-m", "pipeline.build_centrality", "--damping", str(s["centrality_damping"]).strip(),
            *([] if s["centrality_reverse"] else ["--no-reverse"])]


PY = sys.executable

STAGES: list[Stage] = [
    Stage(
        key="download", name="Download dumps",
        description="Fetch the Wikimedia dump files (resumable).",
        command=lambda s: ["bash", "pipeline/download.sh", *download_files(s)],
        inputs=lambda s: [],
        outputs=lambda s: [dump_file(s, f) for f in download_files(s)],
        summary=download_summary,
        parser=download_parser,
        done_check=download_done,
        tools=("bash", "wget", "sha1sum"),
        posix_only=True,
        signature="download.sh",
        hint=download_hint,
        live_progress=download_live_progress,
    ),
    Stage(
        key="core", name="Build core database",
        description="Articles, redirects and links -> wiki.sqlite + graph arrays.",
        command=lambda s: [PY, "-m", "pipeline.build_core", *([] if s["verify"] else ["--skip-verify"])],
        inputs=lambda s: [dump_file(s, f) for f in CORE_DUMPS] + [dump_file(s, "sha1sums.txt")],
        outputs=core_outputs,
        summary=core_summary,
        signature="pipeline.build_core",
    ),
    Stage(
        key="titles", name="Build title index",
        description="Search index for title autocomplete and fuzzy matching.",
        command=lambda s: [PY, "-m", "pipeline.build_titles",
                           *([] if s["titles_redirects"] else ["--no-redirects"])],
        inputs=lambda s: core_outputs(s),
        outputs=lambda s: [titles_path(s)],
        summary=titles_summary,
        signature="pipeline.build_titles",
    ),
    Stage(
        key="centrality", name="Compute centrality",
        description="PageRank, in-degree ranks and reverse PageRank for every article.",
        command=centrality_command,
        inputs=lambda s: core_outputs(s),
        outputs=centrality_outputs,
        summary=centrality_summary,
        signature="pipeline.build_centrality",
    ),
    Stage(
        key="geo", name="Build places index",
        description="Geotagged articles with an R*Tree, for the Nearby project.",
        command=lambda s: [PY, "-m", "pipeline.build_geo"],
        inputs=lambda s: core_outputs(s) + [dump_file(s, "geo_tags.sql.gz")],
        outputs=lambda s: [geo_path(s)],
        summary=geo_summary,
        signature="pipeline.build_geo",
    ),
]

BY_KEY = {st.key: st for st in STAGES}
