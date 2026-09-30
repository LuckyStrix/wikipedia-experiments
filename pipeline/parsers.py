"""Turn a stage's output lines into progress for its card.

Each parser has ``feed(line) -> (pct, text) | None`` and ``hide(line) -> bool`` (lines that exist
only to report progress and shouldn't clutter the log).
"""
from __future__ import annotations

import re

from wikiexp.progress import PREFIX


class ProgressParser:
    """Python stages report with wikiexp.progress: "@progress <pct> <text>"."""

    def __init__(self, settings=None):
        pass

    def feed(self, line: str):
        if line.startswith(PREFIX):
            pct, _, text = line[len(PREFIX):].partition(" ")
            if pct.isdigit():
                return int(pct), text.strip()
        return None

    def hide(self, line: str) -> bool:
        return line.startswith(PREFIX)


class WgetParser:
    """pipeline/download.sh: "(i/N) downloading <file>" headers, then wget's dot progress.

    Overall progress weights each file by its approximate size (``sizes``, file name -> MB; files
    not listed count as 100 MB), since the article text alone is most of the download.
    """
    FILE = re.compile(r"\((\d+)/(\d+)\) downloading (\S+)")
    PCT = re.compile(r"\s(\d{1,3})%\s")
    CHECK = re.compile(r"verifying checksums")

    def __init__(self, sizes: dict | None = None, files: list | None = None):
        self.i, self.n, self.name = 0, 1, ""
        self.sizes, self.files = sizes or {}, files

    def feed(self, line: str):
        if m := self.FILE.search(line):
            self.i, self.n, self.name = int(m.group(1)), int(m.group(2)), m.group(3)
            return self._pct(0), f"{self.name} ({self.i}/{self.n})"
        if self.i and (m := self.PCT.search(line)):
            p = int(m.group(1))
            return self._pct(p), f"{self.name} {p}% ({self.i}/{self.n})"
        if self.CHECK.search(line):
            return 99, "verifying checksums"
        return None

    def _pct(self, file_pct: int) -> int:
        if not self.files:
            return min(99, int(((self.i - 1) + file_pct / 100) / self.n * 100))
        w = [self.sizes.get(f, 100) for f in self.files]
        done = sum(w[:self.i - 1]) + w[self.i - 1] * file_pct / 100 if self.i <= len(w) else sum(w)
        return min(99, int(done / sum(w) * 100))

    def hide(self, line: str) -> bool:
        # wget prints a dot-progress line every 32 MB; keep only every 10th
        m = self.PCT.search(line)
        return bool(m) and self.i > 0 and int(m.group(1)) % 10 != 0
