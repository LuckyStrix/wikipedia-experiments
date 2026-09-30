"""Find articles by title: exact (any case), prefix, substring, and typo-tolerant matches.

Uses data/titles.sqlite from pipeline/build_titles.py. Rows there are in popularity order, so every
query below walks them by id and stops after a few dozen hits: the most-linked matches come first.
"""
from __future__ import annotations

import difflib
import random
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Match:
    title: str           # the title that matched (may be a redirect, e.g. "USA")
    page_id: int         # the article it leads to
    article: str         # that article's title, e.g. "United States"
    links: int           # incoming links to the article (popularity)
    how: str             # "exact", "prefix", "contains" or "similar"

    @property
    def label(self) -> str:
        return self.article if self.title == self.article else f"{self.title} → {self.article}"


def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.replace("_", " ")).strip()


class TitleIndex:
    SCAN = 60   # rows fetched per query before de-duplicating

    def __init__(self, path: Path | None = None):
        path = Path(path or paths.DATA / "titles.sqlite")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - build the title index first "
                                    "(python -m pipeline.build_titles, or the app's Build title index)")
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.lock = threading.Lock()   # the app searches from worker threads

    def _q(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def _article(self, page_id: int) -> str:
        row = self._q("SELECT title FROM titles WHERE target = ? AND redirect = 0", (page_id,))
        return row[0][0] if row else "?"

    def _matches(self, rows, how) -> list[Match]:
        out = []
        for title, target, links, redirect in rows:
            article = self._article(target) if redirect else title
            out.append(Match(title, target, article, links, how))
        return out

    # ── lookups ───────────────────────────────────────────────────────────────
    def exact(self, q: str) -> Match | None:
        """Case-insensitive exact title (or redirect) match; the most-linked one if several."""
        q = normalize(q)
        rows = self._q("SELECT title, target, links, redirect FROM titles "
                       "WHERE title = ? COLLATE NOCASE ORDER BY title = ? DESC, id LIMIT 1", (q, q))
        return self._matches(rows, "exact")[0] if rows else None

    def _like(self, pattern: str, limit: int) -> list[tuple]:
        return self._q("SELECT t.title, t.target, t.links, t.redirect FROM titles_fts f "
                       "JOIN titles t ON t.id = f.rowid WHERE f.title LIKE ? ORDER BY f.rowid LIMIT ?",
                       (pattern, limit))

    def prefix(self, q: str, limit: int = SCAN) -> list[Match]:
        q = normalize(q)
        if len(q) >= 3:  # the trigram index needs 3+ characters
            return self._matches(self._like(_escape_like(q) + "%", limit), "prefix")
        # 1-2 characters: walk the case-insensitive index, then take the most linked
        rows = self._q("SELECT title, target, links, redirect, id FROM titles "
                       "WHERE title >= ?1 COLLATE NOCASE AND title < ?1 || char(1114111) COLLATE NOCASE "
                       "LIMIT 5000", (q,))
        rows.sort(key=lambda r: r[4])
        return self._matches([r[:4] for r in rows[:limit]], "prefix")

    def contains(self, q: str, limit: int = SCAN) -> list[Match]:
        q = normalize(q)
        return self._matches(self._like("%" + _escape_like(q) + "%", limit), "contains") if len(q) >= 3 else []

    def similar(self, q: str, limit: int = 10) -> list[Match]:
        """Typo-tolerant: any single typo leaves one half of each word intact, so gather titles
        containing a word-half, then rank them by string similarity (and popularity)."""
        q = normalize(q)
        pieces = set()
        for w in q.split():
            if len(w) >= 6:
                pieces |= {w[:len(w) // 2], w[len(w) // 2:]}
            elif len(w) >= 3:
                pieces.add(w)
        cands = {}
        for p in pieces:
            for row in self._like("%" + _escape_like(p) + "%", 300):
                cands.setdefault(row[0], row)
        ql = q.lower()
        scored = []
        for title, row in cands.items():
            ratio = difflib.SequenceMatcher(None, ql, title.lower()).ratio()
            if ratio >= 0.6:
                scored.append((ratio, row))
        scored.sort(key=lambda x: (-round(x[0], 2), -x[1][2]))
        return self._matches([row for _, row in scored[:limit * 3]], "similar")

    def search(self, q: str, limit: int = 10) -> list[Match]:
        """Best matches for what the user typed, one per article: exact, then prefix, then
        substring, then typo-tolerant matches, each most-linked first."""
        q = normalize(q)
        if not q:
            return []
        seen, out = set(), []

        def add(matches):
            for m in matches:
                if m.page_id not in seen and len(out) < limit:
                    seen.add(m.page_id)
                    out.append(m)

        e = self.exact(q)
        add([e] if e else [])
        add(self.prefix(q))
        if len(out) < limit:
            add(self.contains(q))
        if len(out) < min(limit, 3):
            add(self.similar(q, limit))
        return out

    def resolve(self, q: str) -> Match | None:
        """The article for q if q names one exactly (any case), else None."""
        return self.exact(q)

    def random_article(self, top: int = 100_000) -> Match:
        """A random article among the `top` most linked (so it's usually a recognisable one)."""
        while True:
            rows = self._q("SELECT title, target, links, redirect FROM titles "
                           "WHERE id >= ? AND redirect = 0 ORDER BY id LIMIT 1", (random.randint(1, top),))
            if rows:
                return self._matches(rows, "exact")[0]


def _escape_like(s: str) -> str:
    # FTS5's trigram LIKE doesn't support ESCAPE, so drop the wildcard characters instead
    return s.replace("%", "").replace("_", " ")
