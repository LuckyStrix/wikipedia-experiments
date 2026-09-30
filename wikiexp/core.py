"""Access to the core outputs of pipeline/build_core.py: data/wiki.sqlite and data/graph/."""
import sqlite3

import numpy as np

from . import paths


def connect():
    """Read-only connection to wiki.sqlite."""
    if not paths.DB.exists():
        raise FileNotFoundError(f"{paths.DB} not found - run: python -m pipeline.build_core")
    return sqlite3.connect(f"file:{paths.DB}?mode=ro", uri=True)


def resolve(db, title):
    """Article page_id for a title or redirect title (spaces or underscores), else None."""
    title = title.replace("_", " ").strip()
    title = title[:1].upper() + title[1:]  # first letter is case-insensitive on Wikipedia
    row = db.execute("SELECT page_id FROM articles WHERE title = ?1 "
                     "UNION ALL SELECT target FROM redirects WHERE title = ?1", (title,)).fetchone()
    return row[0] if row else None


class Graph:
    """Link graph as CSR arrays indexed by articles.idx (memory-mapped, so loading is instant)."""

    def __init__(self):
        load = lambda name: np.load(paths.GRAPH / f"{name}.npy", mmap_mode="r")
        self.out_indptr, self.out_indices = load("out_indptr"), load("out_indices")
        self.in_indptr, self.in_indices = load("in_indptr"), load("in_indices")
        self.page_ids = load("idx_to_page_id")

    def __len__(self):
        return len(self.page_ids)

    def out_links(self, i):
        return self.out_indices[self.out_indptr[i]:self.out_indptr[i + 1]]

    def in_links(self, i):
        return self.in_indices[self.in_indptr[i]:self.in_indptr[i + 1]]

    def idx(self, page_id):
        i = int(np.searchsorted(self.page_ids, page_id))
        return i if i < len(self.page_ids) and self.page_ids[i] == page_id else None
