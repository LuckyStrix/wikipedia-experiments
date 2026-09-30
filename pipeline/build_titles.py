"""Build the title search index used for autocomplete and "did you mean" suggestions.

Reads  data/wiki.sqlite and data/graph/in_indptr.npy
Writes data/titles.sqlite:
  titles      every article title (and redirect title), in popularity order: id 1 is the article
              with the most incoming links. Searching in id order therefore finds popular
              articles first, and can stop early.
  titles_fts  trigram full-text index over titles, for fast substring and prefix search

Usage (from the repo root): python -m pipeline.build_titles [--no-redirects]
"""
import argparse
import os
import sqlite3
import time

import numpy as np

from wikiexp import paths
from wikiexp import progress as prog
from wikiexp.progress import progress

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- target = the article's page_id; links = incoming links to that article; redirect = 1 for aliases
CREATE TABLE titles (id INTEGER PRIMARY KEY, title TEXT NOT NULL, target INTEGER NOT NULL,
                     links INTEGER NOT NULL, redirect INTEGER NOT NULL);
"""
POST_SCHEMA = """
CREATE INDEX titles_nocase ON titles(title COLLATE NOCASE);
CREATE INDEX titles_article ON titles(target) WHERE redirect = 0;
CREATE VIRTUAL TABLE titles_fts USING fts5(title, content='titles', content_rowid='id',
                                           tokenize='trigram');
INSERT INTO titles_fts(titles_fts) VALUES ('rebuild');
"""

T0 = time.time()


def log(msg, pct=None):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)
    if pct is not None:
        progress(pct, msg.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-redirects", action="store_true", help="index article titles only")
    args = ap.parse_args()
    prog.start("titles")

    src = sqlite3.connect(f"file:{paths.DB}?mode=ro", uri=True)
    indptr = np.load(paths.GRAPH / "in_indptr.npy")
    in_degree = np.diff(indptr)

    log("reading article titles", 2)
    rows = src.execute("SELECT page_id, idx, title FROM articles").fetchall()
    titles = [r[2] for r in rows]
    targets = np.array([r[0] for r in rows], dtype=np.int64)
    links = in_degree[np.array([r[1] for r in rows], dtype=np.int64)]
    redirect = np.zeros(len(rows), dtype=np.int8)
    idx_of = {r[0]: r[1] for r in rows}
    del rows
    if not args.no_redirects:
        log("reading redirect titles", 15)
        red = src.execute("SELECT title, target FROM redirects").fetchall()
        titles += [r[0] for r in red]
        red_targets = np.array([r[1] for r in red], dtype=np.int64)
        targets = np.concatenate([targets, red_targets])
        links = np.concatenate([links, in_degree[np.array([idx_of[t] for t in red_targets.tolist()])]])
        redirect = np.concatenate([redirect, np.ones(len(red), dtype=np.int8)])
        del red, red_targets
    del idx_of
    log(f"  {len(titles):,} titles; sorting by popularity", 30)
    # most-linked first; an article before its redirects; shorter titles first
    order = np.lexsort((np.array([len(t) for t in titles]), redirect, -links))

    final = paths.DATA / "titles.sqlite"
    tmp = final.with_name(final.name + ".building")
    tmp.unlink(missing_ok=True)
    os.environ["SQLITE_TMPDIR"] = str(paths.DATA)
    db = sqlite3.connect(tmp)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-1000000;")
    db.executescript(SCHEMA)
    log("writing titles", 35)
    step = 1_000_000
    for i in range(0, len(order), step):
        chunk = order[i:i + step]
        db.executemany("INSERT INTO titles VALUES (?,?,?,?,?)",
                       zip(range(i + 1, i + 1 + len(chunk)), (titles[j] for j in chunk.tolist()),
                           targets[chunk].tolist(), links[chunk].tolist(), redirect[chunk].tolist()))
        log(f"  {min(i + step, len(order)):,} / {len(order):,} titles written",
            35 + 25 * min(i + step, len(order)) / len(order))
    db.commit()
    log("building search indexes (the long step)", 62)
    db.executescript(POST_SCHEMA)
    db.executemany("INSERT INTO meta VALUES (?,?)", [
        ("dump_date", paths.DUMP_DATE), ("built", time.strftime("%Y-%m-%d %H:%M")),
        ("titles", str(len(order))), ("redirects_included", str(not args.no_redirects))])
    db.commit()
    log("optimizing", 95)
    db.execute("INSERT INTO titles_fts(titles_fts) VALUES ('optimize')")
    db.commit()
    db.close()
    os.replace(tmp, final)
    log(f"done: {len(order):,} titles", 100)


if __name__ == "__main__":
    main()
