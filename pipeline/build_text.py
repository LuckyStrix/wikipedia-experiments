"""Extract the clean introduction of every article, for semantic search and other text projects.

Reads  the multistream dump + its index (pages-articles-multistream*.bz2), data/wiki.sqlite and
       data/graph/ (for popularity)
Writes data/text/leads.sqlite:
  leads  rank INTEGER PRIMARY KEY   0 = most linked-to article, so `ORDER BY rank LIMIT n` reads the
                                    n best-known articles (and a partial embed covers them first)
         page_id INTEGER UNIQUE     article id (as in wiki.sqlite)
         idx INTEGER                position in the link graph (articles.idx)
         title TEXT, lead TEXT      lead = intro before the first heading, as plain text, cut at a
                                    sentence end to --lead-chars
         disambig INTEGER           1 for disambiguation pages and set-index pages (kept, flagged)
  meta   key/value (dump date, lead length, counts)
Also caches the dump's index under data/text/ (index_*.npy) so single articles can be fetched later
with wikiexp.wikitext.

Parses the dump in parallel worker processes (WIKI_WORKERS=n to change; plain loop on Windows).

Usage (from the repo root): python -m pipeline.build_text [--lead-chars 1200] [--workers n] [--limit N]
"""
import argparse
import functools
import os
import sqlite3
import time

import numpy as np

from wikiexp import paths
from wikiexp import progress as prog
from wikiexp import wikitext as W
from wikiexp.progress import progress

T0 = time.time()


def log(msg, pct=None):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)
    if pct is not None:
        progress(pct, msg.strip())


def extract(page: W.Page, lead_chars: int):
    """Worker-side: (lead, disambig flag) of one page. A page the cleaner chokes on gets an empty
    lead rather than ending a multi-hour run."""
    try:
        return W.lead(page.text, lead_chars), int(W.is_disambiguation(page.title, page.text))
    except Exception:
        return "", 0


def popularity_ranks(graph_dir=None):
    """(idx_to_page_id, rank_of_idx): rank 0 is the article with the most incoming links."""
    graph_dir = graph_dir or paths.GRAPH
    page_ids = np.load(graph_dir / "idx_to_page_id.npy")
    in_degree = np.diff(np.load(graph_dir / "in_indptr.npy"))
    order = np.argsort(-in_degree, kind="stable")          # ties: lower idx (= lower page_id) first
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return page_ids, rank


RAW_SCHEMA = "CREATE TABLE raw (page_id INTEGER PRIMARY KEY, rank INTEGER, idx INTEGER, title TEXT, lead TEXT, disambig INTEGER)"
SCHEMA = """
CREATE TABLE leads (rank INTEGER PRIMARY KEY, page_id INTEGER NOT NULL UNIQUE, idx INTEGER NOT NULL,
                    title TEXT NOT NULL, lead TEXT NOT NULL, disambig INTEGER NOT NULL);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def build(out_dir, lead_chars=1200, workers=None, limit=None, wt=None, page_ids=None, rank=None,
          say=log):
    """Write <out_dir>/leads.sqlite. `wt`, `page_ids`, `rank` can be passed in (tests do)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "leads.sqlite"
    tmp = final.with_name(final.name + ".building")
    raw_path = out_dir / "leads.raw.building"
    for p in (tmp, raw_path):
        p.unlink(missing_ok=True)

    if page_ids is None:
        say("reading popularity from the link graph", 1)
        page_ids, rank = popularity_ranks()
    if wt is None:
        say("opening the dump index (first time: parses 25M lines, about a minute)", 2)
        wt = W.WikiText(progress=lambda m: say("  " + m))
    total = limit or len(page_ids)

    raw = sqlite3.connect(raw_path)
    raw.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-500000;")
    raw.execute(RAW_SCHEMA)
    say(f"extracting leads from {wt.n_streams:,} dump streams", 3)
    done = disambig = missing = 0
    batch, t0 = [], time.time()
    fn = functools.partial(extract, lead_chars=lead_chars)
    for pid, title, (text, dab) in wt.iter_pages(fn=fn, workers=workers, limit=limit):
        i = int(np.searchsorted(page_ids, pid))
        if i >= len(page_ids) or page_ids[i] != pid:
            missing += 1                       # in the dump but not in wiki.sqlite: dump mismatch
            continue
        batch.append((pid, int(rank[i]), i, title, text, dab))
        disambig += dab
        done += 1
        if len(batch) >= 20000:
            raw.executemany("INSERT INTO raw VALUES (?,?,?,?,?,?)", batch)
            batch.clear()
            rate = done / (time.time() - t0)
            say(f"  {done:,} / {total:,} articles ({rate:,.0f}/s, about "
                f"{max(0, total - done) / rate / 60:.0f} min left)", 3 + 87 * min(1, done / total))
    raw.executemany("INSERT INTO raw VALUES (?,?,?,?,?,?)", batch)
    raw.commit()
    if missing:
        say(f"  note: {missing:,} dump pages are not in wiki.sqlite (different dump dates?)")

    say(f"ordering {done:,} leads by popularity", 91)
    db = sqlite3.connect(tmp)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-500000;")
    db.executescript(SCHEMA)
    db.execute("ATTACH ? AS raw", (str(raw_path),))
    db.execute("CREATE INDEX raw.raw_rank ON raw(rank)")
    # ranks are unique but have gaps (articles missing from the dump); renumber so rank = row number
    db.execute("INSERT INTO leads SELECT row_number() OVER (ORDER BY rank) - 1, page_id, idx, title, lead, disambig "
               "FROM raw ORDER BY rank")
    db.executemany("INSERT INTO meta VALUES (?,?)", [
        ("dump_date", paths.DUMP_DATE), ("built", time.strftime("%Y-%m-%d %H:%M")),
        ("lead_chars", str(lead_chars)), ("articles", str(done)), ("disambig", str(disambig)),
        ("empty_leads", str(db.execute("SELECT count(*) FROM leads WHERE lead = ''").fetchone()[0]))])
    db.commit()
    db.execute("DETACH raw")
    db.close()
    raw.close()
    raw_path.unlink(missing_ok=True)
    os.replace(tmp, final)
    say(f"done: {done:,} articles ({disambig:,} disambiguation) in {(time.time() - t0) / 60:.1f} min", 100)
    return final


def main():
    from pipeline.settings import Settings
    cfg = Settings.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lead-chars", type=int, default=int(cfg["text_lead_chars"]),
                    help="longest lead to keep (default: setting, 1200)")
    ap.add_argument("--workers", type=int, help="parser processes (default: $WIKI_WORKERS or cores - 2, max 8)")
    ap.add_argument("--limit", type=int, help="only the first N articles in dump order (for testing)")
    ap.add_argument("--out", help="output folder (default: <data>/text)")
    args = ap.parse_args()
    prog.start("text")
    from pathlib import Path
    build(Path(args.out) if args.out else paths.DATA / "text", args.lead_chars, args.workers, args.limit)


if __name__ == "__main__":
    main()
