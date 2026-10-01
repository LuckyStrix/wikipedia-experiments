"""Build the keyword (full-text) index over every article's title and introduction.

Reads  data/text/leads.sqlite (the `text` stage)
Writes data/text/fts.sqlite:
  fts   SQLite FTS5 table (contentless) over title + lead with the porter stemmer and unicode61
        tokenizer (accents folded), rowid = leads.rank (0 = most linked-to article), so a hit maps back
        to leads.sqlite by its primary key and the popularity order is free
  meta  key/value (dump date, counts, which leads file it was built from)
Disambiguation pages are left out (their titles are still found by the title index); articles with an
empty lead are indexed by title alone.

Queries rank with bm25 weighting the title above the lead (wikiexp.keyword.TITLE_WEIGHT), which is
why title and lead are separate columns. This index is what makes the RAG assistant work before the
embeddings cover every article: "who designed the Tacoma Narrows Bridge" finds the bridge by its words
even when the bridge is not among the embedded articles.

Usage (from the repo root): python -m pipeline.build_keyword [--limit N] [--no-optimize]
"""
import argparse
import os
import sqlite3
import time
from pathlib import Path

from wikiexp import paths
from wikiexp import progress as prog
from wikiexp.progress import progress

T0 = time.time()
CHUNK = 250_000          # articles per transaction (each flushes one segment; optimize merges them)

TOKENIZER = "porter unicode61 remove_diacritics 2"
SCHEMA = f"""
CREATE VIRTUAL TABLE fts USING fts5(title, lead, content='', tokenize='{TOKENIZER}');
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def log(msg, pct=None):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)
    if pct is not None:
        progress(pct, msg.strip())


def build(data_dir: Path, limit: int | None = None, optimize: bool = True, say=log) -> Path:
    """Write <data_dir>/text/fts.sqlite from <data_dir>/text/leads.sqlite; returns its path."""
    text = Path(data_dir) / "text"
    leads = text / "leads.sqlite"
    if not leads.exists():
        raise FileNotFoundError(f"{leads} not found - run the 'text' stage first")
    final = text / "fts.sqlite"
    tmp = final.with_name(final.name + ".building")
    tmp.unlink(missing_ok=True)
    os.environ.setdefault("SQLITE_TMPDIR", str(text))

    db = sqlite3.connect(tmp, uri=True)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-1000000;")
    db.executescript(SCHEMA)
    db.execute("ATTACH ? AS src", (f"file:{leads}?mode=ro",))
    total = db.execute("SELECT count(*) FROM src.leads" + (" WHERE rank < ?" if limit else ""),
                       (limit,) if limit else ()).fetchone()[0]
    leads_meta = dict(db.execute("SELECT key, value FROM src.meta"))
    say(f"indexing {total:,} articles", 2)
    t0 = time.time()
    done = indexed = 0
    top = limit or total
    for lo in range(0, top, CHUNK):
        hi = min(lo + CHUNK, top)
        cur = db.execute("INSERT INTO fts(rowid, title, lead) SELECT rank, title, lead FROM src.leads "
                         "WHERE rank >= ? AND rank < ? AND disambig = 0 ORDER BY rank", (lo, hi))
        indexed += cur.rowcount
        db.commit()
        done = hi
        rate = done / (time.time() - t0)
        say(f"  {done:,} / {top:,} articles ({rate:,.0f}/s, about {max(0, top - done) / rate / 60:.0f} min left)",
            2 + 80 * done / top)
    if optimize:
        say("merging the index into one segment (the last step)", 85)
        db.execute("INSERT INTO fts(fts) VALUES ('optimize')")
        db.commit()
    db.executemany("INSERT INTO meta VALUES (?,?)", [
        ("dump_date", paths.DUMP_DATE), ("built", time.strftime("%Y-%m-%d %H:%M")),
        ("articles", str(indexed)), ("tokenizer", TOKENIZER),
        ("leads_built", leads_meta.get("built", "")), ("leads_articles", leads_meta.get("articles", ""))])
    db.commit()
    db.execute("DETACH src")
    db.close()
    os.replace(tmp, final)
    say(f"done: {indexed:,} articles, {final.stat().st_size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min", 100)
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="index only the N most popular articles (for testing)")
    ap.add_argument("--no-optimize", action="store_true", help="skip the final merge (faster build, slower queries)")
    ap.add_argument("--out", help="data folder (default: the configured one)")
    args = ap.parse_args()
    prog.start("keyword")
    build(Path(args.out) if args.out else paths.DATA, args.limit, not args.no_optimize)


if __name__ == "__main__":
    main()
