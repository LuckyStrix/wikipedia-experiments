"""Build the places index behind "What's notable near here?": every geotagged article, searchable by area.

Reads  data/wiki.sqlite, data/graph/{idx_to_page_id,in_indptr}.npy
       dumps/enwiki-<date>-geo_tags.sql.gz
       dumps/enwiki-<date>-langlinks.sql.gz  (optional: without it every article has langs = 0)
Writes data/geo.sqlite:
  places     one row per article that has a primary Earth coordinate: page_id, idx (row in the
             link graph), title, lat, lon, type (city, landmark, mountain, ...), pop (population
             where the tag gives one), dim (size in metres), country, region, langs (number of
             other-language editions), links (incoming links)
  places_rt  R*Tree over lat/lon for fast bounding-box queries
  meta       dump date, build time, row counts, and what was skipped

Only primary coordinates are used (a few articles carry several tags; the primary one is the
article's own location). Articles that have only secondary coordinates (a river's mouth, a
railway's stations, a battle's other sites) are skipped: there is no single sensible point for them.
Tags on the Moon, Mars and the like, and on pages that are not articles, are ignored.

Usage (from the repo root): python -m pipeline.build_geo
"""
import functools
import os
import re
import sqlite3
import time

import numpy as np

from wikiexp import paths
from wikiexp import progress as prog
from wikiexp.progress import progress
from wikiexp.sqldump import BLOCK, default_workers, header, map_blocks, parse_rows, rows

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE places (id INTEGER PRIMARY KEY, page_id INTEGER NOT NULL UNIQUE, idx INTEGER NOT NULL,
                     title TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL, type TEXT,
                     pop INTEGER, dim INTEGER, country TEXT, region TEXT,
                     langs INTEGER NOT NULL, links INTEGER NOT NULL);
CREATE VIRTUAL TABLE places_rt USING rtree(id, min_lat, max_lat, min_lon, max_lon);
"""
POST_SCHEMA = "CREATE INDEX places_type ON places(type);"

T0 = time.time()
# decompressed bytes per compressed byte of langlinks.sql.gz (measured); only used for the progress bar
LANGLINKS_EXPANSION = 4.0


def log(msg, pct=None):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)
    if pct is not None:
        progress(pct, msg.strip())


def split_type(raw):
    """gt_type like b'city(334,000' or b'river:us-oh' -> ('city', 334000) / ('river', None)."""
    if not raw:
        return None, None
    text = raw.decode("utf-8", "replace").strip().lower()
    m = re.search(r"\((\d[\d,]*)", text)
    pop = int(m.group(1).replace(",", "")) if m else None
    kind = re.split(r"[(:]", text, maxsplit=1)[0].strip()
    return kind or None, pop


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _text(v):
    return v.decode("utf-8", "replace") if v else None


def count_langs_block(block, ncols):
    """(page_ids, counts): how many langlinks rows each page has in this block (a row = one edition)."""
    ids = np.array([int(r[0]) for r in parse_rows(block, ncols, [0])], dtype=np.int64)
    return np.unique(ids, return_counts=True)


def read_langs(n_pages, workers):
    """uint16 array: number of other-language editions per page_id, or None if langlinks is missing."""
    path = paths.dump("langlinks.sql.gz")
    if not path.exists():
        log("langlinks.sql.gz not found: every article gets langs = 0")
        return None
    ncols = len(header(path)[0])
    langs = np.zeros(n_pages, dtype=np.uint16)
    total = path.stat().st_size * LANGLINKS_EXPANSION
    done = 0
    for ids, counts in map_blocks(path, functools.partial(count_langs_block, ncols=ncols), workers):
        keep = ids < n_pages
        langs[ids[keep]] += counts[keep].astype(np.uint16)
        done += BLOCK
        log(f"  langlinks: {done >> 20:,} MB read", 25 + 55 * min(done / total, 1))
    return langs


def main():
    prog.start("geo")
    workers = default_workers()
    page_ids = np.load(paths.GRAPH / "idx_to_page_id.npy")           # sorted; position = idx
    in_degree = np.diff(np.load(paths.GRAPH / "in_indptr.npy"))
    n_pages = int(page_ids[-1]) + 1

    log("reading geo_tags", 2)
    cols = ["gt_page_id", "gt_globe", "gt_primary", "gt_lat", "gt_lon", "gt_dim", "gt_type",
            "gt_country", "gt_region"]
    primary, secondary = {}, set()
    total = skipped_body = bad_coord = repeated = 0
    for pid, globe, prim, lat, lon, dim, typ, country, region in rows(paths.dump("geo_tags.sql.gz"), cols, workers=workers):
        total += 1
        if globe != b"earth":
            skipped_body += 1
            continue
        pid = int(pid)
        if prim != b"1":
            secondary.add(pid)
            continue
        try:
            la, lo = float(lat), float(lon)
        except (TypeError, ValueError):
            bad_coord += 1
            continue
        if not (-90 <= la <= 90 and -180 <= lo <= 180) or (la == 0 and lo == 0):   # (0, 0) is a placeholder, not a place
            bad_coord += 1
            continue
        if pid in primary:
            repeated += 1       # keep the first primary tag
            continue
        primary[pid] = (la, lo, _int(dim), typ, _text(country), _text(region))
    log(f"  {total:,} tags; {len(primary):,} primary Earth locations", 20)

    # keep only main-namespace articles (those in the link graph), find their idx
    cand = np.fromiter(primary.keys(), dtype=np.int64, count=len(primary))
    pos = np.searchsorted(page_ids, cand).clip(max=len(page_ids) - 1)
    is_article = page_ids[pos] == cand
    not_articles = int((~is_article).sum())
    keep_ids = cand[is_article]
    idx_of = dict(zip(keep_ids.tolist(), pos[is_article].tolist()))
    sec = np.fromiter((p for p in secondary if p not in primary), dtype=np.int64)
    sec_pos = np.searchsorted(page_ids, sec).clip(max=len(page_ids) - 1)
    secondary_only = int((page_ids[sec_pos] == sec).sum())
    del cand, pos
    log(f"  {len(idx_of):,} are articles; {secondary_only:,} articles have only secondary coordinates (skipped)", 22)

    log("reading titles", 23)
    src = sqlite3.connect(f"file:{paths.DB}?mode=ro", uri=True)
    titles = {}
    for pid, title in src.execute("SELECT page_id, title FROM articles"):
        if pid in idx_of:
            titles[pid] = title
    src.close()

    langs = read_langs(n_pages, workers)
    have_langs = langs is not None

    final = paths.DATA / "geo.sqlite"
    tmp = final.with_name(final.name + ".building")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    db.executescript(SCHEMA)
    log("writing places", 82)
    order = sorted(idx_of, key=lambda p: idx_of[p])
    place_rows, rt_rows = [], []
    for i, pid in enumerate(order, 1):
        la, lo, dim, typ, country, region = primary[pid]
        kind, pop = split_type(typ)
        idx = idx_of[pid]
        place_rows.append((i, pid, idx, titles[pid], la, lo, kind, pop, dim, country, region,
                           int(langs[pid]) if have_langs else 0, int(in_degree[idx])))
        rt_rows.append((i, la, la, lo, lo))
    db.executemany("INSERT INTO places VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", place_rows)
    db.executemany("INSERT INTO places_rt VALUES (?,?,?,?,?)", rt_rows)
    db.executescript(POST_SCHEMA)
    db.executemany("INSERT INTO meta VALUES (?,?)", [
        ("dump_date", paths.DUMP_DATE), ("built", time.strftime("%Y-%m-%d %H:%M")),
        ("places", str(len(order))), ("langlinks_used", str(have_langs)),
        ("geo_tags_rows", str(total)), ("skipped_other_bodies", str(skipped_body)),
        ("skipped_not_articles", str(not_articles)), ("skipped_secondary_only", str(secondary_only)),
        ("skipped_bad_coordinates", str(bad_coord)), ("repeated_primary_tags", str(repeated))])
    db.commit()
    db.close()
    os.replace(tmp, final)
    log(f"done: {len(order):,} places ({secondary_only:,} secondary-only articles skipped)", 100)


if __name__ == "__main__":
    main()
