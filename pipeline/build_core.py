"""Build the core Wikipedia database from the SQL dumps.

Reads  dumps/enwiki-<date>-{page,redirect,linktarget,pagelinks}.sql.gz
Writes data/wiki.sqlite  - articles, redirects and links tables (see SCHEMA below)
       data/graph/*.npy  - the same link graph as CSR arrays, for fast in-memory search

Only real articles (namespace 0, not redirects) are kept. Links that point at a redirect
are resolved to the article it redirects to; duplicate links and self-links are dropped.

Usage (from the repo root): python -m pipeline.build_core
"""
import argparse
import os
import sqlite3
import time

import numpy as np

from wikiexp import paths
from wikiexp import progress as prog
from wikiexp.progress import progress
from wikiexp.sqldump import default_workers, header, map_blocks, parse_int_block, rows, title_text, verify

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- idx is the article's row number in data/graph/*.npy
CREATE TABLE articles (page_id INTEGER PRIMARY KEY, idx INTEGER NOT NULL UNIQUE,
                       title TEXT NOT NULL UNIQUE, length INTEGER NOT NULL);
-- target is the article's page_id, with redirect chains already followed
CREATE TABLE redirects (page_id INTEGER PRIMARY KEY, title TEXT NOT NULL UNIQUE,
                        target INTEGER NOT NULL);
CREATE TABLE links (src INTEGER NOT NULL, dst INTEGER NOT NULL,
                    PRIMARY KEY (src, dst)) WITHOUT ROWID;
"""
POST_SCHEMA = """
CREATE INDEX redirects_target ON redirects(target);
CREATE INDEX links_dst ON links(dst);
"""
INPUTS = ["page.sql.gz", "redirect.sql.gz", "linktarget.sql.gz", "pagelinks.sql.gz"]

T0 = time.time()
# pagelinks rows per compressed byte, measured on the 2026-09 dump; only used for the progress bar
LINKS_PER_GZ_BYTE = 0.235
IN_LINKS_CHUNK = 50_000_000


def log(msg, pct=None):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)
    if pct is not None:
        progress(pct, msg.strip())


def load_pages():
    log("reading page table", 3)
    path = paths.dump("page.sql.gz")
    _, auto_inc = header(path)
    title_to_id = {}
    art_ids, art_titles, art_lens, redirect_ids = [], [], [], []
    for pid, ns, title, is_redirect, length in rows(
            path, ["page_id", "page_namespace", "page_title", "page_is_redirect", "page_len"],
            where=("page_namespace", b"0")):
        if ns != b"0":
            continue
        pid = int(pid)
        title_to_id[title] = pid
        if is_redirect == b"1":
            redirect_ids.append(pid)
        else:
            art_ids.append(pid)
            art_titles.append(title)
            art_lens.append(int(length))
    max_pid = max(auto_inc or 0, max(title_to_id.values()) + 1)
    order = np.argsort(np.array(art_ids, dtype=np.int64))
    art_ids = np.array(art_ids, dtype=np.int64)[order]
    art_titles = [art_titles[i] for i in order]
    art_lens = np.array(art_lens, dtype=np.int64)[order]
    page_to_idx = np.full(max_pid + 1, -1, dtype=np.int32)
    page_to_idx[art_ids] = np.arange(len(art_ids), dtype=np.int32)
    log(f"  {len(art_ids):,} articles, {len(redirect_ids):,} redirects", 22)
    return title_to_id, art_ids, art_titles, art_lens, set(redirect_ids), page_to_idx


def resolve_redirects(title_to_id, redirect_ids, page_to_idx):
    """Return {redirect page_id: article idx}, following chains of redirects."""
    log("reading redirect table", 23)
    hop = {}
    for rd_from, ns, title, interwiki in rows(
            paths.dump("redirect.sql.gz"), ["rd_from", "rd_namespace", "rd_title", "rd_interwiki"],
            where=("rd_namespace", b"0")):
        rd_from = int(rd_from)
        if ns == b"0" and not interwiki and rd_from in redirect_ids:
            target = title_to_id.get(title)
            if target is not None:
                hop[rd_from] = target
    resolved = {}
    for start in hop:
        pid, seen = start, 0
        while pid in hop and seen < 10:
            pid, seen = hop[pid], seen + 1
        if page_to_idx[pid] >= 0:
            resolved[start] = int(page_to_idx[pid])
    log(f"  {len(resolved):,} redirects resolved, {len(redirect_ids) - len(resolved):,} broken or dropped")
    return resolved


def map_linktargets(title_to_id, redirect_to_idx, page_to_idx):
    """Array mapping linktarget id -> article idx (-1 if not an article)."""
    log("reading linktarget table", 26)
    path = paths.dump("linktarget.sql.gz")
    _, auto_inc = header(path)
    lt_to_idx = np.full(auto_inc + 1, -1, dtype=np.int32)
    ids, idxs = [], []

    def flush():
        a = np.array(ids, dtype=np.int64)
        if len(a) and a.max() >= len(lt_to_idx):
            raise RuntimeError("linktarget id beyond AUTO_INCREMENT")
        lt_to_idx[a] = np.array(idxs, dtype=np.int32)
        ids.clear(), idxs.clear()

    for lt_id, ns, title in rows(path, ["lt_id", "lt_namespace", "lt_title"], where=("lt_namespace", b"0")):
        if ns != b"0":
            continue
        pid = title_to_id.get(title)
        if pid is None:
            continue  # red link
        idx = page_to_idx[pid] if page_to_idx[pid] >= 0 else redirect_to_idx.get(pid, -1)
        if idx >= 0:
            ids.append(int(lt_id))
            idxs.append(idx)
            if len(ids) >= 5_000_000:
                flush()
    flush()
    log(f"  {np.count_nonzero(lt_to_idx >= 0):,} link targets point at articles")
    return lt_to_idx


# lookup tables for pagelinks workers; set before the workers fork, so they're shared, not copied
_LOOKUP = {}


def _links_block(block):
    """One block of pagelinks -> (raw row count, int64 keys of article->article links)."""
    page_to_idx, lt_to_idx = _LOOKUP["page_to_idx"], _LOOKUP["lt_to_idx"]
    arr = parse_int_block(block, 3)
    n = len(arr)
    arr = arr[(arr[:, 1] == 0) & (arr[:, 0] < len(page_to_idx)) & (arr[:, 2] < len(lt_to_idx))]
    src = page_to_idx[arr[:, 0]]
    dst = lt_to_idx[arr[:, 2]]
    ok = (src >= 0) & (dst >= 0) & (src != dst)
    return n, (src[ok].astype(np.int64) << 32) | dst[ok].astype(np.int64)


def load_links(page_to_idx, lt_to_idx):
    """Sorted, de-duplicated int64 keys (src_idx << 32 | dst_idx)."""
    log(f"reading pagelinks table (the long step; {default_workers()} parser processes)", 38)
    path = paths.dump("pagelinks.sql.gz")
    cols, _ = header(path)
    if cols != ["pl_from", "pl_from_namespace", "pl_target_id"]:
        raise RuntimeError(f"unexpected pagelinks columns: {cols}")
    parts, total = [], 0
    expected = path.stat().st_size * LINKS_PER_GZ_BYTE
    _LOOKUP.update(page_to_idx=page_to_idx, lt_to_idx=lt_to_idx)
    for n, keys in map_blocks(path, _links_block):
        total += n
        parts.append(keys)
        if len(parts) % 50 == 0:
            log(f"  {total:,} raw links read", 38 + min(29, 29 * total / expected))
    _LOOKUP.clear()
    keys = np.concatenate(parts)
    del parts
    if len(keys) == 0:
        raise RuntimeError("no article links found - are the dump files from the same date?")
    log(f"  {total:,} raw links -> {len(keys):,} article links; sorting", 68)
    keys.sort()
    keep = np.empty(len(keys), dtype=bool)
    keep[0] = True
    np.not_equal(keys[1:], keys[:-1], out=keep[1:])
    keys = keys[keep]
    log(f"  {len(keys):,} unique links")
    return keys


def split_keys(keys):
    """(src_idx, dst_idx) int32 arrays from sorted keys; frees the keys as it goes."""
    src = (keys >> 32).astype(np.int32)
    keys &= 0xFFFFFFFF
    dst = keys.astype(np.int32)
    return src, dst


def write_graph(src, dst, art_ids):
    """CSR arrays for outgoing links (edges are already sorted by src) and incoming links.

    Incoming links are placed with a chunked counting sort rather than argsort, which would need
    an 8-byte index per edge (~6 GB for enwiki) on top of everything else.
    """
    log("writing graph arrays", 72)
    out, n = paths.GRAPH, len(art_ids)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "idx_to_page_id.npy", art_ids)
    np.save(out / "out_indptr.npy", np.concatenate(([0], np.cumsum(np.bincount(src, minlength=n)))))
    np.save(out / "out_indices.npy", dst)
    in_indptr = np.concatenate(([0], np.cumsum(np.bincount(dst, minlength=n))))
    np.save(out / "in_indptr.npy", in_indptr)
    in_indices = np.empty(len(dst), dtype=np.int32)
    fill = in_indptr[:-1].copy()   # next free slot for each destination
    step = IN_LINKS_CHUNK
    for i in range(0, len(dst), step):
        d, s = dst[i:i + step], src[i:i + step]
        order = np.argsort(d, kind="stable")
        ds = d[order]
        # rank of each edge among same-destination edges in this chunk
        starts = np.flatnonzero(np.concatenate(([True], ds[1:] != ds[:-1])))
        rank = np.arange(len(ds)) - np.repeat(starts, np.diff(np.append(starts, len(ds))))
        in_indices[fill[ds] + rank] = s[order]
        fill += np.bincount(d, minlength=n)
    np.save(out / "in_indices.npy", in_indices)


def write_sqlite(art_ids, art_titles, art_lens, redirect_titles, src, dst, stats):
    final = paths.DB
    tmp = final.with_name(final.name + ".building")
    tmp.unlink(missing_ok=True)
    os.environ["SQLITE_TMPDIR"] = str(paths.DATA)  # index builds spill to disk; /tmp may be too small
    db = sqlite3.connect(tmp)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-2000000;")
    db.executescript(SCHEMA)
    log("writing articles and redirects", 76)
    db.executemany("INSERT INTO articles VALUES (?,?,?,?)",
                   zip(art_ids.tolist(), range(len(art_ids)), map(title_text, art_titles), art_lens.tolist()))
    db.executemany("INSERT INTO redirects VALUES (?,?,?)",
                   ((pid, title_text(t), int(art_ids[idx])) for pid, t, idx in redirect_titles))
    log("writing links")
    step = 10_000_000
    for i in range(0, len(src), step):
        db.executemany("INSERT INTO links VALUES (?,?)",
                       zip(art_ids[src[i:i + step]].tolist(), art_ids[dst[i:i + step]].tolist()))
        db.commit()
        log(f"  {min(i + step, len(src)):,} / {len(src):,} links written",
            80 + 16 * min(i + step, len(src)) / len(src))
    log("indexing", 97)
    db.executescript(POST_SCHEMA)
    db.executemany("INSERT INTO meta VALUES (?,?)",
                   [("dump_date", paths.DUMP_DATE), ("built", time.strftime("%Y-%m-%d %H:%M"))]
                   + [(k, str(v)) for k, v in stats.items()])
    db.commit()
    db.execute("ANALYZE")
    db.close()
    os.replace(tmp, final)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-verify", action="store_true", help="don't check dump checksums first")
    args = ap.parse_args()
    paths.DATA.mkdir(parents=True, exist_ok=True)
    prog.start("core")

    if not args.skip_verify:
        log("verifying dump checksums", 1)
        verify(INPUTS)

    title_to_id, art_ids, art_titles, art_lens, redirect_ids, page_to_idx = load_pages()
    redirect_to_idx = resolve_redirects(title_to_id, redirect_ids, page_to_idx)
    lt_to_idx = map_linktargets(title_to_id, redirect_to_idx, page_to_idx)
    id_to_title = {pid: t for t, pid in title_to_id.items() if pid in redirect_to_idx}
    redirect_titles = [(pid, id_to_title[pid], idx) for pid, idx in redirect_to_idx.items()]
    del title_to_id, id_to_title, redirect_ids, redirect_to_idx

    keys = load_links(page_to_idx, lt_to_idx)
    del lt_to_idx, page_to_idx
    src, dst = split_keys(keys)
    del keys
    write_graph(src, dst, art_ids)

    stats = {"articles": len(art_ids), "redirects": len(redirect_titles), "links": len(src)}
    write_sqlite(art_ids, art_titles, art_lens, redirect_titles, src, dst, stats)
    log(f"done: {stats}")


if __name__ == "__main__":
    main()
