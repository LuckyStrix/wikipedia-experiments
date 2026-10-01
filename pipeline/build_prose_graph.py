"""Build the prose link graph: only the links editors wrote in an article's running text.

Reads  the multistream dump + its index (pages-articles-multistream*.bz2), data/wiki.sqlite (titles
       and redirects) and data/graph/idx_to_page_id.npy (so both graphs share one idx numbering)
Writes data/prose_graph/, the same CSR layout as data/graph (and the same idx for every article):
  idx_to_page_id.npy, out_indptr.npy, out_indices.npy, in_indptr.npy, in_indices.npy
  out_order.npy    uint16, parallel to out_indices: where in the article the link first appears
                   (0 = its first link; capped at 65535). Rows of out_indices are sorted by target,
                   this tells you the reading order.
  lead_count.npy   int32 per article: how many of its links first appear before the first heading
                   (the lead). Edge e of article i is a lead link iff out_order[e] < lead_count[i].
  first_link.npy   int32 per article: its first link that is not in parentheses or italics, the
                   rule of the "Getting to Philosophy" game; -1 if it has none
  meta.json        counts (with the pagelinks totals for comparison), timings, build info
(see wikiexp/prose_links.py for exactly what counts as a prose link and how titles are resolved.)

Parses the dump in parallel worker processes (WIKI_WORKERS=n or --workers). The title lookup is built
once before they fork, so they share it. About 10 GB of RAM is plenty (the edge arrays are ~3 GB).

Usage (from the repo root): python -m pipeline.build_prose_graph [--workers n] [--limit N] [--out DIR]
"""
import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np

from pipeline import build_core
from wikiexp import paths
from wikiexp import progress as prog
from wikiexp import prose_links as P
from wikiexp import wikitext as W
from wikiexp.progress import progress

T0 = time.time()
FLUSH = 20_000          # articles per chunk of collected results
_LOOKUP = None          # set before the workers fork, so they all share it
_PAGE_IDS = None


def log(msg, pct=None):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)
    if pct is not None:
        progress(pct, msg.strip())


def extract(page: W.Page):
    """Worker-side: (idx, PageLinks) for one article; a page that trips the parser counts as empty
    rather than ending a multi-hour run."""
    i = int(np.searchsorted(_PAGE_IDS, page.page_id))
    if i >= len(_PAGE_IDS) or _PAGE_IDS[i] != page.page_id:
        return -1, P.EMPTY               # in the dump but not in wiki.sqlite: a dump mismatch
    try:
        return i, P.page_links(page.text, _LOOKUP, i)
    except Exception:
        return i, P.EMPTY


def sample_overlap(graph_dir: Path, out_indptr, out_indices, n_sample=5000, seed=1):
    """Share of prose links that are also in the pagelinks graph, over a random sample of articles.
    (Nearly all should be: the rendered page contains every link written in its text.)"""
    full_ptr = np.load(graph_dir / "out_indptr.npy", mmap_mode="r")
    full = np.load(graph_dir / "out_indices.npy", mmap_mode="r")
    have = np.flatnonzero(np.diff(out_indptr) > 0)
    rng = np.random.default_rng(seed)
    total = both = 0
    for i in rng.choice(have, min(n_sample, len(have)), replace=False):
        mine = out_indices[out_indptr[i]:out_indptr[i + 1]]
        theirs = full[full_ptr[i]:full_ptr[i + 1]]
        pos = np.minimum(np.searchsorted(theirs, mine), max(len(theirs) - 1, 0))
        total += len(mine)
        both += int((theirs[pos] == mine).sum()) if len(theirs) else 0
    return both / total if total else 1.0


def build(out_dir: Path, wt, graph_dir: Path | None = None, db_path: Path | None = None,
          workers: int | None = None, limit: int | None = None, say=log) -> Path:
    """Write the prose graph to `out_dir` (via a temporary folder, swapped in at the end)."""
    global _LOOKUP, _PAGE_IDS
    graph_dir = graph_dir or paths.GRAPH
    db_path = db_path or paths.DB
    tmp = out_dir.with_name(out_dir.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)

    _PAGE_IDS = np.load(graph_dir / "idx_to_page_id.npy")
    n = len(_PAGE_IDS)
    say(f"building the title lookup ({n:,} articles + redirects)", 1)
    t = time.time()
    _LOOKUP = P.TitleLookup.from_db(db_path, _PAGE_IDS, lambda m: say(m))
    say(f"  {len(_LOOKUP):,} titles in {time.time() - t:.0f}s", 4)

    counts = np.zeros(n, dtype=np.int64)
    lead_count = np.zeros(n, dtype=np.int32)
    first_link = np.full(n, -1, dtype=np.int32)
    chunks, cur = [], ([], [], [], [])
    done = seen = missing = raw = resolved = 0
    t0 = time.time()

    def flush():
        idxs, cnts, dsts, ords = cur
        if idxs:
            chunks.append((np.array(idxs, dtype=np.int64), np.array(cnts, dtype=np.int64),
                           np.concatenate(dsts), np.concatenate(ords)))
            for part in cur:
                part.clear()

    say(f"extracting prose links from {wt.n_streams:,} dump streams", 5)
    for _, _, (i, pl) in wt.iter_pages(fn=extract, workers=workers, limit=limit):
        seen += 1
        if i < 0:
            missing += 1
            continue
        raw += pl.raw
        resolved += pl.resolved
        counts[i] = len(pl.dst)
        lead_count[i] = pl.n_lead
        first_link[i] = pl.first
        if len(pl.dst):
            cur[0].append(i)
            cur[1].append(len(pl.dst))
            cur[2].append(pl.dst)
            cur[3].append(pl.order)
        if seen % FLUSH == 0:
            flush()
            rate = seen / (time.time() - t0)
            say(f"  {seen:,} / {n:,} articles ({rate:,.0f}/s, about {max(0, n - seen) / rate / 60:.0f} min left, "
                f"{int(counts.sum()):,} links)", 5 + 80 * min(1.0, seen / (limit or n)))
    flush()
    if missing:
        say(f"  note: {missing:,} dump pages are not in wiki.sqlite (different dump dates?)")

    say("placing links in article order", 86)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    total = int(indptr[-1])
    out_indices = np.empty(total, dtype=np.int32)
    out_order = np.empty(total, dtype=np.uint16)
    while chunks:
        idxs, cnts, dst, order = chunks.pop()
        pos = np.repeat(indptr[idxs] - (np.cumsum(cnts) - cnts), cnts) + np.arange(int(cnts.sum()))
        out_indices[pos] = dst
        out_order[pos] = order
        del pos, dst, order
    tmp.mkdir(parents=True)
    np.save(tmp / "out_order.npy", out_order)
    del out_order
    np.save(tmp / "lead_count.npy", lead_count)
    np.save(tmp / "first_link.npy", first_link)

    src = np.repeat(np.arange(n, dtype=np.int32), counts)
    build_core.write_graph(src, out_indices, _PAGE_IDS, tmp, lambda m, p=None: say(m, 90))
    del src

    full_links = int(np.load(graph_dir / "out_indptr.npy", mmap_mode="r")[-1])
    in_pagelinks = sample_overlap(graph_dir, indptr, out_indices)
    out_deg = counts
    meta = {
        "articles": n, "articles_parsed": seen - missing, "dump_pages_not_in_db": missing,
        "links": total, "pagelinks_links": full_links,
        "share_of_pagelinks": round(total / full_links, 4) if full_links else None,
        "articles_with_links": int((out_deg > 0).sum()),
        "articles_with_inlinks": int((np.diff(np.load(tmp / "in_indptr.npy", mmap_mode="r")) > 0).sum()),
        "articles_with_first_link": int((first_link >= 0).sum()),
        "lead_links": int(lead_count.sum()),
        "links_written_before_dedupe": raw, "of_which_resolved_to_articles": resolved,
        "sampled_share_also_in_pagelinks": round(in_pagelinks, 4),
        "max_links_in_one_article": int(out_deg.max()) if n else 0,
        "dump_date": paths.DUMP_DATE, "built": time.strftime("%Y-%m-%d %H:%M"),
        "seconds": round(time.time() - T0, 1),
        "peak_ram_gb": round((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), 2),
        "peak_worker_ram_gb": round((resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6), 2),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))     # last: marks the folder complete
    old = out_dir.with_name(out_dir.name + ".old")
    if out_dir.exists():
        os.replace(out_dir, old)
    os.replace(tmp, out_dir)
    shutil.rmtree(old, ignore_errors=True)
    say(f"done: {total:,} prose links ({100 * total / full_links:.0f}% of the {full_links:,} in pagelinks) "
        f"in {(time.time() - T0) / 60:.1f} min", 100)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, help="parser processes (default: $WIKI_WORKERS or cores - 2, max 8)")
    ap.add_argument("--limit", type=int, help="only the first N articles in dump order (for testing)")
    ap.add_argument("--out", help="output folder (default: <data>/prose_graph)")
    args = ap.parse_args()
    prog.start("prose_graph")
    say = lambda m, p=None: log(m, p)
    say("opening the dump index (first time: parses 25M lines, about a minute)", 0)
    wt = W.WikiText(progress=lambda m: log("  " + m))
    build(Path(args.out) if args.out else paths.DATA / "prose_graph", wt, workers=args.workers,
          limit=args.limit, say=say)


if __name__ == "__main__":
    main()
