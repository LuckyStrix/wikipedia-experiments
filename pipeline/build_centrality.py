"""Compute how central each article is in the link graph: PageRank and "reverse" PageRank.

Reads  data/graph/*.npy (from build_core)
Writes data/centrality/ (all arrays indexed by graph idx, i.e. articles.idx):
  pagerank.npy          float32; a probability distribution (sums to 1). Importance flows along
                        links: an article is important if important articles link to it.
  rank.npy, order.npy   int32 rank (1 = highest PageRank) per idx, and the idx list sorted by it
  indegree_rank.npy, indegree_order.npy   the same for the number of incoming links
  reverse_pagerank.npy, reverse_rank.npy, reverse_order.npy
                        PageRank of the graph with every link reversed: high for "gateway" articles
                        (lists, hubs, overviews) that link out to many important articles
  meta.json             damping, iterations, residual, build time, dump date
With --graph DIR (the prose link graph from build_prose_graph) it also writes, for that graph:
  prose_pagerank.npy    PageRank where only links written in article text count (not the ones
                        templates generate: citations, infoboxes, navboxes). This is the better
                        "how notable is this article" score; the full-graph PageRank ranks ISBN first
  prose_rank.npy, prose_order.npy   its rank and order, like rank.npy / order.npy
  prose_meta.json       damping, iterations, residual and the graph it was computed on

Method: power iteration. Each step, every article hands its score out equally over its outgoing
links; articles with no outgoing links ("dangling") spread theirs uniformly over all articles; and
a fraction (1 - damping) is spread uniformly too. It stops when the L1 change drops below --tol
(default 1e-9) or after --max-iter steps; the residual of every step is logged and drives the
progress bar.

Memory: the link arrays are memory-mapped and read in chunks (a gather plus np.add.reduceat per
chunk), so no sparse matrix is ever built; peak RAM is about 1 GB beyond the page cache. Scores
are kept in float64 while iterating (float32 rounding would stop the residual short of 1e-9) and
saved as float32.

Usage (from the repo root): python -m pipeline.build_centrality [--damping 0.85] [--tol 1e-9]
                            [--max-iter 200] [--no-reverse] [--graph data/prose_graph [--prose-only]]
--prose-only adds the prose metric to an existing build without recomputing (or touching) the rest.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from wikiexp import paths
from wikiexp import progress as prog
from wikiexp.progress import progress

T0 = time.time()
CHUNK_EDGES = 32_000_000   # edges gathered per step: ~256 MB of float64 temporaries


def log(msg, pct=None):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)
    if pct is not None:
        progress(pct, msg.strip())


def pull_sum(indptr, indices, x, out):
    """out[v] = sum of x[u] over the u listed in indices[indptr[v]:indptr[v+1]], in chunks."""
    n = len(indptr) - 1
    out[:] = 0.0
    start = 0
    while start < n:
        # rows start..stop-1 hold about CHUNK_EDGES edges (at least one row)
        stop = int(np.searchsorted(indptr, indptr[start] + CHUNK_EDGES, side="right")) - 1
        stop = min(n, max(stop, start + 1))
        lo, hi = int(indptr[start]), int(indptr[stop])
        if hi > lo:
            gathered = x[indices[lo:hi]]
            ptr = np.asarray(indptr[start:stop + 1], dtype=np.int64) - lo
            rows = np.flatnonzero(ptr[1:] > ptr[:-1])   # reduceat misbehaves on empty rows
            out[start + rows] = np.add.reduceat(gathered, ptr[rows])
        start = stop


def pagerank(pull_indptr, pull_indices, norm_degree, damping=0.85, tol=1e-9, max_iter=100,
             report=None):
    """PageRank where score flows from u to v for every u in v's pull list.

    pull_* is the CSR of "who points at v"; norm_degree[u] is how many links u hands its score
    to (zero = dangling). Returns (scores float64, iterations, residual).
    """
    n = len(norm_degree)
    deg = np.asarray(norm_degree, dtype=np.float64)
    dangling = deg == 0
    inv = np.zeros(n)
    np.divide(1.0, deg, out=inv, where=~dangling)
    pr = np.full(n, 1.0 / n)
    x, new = np.empty(n), np.empty(n)
    residual, it = float("inf"), 0
    for it in range(1, max_iter + 1):
        np.multiply(pr, inv, out=x)
        pull_sum(pull_indptr, pull_indices, x, new)
        base = (1.0 - damping) / n + damping * pr[dangling].sum() / n
        new *= damping
        new += base
        new /= new.sum()     # guards against drift; the sum is 1 up to rounding
        residual = float(np.abs(new - pr).sum())
        pr, new = new, pr
        if report:
            report(it, residual)
        if residual < tol:
            break
    return pr, it, residual


def ranking(score):
    """(rank, order): rank[i] is 1 for the highest score; order lists indices best first.
    Ties go to the lower index, so the result is deterministic."""
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable").astype(np.int32)
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return rank, order


def make_reporter(label, lo, hi, tol):
    """Progress callback: the residual falls roughly geometrically, so map log10(residual)
    between its first value and the tolerance onto lo..hi percent."""
    first = []

    def report(it, residual):
        if not first:
            first.append(residual)
        span = math.log10(first[0]) - math.log10(tol)
        frac = (math.log10(first[0]) - math.log10(max(residual, tol))) / span if span > 0 else 1
        log(f"  {label} iteration {it}: residual {residual:.3e}", lo + (hi - lo) * min(frac, 1))
    return report


def relative_to_data(path):
    """`path` as a string relative to the data folder when it lives there, else absolute."""
    path = os.path.abspath(path)
    try:
        return os.path.relpath(path, paths.DATA) if os.path.commonpath([path, str(paths.DATA)]) == str(paths.DATA) else path
    except ValueError:      # another drive on Windows
        return path


def write_atomically(out, arrays, meta_name, meta):
    """Save the arrays as .tmp files, then swap them in and write the meta file last, because its
    presence is what marks the build complete."""
    for name, arr in arrays.items():
        np.save(out / f"{name}.tmp.npy", arr)
    (out / f"{meta_name}.tmp").write_text(json.dumps(meta, indent=2))
    for name in arrays:
        os.replace(out / f"{name}.tmp.npy", out / f"{name}.npy")
    os.replace(out / f"{meta_name}.tmp", out / meta_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--damping", type=float, default=0.85, help="probability of following a link (default 0.85)")
    ap.add_argument("--tol", type=float, default=1e-9, help="stop when the L1 change is below this")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--no-reverse", action="store_true", help="skip reverse PageRank")
    ap.add_argument("--graph", metavar="DIR", help="also compute prose PageRank on this link graph "
                    "(data/prose_graph, from build_prose_graph)")
    ap.add_argument("--prose-only", action="store_true",
                    help="with --graph: compute only prose PageRank, leaving the other files alone")
    args = ap.parse_args()
    if not 0 < args.damping < 1:
        ap.error("--damping must be between 0 and 1")
    if args.prose_only and not args.graph:
        ap.error("--prose-only needs --graph")
    prog.start("centrality")

    load = lambda name, d=paths.GRAPH: np.load(d / f"{name}.npy", mmap_mode="r")
    out_indptr, in_indptr = load("out_indptr"), load("in_indptr")
    n = len(out_indptr) - 1
    out_deg, in_deg = np.diff(out_indptr), np.diff(in_indptr)
    log(f"{n:,} articles, {int(out_indptr[-1]):,} links; damping {args.damping}", 1)
    prose = Path(args.graph) if args.graph else None
    if prose is not None:
        p_indptr = np.load(prose / "out_indptr.npy", mmap_mode="r")
        if len(p_indptr) - 1 != n:
            ap.error(f"{prose} has {len(p_indptr) - 1:,} articles but the link graph has {n:,}: "
                     "were they built from the same dump?")

    # the work, as (label, relative cost) so the progress bar can be shared out between the steps
    steps = [] if args.prose_only else [("PageRank", 9.4)] + ([] if args.no_reverse else [("Reverse PageRank", 15.3)])
    if prose is not None:
        steps.append(("Prose PageRank", 6.0))
    bounds, acc = {}, 3.0
    for label, cost in steps:
        bounds[label] = (acc, acc + 89.0 * cost / sum(c for _, c in steps))
        acc = bounds[label][1]

    arrays, meta = {}, {}
    if not args.prose_only:
        lo, hi = bounds["PageRank"]
        t = time.time()
        pr, iters, res = pagerank(in_indptr, load("in_indices"), out_deg, args.damping, args.tol, args.max_iter,
                                  make_reporter("PageRank", lo, hi, args.tol))
        log(f"PageRank: {iters} iterations, residual {res:.2e}, {time.time() - t:.0f}s", hi)
        meta = {"damping": args.damping, "tol": args.tol, "iterations": iters, "residual": res}
        arrays["pagerank"] = pr.astype(np.float32)
        del pr
        if not args.no_reverse:
            lo, hi = bounds["Reverse PageRank"]
            t = time.time()
            rpr, riters, rres = pagerank(out_indptr, load("out_indices"), in_deg, args.damping, args.tol,
                                         args.max_iter, make_reporter("Reverse PageRank", lo, hi, args.tol))
            log(f"Reverse PageRank: {riters} iterations, residual {rres:.2e}, {time.time() - t:.0f}s", hi)
            meta["reverse"] = {"iterations": riters, "residual": rres}
            arrays["reverse_pagerank"] = rpr.astype(np.float32)
            del rpr

    prose_arrays, prose_meta = {}, {}
    if prose is not None:
        lo, hi = bounds["Prose PageRank"]
        t = time.time()
        ppr, piters, pres = pagerank(load("in_indptr", prose), load("in_indices", prose),
                                     np.diff(load("out_indptr", prose)), args.damping, args.tol, args.max_iter,
                                     make_reporter("Prose PageRank", lo, hi, args.tol))
        log(f"Prose PageRank: {piters} iterations, residual {pres:.2e}, {time.time() - t:.0f}s", hi)
        prose_arrays["prose_pagerank"] = ppr.astype(np.float32)
        prose_meta = {"damping": args.damping, "tol": args.tol, "iterations": piters, "residual": pres,
                      "graph": relative_to_data(prose), "articles": n,
                      "links": int(load("out_indptr", prose)[-1])}
        del ppr

    log("ranking", 94)
    for score, prefix in [(arrays.get("pagerank"), ""), (None if args.prose_only else in_deg, "indegree_"),
                          (arrays.get("reverse_pagerank"), "reverse_"),
                          (prose_arrays.get("prose_pagerank"), "prose_")]:
        if score is not None:
            target = prose_arrays if prefix == "prose_" else arrays
            target[f"{prefix}rank"], target[f"{prefix}order"] = ranking(score)

    out = paths.DATA / "centrality"
    out.mkdir(exist_ok=True)
    log("writing", 97)
    built = dict(dump_date=paths.DUMP_DATE, built=time.strftime("%Y-%m-%d %H:%M"),
                 seconds=round(time.time() - T0, 1))
    if prose is not None:
        write_atomically(out, prose_arrays, "prose_meta.json", {**prose_meta, **built})
    if not args.prose_only:
        write_atomically(out, arrays, "meta.json", {**meta, "articles": n, **built})
    log(f"done in {time.time() - T0:.0f}s", 100)


if __name__ == "__main__":
    main()
