"""Embed article leads for semantic search, and build the faiss index.

Reads  data/text/leads.sqlite (from build_text; articles are read in popularity order)
Writes data/search/:
  shards/emb_NNNNN.npy   float16 embeddings (rows x dim) of "title: lead" for ranks [N*S, (N+1)*S)
  shards/ids_NNNNN.npy   page_id of each row
  progress.json          which shards are finished: written after each shard, so a stopped run
                         continues where it left off (same model and settings), even on another day
  index.faiss            inner-product index (see below), page_ids.npy, meta.json   -- the result

Disambiguation pages and articles with an empty lead are not embedded.

Index type is picked by size: up to 250,000 vectors a flat (exact) index, which is the fastest and
most accurate at that size and needs N*dim*4 bytes (384 MB at 250k); above that IVF with 4*sqrt(N)
lists and 8-bit scalar quantisation ("IVF{nlist},SQ8"; 7.2M articles = ~2.8 GB, ~10 ms a query at
nprobe=32, recall close to exact).

Runs anywhere (no fork or multiprocessing, so Windows is fine); with an NVIDIA GPU it uses CUDA in
half precision. Install torch for your GPU first (https://pytorch.org/get-started/locally/).

Usage: python -m pipeline.build_embed [--model M] [--device auto|cpu|cuda] [--batch-size 128]
                                      [--limit N] [--restart]
       --limit N embeds only the N most-linked articles (0 = all); a later run with a larger limit
       reuses the finished shards.
"""
import argparse
import json
import math
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from wikiexp import paths
from wikiexp import progress as prog
from wikiexp.progress import progress
from wikiexp.semantic import DEFAULT_MODEL, SentenceTransformerEmbedder, resolve_device

SHARD = 50_000          # articles (ranks) per shard: ~38 MB of float16 at 384 dims
CHUNK_BATCHES = 16      # batches encoded between progress updates
FLAT_MAX = 250_000      # up to this many vectors: exact flat index
NPROBE = 32

T0 = time.time()


def log(msg, pct=None):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)
    if pct is not None:
        progress(pct, msg.strip())


def doc_text(title, lead):
    return f"{title}: {lead}"


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def save_npy(path: Path, arr) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def leads_signature(leads_db) -> dict:
    """What identifies this leads.sqlite: if it's rebuilt, finished shards no longer match."""
    meta = dict(leads_db.execute("SELECT key, value FROM meta"))
    return {"built": meta.get("built"), "articles": meta.get("articles"), "lead_chars": meta.get("lead_chars")}


def embed_shards(out_dir, leads_path, embedder, model_name, limit=0, shard_size=SHARD,
                 batch_size=128, say=log, restart=False):
    """Embed the leads in rank order into out_dir/shards, skipping finished shards.
    Returns (number of ranks covered, state dict)."""
    shards = out_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{leads_path}?mode=ro", uri=True)
    n_leads = db.execute("SELECT count(*) FROM leads").fetchone()[0]
    total = min(limit, n_leads) if limit else n_leads
    sig = leads_signature(db)

    state_path = out_dir / "progress.json"
    state = {"model": model_name, "shard_size": shard_size, "leads": sig, "shards": {}}
    try:
        old = json.loads(state_path.read_text())
    except (OSError, ValueError):
        old = None
    if old and not restart and all(old.get(k) == state[k] for k in ("model", "shard_size", "leads")):
        state["shards"] = old["shards"]
    elif old or any(shards.glob("*.npy")):
        say("settings or leads changed since the last run (or --restart): starting from scratch")
        for f in shards.glob("*.npy"):
            f.unlink()

    n_shards = math.ceil(total / shard_size)
    todo = [i for i in range(n_shards)
            if state["shards"].get(str(i), {}).get("end", -1) < min((i + 1) * shard_size, total)]
    done_rows = sum(min((i + 1) * shard_size, total) - i * shard_size for i in range(n_shards) if i not in todo)
    say(f"{total:,} articles in {n_shards} shards; {n_shards - len(todo)} already finished", 2)
    t0, embedded = time.time(), 0
    for i in todo:
        lo, hi = i * shard_size, min((i + 1) * shard_size, total)
        rows = db.execute("SELECT page_id, title, lead FROM leads WHERE rank >= ? AND rank < ? "
                          "AND disambig = 0 AND lead != '' ORDER BY rank", (lo, hi)).fetchall()
        texts = [doc_text(r[1], r[2]) for r in rows]
        chunk = batch_size * CHUNK_BATCHES
        parts = []
        for j in range(0, len(texts), chunk):
            parts.append(embedder.encode(texts[j:j + chunk]).astype(np.float16))
            frac = (j + len(texts[j:j + chunk])) / max(1, len(texts))
            cur = done_rows + embedded + frac * (hi - lo)
            rate = (embedded + frac * (hi - lo)) / max(1e-9, time.time() - t0)
            progress(2 + 93 * cur / total,
                     f"shard {i + 1}/{n_shards}: {int(cur):,}/{total:,} articles, {rate:,.0f}/s, "
                     f"~{(total - cur) / max(rate, 1e-9) / 60:.0f} min left")
        dim = parts[0].shape[1] if parts else embedder.dim
        emb = np.concatenate(parts) if parts else np.empty((0, dim), dtype=np.float16)
        save_npy(shards / f"ids_{i:05d}.npy", np.array([r[0] for r in rows], dtype=np.int64))
        save_npy(shards / f"emb_{i:05d}.npy", emb)
        state["shards"][str(i)] = {"end": hi, "count": len(rows)}
        write_json(state_path, state)       # the marker comes last: a shard is only done once this lands
        embedded += hi - lo
        say(f"shard {i + 1}/{n_shards} done ({len(rows):,} vectors, {embedded / (time.time() - t0):,.0f} articles/s)")
    return total, state


def build_index(out_dir, state, total, model_name, limit=0, say=log, nprobe=NPROBE):
    """faiss index + page_ids.npy + meta.json from the finished shards."""
    import faiss
    shards = out_dir / "shards"
    order = [i for i in range(math.ceil(total / state["shard_size"]))]
    counts = [state["shards"][str(i)]["count"] for i in order]
    n = sum(counts)
    if n == 0:
        raise SystemExit("nothing to index: no articles with a lead")
    first = np.load(shards / f"emb_{order[next(i for i, c in enumerate(counts) if c)]:05d}.npy", mmap_mode="r")
    dim = first.shape[1]

    def vectors():
        for i in order:
            yield np.asarray(np.load(shards / f"emb_{i:05d}.npy"), dtype=np.float32)

    if n <= FLAT_MAX:
        kind = "flat"
        index = faiss.IndexFlatIP(dim)
        for v in vectors():
            if len(v):
                index.add(v)
    else:
        nlist = int(4 * math.sqrt(n))
        kind = f"IVF{nlist},SQ8"
        index = faiss.index_factory(dim, kind, faiss.METRIC_INNER_PRODUCT)
        # train on a random sample (~50 points per list is enough for k-means)
        sample = min(n, 50 * nlist)
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(n, sample, replace=False))
        train, base = [], 0
        for v, c in zip(vectors(), counts):
            sel = pick[(pick >= base) & (pick < base + c)] - base
            if len(sel):
                train.append(v[sel])
            base += c
        say(f"training {kind} on {sample:,} vectors (a few minutes)", 96)
        index.train(np.concatenate(train))
        del train
        say("adding vectors", 98)
        for v in vectors():
            if len(v):
                index.add(v)
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
    ids = np.concatenate([np.load(shards / f"ids_{i:05d}.npy") for i in order])
    assert index.ntotal == len(ids) == n, (index.ntotal, len(ids), n)

    tmp = out_dir / "index.faiss.building"
    faiss.write_index(index, str(tmp))
    save_npy(out_dir / "page_ids.npy", ids)
    os.replace(tmp, out_dir / "index.faiss")
    meta = {"model": model_name, "dim": dim, "count": n, "limit": limit, "index": kind,
            "nprobe": nprobe if kind != "flat" else None, "built": time.strftime("%Y-%m-%d %H:%M"),
            "leads": state["leads"], "shard_size": state["shard_size"]}
    write_json(out_dir / "meta.json", meta)     # last: its presence means the search data is complete
    return meta


def main():
    from pipeline.settings import Settings
    cfg = Settings.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(cfg["embed_model"]).strip() or DEFAULT_MODEL)
    ap.add_argument("--device", default=str(cfg["embed_device"]).strip(), help="auto, cpu, cuda or cuda:1")
    ap.add_argument("--batch-size", type=int, default=int(cfg["embed_batch"]))
    ap.add_argument("--limit", type=int, default=int(cfg["embed_limit"]), help="most popular N articles (0 = all)")
    ap.add_argument("--restart", action="store_true", help="ignore finished shards")
    ap.add_argument("--data", help="data folder (default: the configured one)")
    args = ap.parse_args()
    data = Path(args.data) if args.data else paths.DATA
    leads = data / "text" / "leads.sqlite"
    if not leads.exists():
        raise SystemExit(f"{leads} not found - run the 'text' stage first (python -m pipeline.build_text)")
    prog.start("embed")
    out = data / "search"
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").unlink(missing_ok=True)   # incomplete until this run finishes

    device = resolve_device(args.device)
    log(f"model {args.model} on {device}, batch size {args.batch_size}", 1)
    embedder = SentenceTransformerEmbedder(args.model, device, args.batch_size)
    total, state = embed_shards(out, leads, embedder, args.model, args.limit, SHARD, args.batch_size,
                                restart=args.restart)
    meta = build_index(out, state, total, args.model, limit=args.limit)
    log(f"done: {meta['count']:,} vectors, {meta['index']} index", 100)


if __name__ == "__main__":
    main()
