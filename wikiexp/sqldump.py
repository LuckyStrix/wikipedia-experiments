"""Streaming readers for Wikimedia's MySQL dump files (*.sql.gz), without importing into MySQL."""
import collections
import functools
import gc
import gzip
import hashlib
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import warnings

import numpy as np

from . import paths

BLOCK = 64 << 20


def verify(names):
    """Check dump files against Wikimedia's sha1sums; raise if missing or incomplete."""
    sums = {}
    with open(paths.dump("sha1sums.txt")) as f:
        for line in f:
            digest, name = line.split()
            sums[name] = digest
    for name in names:
        path = paths.dump(name)
        if not path.exists():
            raise FileNotFoundError(f"missing {path.name} - has pipeline/download.sh finished?")
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while chunk := f.read(BLOCK):
                h.update(chunk)
        if h.hexdigest() != sums[path.name]:
            raise ValueError(f"checksum mismatch for {path.name} - still downloading, or corrupt")


def _open(path):
    if shutil.which("pigz"):
        proc = subprocess.Popen(["pigz", "-dc", str(path)], stdout=subprocess.PIPE, bufsize=BLOCK)
        return proc.stdout
    return gzip.open(path, "rb")


def blocks(path):
    """Yield the decompressed dump in large chunks that end on a line boundary."""
    if not path.exists():
        raise FileNotFoundError(path)
    with _open(path) as f:
        rest = b""
        while chunk := f.read(BLOCK):
            buf = rest + chunk
            cut = buf.rfind(b"\n") + 1
            if cut == 0:
                rest = buf
                continue
            rest = buf[cut:]
            yield buf[:cut]
        if rest:
            yield rest


def header(path):
    """Column names from the CREATE TABLE statement, and its AUTO_INCREMENT value (or None)."""
    for block in blocks(path):
        text = block[: block.find(b"INSERT INTO")].decode()
        cols = re.findall(r"^\s+`(\w+)` ", text, re.M)
        m = re.search(r"AUTO_INCREMENT=(\d+)", text)
        return cols, int(m.group(1)) if m else None
    return [], None


FIELD = rb"(NULL|'(?:[^'\\]|\\.)*'|[-0-9.eE+]+)"
ESCAPES = {b"0": b"\0", b"n": b"\n", b"r": b"\r", b"t": b"\t", b"Z": b"\x1a"}


def unquote(v):
    if v == b"NULL":
        return None
    if v[:1] == b"'":
        v = v[1:-1]
        if b"\\" in v:
            v = re.sub(rb"\\(.)", lambda m: ESCAPES.get(m.group(1), m.group(1)), v)
    return v


# ── parallel block processing ────────────────────────────────────────────────

def default_workers():
    """Parser processes to use: $WIKI_WORKERS, else all but two cores (max 8; beyond that the
    main process, which merges results, becomes the limit)."""
    env = os.environ.get("WIKI_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def map_blocks(path, fn, workers=None):
    """Yield fn(block) for each block of the decompressed dump, in order, computed by worker processes.

    fn must be a picklable top-level function (or functools.partial of one). Workers are forked, so
    they see the parent's module globals (e.g. big numpy lookup tables) without copying them. At most
    2 blocks per worker are in flight, so memory stays bounded however big the dump is. Falls back
    to a plain loop with one worker, or where fork isn't available (Windows).
    """
    workers = workers or default_workers()
    if workers <= 1 or "fork" not in mp.get_all_start_methods():
        for block in blocks(path):
            yield fn(block)
        return
    gc.freeze()   # keep the children's garbage collector from touching (and so copying) parent objects
    try:
        with mp.get_context("fork").Pool(workers) as pool:   # fork before pigz starts
            pending = collections.deque()
            for block in blocks(path):
                pending.append(pool.apply_async(fn, (block,)))
                if len(pending) >= 2 * workers:
                    yield pending.popleft().get()
            while pending:
                yield pending.popleft().get()
    finally:
        gc.unfreeze()


SKIP = rb"(?:NULL|'(?:[^'\\]|\\.)*'|[-0-9.eE+]+)"


@functools.lru_cache(maxsize=32)
def _row_re(ncols, capture):
    """Regex for one row that captures only the columns in `capture` (creating a Python object for
    every field of every row is most of the parsing cost)."""
    return re.compile(rb"\(" + rb",".join(FIELD if i in capture else SKIP for i in range(ncols)) + rb"\)")


def parse_rows(block, ncols, pick, where=None):
    """Rows of one block as tuples of the picked column indexes; where=(index, value) filters on a
    raw column value (e.g. only namespace b"0") before unquoting, to save work and transfer."""
    capture = tuple(sorted(set(pick) | ({where[0]} if where else set())))
    found = _row_re(ncols, capture).findall(block)
    # every row starts on its own line in these dumps; fewer matches means a parse failure
    if len(found) < block.count(b"\n(") - 1:
        raise RuntimeError("row parse failure")
    if len(capture) == 1:
        found = [(f,) for f in found]
    pos = {c: i for i, c in enumerate(capture)}
    if where is not None:
        col, value = pos[where[0]], where[1]
        found = [r for r in found if r[col] == value]
    take = [pos[i] for i in pick]
    return [tuple(unquote(r[i]) for i in take) for r in found]


def rows(path, keep, where=None, workers=None):
    """Yield tuples of the named columns (raw bytes, strings unquoted) from a dump, parsed in
    parallel. where=(column name, raw value) keeps only matching rows, e.g. ("page_namespace", b"0")."""
    cols, _ = header(path)
    pick = [cols.index(c) for c in keep]
    cond = (cols.index(where[0]), where[1]) if where else None
    fn = functools.partial(parse_rows, ncols=len(cols), pick=pick, where=cond)
    for chunk in map_blocks(path, fn, workers):
        yield from chunk


_NOT_ROWS = re.compile(rb"(?m)^[^(\n].*\n?")
_TO_COMMAS = bytes.maketrans(b";", b",")


def parse_int_block(block, ncols):
    """(n, ncols) int64 array from one block of an all-integer dump such as pagelinks."""
    block = _NOT_ROWS.sub(b"", block).translate(_TO_COMMAS, b"()\n").strip(b",")
    if not block:
        return np.empty((0, ncols), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a partial parse raises instead of truncating
        arr = np.fromstring(block, dtype=np.int64, sep=",")
    return arr.reshape(-1, ncols)


def int_rows(path, ncols):
    """Yield (n, ncols) int64 arrays from an all-integer dump (single process; see map_blocks for
    parallel processing)."""
    for block in blocks(path):
        arr = parse_int_block(block, ncols)
        if len(arr):
            yield arr


def title_text(t):
    """Dump titles are bytes with underscores; return the human-readable form."""
    return t.decode("utf-8", "replace").replace("_", " ")
