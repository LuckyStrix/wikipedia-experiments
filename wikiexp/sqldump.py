"""Streaming readers for Wikimedia's MySQL dump files (*.sql.gz), without importing into MySQL."""
import gzip
import hashlib
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


def rows(path, keep):
    """Yield tuples of the named columns (raw bytes, strings unquoted) from a dump."""
    cols, _ = header(path)
    row_re = re.compile(rb"\(" + rb",".join([FIELD] * len(cols)) + rb"\)")
    pick = [cols.index(c) for c in keep]
    for block in blocks(path):
        found = row_re.findall(block)
        # every row starts on its own line in these dumps; fewer matches means a parse failure
        if len(found) < block.count(b"\n(") - 1:
            raise RuntimeError(f"row parse failure in {path}")
        for r in found:
            yield tuple(unquote(r[i]) for i in pick)


def int_rows(path, ncols):
    """Yield (n, ncols) int64 arrays from an all-integer dump such as pagelinks."""
    not_rows = re.compile(rb"(?m)^[^(\n].*\n?")
    to_commas = bytes.maketrans(b";", b",")
    for block in blocks(path):
        block = not_rows.sub(b"", block).translate(to_commas, b"()\n").strip(b",")
        if not block:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a partial parse raises instead of truncating
            arr = np.fromstring(block, dtype=np.int64, sep=",")
        yield arr.reshape(-1, ncols)


def title_text(t):
    """Dump titles are bytes with underscores; return the human-readable form."""
    return t.decode("utf-8", "replace").replace("_", " ")
