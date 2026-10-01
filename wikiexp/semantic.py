"""Offline semantic search over article leads: embed a question, find the nearest articles.

Built by the `embed` pipeline stage (pipeline/build_embed.py) into <data>/search/:

    index.faiss     inner-product faiss index over normalised embeddings of "title: lead"
    page_ids.npy    page_id of every vector, in index order
    meta.json       model, dim, count, limit, index type, ...

and reads titles/leads from <data>/text/leads.sqlite. Designed to be reused (the RAG assistant calls
the same thing):

    ss = SemanticSearch(data_dir)                 # instant; the model and index load on first use
    for hit in ss.search("that battle where the weather decided everything", k=10):
        hit.page_id, hit.title, hit.score, hit.snippet, hit.lead

``embedder`` can be injected (anything with ``encode(texts, query=False) -> float32 array``), which is
how the tests run without downloading a model. The model is loaded lazily and guarded by a lock, so
several threads (e.g. Textual workers) can search at once.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from . import paths
from .wikitext import trim

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SEQ_LENGTH = 256        # tokens; a 1200-character lead is ~300, and the tail adds little
SNIPPET_CHARS = 300


# ── embedders ────────────────────────────────────────────────────────────────

def prefixes(model: str) -> tuple[str, str]:
    """(query prefix, document prefix) the model was trained with. bge models want an instruction
    on short queries (not on documents); e5 models want "query: " / "passage: " on both."""
    name = model.lower()
    if "bge" in name and "-en" in name:
        return "Represent this sentence for searching relevant passages: ", ""
    if "e5" in name:
        return "query: ", "passage: "
    return "", ""


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str], query: bool = False) -> np.ndarray:
        """float32 array (len(texts), dim) of unit vectors; query=True for search questions."""


class SentenceTransformerEmbedder:
    """sentence-transformers model, loaded on first use. On CUDA it runs in half precision."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto", batch_size: int = 64):
        self.model_name, self.device_name, self.batch_size = model, device, batch_size
        self.query_prefix, self.doc_prefix = prefixes(model)
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self.device = resolve_device(self.device_name)
                try:    # offline first: no network round trips (or hangs) when the model is already cached
                    m = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
                except Exception:
                    m = SentenceTransformer(self.model_name, device=self.device)   # first use: download
                m.max_seq_length = min(MAX_SEQ_LENGTH, m.max_seq_length or MAX_SEQ_LENGTH)
                if self.device.startswith("cuda"):
                    m.half()
                self._model = m
            return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str], query: bool = False) -> np.ndarray:
        prefix = self.query_prefix if query else self.doc_prefix
        model = self.model
        with self._lock:     # a model isn't safe to run from several threads at once
            out = model.encode([prefix + t for t in texts], batch_size=self.batch_size,
                               normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)


# ── search ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Hit:
    page_id: int
    title: str
    score: float          # cosine similarity (inner product of unit vectors)
    snippet: str          # start of the lead, ~300 characters
    lead: str             # the whole stored lead (up to the build's lead length)


def search_dir(data_dir: Path | None = None) -> Path:
    return Path(data_dir or paths.DATA) / "search"


def read_meta(data_dir: Path | None = None) -> dict:
    return json.loads((search_dir(data_dir) / "meta.json").read_text())


class SemanticSearch:
    def __init__(self, data_dir: Path | None = None, embedder: Embedder | None = None,
                 nprobe: int | None = None):
        self.data_dir = Path(data_dir or paths.DATA)
        self.dir = search_dir(self.data_dir)
        self._embedder = embedder
        self._nprobe = nprobe
        self._lock = threading.Lock()
        self._loaded = False
        self._db: sqlite3.Connection | None = None
        self._db_lock = threading.Lock()

    # -- loading
    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            import faiss
            for name in ("index.faiss", "page_ids.npy", "meta.json"):
                if not (self.dir / name).exists():
                    raise FileNotFoundError(f"{self.dir / name} not found - run the 'embed' stage "
                                            f"(python -m pipeline.build_embed)")
            self.meta = json.loads((self.dir / "meta.json").read_text())
            self.page_ids = np.load(self.dir / "page_ids.npy", mmap_mode="r")
            self.index = faiss.read_index(str(self.dir / "index.faiss"))
            nprobe = self._nprobe or self.meta.get("nprobe")
            if nprobe:
                faiss.ParameterSpace().set_index_parameter(self.index, "nprobe", int(nprobe))
            if self._embedder is None:
                self._embedder = SentenceTransformerEmbedder(self.meta["model"], device="auto")
            self._db = sqlite3.connect(f"file:{self.data_dir / 'text' / 'leads.sqlite'}?mode=ro", uri=True,
                                       check_same_thread=False)
            self._loaded = True

    def load(self) -> "SemanticSearch":
        """Load everything now (the first search does it otherwise); returns self."""
        self._load()
        return self

    def available(self) -> bool:
        return all((self.dir / n).exists() for n in ("index.faiss", "page_ids.npy", "meta.json")) \
            and (self.data_dir / "text" / "leads.sqlite").exists()

    def __len__(self) -> int:
        self._load()
        return int(self.index.ntotal)

    # -- querying
    def encode_query(self, query: str) -> np.ndarray:
        self._load()
        return self._embedder.encode([query], query=True)

    def search_vector(self, vec: np.ndarray, k: int = 10) -> list[Hit]:
        """Nearest articles to an already-embedded query (shape (dim,) or (1, dim))."""
        self._load()
        vec = np.ascontiguousarray(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        scores, pos = self.index.search(vec, k)
        pairs = [(int(self.page_ids[p]), float(s)) for p, s in zip(pos[0], scores[0]) if p >= 0]
        return self._hits(pairs)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The k articles whose lead is closest in meaning to the query, best first."""
        query = query.strip()
        if not query or k <= 0:
            return []
        return self.search_vector(self.encode_query(query), k)

    def leads(self, page_ids: Sequence[int]) -> dict[int, tuple[str, str]]:
        """{page_id: (title, lead)} for the ids that are in leads.sqlite."""
        self._load()
        ids = [int(p) for p in page_ids]
        with self._db_lock:
            rows = self._db.execute(
                f"SELECT page_id, title, lead FROM leads WHERE page_id IN ({','.join('?' * len(ids))})",
                ids).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def _hits(self, pairs: list[tuple[int, float]]) -> list[Hit]:
        info = self.leads([p for p, _ in pairs])
        out = []
        for pid, score in pairs:
            title, lead = info.get(pid, (f"(page {pid})", ""))
            out.append(Hit(pid, title, score, trim(lead, SNIPPET_CHARS), lead))
        return out
