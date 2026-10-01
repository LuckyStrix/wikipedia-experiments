"""Hybrid article retrieval: meaning (embeddings) + words (FTS5 bm25) + names (title index).

    hs = HybridSearch(data_dir)                    # instant; everything loads on first use
    for r in hs.search("who designed the Tacoma Narrows Bridge?", k=10):
        r.page_id, r.title, r.score, r.snippet, r.via     # via: ("keyword", "title", "semantic")

Three ranked lists are fused with reciprocal rank fusion (RRF: an article scores sum(w / (60 + rank))
over the lists it appears in, so agreement between independent retrievers beats one very high score,
and the incomparable scores - cosine, bm25 - never need to be normalised):

- **semantic**: nearest embeddings of the question (wikiexp.semantic). Optional: skipped when the
  embeddings aren't built. Great at "which article is about X"; blind to articles it hasn't embedded.
- **keyword**: bm25 over title+lead (wikiexp.keyword), covers all 7.2M articles and finds the exact
  words of a question.
- **title**: entity names found in the question ("Tacoma Narrows Bridge", "Napoleon") looked up in
  the title index, longest names first, so a named article is always a candidate.

A mild popularity prior (up to +15% for the most linked-to articles) breaks ties toward the article
people mean ("Mercury" the planet over an obscure ship of that name).
"""
from __future__ import annotations

import math
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from . import paths
from .keyword import STOP_WORDS, KeywordIndex
from .semantic import SNIPPET_CHARS, SemanticSearch
from .titles import TitleIndex
from .wikitext import trim

MODES = ("hybrid", "semantic", "keyword")
RRF_K = 60
WEIGHTS = {"semantic": 1.0, "keyword": 1.0, "title": 1.5}
POPULARITY_BOOST = 0.15          # best case multiplier is 1 + this
CANDIDATES = 30                  # per retriever
MAX_NAME_WORDS = 6


@dataclass
class Result:
    page_id: int
    title: str
    score: float                 # fused score (hybrid) or the retriever's own score
    snippet: str
    lead: str
    via: tuple[str, ...] = ()    # which retrievers found it
    ranks: dict[str, int] = field(default_factory=dict)    # 1-based rank in each retriever's list


def rrf(lists: dict[str, Sequence[int]], weights: dict[str, float] | None = None, k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion of ranked id lists: {id: sum of weight / (k + rank)}."""
    weights = weights or {}
    out: dict[int, float] = {}
    for name, ids in lists.items():
        w = weights.get(name, 1.0)
        seen = set()
        for rank, i in enumerate(ids, 1):
            if i in seen:
                continue
            seen.add(i)
            out[i] = out.get(i, 0.0) + w / (k + rank)
    return out


def popularity_prior(rank: int | None, total: int) -> float:
    """1.0 for the most linked-to article falling to 0.0 for the least (log scale); 0 if unknown."""
    if rank is None or total <= 1:
        return 0.0
    return max(0.0, 1.0 - math.log1p(rank) / math.log1p(total))


def name_candidates(question: str) -> list[str]:
    """Phrases of the question that might be article titles, longest first (the caller keeps the
    first match of overlapping ones). A phrase qualifies only if it has a capitalised word (not the
    question's first), since "wrote the novel" or "battle" alone name articles nobody means; in an
    all-lowercase question only phrases of two or more words are tried."""
    words = re.findall(r"[^\W_]+(?:['’.\-][^\W_]+)*", question)
    spans = []
    any_caps = any(w[0].isupper() for w in words[1:])      # an all-lowercase question gets no capital-letter test
    for n in range(min(MAX_NAME_WORDS, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            chunk = words[i:i + n]
            low = [w.lower() for w in chunk]
            if low[0] in STOP_WORDS or low[-1] in STOP_WORDS:
                continue                       # a title doesn't start or end on "who", "of", ...
            caps = any(w[0].isupper() for j, w in enumerate(chunk) if i + j > 0)
            if (n == 1 or any_caps) and not caps:
                continue                       # "wrote the novel" is a redirect nobody means
            spans.append((i, i + n, " ".join(chunk)))
    return spans


class HybridSearch:
    def __init__(self, data_dir: Path | None = None, semantic: SemanticSearch | None = None,
                 keyword: KeywordIndex | None = None, titles: TitleIndex | None = None):
        self.data_dir = Path(data_dir or paths.DATA)
        self.semantic = semantic if semantic is not None else SemanticSearch(self.data_dir)
        self.keyword = keyword if keyword is not None else KeywordIndex(self.data_dir)
        self._titles = titles
        self._titles_tried = titles is not None
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._total = 0

    # -- availability
    def has_semantic(self) -> bool:
        return self.semantic.available()

    def has_keyword(self) -> bool:
        return self.keyword.available()

    def modes(self) -> tuple[str, ...]:
        """The modes that can run with the data on disk (hybrid needs at least one retriever)."""
        out = []
        if self.has_semantic() or self.has_keyword():
            out.append("hybrid")
        if self.has_semantic():
            out.append("semantic")
        if self.has_keyword():
            out.append("keyword")
        return tuple(out)

    @property
    def titles(self) -> TitleIndex | None:
        with self._lock:
            if not self._titles_tried:
                self._titles_tried = True
                try:
                    self._titles = TitleIndex(self.data_dir / "titles.sqlite")
                except FileNotFoundError:
                    self._titles = None
            return self._titles

    def warm_up(self) -> None:
        """Load the models and indexes now so the first real question isn't slow."""
        if self.has_semantic():
            self.semantic.search("warm up", 1)
        if self.has_keyword():
            self.keyword.search("warm up", 1)
        _ = self.titles

    # -- leads (title, lead, popularity rank) for any article
    def _leads(self, page_ids: Iterable[int]) -> dict[int, tuple[int, str, str, int]]:
        """{page_id: (rank, title, lead, disambig)}."""
        ids = [int(p) for p in page_ids]
        if not ids:
            return {}
        with self._lock:
            if self._db is None:
                self._db = sqlite3.connect(f"file:{self.data_dir / 'text' / 'leads.sqlite'}?mode=ro", uri=True,
                                           check_same_thread=False)
                self._total = self._db.execute("SELECT count(*) FROM leads").fetchone()[0]
            rows = self._db.execute(
                f"SELECT page_id, rank, title, lead, disambig FROM leads WHERE page_id IN ({','.join('?' * len(ids))})",
                ids).fetchall()
        return {r[0]: (r[1], r[2], r[3], r[4]) for r in rows}

    # -- the three retrievers
    def title_matches(self, question: str, limit: int = 8) -> list[int]:
        """Page ids of articles named in the question, longest names first."""
        ti = self.titles
        if ti is None:
            return []
        spans = sorted(name_candidates(question), key=lambda s: (-(s[1] - s[0]), s[0]))
        taken: list[tuple[int, int]] = []
        out: list[int] = []
        for a, b, text in spans:
            if any(a < tb and ta < b for ta, tb in taken):
                continue
            m = ti.exact(text)
            if m is None or m.page_id in out:
                continue
            taken.append((a, b))
            out.append(m.page_id)
            if len(out) >= limit:
                break
        return out

    # -- fusion
    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[Result]:
        query = query.strip()
        if not query or k <= 0:
            return []
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "semantic":
            return self._single("semantic", self.semantic.search(query, k))
        if mode == "keyword":
            return self._single("keyword", self.keyword.search(query, k))
        lists: dict[str, list[int]] = {}
        info: dict[int, tuple[str, str]] = {}      # page_id -> (title, lead) from retrievers
        if self.has_semantic():
            hits = self.semantic.search(query, CANDIDATES)
            lists["semantic"] = [h.page_id for h in hits]
            info.update({h.page_id: (h.title, h.lead) for h in hits})
        if self.has_keyword():
            hits = self.keyword.search(query, CANDIDATES)
            lists["keyword"] = [h.page_id for h in hits]
            info.update({h.page_id: (h.title, h.lead) for h in hits})
        lists["title"] = self.title_matches(query)
        if not any(lists.values()):
            return []
        fused = rrf(lists, WEIGHTS)
        leads = self._leads(fused)
        results = []
        for pid, score in fused.items():
            rank, title, lead, dab = leads.get(pid, (None, *info.get(pid, (f"(page {pid})", "")), 0))
            if dab and pid not in info:        # a disambiguation page named in the question: not an answer
                continue
            score *= 1.0 + POPULARITY_BOOST * popularity_prior(rank, self._total)
            via = tuple(n for n in ("semantic", "keyword", "title") if pid in lists.get(n, ()))
            ranks = {n: lists[n].index(pid) + 1 for n in via}
            results.append(Result(pid, title, score, trim(lead, SNIPPET_CHARS), lead, via, ranks))
        results.sort(key=lambda r: -r.score)
        return results[:k]

    @staticmethod
    def _single(name: str, hits) -> list[Result]:
        return [Result(h.page_id, h.title, h.score, h.snippet, h.lead, (name,), {name: i})
                for i, h in enumerate(hits, 1)]
