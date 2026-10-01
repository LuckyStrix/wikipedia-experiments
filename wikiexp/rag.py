"""Retrieval-augmented question answering over the local Wikipedia.

    rag = RAG(data_dir, backend)                       # backend: wikiexp.llm (ollama / claude / a fake)
    conv = Conversation()
    for ev in rag.ask("Who designed the Eiffel Tower?", conv):
        if ev.kind == "token":   print(ev.data, end="")
        elif ev.kind == "sources": ...                  # list[Source], numbered from 1
        elif ev.kind == "done":  answer = ev.data       # Answer: text, cited, invalid, sources, timings

The pipeline, step by step:

1. **Standalone query.** A follow-up ("and when was it built?") is rewritten into a question that
   makes sense alone, by the LLM when there is history (a plain heuristic if that fails).
2. **Candidates.** ``HybridSearch`` (wikiexp.retrieval) returns the ~15 best articles; the best few
   are fetched in full from the dump (``WikiText.fetch``, ~40 ms each) and cleaned by
   ``wikitext.sections``.
3. **Passages.** Each article is cut into passages of ~120-220 words that never cross a section and
   remember "Article - Section". Huge articles are pre-filtered by word overlap so a few hundred
   passages at most go on.
4. **Rerank.** The bge embedder scores every passage against the standalone query (CPU is fine for a
   few hundred); the model is loaded once and shared with semantic search.
5. **Pack.** The best passages go into the prompt, at most a few per article, until the token budget
   is spent. They are numbered [1], [2], ... in score order.
6. **Answer.** The LLM is told to use only those sources and to cite them; afterwards every [n] is
   checked against the sources actually given, and numbers that don't exist are stripped and reported.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from . import paths
from . import wikitext as W
from .keyword import STOP_WORDS
from .llm import Backend, GenInfo, LLMError, complete
from .retrieval import HybridSearch, Result
from .semantic import DEFAULT_MODEL, Embedder, SentenceTransformerEmbedder, read_meta

MIN_WORDS, MAX_WORDS = 120, 220          # passage size
MAX_PASSAGES_PER_ARTICLE = 30            # before embedding (lexical pre-filter), the lead always stays
MAX_PER_ARTICLE_IN_PROMPT = 3
SCORE_MARGIN = 0.10                      # keep passages within this of the best score
HISTORY_TURNS = 3
NO_ANSWER = "The sources I found don't contain the answer to that."


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class Passage:
    page_id: int
    title: str
    section: str                 # "" for the lead
    text: str
    index: int = 0               # position within the article
    score: float = 0.0

    @property
    def path(self) -> str:
        return f"{self.title} — {self.section}" if self.section else self.title

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Source:
    """A passage as the model saw it: numbered from 1."""
    n: int
    passage: Passage

    @property
    def title(self) -> str:
        return self.passage.title

    @property
    def page_id(self) -> int:
        return self.passage.page_id

    @property
    def label(self) -> str:
        return self.passage.path


@dataclass
class Turn:
    question: str
    answer: str


class Conversation:
    """The turns so far; ``reset()`` starts a new conversation."""

    def __init__(self):
        self.turns: list[Turn] = []

    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question, answer))

    def reset(self) -> None:
        self.turns.clear()


@dataclass
class Answer:
    question: str
    query: str                           # the standalone question used for retrieval
    text: str                            # the answer with invalid citations removed
    raw_text: str                        # as generated
    sources: list[Source]
    cited: list[int]                     # source numbers the answer cites (valid ones)
    invalid: list[int]                   # numbers cited that don't exist (stripped from `text`)
    candidates: list[Result] = field(default_factory=list)      # articles retrieval considered
    retrieval_ms: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)     # ms per retrieval step
    generation: GenInfo = field(default_factory=GenInfo)
    backend: str = ""
    model: str = ""
    error: str = ""                      # set when generation failed part-way (text is what arrived)
    notes: list[str] = field(default_factory=list)


@dataclass
class Event:
    kind: str                            # "status" | "sources" | "token" | "done"
    data: object = None


# ── passages ─────────────────────────────────────────────────────────────────

_SENTENCE = re.compile(r"(?<=[.!?])[\"')\]”’]*\s+(?=[\"'(“‘]?[A-Z0-9])")


def _chunk_long(paragraph: str, max_words: int) -> list[str]:
    """A very long paragraph cut at sentence ends into pieces of at most max_words."""
    out, cur, count = [], [], 0
    for s in _SENTENCE.split(paragraph):
        n = len(s.split())
        if cur and count + n > max_words:
            out.append(" ".join(cur))
            cur, count = [], 0
        cur.append(s)
        count += n
        while count > max_words * 1.5 and len(cur) == 1:       # one sentence longer than the limit: hard cut
            words = cur[0].split()
            out.append(" ".join(words[:max_words]))
            cur, count = [" ".join(words[max_words:])], len(words) - max_words
    if cur:
        out.append(" ".join(cur))
    return out


def split_passages(title: str, page_id: int, secs: Sequence[tuple[str, str]],
                   min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[Passage]:
    """Cut an article's ``sections()`` into passages of ~min..max words that never cross a section.

    Paragraphs (lines) are added until the passage has min_words; one that would push it past
    max_words starts the next passage; a paragraph longer than max_words is cut at sentence ends. A
    short tail of a section is merged into the previous passage of that section when it fits."""
    out: list[Passage] = []
    for heading, text in secs:
        mine: list[str] = []
        buf: list[str] = []
        count = 0

        def flush():
            nonlocal buf, count
            if buf:
                mine.append("\n".join(buf))
            buf, count = [], 0

        for para in (p.strip() for p in text.split("\n")):
            if not para:
                continue
            n = len(para.split())
            pieces = _chunk_long(para, max_words) if n > max_words else [para]
            for piece in pieces:
                pn = len(piece.split())
                if buf and count + pn > max_words:
                    flush()
                buf.append(piece)
                count += pn
                if count >= min_words:
                    flush()
        if buf:
            tail_words = count
            if mine and tail_words < min_words // 3 and len(mine[-1].split()) + tail_words <= max_words * 1.3:
                mine[-1] += "\n" + "\n".join(buf)
                buf, count = [], 0
            else:
                flush()
        for t in mine:
            if len(t.split()) >= 8:
                out.append(Passage(page_id, title, heading, t, index=len(out)))
    return out


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W_]+", text.lower()) if w not in STOP_WORDS and len(w) > 1}


def lexical_overlap(query_tokens: set[str], text: str) -> float:
    """Share of the query's content words that appear in the text (0..1)."""
    return len(query_tokens & _tokens(text)) / len(query_tokens) if query_tokens else 0.0


def prefilter(passages: list[Passage], query: str, keep: int = MAX_PASSAGES_PER_ARTICLE) -> list[Passage]:
    """Keep the lead passages and the `keep` best by word overlap, in article order."""
    if len(passages) <= keep:
        return passages
    q = _tokens(query)
    ranked = sorted(passages, key=lambda p: -lexical_overlap(q, p.path + " " + p.text))
    chosen = {id(p) for p in ranked[:keep]} | {id(p) for p in passages if p.index == 0}
    return [p for p in passages if id(p) in chosen]


# ── packing and prompt ───────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token count (English wikitext is ~1.35 tokens per word); good enough for budgeting."""
    return int(len(text.split()) * 1.35) + 1


def pack(passages: Sequence[Passage], budget_tokens: int, per_article: int = MAX_PER_ARTICLE_IN_PROMPT,
         max_sources: int = 8, margin: float = SCORE_MARGIN, min_sources: int = 2) -> list[Source]:
    """The best passages (by .score) that fit the token budget, at most `per_article` from one article,
    numbered from 1 in score order. Passages scoring more than `margin` below the best are left out
    (after `min_sources`): padding the prompt with weak matches slows a small model and distracts it.
    The best passage is always included, even if it alone is over budget."""
    out: list[Source] = []
    used = 0
    per: dict[int, int] = {}
    ranked = sorted(passages, key=lambda p: -p.score)
    for p in ranked:
        if out and len(out) >= min_sources and p.score < ranked[0].score - margin:
            break
        cost = estimate_tokens(p.path + "\n" + p.text) + 6
        if per.get(p.page_id, 0) >= per_article:
            continue
        if out and used + cost > budget_tokens:
            continue
        out.append(Source(len(out) + 1, p))
        used += cost
        per[p.page_id] = per.get(p.page_id, 0) + 1
        if len(out) >= max_sources:
            break
    return out


SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources the user provides, which are passages from Wikipedia.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge, even if you are sure of it.
- After each fact or sentence, cite the source(s) it came from like [1] or [2][3]. Only cite numbers that appear in the sources.
- If the sources do not contain the answer, say so plainly (for example "The sources don't say.") and mention what they do cover. Do not guess.
- If the sources disagree, say so and cite both.
- Be concise and direct: start with the answer. No preamble, and do not list the sources at the end."""

REWRITE_PROMPT = """Rewrite the user's last message as one standalone question for searching an encyclopedia. \
Replace pronouns and references like "it", "he", "that" or "and what about X" with the names they refer to in the \
conversation. Keep it short. Reply with the question only."""


def build_user_message(question: str, sources: Sequence[Source]) -> str:
    """The final user turn: numbered sources, then the question."""
    if not sources:
        return f"Sources: (none found)\n\nQuestion: {question}"
    blocks = [f"[{s.n}] {s.label}\n{s.passage.text}" for s in sources]
    return ("Sources:\n\n" + "\n\n".join(blocks) +
            f"\n\nQuestion: {question}\n\nAnswer using only the sources above, citing them like [1].")


def build_messages(question: str, sources: Sequence[Source], history: Sequence[Turn] = ()) -> list[dict]:
    """Chat messages: recent turns (answers without their citation numbers, which refer to sources
    that are no longer shown), then the sources and the new question."""
    msgs: list[dict] = []
    for t in list(history)[-HISTORY_TURNS:]:
        msgs.append({"role": "user", "content": t.question})
        msgs.append({"role": "assistant", "content": strip_citations(t.answer)})
    msgs.append({"role": "user", "content": build_user_message(question, sources)})
    return msgs


# ── citations ────────────────────────────────────────────────────────────────

_CITE = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+|\s*[-–]\s*\d+)*)\s*\]")


def _numbers(group: str) -> list[int]:
    nums: list[int] = []
    for part in re.split(r"\s*[,;]\s*", group.strip()):
        if re.search(r"[-–]", part):
            a, b = (int(x) for x in re.split(r"\s*[-–]\s*", part))
            nums += list(range(a, b + 1)) if 0 < b - a < 20 else [a, b]
        else:
            nums.append(int(part))
    return nums


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\s*\d+(?:\s*[,;\-–]\s*\d+)*\s*\]", "", text).strip()


def validate_citations(text: str, n_sources: int) -> tuple[str, list[int], list[int]]:
    """(text with nonexistent citations removed, valid numbers cited in order of first use,
    invalid numbers cited). "[1, 7]" with 3 sources becomes "[1]"; "[7]" disappears."""
    cited: list[int] = []
    invalid: list[int] = []

    def fix(m: re.Match) -> str:
        nums = _numbers(m.group(1))
        good = [n for n in nums if 1 <= n <= n_sources]
        for n in nums:
            if n in good:
                if n not in cited:
                    cited.append(n)
            elif n not in invalid:
                invalid.append(n)
        if len(good) == len(nums):
            return m.group(0)
        return "[" + ", ".join(str(n) for n in good) + "]" if good else ""

    fixed = _CITE.sub(fix, text)
    fixed = re.sub(r"[ \t]+([.,;:!?])", r"\1", fixed)         # "word [7]." leaves "word ."
    fixed = re.sub(r"[ \t]{2,}", " ", fixed)
    return fixed, cited, invalid


# ── query rewriting ──────────────────────────────────────────────────────────

_REFERS = re.compile(r"\b(he|she|it|they|them|him|her|his|hers|its|their|theirs|this|that|these|those|there|then|"
                     r"former|latter|same|such|one|ones)\b", re.I)
_FOLLOWUP_START = re.compile(r"^\s*(and|but|also|what about|how about|why|how so|where|when|who|which)\b", re.I)


def looks_like_followup(question: str) -> bool:
    """Heuristic: short, or leans on a pronoun / "and ...", so it can't be searched on its own."""
    words = question.split()
    named = any(w[0].isupper() for w in words[1:] if w[:1].isalpha())
    if _REFERS.search(question) and not named:
        return True
    if _FOLLOWUP_START.match(question) and len(words) <= 5 and not named:
        return True
    return len(words) <= 3 and not named


def heuristic_rewrite(question: str, history: Sequence[Turn]) -> str:
    """Without an LLM: put the previous question in front, so its names are searched too."""
    if not history or not looks_like_followup(question):
        return question
    return f"{history[-1].question.rstrip('?. ')} {question}"


def rewrite_query(question: str, history: Sequence[Turn], backend: Backend | None = None) -> str:
    """A standalone version of a follow-up question (the question itself when there is no history
    or it already stands alone). Uses the LLM when given one; any failure falls back to the heuristic."""
    if not history or not looks_like_followup(question):
        return question
    if backend is not None:
        convo = "\n".join(f"User: {t.question}\nAssistant: {strip_citations(t.answer)[:300]}"
                          for t in list(history)[-2:])
        try:
            out = complete(backend, REWRITE_PROMPT,
                           [{"role": "user", "content": f"Conversation:\n{convo}\n\nLast message: {question}"}],
                           max_tokens=80)
            out = out.strip().splitlines()[0].strip().strip('"“”').strip() if out.strip() else ""
            if 3 <= len(out) <= 300:
                return out
        except LLMError:
            pass
    return heuristic_rewrite(question, history)


# ── the pipeline ─────────────────────────────────────────────────────────────

_EMBEDDERS: dict[str, SentenceTransformerEmbedder] = {}
_EMBEDDERS_LOCK = threading.Lock()


def shared_embedder(model: str = DEFAULT_MODEL) -> SentenceTransformerEmbedder:
    """One loaded embedding model per process, shared by semantic search and passage reranking."""
    with _EMBEDDERS_LOCK:
        if model not in _EMBEDDERS:
            _EMBEDDERS[model] = SentenceTransformerEmbedder(model, device="cpu")
        return _EMBEDDERS[model]


def embedder_model(data_dir: Path) -> str:
    try:
        return read_meta(data_dir)["model"]
    except (OSError, ValueError, KeyError):
        return DEFAULT_MODEL


def make_hybrid(data_dir: Path | None = None, embedder: Embedder | None = None) -> HybridSearch:
    """HybridSearch whose semantic part uses the shared embedder (so the model is loaded once)."""
    from .semantic import SemanticSearch
    data_dir = Path(data_dir or paths.DATA)
    emb = embedder or shared_embedder(embedder_model(data_dir))
    return HybridSearch(data_dir, semantic=SemanticSearch(data_dir, embedder=emb))


class RAG:
    def __init__(self, data_dir: Path | None = None, backend: Backend | None = None,
                 hybrid: HybridSearch | None = None, embedder: Embedder | None = None,
                 wikitext: "W.WikiText | None" = None, n_articles: int = 4, n_candidates: int = 15,
                 budget_tokens: int = 2400, max_answer_tokens: int = 700):
        self.data_dir = Path(data_dir or paths.DATA)
        self.backend = backend
        self.embedder = embedder or shared_embedder(embedder_model(self.data_dir))
        self.hybrid = hybrid or make_hybrid(self.data_dir, self.embedder)
        self._wt = wikitext
        self._wt_lock = threading.Lock()
        self.n_articles, self.n_candidates = n_articles, n_candidates
        self.budget_tokens, self.max_answer_tokens = budget_tokens, max_answer_tokens

    @property
    def wikitext(self) -> "W.WikiText":
        with self._wt_lock:
            if self._wt is None:
                self._wt = W.WikiText(cache_dir=self.data_dir / "text")
            return self._wt

    def warm_up(self) -> None:
        self.hybrid.warm_up()
        self.embedder.encode(["warm up"], query=True)

    # -- steps (public so the screen/CLI and tests can use them separately)
    def article_passages(self, result: Result, query: str) -> list[Passage]:
        raw = self.wikitext.fetch(result.page_id)
        if not raw:
            return []
        secs = W.sections(raw, W.DROP_SECTIONS)
        return prefilter(split_passages(result.title, result.page_id, secs), query)

    def rank_passages(self, query: str, passages: list[Passage]) -> list[Passage]:
        """Score passages (in place) against the query: embedding cosine plus a little word overlap."""
        if not passages:
            return []
        q = self.embedder.encode([query], query=True)[0]
        docs = self.embedder.encode([f"{p.path}: {p.text}" for p in passages])
        cos = docs @ q
        qt = _tokens(query)
        for p, c in zip(passages, np.asarray(cos).tolist()):
            p.score = float(c) + 0.1 * lexical_overlap(qt, p.text)
        return sorted(passages, key=lambda p: -p.score)

    def retrieve(self, query: str, timings: dict | None = None) -> tuple[list[Source], list[Result]]:
        """(numbered sources for the prompt, all candidate articles considered). `timings` gets the
        milliseconds of each step."""
        timings = timings if timings is not None else {}
        t = time.time()
        candidates = self.hybrid.search(query, self.n_candidates)
        timings["search"], t = (time.time() - t) * 1000, time.time()
        passages: list[Passage] = []
        for r in candidates[:self.n_articles]:
            passages += self.article_passages(r, query)
        timings["read"], t = (time.time() - t) * 1000, time.time()
        timings["passages"] = len(passages)
        ranked = self.rank_passages(query, passages)
        timings["rerank"] = (time.time() - t) * 1000
        return pack(ranked, self.budget_tokens), candidates

    # -- the whole thing
    def ask(self, question: str, conversation: Conversation | None = None) -> Iterator[Event]:
        """Answer a question, yielding status / sources / token events and finally ("done", Answer)."""
        question = question.strip()
        history = list(conversation.turns) if conversation else []
        if not question:
            return
        if self.backend is None:
            raise LLMError("No language model is configured")
        t0 = time.time()
        yield Event("status", "Understanding the question…" if history else "Searching Wikipedia…")
        query = rewrite_query(question, history, self.backend)
        if query != question:
            yield Event("status", f"Searching for: {query}")
        timings: dict[str, float] = {}
        sources, candidates = self.retrieve(query, timings)
        retrieval_ms = (time.time() - t0) * 1000
        yield Event("sources", sources)
        answer = Answer(question, query, "", "", sources, [], [], candidates, retrieval_ms, timings,
                        backend=self.backend.name, model=self.backend.model)
        if not sources:
            answer.text = answer.raw_text = "I couldn't find anything about that in the local Wikipedia."
            yield Event("token", answer.text)
            yield Event("done", answer)
            if conversation is not None:
                conversation.add(question, answer.text)
            return
        yield Event("status", f"Asking {self.backend.name} ({self.backend.model})…")
        parts: list[str] = []
        try:
            for piece in self.backend.stream(SYSTEM_PROMPT, build_messages(question, sources, history),
                                             self.max_answer_tokens):
                parts.append(piece)
                yield Event("token", piece)
        except LLMError as e:
            answer.error = str(e)
        answer.generation = self.backend.info
        answer.notes = list(self.backend.info.notes)
        answer.raw_text = "".join(parts).strip()
        answer.text, answer.cited, answer.invalid = validate_citations(answer.raw_text, len(sources))
        if answer.invalid:
            answer.notes.append("removed citations to sources that don't exist: " +
                                ", ".join(f"[{n}]" for n in answer.invalid))
        if answer.text and sources and not answer.cited and not answer.error:
            answer.notes.append("the answer cites no sources")
        yield Event("done", answer)
        if conversation is not None and answer.text:
            conversation.add(question, answer.text)

    def answer(self, question: str, conversation: Conversation | None = None) -> Answer:
        """Blocking version of ask(): the final Answer."""
        final = None
        for ev in self.ask(question, conversation):
            if ev.kind == "done":
                final = ev.data
        return final
