"""Keyword search over article titles and leads (SQLite FTS5, built by pipeline/build_keyword.py).

    kw = KeywordIndex(data_dir)
    for hit in kw.search("who designed the Tacoma Narrows Bridge", k=10):
        hit.page_id, hit.title, hit.score, hit.snippet     # same Hit as wikiexp.semantic

Questions are not search syntax, so the query is built from the question's words: stop words ("who",
"the", ...) go, each remaining word is looked up in the index vocabulary (stemmed exactly as the index
stems it) to learn how common it is, and words that appear in more than ~5% of articles go too unless
nothing else is left. The rarest few are then ANDed together (every word must appear); if that finds
fewer than k articles, the rarest ORed (any may appear) tops it up. Both rank by bm25 with the title
column weighted above the lead. That keeps every query to a handful of short posting lists, so it
takes milliseconds even over all 7.2M articles.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path

from . import paths
from .semantic import SNIPPET_CHARS, Hit
from .wikitext import trim

TITLE_WEIGHT = 8.0       # bm25 column weights: a word in the title counts 8x one in the lead
COMMON_FRACTION = 0.05   # a word in more than this share of articles is treated as a stop word
MAX_AND, MAX_OR = 6, 4

STOP_WORDS = frozenset("""
a about above after again all also am an and any are aren't as at be because been before being below between
both but by can can't cannot could did didn't do does doesn't doing don't down during each few for from
further get gets had has have having he her here hers him his how i if in into is isn't it its just let me
more most my no nor not of off on once only or other our out over own same she should so some such than that
the their them then there these they this those through to too under until up us very was we were what
when where which while who whom whose why will with would you your yours tell give name explain describe
many much did like known
""".split())
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.U)


def fts_path(data_dir: Path | None = None) -> Path:
    return Path(data_dir or paths.DATA) / "text" / "fts.sqlite"


def question_words(text: str) -> list[str]:
    """The distinct content words of a question (lower case, original order)."""
    seen, out = set(), []
    for w in _WORD.findall(text.lower()):
        w = w.replace("’", "'")
        if w not in STOP_WORDS and re.sub(r"'s$", "", w) not in STOP_WORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


class KeywordIndex:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or paths.DATA)
        self.path = fts_path(self.data_dir)
        self.leads_path = self.data_dir / "text" / "leads.sqlite"
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self.articles = 0

    def available(self) -> bool:
        return self.path.exists() and self.leads_path.exists()

    def _connect(self) -> sqlite3.Connection:
        if self._db is not None:
            return self._db
        if not self.available():
            raise FileNotFoundError(f"{self.path} not found - run the 'keyword' stage "
                                    f"(python -m pipeline.build_keyword)")
        db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        db.execute("ATTACH ? AS leads", (f"file:{self.leads_path}?mode=ro",))
        # vocabulary of the index, and a scratch FTS table with the same tokenizer to stem query words
        db.execute("CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main, fts, 'row')")
        tok = db.execute("SELECT value FROM main.meta WHERE key = 'tokenizer'").fetchone()[0]
        db.execute(f"CREATE VIRTUAL TABLE temp.q USING fts5(t, tokenize='{tok}')")
        db.execute("CREATE VIRTUAL TABLE temp.qvocab USING fts5vocab(temp, q, 'row')")
        self.articles = int(db.execute("SELECT value FROM main.meta WHERE key = 'articles'").fetchone()[0])
        built = db.execute("SELECT value FROM main.meta WHERE key = 'leads_built'").fetchone()
        theirs = db.execute("SELECT value FROM leads.meta WHERE key = 'built'").fetchone()
        if built and theirs and built[0] != theirs[0]:
            raise RuntimeError("fts.sqlite was built from an older leads.sqlite; rebuild it "
                               "(python -m pipeline.build_keyword)")
        self._db = db
        return db

    def __len__(self) -> int:
        with self._lock:
            self._connect()
        return self.articles

    def doc_freq(self, word: str) -> int:
        """Number of articles containing the word (any inflection the stemmer folds together)."""
        with self._lock:
            return self._doc_freq(self._connect(), word)

    def _doc_freq(self, db, word: str) -> int:
        db.execute("DELETE FROM temp.q")
        db.execute("INSERT INTO temp.q(t) VALUES (?)", (word,))
        stems = [r[0] for r in db.execute("SELECT term FROM temp.qvocab")]
        if not stems:
            return 0
        # a hyphenated word is several stems that must all be present: its rarest one bounds it
        return min((db.execute("SELECT doc FROM temp.vocab WHERE term = ?", (s,)).fetchone() or (0,))[0]
                   for s in stems)

    def plan(self, query: str) -> list[tuple[str, int]]:
        """[(word, doc frequency)] of the words the search will use, rarest first."""
        with self._lock:
            db = self._connect()
            words = [(w, self._doc_freq(db, w)) for w in question_words(query)]
        words = [(w, df) for w, df in words if df > 0]
        rare = [(w, df) for w, df in words if df <= COMMON_FRACTION * self.articles]
        use = rare if rare else words
        use.sort(key=lambda x: x[1])
        return use[:MAX_AND]

    def _match(self, db, expr: str, limit: int) -> list[tuple[int, float]]:
        rows = db.execute("SELECT rowid, bm25(fts, ?, 1.0) AS s FROM fts WHERE fts MATCH ? "
                          "ORDER BY s LIMIT ?", (TITLE_WEIGHT, expr, limit)).fetchall()
        return [(r[0], -r[1]) for r in rows]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The k articles whose title/lead best match the question's words, best first."""
        if k <= 0 or not query.strip():
            return []
        words = self.plan(query)
        if not words:
            return []
        quote = lambda w: '"' + w.replace('"', "") + '"'      # noqa: E731
        with self._lock:
            db = self._connect()
            found = self._match(db, " AND ".join(quote(w) for w, _ in words), k)
            if len(found) < k and len(words) > 1:
                seen = {r for r, _ in found}
                extra = self._match(db, " OR ".join(quote(w) for w, _ in words[:MAX_OR]), k * 2)
                # ORed matches never outrank articles that contain every word
                floor = min((s for _, s in found), default=None)
                for rank, s in extra:
                    if rank not in seen and len(found) < k:
                        found.append((rank, s if floor is None else min(s, floor * 0.999)))
                        seen.add(rank)
            return self._hits(db, found)

    def _hits(self, db, found: list[tuple[int, float]]) -> list[Hit]:
        if not found:
            return []
        ranks = [r for r, _ in found]
        rows = db.execute(f"SELECT rank, page_id, title, lead FROM leads.leads "
                          f"WHERE rank IN ({','.join('?' * len(ranks))})", ranks).fetchall()
        by_rank = {r[0]: r for r in rows}
        out = []
        for rank, score in found:
            if rank in by_rank:
                _, pid, title, lead = by_rank[rank]
                out.append(Hit(pid, title, score, trim(lead, SNIPPET_CHARS), lead))
        return out

    def popularity_rank(self, page_ids) -> dict[int, int]:
        """{page_id: rank} (0 = most linked-to article) for ids in leads.sqlite."""
        ids = [int(p) for p in page_ids]
        if not ids:
            return {}
        with self._lock:
            db = self._connect()
            rows = db.execute(f"SELECT page_id, rank FROM leads.leads WHERE page_id IN ({','.join('?' * len(ids))})",
                              ids).fetchall()
        return {r[0]: r[1] for r in rows}
