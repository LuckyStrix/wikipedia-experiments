"""Links an editor wrote by hand in an article's prose, as opposed to every link on the page.

The pagelinks dump counts a link whenever the *rendered* page contains it, so a citation template
that links ISBN, an infobox that links its country, and a navbox with 300 links all count the same
as a link in a sentence. This module reads the raw wikitext and keeps only the ``[[Target|text]]``
links that sit in running text: not inside ``{{templates}}``, ``<ref>`` notes, ``{| tables |}``,
``<gallery>``/``<imagemap>``/``<math>`` and friends, ``[[File:...]]`` captions or HTML comments, and
not ``[[Category:...]]`` or interlanguage links.

    lookup = TitleLookup.from_titles({"Paris": 7, "Eiffel Tower": 9})        # or .from_db(...)
    links = page_links(wikitext, lookup, self_idx=9)
    links.dst       # unique target idx, sorted (so it drops straight into a CSR row)
    links.order     # rank of each dst's first appearance in the article (0 = first)
    links.n_lead    # how many of them first appear before the first heading (order < n_lead)
    links.first     # idx of the first link outside parentheses/italics ("Getting to Philosophy"), or -1

``extract_links(wikitext)`` is the same without the lookup: the normalised titles in order, each with
its lead and first-link flags. Used by pipeline/build_prose_graph.py.

Targets are normalised like MediaWiki does: HTML entities and %XX decoded, underscores and runs of
spaces collapsed, ``#section`` dropped, a leading ``:`` dropped, the first letter capitalised. They
are then resolved to a graph idx through article titles and redirects. Titles are looked up by a
64-bit hash in a sorted numpy array (19 M titles take ~230 MB and, unlike a dict, forked workers
share it without ever copying a page). Hashes use Python's per-process string hash, so a lookup is
only valid inside the process tree that built it: never save one.
"""
from __future__ import annotations

import html
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, NamedTuple
from urllib.parse import unquote

import numpy as np

# ── stripping what isn't prose ───────────────────────────────────────────────

_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.S)
_NOWIKI = re.compile(r"<nowiki\b[^>]*?(?:/>|>.*?(?:</nowiki\s*>|\Z))", re.S | re.I)
# tags whose whole content is dropped (an unterminated <ref> is shown literally by MediaWiki, so it stays)
_DROP_TAGS = re.compile(r"<(ref|gallery|math|timeline|imagemap|score|syntaxhighlight|source|hiero|"
                        r"templatedata|mapframe|maplink|references|chem|ce|pre|graph|inputbox)\b[^>]*?"
                        r"(?:/>|>.*?</\1\s*>)", re.S | re.I)
_BRACES = re.compile(r"\{\{|\}\}")
_FILE_START = re.compile(r"\[\[\s*(?:file|image|media)\s*:", re.I)
_BRACKETS = re.compile(r"\[\[|\]\]")


def drop_templates(text: str) -> str:
    """Remove every outermost ``{{...}}`` (however deeply nested inside). An opening ``{{`` that is
    never closed is literal text, as on the real site, and only that brace pair is kept."""
    if "{{" not in text:
        return text
    out, pos, depth, start = [], 0, 0, 0
    while True:
        for m in _BRACES.finditer(text, pos):
            if m.group() == "{{":
                if depth == 0:
                    out.append(text[pos:m.start()])
                    start = m.start()
                depth += 1
            elif depth:
                depth -= 1
                if depth == 0:
                    pos = m.end()
        if depth == 0:
            out.append(text[pos:])
            return "".join(out)
        text, pos, depth = text[start + 2:], 0, 0     # the outermost opener never closed: it's literal


def drop_tables(text: str) -> str:
    """Remove ``{| ... |}`` tables (they nest); both markers must start their line."""
    if "{|" not in text:
        return text
    out, depth = [], 0
    for line in text.split("\n"):
        s = line.lstrip()
        if s.startswith("{|"):
            depth += 1
        elif depth:
            if s.startswith("|}"):
                depth -= 1
        else:
            out.append(line)
    return "\n".join(out)


def drop_file_links(text: str) -> str:
    """Remove ``[[File:...]]``/``[[Image:...]]``, whose captions hold links of their own."""
    out, pos = [], 0
    while (m := _FILE_START.search(text, pos)):
        depth, end = 0, -1
        for b in _BRACKETS.finditer(text, m.start()):
            depth += 1 if b.group() == "[[" else -1
            if depth == 0:
                end = b.end()
                break
        if end < 0:                       # never closed: drop the rest of that line only
            nl = text.find("\n", m.end())
            end = len(text) if nl < 0 else nl
        out.append(text[pos:m.start()])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def strip_wikitext(text: str) -> str:
    """The wikitext with comments, nowiki, refs and other dropped tags, templates, tables and file
    links removed; what remains is headings, running text and the links in it."""
    text = _COMMENT.sub("", text)
    if "<" in text:
        text = _NOWIKI.sub("", text)
        text = _DROP_TAGS.sub("", text)
    text = drop_templates(text)
    text = drop_tables(text)
    if "[[" in text:
        text = drop_file_links(text)
    return text


# ── finding and normalising links ────────────────────────────────────────────

_LINK = re.compile(r"\[\[([^\[\]|{}<>\n]+)(?:\|[^\[\]]*)?\]\]")
_HEADING = re.compile(r"^={2,6}[^=\n]", re.M)
_APOSTROPHES = re.compile(r"'{2,}")
_SPACES = re.compile(r"[ _\t   -​‎‏  　]+")
_PERCENT = re.compile(r"%[0-9A-Fa-f]{2}")
# namespaces whose links never lead to an article, and interlanguage prefixes (`fr:`, `zh-min-nan:`;
# lowercase only, so titles like "CSI: Miami" are left alone). Anything else that isn't an article
# (wikt:, commons:, ...) simply fails the lookup.
_NOT_ARTICLE = re.compile(r"(?:category|file|image|media|template|wikipedia|wp|help|portal|talk|user|draft|"
                          r"special|module|mediawiki)\s*:", re.I)
_INTERWIKI = re.compile(r"[a-z]{2,3}(?:-[a-z]+)*\s*:")


def normalize_title(raw: str) -> str | None:
    """The article title a link target points at, or None if it can't be an article (an in-page
    ``#section`` link, a namespace or interlanguage link, an empty or impossible title)."""
    t = raw
    if "&" in t:
        t = html.unescape(t)
    if "%" in t and _PERCENT.search(t):
        t = unquote(t)
    if "#" in t:
        t = t.partition("#")[0]
    t = _SPACES.sub(" ", t).strip()
    if t.startswith(":"):
        t = t[1:].lstrip()       # [[:Foo]]: a plain link even where [[Foo]] would mean something else
    if ":" in t and (_NOT_ARTICLE.match(t) or _INTERWIKI.match(t)):
        return None
    if not t or len(t) > 255 or t[0] == "/":
        return None
    if not t.isascii():
        t = unicodedata.normalize("NFC", t)
    return t[0].upper() + t[1:]


class Link(NamedTuple):
    target: str        # normalised title
    lead: bool         # before the first heading
    plain: bool        # not in parentheses, italics or an indented (hatnote-style) line


def extract_links(wikitext: str) -> list[Link]:
    """Every prose link of an article in order of appearance (repeats included).

    ``plain`` follows the rules of the "Getting to Philosophy" game: a link counts unless it is
    inside parentheses or italics, or on a line that starts with ``:`` or ``;`` (old-style hatnotes).
    Parentheses and quote marks are tracked per line, so one stray ``(`` can't hide the rest of an
    article."""
    text = strip_wikitext(wikitext)
    if "[[" not in text:
        return []
    m = _HEADING.search(text)
    lead_end = m.start() if m else len(text)
    out: list[Link] = []
    depth, italic, bold, prev = 0, False, False, 0
    indent = text[:1] in (":", ";")
    for m in _LINK.finditer(text):
        gap = text[prev:m.start()]
        prev = m.end()
        nl = gap.rfind("\n")
        if nl >= 0:
            depth, italic, bold = 0, False, False
            gap = gap[nl + 1:]
            indent = gap[:1] in (":", ";") if gap else False
        if gap:
            depth = max(0, depth + gap.count("(") - gap.count(")"))
            if "''" in gap:
                for a in _APOSTROPHES.findall(gap):
                    n = len(a)
                    if n == 2 or n == 5:
                        italic = not italic
                    if n == 3 or n == 4 or n == 5:
                        bold = not bold
        target = normalize_title(m.group(1))
        if target is not None:
            plain = depth == 0 and not italic and not indent and "''" not in m.group(0)
            out.append(Link(target, m.start() < lead_end, plain))
    return out


# ── resolving titles to graph idx ────────────────────────────────────────────

class TitleLookup:
    """title -> idx for article titles and redirect titles, via sorted 64-bit hashes.

        lookup = TitleLookup.from_db(wiki_sqlite, page_ids)
        lookup(["Paris", "Mitochondria", "No such page"])    # int32 array: [5, 9, -1]
    """

    def __init__(self, hashes: np.ndarray, idxs: np.ndarray):
        order = np.argsort(hashes, kind="stable")
        self.hashes = hashes[order]
        self.idxs = idxs[order].astype(np.int32)

    def __len__(self) -> int:
        return len(self.hashes)

    @classmethod
    def from_titles(cls, articles: dict[str, int], redirects: dict[str, int] | None = None) -> "TitleLookup":
        """From {title: idx} for articles and {redirect title: idx of its target article}."""
        items = list(articles.items()) + list((redirects or {}).items())
        return cls(np.array([hash(t) for t, _ in items], dtype=np.int64),
                   np.array([i for _, i in items], dtype=np.int32))

    @classmethod
    def from_db(cls, db_path: Path | str, page_ids: np.ndarray,
                say: Callable[[str], None] | None = None) -> "TitleLookup":
        """From wiki.sqlite (articles.title -> idx, redirects.title -> idx of target) where
        ``page_ids[idx]`` is each article's page id, ascending (graph/idx_to_page_id.npy)."""
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        hashes, idxs = [], []

        def add(titles, ids):
            hashes.append(np.fromiter((hash(t) for t in titles), dtype=np.int64, count=len(titles)))
            idxs.append(np.asarray(ids, dtype=np.int32))

        cur = db.execute("SELECT title, idx FROM articles")
        while rows := cur.fetchmany(500_000):
            add([r[0] for r in rows], [r[1] for r in rows])
        n_articles = sum(len(h) for h in hashes)
        if say:
            say(f"  {n_articles:,} article titles")
        cur = db.execute("SELECT title, target FROM redirects")
        while rows := cur.fetchmany(500_000):
            pos = np.searchsorted(page_ids, np.array([r[1] for r in rows], dtype=np.int64))
            pos = np.minimum(pos, len(page_ids) - 1)
            ok = page_ids[pos] == [r[1] for r in rows]       # a redirect to a page that isn't an article
            add([r[0] for r, k in zip(rows, ok) if k], pos[ok])
        if say:
            say(f"  {sum(len(h) for h in hashes) - n_articles:,} redirect titles")
        db.close()
        return cls(np.concatenate(hashes), np.concatenate(idxs))

    def __call__(self, titles: list[str]) -> np.ndarray:
        """int32 idx of each normalised title, -1 where there is no such article or redirect."""
        h = np.fromiter((hash(t) for t in titles), dtype=np.int64, count=len(titles))
        if not len(self.hashes):
            return np.full(len(titles), -1, dtype=np.int32)
        pos = np.searchsorted(self.hashes, h)
        pos[pos == len(self.hashes)] = 0
        return np.where(self.hashes[pos] == h, self.idxs[pos], -1).astype(np.int32)


@dataclass
class PageLinks:
    dst: np.ndarray       # int32, unique target idx, ascending
    order: np.ndarray     # uint16, appearance rank of each dst (0 = the article's first link; capped)
    n_lead: int           # links whose first appearance is before the first heading: order < n_lead
    first: int            # idx of the first plain link (not in parentheses/italics), or -1
    raw: int = 0          # prose links found, repeats and red links included
    resolved: int = 0     # of those, how many pointed at an article (or a redirect to one)


EMPTY = PageLinks(np.empty(0, np.int32), np.empty(0, np.uint16), 0, -1)


def page_links(wikitext: str, lookup: TitleLookup, self_idx: int = -1) -> PageLinks:
    """Resolve an article's prose links. Links to itself, to red links and to non-articles are
    dropped; a repeated link keeps its first appearance."""
    links = extract_links(wikitext)
    if not links:
        return EMPTY
    idx = lookup([l.target for l in links])
    hit = idx >= 0
    resolved = int(hit.sum())
    keep = hit & (idx != self_idx)
    if not keep.any():
        return PageLinks(EMPTY.dst, EMPTY.order, 0, -1, len(links), resolved)
    idx = idx[keep]
    lead = np.fromiter((l.lead for l, k in zip(links, keep) if k), dtype=bool, count=len(idx))
    plain = np.fromiter((l.plain for l, k in zip(links, keep) if k), dtype=bool, count=len(idx))
    uniq, first_pos = np.unique(idx, return_index=True)      # uniq ascending, first_pos = first occurrence
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[np.argsort(first_pos, kind="stable")] = np.arange(len(uniq))
    first = int(idx[np.argmax(plain)]) if plain.any() else -1
    return PageLinks(uniq, np.minimum(rank, 65535).astype(np.uint16), int(lead[first_pos].sum()), first,
                     len(links), resolved)
