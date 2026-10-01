"""Article wikitext from the multistream dump, and turning it into clean plain text.

``pages-articles-multistream.xml.bz2`` is many independent bz2 streams of ~100 pages each, and
``pages-articles-multistream-index.txt.bz2`` lists ``offset:page_id:title`` for every page. So one
article can be read by decompressing just its stream (~1 MB of XML, a few tens of ms), and the whole
dump can be streamed by many processes at once, each taking different streams.

    wt = WikiText()                      # reads (and caches) the index
    wt.fetch(12)                         # wikitext of page 12, random access
    for page_id, title, lead in wt.iter_pages(fn=lead_of):   # every article, in parallel
        ...
    to_text(wikitext)                    # clean plain text
    sections(wikitext)                   # [(heading, text), ...]; the first heading is ""

Cleaning (``clean``/``to_text``/``sections``) drops templates, references, files/images, categories,
tables, galleries, math and comments, and keeps the text of links. A handful of templates that carry
prose are rendered instead of dropped (``{{lang|fr|bonjour}}``, ``{{convert|5|km|mi}}``, dates, ...).
"""
from __future__ import annotations

import bz2
import collections
import functools
import gc
import html
import json
import multiprocessing as mp
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import mwparserfromhell
import numpy as np

from . import paths
from .sqldump import default_workers

DUMP_NAME = "pages-articles-multistream.xml.bz2"
INDEX_NAME = "pages-articles-multistream-index.txt.bz2"


@dataclass(frozen=True)
class Page:
    page_id: int
    title: str
    ns: int
    redirect: str | None      # target title if this page is a redirect
    text: str                 # wikitext


# ── reading the dump ─────────────────────────────────────────────────────────

_ENTITIES = (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'"), ("&amp;", "&"))


def _unescape(s: str) -> str:
    """Undo the XML escaping of the dump (once; `&amp;` goes last so `&amp;lt;` stays `&lt;`)."""
    if "&" in s:
        for a, b in _ENTITIES:
            s = s.replace(a, b)
    return s


_HEAD = re.compile(rb"<title>(.*?)</title>\s*<ns>(\d+)</ns>\s*<id>(\d+)</id>\s*(?:<redirect title=\"(.*?)\"\s*/>)?")


def parse_stream(xml: bytes) -> Iterator[Page]:
    """The pages in one decompressed stream (a run of `<page>…</page>` elements)."""
    pos = 0
    while (start := xml.find(b"<page>", pos)) >= 0:
        end = xml.find(b"</page>", start)
        if end < 0:
            break
        pos = end + 7
        m = _HEAD.search(xml, start, min(end, start + 2000))
        if not m:
            continue
        # <text bytes="..." xml:space="preserve">…</text>, or <text … /> for an empty page
        t0 = xml.find(b"<text", m.end(), end)
        text = ""
        if t0 >= 0:
            gt = xml.find(b">", t0, end)
            if xml[gt - 1:gt] != b"/":
                text = xml[gt + 1:xml.rfind(b"</text>", gt, end)].decode("utf-8", "replace")
        red = m.group(4)
        yield Page(int(m.group(3)), _unescape(m.group(1).decode("utf-8", "replace")), int(m.group(2)),
                   _unescape(red.decode("utf-8", "replace")) if red is not None else None,
                   _unescape(text))


def _read_range(path: Path, start: int, length: int | None) -> bytes:
    with open(path, "rb") as f:        # opened per call: safe from any thread, cheap
        f.seek(start)
        return f.read() if length is None else f.read(length)


# ── the index ────────────────────────────────────────────────────────────────

_INDEX_LINE = re.compile(rb"^(\d+):(\d+):[^\n]*\n", re.M)


def _parse_index(path: Path, progress: Callable[[str], None] | None = None):
    """(page_ids, stream_numbers, stream_offsets) from the bz2 index, ids sorted ascending."""
    ids, offs = [], []
    rest = b""
    n = 0
    with bz2.open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            buf = rest + chunk
            cut = buf.rfind(b"\n") + 1
            rest = buf[cut:]
            # keep "offset id" of every line, as text numpy can parse in one go
            nums = np.fromstring(_INDEX_LINE.sub(rb"\1 \2 ", buf[:cut]), dtype=np.int64, sep=" ") \
                if cut else np.empty(0, dtype=np.int64)
            offs.append(nums[0::2])
            ids.append(nums[1::2])
            n += len(nums) // 2
            if progress:
                progress(f"{n:,} index entries")
    offs, ids = np.concatenate(offs), np.concatenate(ids)
    stream_offsets, stream_no = np.unique(offs, return_inverse=True)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    return (ids.astype(np.int32 if ids.max() < 2 ** 31 else np.int64),
            stream_no[order].astype(np.uint32), stream_offsets)


class WikiText:
    """Random access to, and streaming of, the multistream dump through its index.

    The index (25 M lines) is parsed once into three numpy arrays cached under ``<data>/text/``
    and memory-mapped afterwards, so opening is instant. Safe to use from several threads.
    """

    def __init__(self, dump: Path | None = None, index: Path | None = None,
                 cache_dir: Path | None = None, progress: Callable[[str], None] | None = None):
        self.dump = Path(dump or paths.dump(DUMP_NAME))
        self.index = Path(index or paths.dump(INDEX_NAME))
        self.cache_dir = Path(cache_dir or paths.DATA / "text")
        for p in (self.dump, self.index):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found - download the article text dumps first")
        self.ids, self.stream_no, self.offsets = self._load_index(progress)
        self.dump_size = self.dump.stat().st_size
        self._fetch_stream = functools.lru_cache(maxsize=16)(self._fetch_stream_uncached)

    # -- index cache
    def _load_index(self, progress):
        d = self.cache_dir
        names = ("index_ids", "index_stream", "index_offsets")
        key = {"index": [self.index.stat().st_size, self.index.stat().st_mtime_ns],
               "dump": self.dump.stat().st_size}
        try:
            if json.loads((d / "index_meta.json").read_text()) == key:
                return tuple(np.load(d / f"{n}.npy", mmap_mode="r") for n in names)
        except (OSError, ValueError):
            pass
        t0 = time.time()
        arrays = _parse_index(self.index, progress)
        try:
            d.mkdir(parents=True, exist_ok=True)
            for n, a in zip(names, arrays):
                np.save(d / f"{n}.tmp.npy", a)
                os.replace(d / f"{n}.tmp.npy", d / f"{n}.npy")
            (d / "index_meta.json").write_text(json.dumps(key))   # last: marks the cache complete
            if progress:
                progress(f"index cached in {time.time() - t0:.0f}s")
        except OSError:
            pass   # read-only data dir: just keep it in memory
        return arrays

    # -- lookups
    def __len__(self) -> int:
        return len(self.ids)

    @property
    def n_streams(self) -> int:
        return len(self.offsets)

    def __contains__(self, page_id: int) -> bool:
        return self.stream_of(page_id) is not None

    def stream_of(self, page_id: int) -> int | None:
        i = int(np.searchsorted(self.ids, page_id))
        return int(self.stream_no[i]) if i < len(self.ids) and self.ids[i] == page_id else None

    def _stream_bytes(self, n: int) -> bytes:
        start = int(self.offsets[n])
        end = int(self.offsets[n + 1]) if n + 1 < len(self.offsets) else self.dump_size
        return _read_range(self.dump, start, end - start)

    def _fetch_stream_uncached(self, n: int) -> dict[int, Page]:
        return {p.page_id: p for p in parse_stream(bz2.decompress(self._stream_bytes(n)))}

    def stream_pages(self, n: int) -> list[Page]:
        """All pages (every namespace, redirects too) of stream number `n`."""
        return list(self._fetch_stream(n).values())

    def page(self, page_id: int) -> Page | None:
        """The page with this id (any namespace, redirects included), or None."""
        n = self.stream_of(page_id)
        return None if n is None else self._fetch_stream(n).get(int(page_id))

    def fetch(self, page_id: int) -> str | None:
        """Wikitext of one page, or None if the id isn't in the dump. Decompresses one stream
        (the last 16 are cached, so neighbouring articles and repeat calls are free)."""
        p = self.page(page_id)
        return p.text if p else None

    def fetch_many(self, page_ids) -> dict[int, str]:
        """{page_id: wikitext}, reading each needed stream once."""
        out = {}
        for pid in page_ids:
            if (t := self.fetch(pid)) is not None:
                out[int(pid)] = t
        return out

    # -- streaming everything
    def iter_pages(self, fn: Callable[[Page], object] | None = None, workers: int | None = None,
                   streams: range | None = None, namespace: int | None = 0, redirects: bool = False,
                   limit: int | None = None) -> Iterator[tuple[int, str, object]]:
        """Yield ``(page_id, title, fn(page))`` for main-namespace non-redirect pages, in dump order.

        ``fn`` (default: the wikitext) runs inside the worker processes, so use it to boil a page
        down (a lead, a length, ...) rather than shipping 100 GB of wikitext to the parent. Results
        that are None are skipped. ``streams`` restricts to a range of stream numbers, ``namespace``
        None means every namespace, ``redirects=True`` includes redirects, ``limit`` stops early.

        Streams are spread over forked workers (at most 2 batches per worker in flight, so memory
        stays bounded); with one worker, or where fork doesn't exist (Windows), it's a plain loop.
        """
        streams = streams if streams is not None else range(self.n_streams)
        workers = workers or default_workers()
        job = _Job(self, fn, namespace, redirects)
        batches = [streams[i:i + BATCH] for i in range(0, len(streams), BATCH)]
        count = 0
        if workers <= 1 or "fork" not in mp.get_all_start_methods() or len(batches) <= 1:
            results = (_run_batch(job, b) for b in batches)
            pool = None
        else:
            global _JOB
            _JOB = job                 # forked workers inherit it, so nothing big is pickled
            gc.freeze()
            pool = mp.get_context("fork").Pool(workers)
            pending = collections.deque()

            def gen():
                for b in batches:
                    pending.append(pool.apply_async(_pool_batch, (b,)))
                    if len(pending) >= 2 * workers:
                        yield pending.popleft().get()
                while pending:
                    yield pending.popleft().get()
            results = gen()
        try:
            for rows in results:
                for row in rows:
                    yield row
                    count += 1
                    if limit and count >= limit:
                        return
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
                gc.unfreeze()


BATCH = 8     # streams per task: big enough to amortise the hand-off, small enough to balance load
_JOB = None


@dataclass
class _Job:
    wt: WikiText
    fn: Callable | None
    namespace: int | None
    redirects: bool


def _run_batch(job: _Job, batch) -> list[tuple[int, str, object]]:
    out = []
    for n in batch:
        xml = bz2.decompress(job.wt._stream_bytes(n))
        for p in parse_stream(xml):
            if (job.namespace is not None and p.ns != job.namespace) or (p.redirect is not None and not job.redirects):
                continue
            r = job.fn(p) if job.fn else p.text
            if r is not None:
                out.append((p.page_id, p.title, r))
    return out


def _pool_batch(batch):
    return _run_batch(_JOB, batch)


# ── cleaning wikitext ────────────────────────────────────────────────────────

_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.S)
_HEADING = re.compile(r"^(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*$", re.M)
# tags whose whole content is dropped (the rest keep their content)
_DROP_TAGS = re.compile(r"<(ref|gallery|math|timeline|imagemap|score|syntaxhighlight|source|hiero|"
                        r"templatedata|mapframe|maplink|references|chem|ce|pre|graph)\b[^>]*?"
                        r"(?:/>|>.*?</\1\s*>)", re.S | re.I)
_REF_OPEN = re.compile(r"<ref\b[^>/]*>.*\Z", re.S | re.I)     # unterminated <ref>
_INNER_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")
_PROTECT_LINK_PIPES = re.compile(r"\[\[[^\[\]]*\]\]")
_MAGIC = re.compile(r"__[A-Z_]+__")
_INCLUDE_TAGS = re.compile(r"</?(?:noinclude|includeonly|onlyinclude|nowiki)\s*>", re.I)
_LIST_MARK = re.compile(r"^[*:;]+\s*")
_APOSTROPHES = re.compile(r"'{2,}")      # bold/italic marks left behind when their content was a dropped template
_SPACES = re.compile("[ \t\u00a0\u2000-\u200a\u202f]+")
_NON_ARTICLE_LINK = re.compile(r"^\s*(file|image|media|category)\s*:", re.I)
_INTERWIKI = re.compile(r"^\s*(?:[a-z]{2,3}|simple|zh-[a-z-]+|be-tarask|roa-[a-z]+|fiu-vro|map-bms|"
                        r"bat-smg|nds-nl|cbk-zam):[^\s]", re.I)

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def _split_args(body: str) -> tuple[str, list[str], dict[str, str]]:
    """Template body 'name|a|b=c' -> (name, positional args, named args), ignoring pipes in [[links]]."""
    body = _PROTECT_LINK_PIPES.sub(lambda m: m.group(0).replace("|", "\x00"), body)
    parts = [p.replace("\x00", "|") for p in body.split("|")]
    pos, named = [], {}
    for p in parts[1:]:
        k, eq, v = p.partition("=")
        if eq and re.fullmatch(r"\s*[\w -]+\s*", k):
            named[k.strip().lower()] = v.strip()
        else:
            pos.append(p.strip())
    return parts[0].strip(), pos, named


def _first(pos, named, *keys):
    return pos[0] if pos else next((named[k] for k in keys if named.get(k)), "")


_NUM = re.compile(r"[-+\d.,/ ]+")
_RANGE = {"to", "and", "or", "by", "x", "×", "-", "–", "+", "±", "to about", "and about"}


def _convert(pos, named):
    """{{convert|10|to|20|km|mi}} -> "10 to 20 km": the numbers up to and including the first unit."""
    out = []
    for tok in pos:
        out.append(tok)
        if not (_NUM.fullmatch(tok) or tok in _RANGE):
            break
    return " ".join(out)


def _as_of(pos, named):
    if named.get("alt"):
        return named["alt"]
    if not pos:
        return ""
    word = "since" if named.get("since") == "y" else "as of"
    return f"{word if named.get('lc') == 'y' else word.capitalize()} {pos[0]}"


def _nihongo(pos, named):
    """{{nihongo|English|kanji|romaji}}: just the English text (plus the template's `post=` comma)."""
    return (pos[0] if pos else "") + named.get("post", "")


def _date(pos, named):
    nums = [int(p) for p in pos if p.isdigit()]
    if not nums:
        return ""
    if len(nums) >= 3 and 1 <= nums[1] <= 12:
        return f"{MONTHS[nums[1] - 1]} {nums[2]}, {nums[0]}"
    if len(nums) == 2 and 1 <= nums[1] <= 12:
        return f"{MONTHS[nums[1] - 1]} {nums[0]}"
    return str(nums[0])


def _lang(pos, named):
    return pos[1] if len(pos) > 1 else named.get("text", "")


_DATE_NAMES = ("birth date and age", "birth date", "death date and age", "death date", "bda", "dob",
               "birth-date", "start date", "end date", "start date and age", "end date and age",
               "birth year and age", "death year and age", "birth based on age as of date")
_FIRST_ARG = ("nowrap", "nobr", "small", "smaller", "larger", "big", "huge", "tiny", "em", "strong",
              "sc", "smallcaps", "abbr", "sup", "sub", "lower", "upper", "math", "mvar", "var", "nobold",
              "noitalic", "no wrap", "nowrap begin", "bold", "italic", "italics", "bi", "b", "i", "u", "mono",
              "ill", "interlanguage link", "interlanguage link multi", "iw", "link-interwiki",
              "linktext", "lang-en", "unicode", "script", "wikt", "sic", "not a typo", "typo", "tooltip", "ruby")
_TEMPLATES: dict[str, Callable] = {
    "lang": _lang, "langx": _lang, "-\"": lambda p, n: '"', "\"": lambda p, n: '"', "-'": lambda p, n: "'",
    "pi": lambda p, n: "π", "tau": lambda p, n: "τ", "!": lambda p, n: "|", "=": lambda p, n: "=", "transl": lambda p, n: p[-1] if p else "", "transliteration": lambda p, n: p[-1] if p else "",
    "nihongo": _nihongo, "nihongo foot": _nihongo, "nihongo2": _nihongo, "nihongo3": _nihongo,
    "convert": _convert, "cvt": _convert, "val": lambda p, n: (p[0] + (" " + n["u"] if n.get("u") else "")) if p else "",
    "circa": lambda p, n: "c. " + p[0] if p else "c.", "c.": lambda p, n: "c. " + p[0] if p else "c.",
    "frac": lambda p, n: "/".join(p[:2]) if len(p) > 1 else (p[0] if p else ""),
    "sfrac": lambda p, n: "/".join(p[-2:]) if len(p) > 1 else (p[0] if p else ""),
    "snd": lambda p, n: " – ", "spnd": lambda p, n: " – ", "spaced ndash": lambda p, n: " – ",
    "spaced en dash": lambda p, n: " – ", "ndash": lambda p, n: "–", "mdash": lambda p, n: "—",
    "nbsp": lambda p, n: " ", "spaces": lambda p, n: " ", "'": lambda p, n: "'", "-": lambda p, n: "",
    "as of": _as_of,
    "quote": lambda p, n: _first(p, n, "text", "quote"), "cquote": lambda p, n: _first(p, n, "text", "quote"),
    "blockquote": lambda p, n: _first(p, n, "text", "quote"), "quotation": lambda p, n: _first(p, n, "text", "quote"),
    "plainlist": lambda p, n: p[0] if p else "", "flatlist": lambda p, n: p[0] if p else "",
    "unbulleted list": lambda p, n: ", ".join(p), "ubl": lambda p, n: ", ".join(p),
    "hlist": lambda p, n: ", ".join(p), "flag": lambda p, n: p[0] if p else "",
    "sortname": lambda p, n: " ".join(p[:2]), "sort": lambda p, n: p[1] if len(p) > 1 else "",
    "us$": lambda p, n: "US$" + p[0] if p else "", "usd": lambda p, n: "US$" + p[0] if p else "",
    "gbp": lambda p, n: "£" + p[0] if p else "", "£": lambda p, n: "£" + p[0] if p else "",
    "eur": lambda p, n: "€" + p[0] if p else "", "€": lambda p, n: "€" + p[0] if p else "",
}
_TEMPLATES.update({n: (lambda p, n_: _date(p, n_)) for n in _DATE_NAMES})
_TEMPLATES.update({n: (lambda p, n_: p[0] if p else n_.get("1", "")) for n in _FIRST_ARG})


def _render_template(m: re.Match) -> str:
    name, pos, named = _split_args(m.group(1))
    key = re.sub(r"[\s_]+", " ", name).strip().lower()
    h = _TEMPLATES.get(key)
    if h is None and key.startswith("lang-"):
        return pos[0] if pos else named.get("text", "")
    if h is None:
        return ""
    try:
        return h(pos, named)
    except Exception:
        return ""


def _drop_templates(text: str) -> str:
    """Replace each template by its rendering (or nothing), innermost first, so nesting just works."""
    for _ in range(12):
        if "{{" not in text:
            break
        new = _INNER_TEMPLATE.sub(_render_template, text)
        if new == text:
            break
        text = new
    return text     # anything still in braces (a stray `{` inside a template) is left to mwparserfromhell


_FILE_START = re.compile(r"\[\[\s*(?:file|image|media|category)\s*:", re.I)
_BRACKETS = re.compile(r"\[\[|\]\]")


def _drop_file_links(text: str) -> str:
    """Remove `[[File:…]]`/`[[Category:…]]`, whose captions can hold nested links and stray markup."""
    out, pos = [], 0
    while (m := _FILE_START.search(text, pos)):
        depth, end = 0, len(text)
        for b in _BRACKETS.finditer(text, m.start()):
            depth += 1 if b.group() == "[[" else -1
            if depth == 0:
                end = b.end()
                break
        out.append(text[pos:m.start()])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _drop_tables(text: str) -> str:
    """Remove `{| … |}` tables (they may nest) line by line."""
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


_EMPTY_PARENS = [
    (re.compile(r"\(\s*\.?\s*[;,:]\s*"), "("), (re.compile(r"\s*[;,:]\s*\)"), ")"),
    (re.compile(r"\(\s*[$£€]?\s*(?:million |billion |trillion )?(?:when )?adjusted for inflation\s*\)"), ""),
    (re.compile(r"\(\s*\)|\[\s*\]"), ""), (re.compile(r"\s+([,.;:!?)])"), r"\1"),
    (re.compile(r"\(\s+"), "("), (re.compile(r",\s*,"), ","), (re.compile(r"\s+,"), ","),
]


def clean(wikitext: str) -> str:
    """Plain text of a piece of wikitext (no headings handling): no templates, refs, files, tables."""
    text = _COMMENT.sub("", wikitext)
    text = _DROP_TAGS.sub("", text)
    if "<ref" in text.lower():
        text = _REF_OPEN.sub("", text)
    text = _drop_tables(text)
    text = _drop_templates(text)
    text = _drop_file_links(text)
    code = mwparserfromhell.parse(text)
    for link in code.filter_wikilinks():
        if _NON_ARTICLE_LINK.match(str(link.title)) or _INTERWIKI.match(str(link.title)):
            try:
                code.remove(link)      # a file's caption holds nested links; those go with it
            except ValueError:
                pass
    text = code.strip_code(normalize=True, collapse=True)
    for junk in ("{{", "}}", "[[", "]]"):             # broken markup: don't show the brackets
        if junk in text:
            text = text.replace(junk, "")
    if "&" in text:
        text = html.unescape(text)                     # entities mwparserfromhell doesn't know, e.g. &hairsp;
    text = _INCLUDE_TAGS.sub("", text)
    text = _MAGIC.sub("", text)
    text = _APOSTROPHES.sub("", text)
    for pat, repl in _EMPTY_PARENS:
        text = pat.sub(repl, text)
    lines = []
    for line in text.split("\n"):
        line = _LIST_MARK.sub("", _SPACES.sub(" ", line).strip())
        if line and not line.startswith("|") and "||" not in line \
                and not re.fullmatch(r"[|!{}\-=*#:;~ ]+|thumb(nail)?(\|.*)?", line):    # table debris
            lines.append(line)
    return "\n".join(lines)


def _headings(text: str):
    """Heading lines that aren't inside a `{{template}}` (infoboxes can hold `==` lines)."""
    depth = last = 0
    for m in _HEADING.finditer(text):
        depth += text.count("{{", last, m.start()) - text.count("}}", last, m.start())
        last = m.start()
        if depth <= 0:
            yield m


def split_sections(wikitext: str) -> list[tuple[str, str]]:
    """[(heading, raw wikitext)] split at `== Heading ==` lines; the first heading is "" (the lead)."""
    wikitext = _COMMENT.sub("", wikitext)
    out, last, heading = [], 0, ""
    for m in _headings(wikitext):
        out.append((heading, wikitext[last:m.start()]))
        heading, last = m.group(2), m.end()
    out.append((heading, wikitext[last:]))
    return out


def lead_wikitext(wikitext: str) -> str:
    """Raw wikitext before the first heading (cheap: used to avoid cleaning whole articles)."""
    wikitext = _COMMENT.sub("", wikitext)
    m = next(_headings(wikitext), None)
    return wikitext[:m.start()] if m else wikitext


# sections that are lists of references/links rather than prose
DROP_SECTIONS = frozenset({"references", "external links", "see also", "further reading", "notes",
                           "bibliography", "footnotes", "sources", "citations", "works cited",
                           "explanatory notes", "general references", "gallery"})


def sections(wikitext: str, drop: frozenset[str] = frozenset()) -> list[tuple[str, str]]:
    """[(heading, plain text)]; the first heading is "" (the lead). Sections whose lowercase
    heading is in `drop` (e.g. DROP_SECTIONS) and sections with no text are left out."""
    out = []
    for i, (h, body) in enumerate(split_sections(wikitext)):
        h = clean(h)
        if h.lower() in drop:
            continue
        text = clean(body)
        if text:
            out.append((h, text))
    return out


def to_text(wikitext: str, drop: frozenset[str] = DROP_SECTIONS, headings: bool = True) -> str:
    """Whole article as plain text, sections separated by blank lines, each headed by its title
    (``headings=False`` leaves the titles out). Reference-like sections are dropped by default."""
    parts = []
    for h, text in sections(wikitext, drop):
        parts.append(f"{h}\n{text}" if h and headings else text)
    return "\n\n".join(parts)


# ── lead text ────────────────────────────────────────────────────────────────

_ABBREV = frozenset("mr mrs ms dr st jr sr vs no inc ltd co corp mt ft gen col lt sgt capt prof rev hon gov "
                    "sen rep etc approx ca c cf fig vol ed eds al ave blvd dept est lat lon min sec "
                    "jan feb mar apr jun jul aug sep sept oct nov dec bros e.g i.e".split())
_SENT_END = re.compile(r"[.!?][\"')\]”’]*\s+(?=[\"'(\[“‘]?[A-Z0-9])")


def sentence_end(text: str, limit: int, floor: int = 0) -> int:
    """Index just after the last sentence end at or before `limit` (and after `floor`), or -1."""
    best = -1
    for m in _SENT_END.finditer(text, 0, limit + 1):
        end = m.start() + 1
        word = re.search(r"([\w.]*)$", text[max(0, m.start() - 12):m.start()])
        w = word.group(1).lower().rstrip(".") if word else ""
        if text[m.start()] == "." and (w in _ABBREV or (len(w) == 1 and w.isalpha()) or re.fullmatch(r"(?:\w\.)+\w", w)):
            continue
        if m.start() + 1 <= limit and end > floor:
            best = m.start() + len(m.group(0).rstrip())
    return best


def trim(text: str, max_chars: int) -> str:
    """At most `max_chars` of text, cut at a sentence end (or a word boundary with "…")."""
    if len(text) <= max_chars:
        return text
    cut = sentence_end(text, max_chars, max_chars // 3)
    if cut > 0:
        return text[:cut].rstrip()
    cut = text.rfind(" ", 0, max_chars)
    return text[:cut if cut > 0 else max_chars].rstrip(" ,;:") + "…"


_HATNOTE = re.compile(r"^:+\s*''.*''\s*$", re.M)
DISAMBIG = re.compile(r"\{\{\s*(?:disambiguation|disambig|dab|geodis|hndis|hndis-cleanup|"
                      r"(?:given name|surname|school|number|letter-number combination|"
                      r"airport|call sign|chinese|mountain|road|ship|species|wp)?\s*disambiguation|"
                      r"set index(?: article)?|shipindex|lake index|mountainindex|roadindex|"
                      r"place name disambiguation|human name disambiguation|biology disambiguation)"
                      r"\s*(?:\||\}\})", re.I)


def is_disambiguation(title: str, wikitext: str) -> bool:
    return title.endswith("(disambiguation)") or bool(DISAMBIG.search(wikitext))


def lead(wikitext: str, max_chars: int = 1200) -> str:
    """Clean introduction (text before the first heading), cut to `max_chars` at a sentence end.
    A page that opens with a heading (many lists and set indexes do) uses its first section."""
    text = clean(_HATNOTE.sub("", lead_wikitext(wikitext)))
    if not text:
        text = next((t for _, t in sections(wikitext) if t), "")
    return trim(text, max_chars)
