"""wikiexp.wikitext: the multistream dump (index, random access, streaming) and wikitext cleaning.
Uses a tiny hand-made multistream file, no dumps needed."""
import bz2
import multiprocessing as mp
from xml.sax.saxutils import escape

import numpy as np
import pytest

from wikiexp import wikitext as W

WATERLOO = """{{Short description|1815 battle}}
{{Infobox military conflict
| name = Waterloo
| notes = {x} braces {y}
}}
[[File:Waterloo.jpg|thumb|The [[Duke of Wellington]] at ''Waterloo''<ref>caption note</ref>]]
The '''Battle of Waterloo'''<ref name=a>{{cite book|title=Foo}}</ref> was fought on 18 June 1815 near [[Waterloo, Belgium|Waterloo]]. \
Heavy rain turned the fields to mud and delayed the [[French Army|French]] artillery.{{efn|A note.}} <!-- hidden -->

== Prelude ==
Napoleon returned from [[Elba]].
{| class="wikitable"
|-
| table || cell
|}

== Aftermath ==
The war ended.

== References ==
{{reflist}}
* [[Some book]]

[[Category:1815 in Europe]]
"""

PAGES = [
    # (id, title, ns, redirect, wikitext)
    (10, "Battle of Waterloo", 0, None, WATERLOO),
    (11, "Waterloo", 0, "Battle of Waterloo", "#REDIRECT [[Battle of Waterloo]]"),
    (12, "Photosynthesis", 0, None, "'''Photosynthesis''' is how plants make sugar from light &amp; water.\n\n== Light reactions ==\nChlorophyll absorbs light."),
    (13, "Talk:Photosynthesis", 1, None, "Discussion page."),
    (20, "Eiffel Tower", 0, None, "The '''Eiffel Tower''' is a lattice tower in [[Paris]]. It is 330 m tall."),
    (21, "Mercury (disambiguation)", 0, None, "'''Mercury''' may refer to:\n* [[Mercury (planet)]]\n* [[Mercury (element)]]\n\n{{disambiguation}}"),
    (30, "Empty page", 0, None, ""),
    (31, "Heading first", 0, None, "== History ==\nIt started long ago in [[Rome]]."),
]
STREAMS = [PAGES[0:3], PAGES[3:5], PAGES[5:6], PAGES[6:8]]


def page_xml(pid, title, ns, redirect, text):
    red = f'    <redirect title="{escape(redirect)}" />\n' if redirect else ""
    body = f'<text bytes="{len(text)}" xml:space="preserve">{escape(text)}</text>' if text else '<text bytes="0" xml:space="preserve" />'
    return (f"  <page>\n    <title>{escape(title)}</title>\n    <ns>{ns}</ns>\n    <id>{pid}</id>\n{red}"
            f"    <revision>\n      <id>{pid * 100}</id>\n      {body}\n    </revision>\n  </page>\n")


def write_dump(dump, index, streams=STREAMS):
    """Multistream bz2 (a header stream the index doesn't list, then one stream per group) + its index."""
    out, lines = bytearray(bz2.compress(b"<mediawiki>\n  <siteinfo></siteinfo>\n")), []
    for n, group in enumerate(streams):
        xml = "".join(page_xml(*p) for p in group)
        if n == len(streams) - 1:
            xml += "</mediawiki>\n"
        lines += [f"{len(out)}:{p[0]}:{p[1]}" for p in group]
        out += bz2.compress(xml.encode())
    dump.write_bytes(bytes(out))
    index.write_bytes(bz2.compress(("\n".join(lines) + "\n").encode()))


@pytest.fixture
def dump(tmp_path):
    d, i = tmp_path / "dump.xml.bz2", tmp_path / "index.txt.bz2"
    write_dump(d, i)
    return d, i


@pytest.fixture
def wt(dump, tmp_path):
    return W.WikiText(dump[0], dump[1], cache_dir=tmp_path / "cache")


# ── index, fetch ─────────────────────────────────────────────────────────────

def test_index_and_fetch(wt):
    assert len(wt) == len(PAGES) and wt.n_streams == 4
    assert 12 in wt and 99 not in wt
    assert wt.fetch(12).startswith("'''Photosynthesis''' is how plants make sugar from light &amp; water.")
    assert wt.fetch(99) is None
    assert wt.fetch(30) == ""                      # an empty <text /> is empty, not missing
    p = wt.page(11)
    assert (p.title, p.ns, p.redirect) == ("Waterloo", 0, "Battle of Waterloo")
    assert wt.page(13).ns == 1
    assert wt.fetch_many([10, 20, 99]).keys() == {10, 20}


def test_index_is_cached_and_rebuilt_when_the_dump_changes(dump, tmp_path):
    cache = tmp_path / "cache"
    msgs = []
    W.WikiText(dump[0], dump[1], cache_dir=cache, progress=msgs.append)
    assert any("cached" in m for m in msgs) and (cache / "index_meta.json").exists()
    msgs.clear()
    again = W.WikiText(dump[0], dump[1], cache_dir=cache, progress=msgs.append)
    assert msgs == [] and again.fetch(20).startswith("The '''Eiffel")        # loaded from the cache
    # a different index (e.g. a newer dump under the same name) must not use the old cache
    write_dump(dump[0], dump[1], streams=[PAGES[:4], PAGES[4:]])
    newer = W.WikiText(dump[0], dump[1], cache_dir=cache)
    assert newer.n_streams == 2 and newer.fetch(31).startswith("== History")


def test_unwritable_cache_dir_still_works(dump, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    wt = W.WikiText(dump[0], dump[1], cache_dir=blocker / "sub")     # can't mkdir under a file
    assert wt.fetch(10) is not None


def test_missing_dump_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        W.WikiText(tmp_path / "nope.bz2", tmp_path / "nope-index.bz2", cache_dir=tmp_path)


# ── streaming ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("workers", [1, 3])
def test_iter_pages_main_namespace_non_redirects(wt, workers):
    if workers > 1 and "fork" not in mp.get_all_start_methods():
        pytest.skip("needs fork")
    W.BATCH, old = 1, W.BATCH                      # one stream per task so the pool really splits the work
    try:
        got = list(wt.iter_pages(workers=workers))
    finally:
        W.BATCH = old
    assert [g[0] for g in got] == [10, 12, 20, 21, 30, 31]      # dump order; no redirect (11), no Talk (13)
    assert got[0][1] == "Battle of Waterloo" and "Heavy rain" in got[0][2]


def test_iter_pages_fn_runs_in_workers_and_none_is_skipped(wt):
    got = list(wt.iter_pages(fn=lambda p: len(p.text) if p.text else None, workers=2))
    assert [g[0] for g in got] == [10, 12, 20, 21, 31]            # the empty page returned None
    assert got[1][2] == len(PAGES[2][4])


def test_iter_pages_options(wt):
    assert [g[0] for g in wt.iter_pages(workers=1, redirects=True)][:2] == [10, 11]
    assert {g[0] for g in wt.iter_pages(workers=1, namespace=None)} >= {13}
    assert [g[0] for g in wt.iter_pages(workers=1, limit=2)] == [10, 12]
    assert [g[0] for g in wt.iter_pages(workers=1, streams=range(2, 4))] == [21, 30, 31]


def test_iter_pages_stops_cleanly_when_abandoned(wt):
    it = wt.iter_pages(workers=2)
    next(it)
    it.close()            # must terminate the pool without hanging


# ── cleaning ─────────────────────────────────────────────────────────────────

def test_to_text_keeps_prose_and_drops_the_rest():
    text = W.to_text(WATERLOO)
    assert "The Battle of Waterloo was fought on 18 June 1815 near Waterloo." in text
    assert "Heavy rain turned the fields to mud and delayed the French artillery." in text
    for junk in ("{{", "}}", "[[", "<ref", "File:", "Category", "hidden", "thumb", "caption", "cite book",
                 "cell", "braces", "Infobox", "A note", "reflist", "Some book", "Duke of Wellington"):
        assert junk not in text, junk
    assert "Prelude\nNapoleon returned from Elba." in text and "Aftermath\nThe war ended." in text
    assert "References" not in text                       # reference-like sections are dropped by default


def test_sections():
    secs = W.sections(WATERLOO)
    assert [h for h, _ in secs] == ["", "Prelude", "Aftermath", "References"]
    assert secs[0][1].startswith("The Battle of Waterloo")
    assert secs[1][1] == "Napoleon returned from Elba."
    assert [h for h, _ in W.sections(WATERLOO, drop=W.DROP_SECTIONS)] == ["", "Prelude", "Aftermath"]
    # an infobox may contain "== x ==" lines; they are not headings
    assert [h for h, _ in W.sections("{{Box\n== not a heading ==\n}}\nText.\n== Real ==\nMore.")] == ["", "Real"]
    assert W.to_text(WATERLOO, headings=False).count("Napoleon returned") == 1


@pytest.mark.parametrize("wikitext,expected", [
    ("[[Paris|the capital]] and [[Rome]]", "the capital and Rome"),
    ("'''bold''' and ''italic''", "bold and italic"),
    ("a<ref>x</ref>b<ref name=n/>c<ref name=\"m\">y {{cite|z}}</ref>d", "abcd"),
    ("a {{Infobox x|a={{nested|b}}|c=d}} b", "a b"),
    ("born {{birth date and age|1950|3|7}}", "born March 7, 1950"),
    ("{{lang|fr|bonjour}} and {{lang-de|Hallo}}", "bonjour and Hallo"),
    ("{{convert|10|km|mi}} and {{convert|5|to|9|ft}}", "10 km and 5 to 9 ft"),
    ("Tokyo ({{IPA|ja|x}}; {{nihongo|Tokyo|東京|Tōkyō}}) is", "Tokyo (Tokyo) is"),
    ("a (  ) b []", "a b"),
    ("x [[File:A.png|thumb|[[nested]] caption]] y [[Category:Z]] [[:Category:Kept|link]]", "x y link"),
    ("see [http://example.com the site] or [http://a.b]", "see the site or"),
    ("a<br>b<small>c</small> &nbsp;&amp; d", "abc & d"),
    ("a <math>x^2</math> b <gallery>\nFile:a.jpg\n</gallery> c", "a b c"),
    ("{|\n| cell\n{|\n| inner\n|}\n|}\nafter", "after"),
    ("* one\n* two\n# three", "one\ntwo\nthree"),
    ("text <!-- gone --> more <!-- unterminated", "text more"),
    ("unbalanced {{template and [[link", "unbalanced template and link"),
    ("a <nowiki>[[x]]</nowiki> b", "a x b"),
    ("{{pi}} is '''{{pi}}'''", "π is π"),
    ("AT&amp;T &hairsp;x", "AT&T x"),
])
def test_clean(wikitext, expected):
    assert W.clean(wikitext) == expected


def test_clean_survives_garbage():
    for bad in ["", "{{", "}}", "[[[[", "<ref>", "{|", "'''''", "==", "{{{{{{x", "<!--", "|}"]:
        assert isinstance(W.clean(bad), str)


def test_lead_trims_at_sentence_end():
    text = "Dr. J. Smith (born 1900) was an actor. He lived in the U.S. for years. " * 30
    out = W.lead(text, 200)
    assert len(out) <= 200 and out.endswith(".")
    assert not out.endswith(("Dr.", "U.S.", "J."))
    assert W.lead("Short text.", 200) == "Short text."
    # no sentence end in range: cut at a word, with an ellipsis
    assert W.lead("word " * 100, 50).endswith("word…")


def test_lead_uses_first_section_when_there_is_no_intro():
    assert W.lead(PAGES[7][4]) == "It started long ago in Rome."
    assert W.lead("") == ""


def test_lead_skips_hatnote_lines_and_infobox_headings():
    wt = ":''For other uses, see [[X]].''\nThe '''thing''' is a thing.\n== Later ==\nx"
    assert W.lead(wt) == "The thing is a thing."


def test_disambiguation_detection():
    assert W.is_disambiguation("Mercury", "Mercury may refer to:\n{{disambiguation}}")
    assert W.is_disambiguation("Mercury", "x {{Disambig|geo}} y")
    assert W.is_disambiguation("Smith", "{{surname disambiguation}}") or W.is_disambiguation("Smith", "{{hndis|Smith}}")
    assert W.is_disambiguation("ISEL", "{{set index article}}")
    assert W.is_disambiguation("Foo (disambiguation)", "anything")
    assert not W.is_disambiguation("Mercury (planet)", "The planet. See {{disambiguation needed}}.")


def test_page_parsing_handles_escaped_markup_and_empty_text():
    xml = (page_xml(1, "A & B", 0, None, "x <b> &amp; y") + page_xml(2, "Empty", 0, None, "")).encode()
    a, b = W.parse_stream(xml)
    assert (a.title, a.text) == ("A & B", "x <b> &amp; y") and b.text == ""
