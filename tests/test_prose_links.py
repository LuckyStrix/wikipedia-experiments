"""Prose link extraction (wikiexp/prose_links.py) and the prose graph build, on tiny hand-made wikitext."""
import json
import sys

import numpy as np
import pytest

from pipeline import build_prose_graph as bpg
from pipeline import settings as S
from pipeline import stages as ST
from tests.conftest import ARTICLES
from tests.test_wikitext import write_dump
from wikiexp import core
from wikiexp import prose_links as P
from wikiexp import wikitext as W


def targets(text, plain_only=False):
    return [l.target for l in P.extract_links(text) if l.plain or not plain_only]


# ── what is and isn't prose ──────────────────────────────────────────────────

def test_plain_links_piped_links_and_repeats():
    assert targets("A [[Paris]], the [[France|French]] capital, near [[Paris|the city]].") == \
        ["Paris", "France", "Paris"]


def test_templates_are_excluded_at_any_depth():
    text = ("{{Infobox person | name = [[Hidden 1]] | birth = {{birth date|1879|3|14}} {{nowrap|[[Hidden 2]]}}\n"
            "| spouse = {{plainlist|\n* [[Hidden 3]]\n* {{small|[[Hidden 4]]}}\n}}\n}}\n"
            "He met [[Visible]] {{cite web|title=[[Hidden 5]]}} and {{sfn|X}} [[Also visible]].")
    assert targets(text) == ["Visible", "Also visible"]


def test_unbalanced_braces_dont_swallow_the_article():
    assert targets("{{unclosed [[In literal]] text and [[More]]") == ["In literal", "More"]
    assert targets("[[A]] }} stray {{b|[[Hidden]]}} [[C]]") == ["A", "C"]
    assert targets("{{a {{b}} [[Kept]]") == ["Kept"]        # the outer opener is literal, {{b}} still drops


def test_refs_are_excluded():
    text = ('Fact<ref name="a">See [[In ref]] and {{cite|[[In cite]]}}</ref> then [[Real 1]].<ref name=b />'
            '<ref group=n>[[Note]]</ref> [[Real 2]]<REF>[[Upper]]</REF>')
    assert targets(text) == ["Real 1", "Real 2"]


def test_an_unterminated_ref_is_literal_text_like_on_the_real_site():
    assert targets("Text <ref>never closed [[Kept]] more [[Also]]") == ["Kept", "Also"]


def test_tables_galleries_math_and_other_blocks_are_excluded():
    text = ("[[Before]]\n{| class=\"wikitable\"\n|-\n| [[Cell]] || {{flag|X}}\n{| nested\n| [[Inner]]\n|}\n| [[Cell 2]]\n|}\n"
            "[[Between]]\n<gallery>\nFile:A.jpg|[[Gallery link]]\n</gallery>\n<math>[[Math]]</math>"
            "<imagemap>Image:A.jpg\n[[Map link]]\n</imagemap>[[After]]")
    assert targets(text) == ["Before", "Between", "After"]


def test_file_captions_categories_and_interlanguage_links_are_excluded():
    text = ("[[File:Einstein.jpg|thumb|left|[[Hidden caption]] with [[nested [[deep]] link]] and ''x'']]"
            "[[Image:B.png|[[Hidden 2]]]] [[Visible]] [[Category:Physicists]] [[fr:Paris]] [[zh-min-nan:Pa-lê]]"
            " [[:Category:Linked category]] [[:fr:Link]] [[Wikipedia:Policy]] [[Help:Contents]] [[Talk:Foo]]"
            " [[file:lower.jpg|[[Hidden 3]]]] [[Also visible]]")
    assert targets(text) == ["Visible", "Also visible"]
    assert targets("[[File:Never closed|[[Hidden]] caption\n[[Next line]]") == ["Next line"]


def test_titles_with_colons_are_not_mistaken_for_namespaces():
    assert targets("[[CSI: Miami]], [[Star Wars: A New Hope]], [[Mission: Impossible]], [[AI: More]]") == \
        ["CSI: Miami", "Star Wars: A New Hope", "Mission: Impossible", "AI: More"]


def test_comments_and_nowiki_are_ignored():
    text = "[[One]] <!-- [[Two]] --> [[Three]]<nowiki>[[Four]]</nowiki> <nowiki/> <!-- unterminated [[Five]]"
    assert targets(text) == ["One", "Three"]


def test_links_in_headings_and_lists_count():
    assert targets("== The [[Heading link]] ==\n* [[Item]]\n# [[Numbered]]") == ["Heading link", "Item", "Numbered"]


def test_empty_and_linkless_text():
    assert P.extract_links("") == [] and P.extract_links("no links {{x|[[y]]}}") == []
    assert P.extract_links("[[]] [[ ]] [[|x]] [[#Section]] [[a|") == []


# ── normalisation ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("Paris", "Paris"),
    ("paris", "Paris"),                                  # first letter is case-insensitive
    ("new  york_city", "New york city"),                 # underscores, runs of spaces; only the 1st letter changes
    ("Albert_Einstein#Early life", "Albert Einstein"),
    ("  Cell (biology)  ", "Cell (biology)"),
    ("AT&amp;T", "AT&T"),
    ("Caf&eacute;", "Café"),
    ("Space&nbsp;Shuttle", "Space Shuttle"),
    ("Foo%20Bar", "Foo Bar"),
    ("100% sure", "100% sure"),                          # a lone % is not an escape
    (":Foo", "Foo"),
    ("Café", "Café"),                              # NFC
    ("#Section", None), ("", None), ("   ", None), (":", None),
    ("/Subpage", None), ("Category:Foo", None), ("File:A.jpg", None), ("fr:Paris", None), ("x" * 300, None),
])
def test_normalize_title(raw, expected):
    assert P.normalize_title(raw) == expected


def test_normalisation_inside_extraction():
    assert targets("[[theoretical_physics#History|theory]] [[Foo&nbsp;Bar]] [[:Colon]]") == \
        ["Theoretical physics", "Foo Bar", "Colon"]


# ── lead section and the "plain" (first-link) flag ───────────────────────────

def test_lead_flag_uses_the_first_real_heading():
    text = ("{{Infobox\n== not a heading ==\n[[Hidden]]\n}}\n[[Lead 1]] text [[Lead 2]]\n"
            "== History ==\n[[Later]]\n=== Sub ===\n[[Later 2]]")
    assert [(l.target, l.lead) for l in P.extract_links(text)] == \
        [("Lead 1", True), ("Lead 2", True), ("Later", False), ("Later 2", False)]
    assert [l.lead for l in P.extract_links("== First ==\n[[A]]")] == [False]     # no lead at all
    assert [l.lead for l in P.extract_links("[[A]] [[B]]")] == [True, True]       # no headings


def plain(text):
    return [(l.target, l.plain) for l in P.extract_links(text)]


def test_plain_skips_parentheses_italics_and_indented_lines():
    text = ("'''Einstein''' (born in [[Ulm]] ([[Nested]]); [[Paren 2]]) was a [[Physicist]] "
            "and ''the [[Italic]] one'' and '''[[Bold link]]''' and ''[[Italic link]]'' then [[Last]].")
    assert plain(text) == [("Ulm", False), ("Nested", False), ("Paren 2", False), ("Physicist", True),
                           ("Italic", False), ("Bold link", True), ("Italic link", False), ("Last", True)]
    assert plain(":''For other uses see [[Hatnote]].''\n;[[Term]]\n[[Body]]") == \
        [("Hatnote", False), ("Term", False), ("Body", True)]


def test_parentheses_inside_link_targets_and_stray_marks_dont_confuse_it():
    assert plain("A [[Cell (biology)|cell]] and [[Mercury (planet)]] then (a [[Inside]]) [[Out]]") == \
        [("Cell (biology)", True), ("Mercury (planet)", True), ("Inside", False), ("Out", True)]
    # an unbalanced "(" or italic mark only lasts to the end of its line
    assert plain("a smiley :( and [[Same line]]\n[[Next line]]") == [("Same line", False), ("Next line", True)]
    assert plain("''unclosed italic [[Same]]\n[[Next]] and ) stray [[Third]]") == \
        [("Same", False), ("Next", True), ("Third", True)]
    assert plain("[[Piped|''text'']] [[Plain]]") == [("Piped", False), ("Plain", True)]


def test_a_template_removed_from_parentheses_leaves_them_empty():
    assert plain("He ({{IPA|x}}; born [[Ulm]]) lived in [[Bern]].") == [("Ulm", False), ("Bern", True)]


# ── resolving to idx ─────────────────────────────────────────────────────────

@pytest.fixture
def lookup():
    return P.TitleLookup.from_titles(
        {"Paris": 0, "France": 1, "Einstein": 2, "Theoretical physics": 3, "Café": 4},
        {"Albert Einstein": 2, "Cafe": 4, "Theory": 3})


def test_lookup_resolves_articles_redirects_and_misses(lookup):
    assert lookup(["Paris", "Albert Einstein", "Cafe", "No such", ""]).tolist() == [0, 2, 4, -1, -1]
    assert lookup([]).tolist() == []
    assert P.TitleLookup(np.empty(0, np.int64), np.empty(0, np.int32))(["x"]).tolist() == [-1]


def test_page_links_dedupes_through_redirects_and_orders_by_first_appearance(lookup):
    text = ("[[Albert Einstein]] lived in [[Paris]] ([[Red link]]).\n"
            "== Later ==\n[[Einstein]] [[France]] [[Theory]] [[Paris]] [[Theoretical physics]]")
    pl = P.page_links(text, lookup, self_idx=-1)
    assert pl.dst.tolist() == [0, 1, 2, 3]                  # unique, ascending idx
    assert pl.order.tolist() == [1, 2, 0, 3]                # Einstein first, then Paris, France, Theory
    assert pl.n_lead == 2                                   # Einstein and Paris first appear before the heading
    assert pl.first == 2                                    # [[Albert Einstein]] is not in parentheses
    assert pl.raw == 8 and pl.resolved == 7                 # one red link
    assert pl.order.dtype == np.uint16 and pl.dst.dtype == np.int32


def test_page_links_drops_self_links_and_finds_first_plain(lookup):
    pl = P.page_links("'''Paris''' ([[France]]) is [[Paris]] in [[Theory|a theory]].", lookup, self_idx=0)
    assert pl.dst.tolist() == [1, 3] and pl.first == 3 and pl.order.tolist() == [0, 1]
    only_paren = P.page_links("A (see [[France]]) only.", lookup)
    assert only_paren.dst.tolist() == [1] and only_paren.first == -1
    none = P.page_links("[[Nowhere]] and [[Paris]]", lookup, self_idx=0)
    assert len(none.dst) == 0 and none.first == -1 and none.raw == 2 and none.resolved == 1
    assert len(P.page_links("", lookup).dst) == 0


def test_title_lookup_from_db(tiny_data):
    page_ids = np.load(tiny_data / "graph" / "idx_to_page_id.npy")
    lk = P.TitleLookup.from_db(tiny_data / "wiki.sqlite", page_ids)
    assert len(lk) == len(ARTICLES) + 3
    assert lk(["Mitochondria", "Mitochondrion", "Movie", "Bacon, Kevin", "Nope"]).tolist() == \
        [ARTICLES.index("Mitochondrion"), ARTICLES.index("Mitochondrion"), ARTICLES.index("Film"),
         ARTICLES.index("Kevin Bacon"), -1]


# ── the build ────────────────────────────────────────────────────────────────

def page(i, title, text):
    return (100 + i, title, 0, None, text)


TEXTS = {
    "Kevin Bacon": "'''Kevin Bacon''' (born [[Bacon, Kevin|himself]]) starred in [[Footloose]]{{cite|[[Biology]]}}.\n"
                   "== Career ==\n[[Film|Films]] and [[Movie]] again, plus [[Red link]].",
    "Footloose": "[[Film|A film]] with [[Kevin Bacon]] and [[Footloose]] itself.",
    "Film": "{{Infobox film|director=[[Kevin Bacon]]}} Made of [[Biology]].",
    "Biology": "[[Cell (biology)]] and [[Mitochondria]] are [[Mitochondrion|studied]].",
    "Cell (biology)": "Has [[Mitochondrion]].",
    "Mitochondrion": "See [[Biology]].",
    "Isolated article": "Nothing to link here.",
    "Bacon": "",
}


@pytest.fixture
def dump(tiny_data, tmp_path):
    pages = [page(i, t, TEXTS[t]) for i, t in enumerate(ARTICLES)]
    pages += [(900, "Mitochondria", 0, "Mitochondrion", "#REDIRECT [[Mitochondrion]]"),
              (901, "Talk:Film", 1, None, "[[Kevin Bacon]]"),
              (999, "Not in db", 0, None, "[[Film]]")]               # a dump page the database doesn't know
    d, i = tmp_path / "dump.xml.bz2", tmp_path / "index.txt.bz2"
    write_dump(d, i, [pages[:4], pages[4:8], pages[8:]])
    return W.WikiText(d, i, cache_dir=tmp_path / "cache")


def run_build(tiny_data, dump, **kw):
    return bpg.build(tiny_data / "prose_graph", dump, graph_dir=tiny_data / "graph",
                     db_path=tiny_data / "wiki.sqlite", say=lambda *a, **k: None, **kw)


@pytest.mark.parametrize("workers", [1, 2])
def test_build_writes_a_graph_in_the_core_layout(tiny_data, dump, workers):
    out = run_build(tiny_data, dump, workers=workers)
    g, full = core.Graph(out), core.Graph(tiny_data / "graph")
    assert len(g) == len(full) and (g.page_ids == full.page_ids).all()
    ix = {t: i for i, t in enumerate(ARTICLES)}
    expect = {"Kevin Bacon": ["Footloose", "Film"], "Footloose": ["Film", "Kevin Bacon"], "Film": ["Biology"],
              "Biology": ["Cell (biology)", "Mitochondrion"], "Cell (biology)": ["Mitochondrion"],
              "Mitochondrion": ["Biology"], "Isolated article": [], "Bacon": []}
    for t, outs in expect.items():
        assert g.out_links(ix[t]).tolist() == sorted(ix[o] for o in outs), t
    for i in range(len(g)):                                     # incoming rows are the transpose
        assert g.in_links(i).tolist() == sorted(j for j in range(len(g)) if i in g.out_links(j).tolist())
    for name in ("out_indptr", "out_indices", "in_indptr", "in_indices", "idx_to_page_id"):
        assert np.load(out / f"{name}.npy").dtype == np.load(tiny_data / "graph" / f"{name}.npy").dtype, name


def test_build_order_lead_and_first_link(tiny_data, dump):
    out = run_build(tiny_data, dump, workers=1)
    ix = {t: i for i, t in enumerate(ARTICLES)}
    order, lead = np.load(out / "out_order.npy"), np.load(out / "lead_count.npy")
    ptr = np.load(out / "out_indptr.npy")
    assert order.dtype == np.uint16 and len(order) == ptr[-1]
    kb = ix["Kevin Bacon"]
    row = slice(ptr[kb], ptr[kb + 1])
    g = core.Graph(out)
    reading_order = g.out_links(kb)[np.argsort(order[row])]
    assert [ARTICLES[i] for i in reading_order] == ["Footloose", "Film"]    # himself is a self-redirect: dropped
    assert lead[kb] == 1 and lead[ix["Footloose"]] == 2 and lead[ix["Isolated article"]] == 0
    first = np.load(out / "first_link.npy")
    assert first.dtype == np.int32
    assert first[kb] == ix["Footloose"]                  # "himself" is inside parentheses and a self link anyway
    assert first[ix["Biology"]] == ix["Cell (biology)"]
    assert first[ix["Isolated article"]] == -1 and first[ix["Bacon"]] == -1


def test_build_meta_compares_with_pagelinks_and_leaves_no_temp_files(tiny_data, dump):
    out = run_build(tiny_data, dump, workers=1)
    meta = json.loads((out / "meta.json").read_text())
    full = core.Graph(tiny_data / "graph")
    assert meta["articles"] == len(ARTICLES) and meta["links"] == int(core.Graph(out).out_indptr[-1])
    assert meta["pagelinks_links"] == int(full.out_indptr[-1])
    assert meta["dump_pages_not_in_db"] == 1 and meta["articles_with_links"] == 6
    assert meta["articles_with_first_link"] == 6 and meta["sampled_share_also_in_pagelinks"] <= 1.0
    # written: 14 (one red link included; the link in a template is not); 13 resolve; 9 unique, 8 in leads
    assert meta["links_written_before_dedupe"] == 14 and meta["of_which_resolved_to_articles"] == 13
    assert meta["links"] == 9 and meta["lead_links"] == 8
    assert not [p for p in tiny_data.iterdir() if ".building" in p.name or p.name.endswith(".old")]


def test_rebuild_replaces_the_old_graph(tiny_data, dump):
    run_build(tiny_data, dump, workers=1)
    (tiny_data / "prose_graph" / "stale.txt").write_text("x")
    run_build(tiny_data, dump, workers=1, limit=2)                  # only the first two articles this time
    assert not (tiny_data / "prose_graph" / "stale.txt").exists()
    assert json.loads((tiny_data / "prose_graph" / "meta.json").read_text())["articles_parsed"] == 2


def test_a_page_that_breaks_the_parser_just_has_no_links(tiny_data, dump, monkeypatch):
    real = P.page_links

    def flaky(text, lookup, self_idx=-1):
        if text.startswith("Has "):
            raise RecursionError("boom")
        return real(text, lookup, self_idx)
    monkeypatch.setattr(P, "page_links", flaky)
    out = run_build(tiny_data, dump, workers=1)
    assert core.Graph(out).out_links(ARTICLES.index("Cell (biology)")).tolist() == []
    assert len(core.Graph(out).out_links(ARTICLES.index("Biology"))) == 2


def test_main_runs_from_the_command_line(tiny_data, dump, monkeypatch):
    monkeypatch.setattr(bpg, "log", lambda *a, **k: None)
    monkeypatch.setattr(W, "WikiText", lambda **k: dump)
    monkeypatch.setattr(sys, "argv", ["build_prose_graph", "--workers", "1"])
    bpg.main()
    assert (tiny_data / "prose_graph" / "meta.json").exists()


# ── the stage ────────────────────────────────────────────────────────────────

def test_stage_is_registered_and_follows_the_data(tiny_data):
    st = ST.BY_KEY["prose_graph"]
    s = S.Settings({"data_dir": str(tiny_data)})
    assert st.command(s)[-1] == "pipeline.build_prose_graph"
    assert "pagelinks.sql.gz" not in " ".join(p.name for p in st.inputs(s))     # needs the text dumps
    assert any(p.name.endswith("pages-articles-multistream.xml.bz2") for p in st.inputs(s))
    assert not st.is_done(s) and st.summary(s) == []
    (tiny_data / "prose_graph").mkdir()
    for f in ST.GRAPH_FILES + ["out_order.npy", "lead_count.npy", "first_link.npy", "meta.json"]:
        (tiny_data / "prose_graph" / f).write_bytes(b"x")
    (tiny_data / "prose_graph" / "meta.json").write_text('{"links": 5}')
    assert st.is_done(s) and any("links: 5" in l for l in st.summary(s))
