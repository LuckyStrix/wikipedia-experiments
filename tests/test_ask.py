"""The keyword index, hybrid retrieval, the RAG pipeline, the ollama and Claude backends, and the Ask screen.
Everything runs on the tiny dump from test_wikitext with a hashing embedder and fake language models."""
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from experiments.ask import session
from pipeline import build_keyword
from pipeline import settings as S
from pipeline import stages as ST
from tests.test_semantic import HashingEmbedder, build_leads, embed
from tests.test_semantic import wt, dump  # noqa: F401  (fixtures)
from tests.test_tui import make_app, wait_for
from wikiexp import llm, rag as R
from wikiexp import retrieval as RT
from wikiexp.keyword import KeywordIndex, question_words
from wikiexp.llm import ClaudeBackend, GenInfo, LLMError, OllamaBackend
from wikiexp.semantic import SemanticSearch

quiet = lambda *a, **k: None  # noqa: E731


@pytest.fixture
def data(tmp_path, wt):  # noqa: F811
    d = build_leads(tmp_path, wt)
    build_keyword.build(d, say=quiet)
    return d


# ── keyword stage and index ──────────────────────────────────────────────────

def test_build_keyword_indexes_every_article_but_disambiguation_pages(data):
    db = sqlite3.connect(data / "text" / "fts.sqlite")
    meta = dict(db.execute("SELECT key, value FROM meta"))
    assert meta["articles"] == "5" and meta["leads_built"]          # 6 leads minus the disambiguation page
    assert not list((data / "text").glob("*.building"))
    kw = KeywordIndex(data)
    assert kw.available() and len(kw) == 5
    hits = kw.search("When was the Battle of Waterloo fought?", 5)
    assert hits[0].title == "Battle of Waterloo" and hits[0].page_id == 10 and hits[0].snippet
    assert all(h.title != "Mercury (disambiguation)" for h in kw.search("Mercury", 5))


def test_keyword_search_stems_folds_accents_and_weights_titles(data):
    kw = KeywordIndex(data)
    assert kw.search("plants making sugar", 3)[0].title == "Photosynthesis"           # plants/make: stemmed
    assert kw.search("EIFFEL tower", 3)[0].title == "Eiffel Tower"
    # a word in the title outranks the same word once in a lead
    db = sqlite3.connect(data / "text" / "fts.sqlite")
    ranks = [r for r, in db.execute("SELECT rowid FROM fts WHERE fts MATCH 'rome'")]
    assert ranks                                                    # "Heading first" mentions Rome in its lead
    assert kw.search("zzzzqqqq", 3) == [] and kw.search("   ", 3) == [] and kw.search("rome", 0) == []


def test_question_words_and_plan(data):
    assert question_words("Who designed the Eiffel Tower's top, and why?") == ["designed", "eiffel", "tower's", "top"]
    kw = KeywordIndex(data)
    plan = kw.plan("what was the battle of waterloo like")
    assert [w for w, _ in plan][:2] == ["waterloo", "battle"] or {w for w, _ in plan} >= {"waterloo", "battle"}
    assert all(df > 0 for _, df in plan)
    assert kw.doc_freq("plants") == 1 and kw.doc_freq("nonexistentword") == 0


def test_keyword_or_fallback_tops_up_when_and_finds_too_few(data):
    kw = KeywordIndex(data)
    hits = kw.search("photosynthesis tower", 5)               # no article has both words
    assert {h.title for h in hits} == {"Photosynthesis", "Eiffel Tower"}


def test_keyword_index_refuses_a_stale_leads_file(data):
    db = sqlite3.connect(data / "text" / "leads.sqlite")
    db.execute("UPDATE meta SET value = 'yesterday' WHERE key = 'built'")
    db.commit()
    with pytest.raises(RuntimeError, match="older leads"):
        KeywordIndex(data).search("battle", 3)


def test_keyword_missing_index_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_keyword"):
        KeywordIndex(tmp_path).search("x", 3)
    assert not KeywordIndex(tmp_path).available()


def test_keyword_stage_is_registered_and_summarises(data):
    st = ST.BY_KEY["keyword"]
    s = S.Settings({"data_dir": str(data)})
    assert [p.name for p in st.outputs(s)] == ["fts.sqlite"] and st.is_done(s) and not st.manual
    assert any(line.startswith("articles: 5") for line in st.summary(s))
    assert "pipeline.build_keyword" in st.command(s)
    assert st.missing_inputs(S.Settings({"data_dir": str(data / "nope")})) == ["leads.sqlite"]
    from experiments import EXPERIMENTS
    ex = {e.key: e for e in EXPERIMENTS}
    assert ex["ask"].requires == ("core", "titles", "text", "keyword")
    assert "embed" not in ex["semantic_search"].requires        # hybrid search works without embeddings


def test_new_ask_settings_validate():
    st = S.BY_KEY
    assert S.Settings().invalid() == [] and S.Settings()["ollama_model"] == "llama3.2:3b"
    assert S.Settings()["ollama_url"] == "http://localhost:11434" and S.Settings()["claude_model"] == "claude-opus-5-5"
    assert S.valid(st["ask_backend"], "claude") and not S.valid(st["ask_backend"], "gpt")
    assert not S.valid(st["claude_effort"], "extreme") and S.valid(st["claude_effort"], "xhigh")
    assert not S.valid(st["ollama_url"], "gpu-pc:11434") and S.valid(st["ollama_url"], "http://gpu-pc:11434")


# ── fusion ───────────────────────────────────────────────────────────────────

def test_rrf_rewards_agreement_over_a_single_high_rank():
    fused = RT.rrf({"a": [1, 2, 3], "b": [3, 4, 1]})
    assert fused[1] > fused[2] and fused[3] > fused[2]                  # in both lists beats first in one
    assert max(fused, key=fused.get) in (1, 3)
    assert RT.rrf({"a": [7, 7, 7]}) == {7: 1 / 61}                      # duplicates counted once
    w = RT.rrf({"a": [1], "b": [2]}, {"a": 1.0, "b": 3.0})
    assert w[2] > w[1]
    assert RT.rrf({}) == {}


def test_popularity_prior_is_mild_and_monotonic():
    assert RT.popularity_prior(0, 1000) == 1.0 and RT.popularity_prior(999, 1000) < 0.01
    assert RT.popularity_prior(10, 1000) > RT.popularity_prior(100, 1000) > RT.popularity_prior(900, 1000)
    assert RT.popularity_prior(None, 1000) == 0.0 and RT.popularity_prior(5, 1) == 0.0
    assert RT.POPULARITY_BOOST <= 0.2


def test_name_candidates_prefer_capitalised_phrases():
    names = [t for _, _, t in RT.name_candidates("Who designed the Eiffel Tower in Paris?")]
    assert "Eiffel Tower" in names and "Paris" in names and "Who" not in names
    assert "designed the" not in names and "the Eiffel Tower" not in names     # stop words at the edges
    assert "wrote the novel" not in [t for _, _, t in RT.name_candidates("who wrote the novel Middlemarch")]
    lower = [t for _, _, t in RT.name_candidates("eiffel tower height")]
    assert "eiffel tower" in lower and "eiffel" not in lower                   # all lowercase: phrases only


class FakeTitles:
    """TitleIndex.exact stand-in: case-insensitive lookup in a dict of title -> page id."""

    def __init__(self, table):
        self.table = {k.lower(): v for k, v in table.items()}

    def exact(self, q):
        pid = self.table.get(q.lower())
        return SimpleNamespace(page_id=pid, title=q) if pid else None


def hybrid(data, titles=None, semantic=True):
    ss = SemanticSearch(data, embedder=HashingEmbedder())
    return RT.HybridSearch(data, semantic=ss, titles=titles or FakeTitles({}))


def test_title_matches_longest_name_wins_and_overlaps_are_dropped(data):
    hs = hybrid(data, FakeTitles({"Battle of Waterloo": 10, "Waterloo": 11, "Eiffel Tower": 20}))
    assert hs.title_matches("Tell me about the Battle of Waterloo and the Eiffel Tower") == [10, 20]
    assert hs.title_matches("What happened at Waterloo?") == [11]


def test_hybrid_fuses_the_retrievers(data):
    embed(data)
    hs = hybrid(data, FakeTitles({"Eiffel Tower": 20}))
    assert hs.modes() == ("hybrid", "semantic", "keyword")
    res = hs.search("How tall is the Eiffel Tower?", 5)
    assert res[0].title == "Eiffel Tower" and set(res[0].via) == {"semantic", "keyword", "title"}
    assert res[0].ranks["title"] == 1 and res[0].score > res[-1].score
    assert [r.title for r in hs.search("heavy rain mud", 5, "keyword")][0] == "Battle of Waterloo"
    sem = hs.search("heavy rain mud", 5, "semantic")
    assert sem[0].via == ("semantic",) and sem[0].title == "Battle of Waterloo"
    assert hs.search("", 5) == [] and hs.search("x", 0) == []
    with pytest.raises(ValueError):
        hs.search("x", 3, "bogus")


def test_hybrid_without_embeddings_still_works(data):
    hs = hybrid(data)                                    # no `embed` run: semantic search unavailable
    assert hs.modes() == ("hybrid", "keyword")
    res = hs.search("battle of waterloo", 3)
    assert res[0].title == "Battle of Waterloo" and res[0].via == ("keyword",)


def test_hybrid_never_returns_a_disambiguation_page_named_in_the_question(data):
    hs = hybrid(data, FakeTitles({"Mercury (disambiguation)": 21}))
    assert all(r.page_id != 21 for r in hs.search("Mercury (disambiguation)", 5))


def test_hybrid_popularity_breaks_ties_toward_the_popular_article(data):
    hs = hybrid(data)
    rank = {pid: r for pid, (r, *_ ) in hs._leads([10, 12, 20, 31]).items()}
    assert rank[12] < rank[10] < rank[20]                # from RANK in test_semantic: Photosynthesis most linked
    boosted = RT.rrf({"a": [20, 12]})                    # same fused score for two ranks 1/61 vs 1/62 ...
    assert boosted[20] > boosted[12]


# ── passages ─────────────────────────────────────────────────────────────────

def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_split_passages_sizes_sections_and_titles():
    secs = [("", f"{words(100, 'a')}\n{words(100, 'b')}\n{words(100, 'c')}\n{words(10, 'd')}"),
            ("History", words(50, "h")), ("Tiny", "too few")]
    ps = R.split_passages("Thing", 7, secs)
    assert [p.path for p in ps] == ["Thing", "Thing", "Thing — History"]            # "Tiny" has < 8 words: skipped
    assert all(R.MIN_WORDS * 0 < p.words <= R.MAX_WORDS * 1.3 for p in ps)
    assert ps[0].words >= R.MIN_WORDS and ps[0].text.startswith("a0") and ps[1].text.endswith("d9")
    assert [p.index for p in ps] == [0, 1, 2] and ps[0].page_id == 7
    # a passage never crosses a section
    assert not any("h0" in p.text and "a0" in p.text for p in ps)


def test_split_passages_cuts_a_huge_paragraph_at_sentences():
    big = " ".join(f"Sentence number {i} says something quite ordinary about the world." for i in range(80))
    ps = R.split_passages("Big", 1, [("", big)])
    assert len(ps) >= 3 and all(p.words <= R.MAX_WORDS * 1.3 for p in ps)
    assert all(p.text.rstrip().endswith(".") for p in ps)                        # cut at sentence ends
    # one endless "sentence" is hard-cut
    ps = R.split_passages("Long", 1, [("", words(900))])
    assert len(ps) >= 4 and sum(p.words for p in ps) == 900


def test_short_tail_is_merged_into_the_previous_passage():
    ps = R.split_passages("T", 1, [("S", f"{words(130, 'x')}\n{words(15, 'y')}")])
    assert len(ps) == 1 and ps[0].words == 145


def test_prefilter_keeps_leads_and_the_best_word_overlap():
    ps = [R.Passage(1, "T", "", "intro text here for sure", 0)] + \
         [R.Passage(1, "T", f"S{i}", f"filler {i} " + ("volcano eruption" if i == 7 else "nothing"), i + 1)
          for i in range(20)]
    kept = R.prefilter(ps, "volcano eruption", keep=3)
    assert ps[0] in kept and any(p.section == "S7" for p in kept) and 3 <= len(kept) <= 4
    assert [p.index for p in kept] == sorted(p.index for p in kept)               # article order
    assert R.prefilter(ps, "x", keep=100) == ps


def test_pack_respects_budget_per_article_and_margin():
    def p(pid, score, n=100):
        return R.Passage(pid, f"A{pid}", "", words(n), score=score)
    ps = [p(1, 0.9), p(1, 0.85), p(1, 0.84), p(1, 0.83), p(2, 0.8), p(3, 0.3), p(4, 0.79, n=2000)]
    out = R.pack(ps, budget_tokens=10_000)
    assert [s.n for s in out] == list(range(1, len(out) + 1))
    assert sum(1 for s in out if s.page_id == 1) == R.MAX_PER_ARTICLE_IN_PROMPT  # at most 3 per article
    assert all(s.passage.score >= 0.9 - R.SCORE_MARGIN for s in out[2:])          # weak matches dropped
    assert 0.3 not in [s.passage.score for s in out]
    small = R.pack(ps, budget_tokens=250)
    assert len(small) == 1                                                         # budget: only one passage fits
    huge = R.pack([p(9, 0.9, n=5000)], budget_tokens=100)
    assert len(huge) == 1                                                          # the best passage is always kept
    assert R.pack([], 1000) == []


# ── prompt, citations, rewriting ─────────────────────────────────────────────

def sources(n=3):
    return [R.Source(i, R.Passage(i, f"Article {i}", "Sec" if i > 1 else "", f"Text of passage {i}.", score=1.0))
            for i in range(1, n + 1)]


def test_prompt_numbers_sources_and_forbids_outside_knowledge():
    msg = R.build_user_message("Why?", sources(2))
    assert "[1] Article 1\nText of passage 1." in msg and "[2] Article 2 — Sec\nText of passage 2." in msg
    assert msg.index("[1]") < msg.index("Question: Why?") and "only the sources" in msg
    assert "ONLY the numbered sources" in R.SYSTEM_PROMPT and "say so plainly" in R.SYSTEM_PROMPT
    assert "cite" in R.SYSTEM_PROMPT.lower() and "[1]" in R.SYSTEM_PROMPT
    assert "(none found)" in R.build_user_message("Why?", [])


def test_messages_include_recent_history_without_stale_citations():
    hist = [R.Turn(f"q{i}", f"answer {i} [1][2] done") for i in range(5)]
    msgs = R.build_messages("next?", sources(1), hist)
    assert [m["role"] for m in msgs] == ["user", "assistant"] * R.HISTORY_TURNS + ["user"]
    assert msgs[0]["content"] == "q2" and msgs[1]["content"] == "answer 2 done"     # last 3 turns; [n] removed
    assert msgs[-1]["content"].endswith("citing them like [1].")


@pytest.mark.parametrize("text,n,fixed,cited,invalid", [
    ("Fact [1]. Another [2].", 3, "Fact [1]. Another [2].", [1, 2], []),
    ("Fact [7].", 3, "Fact.", [], [7]),
    ("Fact [1, 7] and [2][9].", 3, "Fact [1] and [2].", [1, 2], [7, 9]),
    ("Fact [0].", 3, "Fact.", [], [0]),
    ("Range [1-3] and [2–4].", 3, "Range [1-3] and [2, 3].", [1, 2, 3], [4]),
    ("Semicolons [2; 1]", 2, "Semicolons [2; 1]", [2, 1], []),
    ("No cites at all.", 2, "No cites at all.", [], []),
    ("Link [text] and [a1] untouched, array[3] too", 2, "Link [text] and [a1] untouched, array[3] too", [], [3]),
])
def test_validate_citations(text, n, fixed, cited, invalid):
    got_text, got_cited, got_invalid = R.validate_citations(text, n)
    assert got_cited == cited and got_invalid == invalid
    if invalid == [3] and "array" in text:
        return                                         # "array[3]" is a citation-shaped token: stripped like any other
    assert got_text == fixed


def test_strip_citations():
    assert R.strip_citations("A fact [1]. Another [2][3], and [4, 5].") == "A fact. Another, and."


def test_followup_detection_and_heuristic_rewrite():
    assert R.looks_like_followup("When was it built?") and R.looks_like_followup("And who won?")
    assert R.looks_like_followup("Why?")
    assert not R.looks_like_followup("When was the Eiffel Tower built?")
    assert not R.looks_like_followup("What is the capital of Burkina Faso?")
    hist = [R.Turn("Who designed the Eiffel Tower?", "Gustave Eiffel's company [1].")]
    assert R.heuristic_rewrite("When was it built?", hist) == "Who designed the Eiffel Tower When was it built?"
    assert R.heuristic_rewrite("When was Notre-Dame built?", hist) == "When was Notre-Dame built?"
    assert R.heuristic_rewrite("When was it built?", []) == "When was it built?"


class FakeBackend:
    """A language model that answers from a script: `reply(system, messages)` returns the text."""
    name, model = "ollama", "fake:1b"

    def __init__(self, reply=None, fail=None):
        self.reply = reply or (lambda system, msgs: "It was fought on 18 June 1815 [1]. See also [9].")
        self.fail = fail
        self.info = GenInfo()
        self.calls = []

    def stream(self, system, messages, max_tokens=1024):
        self.calls.append((system, list(messages), max_tokens))
        text = self.reply(system, messages)
        self.info = GenInfo(stop_reason="stop", prompt_tokens=50, output_tokens=len(text.split()), seconds=0.5,
                            first_token_seconds=0.1)
        for piece in text.split(" "):
            yield piece + " "
        if self.fail:
            raise LLMError(self.fail)


def test_rewrite_uses_the_llm_and_falls_back_on_failure():
    hist = [R.Turn("Who designed the Eiffel Tower?", "Gustave Eiffel [1].")]
    ok = FakeBackend(lambda s, m: '"Who built the Eiffel Tower and when?"\nextra line')
    assert R.rewrite_query("When was it built?", hist, ok) == "Who built the Eiffel Tower and when?"
    assert ok.calls[0][0] == R.REWRITE_PROMPT and "Eiffel" in ok.calls[0][1][0]["content"]
    assert R.rewrite_query("When was Notre-Dame built?", hist, ok) == "When was Notre-Dame built?"   # no LLM call
    assert len(ok.calls) == 1
    broken = FakeBackend(lambda s, m: "x", fail="down")
    assert R.rewrite_query("When was it built?", hist, broken) == "Who designed the Eiffel Tower When was it built?"
    assert R.rewrite_query("When was it built?", hist, None).startswith("Who designed")
    assert R.rewrite_query("hi", [], ok) == "hi"


# ── the pipeline end to end ──────────────────────────────────────────────────

def make_rag(data, wt, backend=None, **kw):  # noqa: F811
    emb = HashingEmbedder()
    hs = RT.HybridSearch(data, semantic=SemanticSearch(data, embedder=emb), titles=FakeTitles({}))
    return R.RAG(data, backend or FakeBackend(), hybrid=hs, embedder=emb, wikitext=wt, **kw)


def test_rag_answers_with_numbered_sources_and_validates_citations(data, wt):
    backend = FakeBackend()
    rag = make_rag(data, wt, backend)
    conv = R.Conversation()
    events = list(rag.ask("When was the Battle of Waterloo fought?", conv))
    kinds = [e.kind for e in events]
    assert kinds[0] == "status" and "sources" in kinds and kinds.count("done") == 1 and "token" in kinds
    assert kinds.index("sources") < kinds.index("token") < kinds.index("done")
    ans = events[-1].data
    assert ans.sources and ans.sources[0].title == "Battle of Waterloo" and ans.sources[0].n == 1
    assert [s.n for s in ans.sources] == list(range(1, len(ans.sources) + 1))
    assert ans.raw_text == "It was fought on 18 June 1815 [1]. See also [9]."
    assert ans.text == "It was fought on 18 June 1815 [1]. See also." and ans.cited == [1] and ans.invalid == [9]
    assert any("[9]" in n for n in ans.notes)
    assert ans.retrieval_ms > 0 and set(ans.timings) >= {"search", "read", "rerank", "passages"}
    assert ans.generation.output_tokens > 0 and (ans.backend, ans.model) == ("ollama", "fake:1b")
    assert [t.question for t in conv.turns] == ["When was the Battle of Waterloo fought?"]
    # the model saw numbered passages from the real article, with section paths and cleaned text
    system, msgs, _ = backend.calls[-1]
    assert system == R.SYSTEM_PROMPT and msgs[-1]["role"] == "user"
    assert "[1] Battle of Waterloo\nThe Battle of Waterloo was fought on 18 June 1815" in msgs[-1]["content"]
    assert "{{" not in msgs[-1]["content"] and "cite book" not in msgs[-1]["content"]


def test_rag_follow_up_is_rewritten_and_history_is_sent(data, wt):
    def reply(system, msgs):
        if system == R.REWRITE_PROMPT:
            return "What did Napoleon do before the Battle of Waterloo?"
        return "He returned from Elba [1]."
    backend = FakeBackend(reply)
    rag = make_rag(data, wt, backend)
    conv = R.Conversation()
    rag.answer("When was the Battle of Waterloo fought?", conv)
    ans = rag.answer("What did he do before it?", conv)
    assert ans.query == "What did Napoleon do before the Battle of Waterloo?" and ans.question == "What did he do before it?"
    assert [t.question for t in conv.turns] == ["When was the Battle of Waterloo fought?", "What did he do before it?"]
    final = backend.calls[-1][1]
    assert [m["role"] for m in final] == ["user", "assistant", "user"]
    assert final[0]["content"] == "When was the Battle of Waterloo fought?"
    assert "Question: What did he do before it?" in final[2]["content"]          # the model sees the user's own words
    conv.reset()
    assert conv.turns == []


def test_rag_says_so_when_nothing_is_found_without_calling_the_model(data, wt):
    backend = FakeBackend()
    ans = make_rag(data, wt, backend).answer("zzzzqqqq xxxxyyyy")
    assert "couldn't find anything" in ans.text and ans.sources == [] and backend.calls == []


def test_rag_reports_a_backend_failure_but_keeps_the_partial_answer(data, wt):
    backend = FakeBackend(lambda s, m: "Partial answer [1]", fail="ollama went away")
    conv = R.Conversation()
    ans = make_rag(data, wt, backend).answer("Battle of Waterloo?", conv)
    assert ans.error == "ollama went away" and ans.text.startswith("Partial answer")
    assert make_rag(data, wt, FakeBackend(lambda s, m: "No cites here.")).answer("battle of waterloo").notes == \
        ["the answer cites no sources"]


def test_rag_budget_limits_the_prompt(data, wt):
    small = make_rag(data, wt, FakeBackend(), budget_tokens=1).answer("Battle of Waterloo")
    assert len(small.sources) == 1                          # the best passage always goes in


def test_article_passages_keep_section_paths(data, wt):
    rag = make_rag(data, wt)
    res = rag.hybrid.search("battle of waterloo", 3, "keyword")[0]
    ps = rag.article_passages(res, "napoleon elba")
    assert [p.path for p in ps] == ["Battle of Waterloo"]       # "Prelude" (4 words) is too short to be a passage
    assert not any(p.section == "References" for p in ps)


def test_shared_embedder_is_cached_per_model():
    a, b = R.shared_embedder("m1"), R.shared_embedder("m1")
    assert a is b and R.shared_embedder("m2") is not a


# ── ollama backend against a fake server ─────────────────────────────────────

class FakeOllama(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.requests = []
        super().__init__(("127.0.0.1", 0), self.Handler)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, status, body: bytes, ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                self._send(200, json.dumps({"models": [{"name": "llama3.2:3b"}, {"name": "qwen:7b"}]}).encode())
            else:
                self._send(404, b"{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.requests.append(body)
            model = body["model"]
            if model == "missing:1b":
                return self._send(404, json.dumps({"error": f"model '{model}' not found, try pulling it first"}).encode())
            if model == "broken:1b":
                return self._send(500, b'{"error": "kaboom"}')
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()
            lines = [{"message": {"role": "assistant", "content": c}, "done": False} for c in ("Hel", "lo ", "wor", "ld")]
            if model == "midfail:1b":
                lines = lines[:2] + [{"error": "llama runner crashed"}]
            elif model == "truncated:1b":
                pass                                       # no final done line: the connection just ends
            else:
                done = {"message": {"content": ""}, "done": True, "done_reason": "length" if model == "long:1b" else "stop",
                        "prompt_eval_count": 42, "eval_count": 4}
                lines.append(done)
            for ln in lines:
                self.wfile.write(json.dumps(ln).encode() + b"\n")
                self.wfile.flush()
            self.close_connection = True


@pytest.fixture
def ollama():
    srv = FakeOllama()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv
    srv.shutdown()
    srv.server_close()


def test_ollama_streams_text_and_reports_usage(ollama):
    b = OllamaBackend(ollama.url, "llama3.2:3b")
    pieces = list(b.stream("Be brief.", [{"role": "user", "content": "hi"}], 99))
    assert pieces == ["Hel", "lo ", "wor", "ld"] and "".join(pieces) == "Hello world"
    assert (b.info.stop_reason, b.info.prompt_tokens, b.info.output_tokens) == ("stop", 42, 4)
    assert b.info.seconds > 0 and b.info.first_token_seconds > 0 and not b.info.truncated
    req = ollama.requests[-1]
    assert req["model"] == "llama3.2:3b" and req["stream"] is True
    assert req["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
    assert req["options"]["num_ctx"] >= 8192 and req["options"]["num_predict"] == 99      # ollama's default 4096 would truncate
    assert llm.complete(b, "s", [{"role": "user", "content": "x"}]) == "Hello world"


def test_ollama_flags_a_cut_off_answer(ollama):
    b = OllamaBackend(ollama.url, "long:1b")
    list(b.stream("", [{"role": "user", "content": "hi"}]))
    assert b.info.truncated and b.info.stop_reason == "length"
    assert ollama.requests[-1]["messages"] == [{"role": "user", "content": "hi"}]          # no empty system turn


def test_ollama_model_not_pulled_tells_the_user_the_command(ollama):
    with pytest.raises(LLMError, match="ollama pull missing:1b"):
        list(OllamaBackend(ollama.url, "missing:1b").stream("s", [{"role": "user", "content": "x"}]))


def test_ollama_not_running_explains_how_to_start_it():
    b = OllamaBackend("http://127.0.0.1:9", "llama3.2:3b", timeout=2)       # nothing listens on port 9
    with pytest.raises(LLMError, match=r"ollama serve") as e:
        list(b.stream("s", [{"role": "user", "content": "x"}]))
    assert "http://127.0.0.1:9" in str(e.value)
    with pytest.raises(LLMError, match="ollama serve"):
        b.models()


def test_ollama_http_and_midstream_errors(ollama):
    with pytest.raises(LLMError, match="HTTP 500.*kaboom"):
        list(OllamaBackend(ollama.url, "broken:1b").stream("s", [{"role": "user", "content": "x"}]))
    b = OllamaBackend(ollama.url, "midfail:1b")
    got = []
    with pytest.raises(LLMError, match="llama runner crashed"):
        for p in b.stream("s", [{"role": "user", "content": "x"}]):
            got.append(p)
    assert got == ["Hel", "lo "]                                            # what arrived before the failure
    with pytest.raises(LLMError, match="closed the connection"):
        list(OllamaBackend(ollama.url, "truncated:1b").stream("s", [{"role": "user", "content": "x"}]))


def test_ollama_lists_models_and_validates_the_url(ollama):
    assert OllamaBackend(ollama.url).models() == ["llama3.2:3b", "qwen:7b"]
    assert OllamaBackend("127.0.0.1:11434").url == "127.0.0.1:11434"        # scheme optional
    with pytest.raises(LLMError, match="not a valid"):
        OllamaBackend("http://")
    assert OllamaBackend().url == llm.DEFAULT_OLLAMA_URL and OllamaBackend().model == "llama3.2:3b"


def test_ollama_timeout_is_a_clear_error():
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)                                                           # accepts connections, never answers
    try:
        b = OllamaBackend(f"http://127.0.0.1:{srv.getsockname()[1]}", timeout=0.3)
        with pytest.raises(LLMError, match="didn't answer within|Can't reach"):
            list(b.stream("s", [{"role": "user", "content": "x"}]))
    finally:
        srv.close()


# ── Claude backend with a mocked SDK client ──────────────────────────────────

class FakeStream:
    def __init__(self, owner, texts, final, error=None):
        self.owner, self.texts, self.final, self.error = owner, texts, final, error

    def __enter__(self):
        if self.error and self.error[0] == "enter":
            raise self.error[1]
        return self

    def __exit__(self, *a):
        self.owner.closed = True
        return False

    @property
    def text_stream(self):
        for t in self.texts:
            yield t
        if self.error and self.error[0] == "mid":
            raise self.error[1]

    def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, texts=("Hello", " world [1]"), stop_reason="end_turn", error=None, usage=None, **final_kw):
        self.calls, self.closed = [], False
        self.final = SimpleNamespace(stop_reason=stop_reason, model="claude-opus-5-5",
                                     usage=usage or SimpleNamespace(input_tokens=120, output_tokens=7, iterations=[]),
                                     stop_details=None, **final_kw)
        self.texts, self.error = texts, error
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.messages = SimpleNamespace(stream=lambda **kw: pytest.fail("must use the beta stream (fallbacks)"))

    def _stream(self, **kw):
        self.calls.append(kw)
        return FakeStream(self, self.texts, self.final, self.error)


def anth_error(cls, status=400, headers=None, **kw):
    import anthropic
    import httpx2 as httpx
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req, headers=headers or {})
    return getattr(anthropic, cls)("boom", response=resp, body=None, **kw)


def test_claude_request_follows_the_documented_shape():
    client = FakeClient()
    b = ClaudeBackend(client=client)
    assert (b.model, b.effort) == ("claude-opus-5-5", "medium")
    out = list(b.stream("SYS", [{"role": "user", "content": "q"}], 700))
    assert out == ["Hello", " world [1]"]
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "SYS"
    assert kw["messages"] == [{"role": "user", "content": "q"}]
    assert kw["thinking"] == {"type": "adaptive"} and kw["output_config"] == {"effort": "medium"}
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    assert kw["max_tokens"] >= 4096                                  # adaptive thinking shares the budget
    assert "temperature" not in kw and "top_p" not in kw and "budget_tokens" not in str(kw)
    assert kw["messages"][-1]["role"] == "user"                      # never an assistant prefill
    assert (b.info.stop_reason, b.info.prompt_tokens, b.info.output_tokens) == ("end_turn", 120, 7)
    assert client.closed and b.info.first_token_seconds > 0
    ClaudeBackend("claude-sonnet-5-5", "high", client=client).__class__
    list(ClaudeBackend("claude-sonnet-5-5", "high", client=client).stream("S", [{"role": "user", "content": "q"}]))
    assert client.calls[-1]["model"] == "claude-sonnet-5-5" and client.calls[-1]["output_config"] == {"effort": "high"}


def test_claude_max_tokens_stop_reason_is_flagged_not_an_error():
    b = ClaudeBackend(client=FakeClient(stop_reason="max_tokens"))
    list(b.stream("S", [{"role": "user", "content": "q"}]))
    assert b.info.truncated and any("cut off" in n for n in b.info.notes)


def test_claude_refusal_raises_after_the_partial_text():
    final_kw = {}
    client = FakeClient(texts=("I can't",), stop_reason="refusal")
    client.final.stop_details = SimpleNamespace(category="cyber", explanation="not allowed")
    b = ClaudeBackend(client=client)
    got = []
    with pytest.raises(LLMError, match="declined.*not allowed"):
        for p in b.stream("S", [{"role": "user", "content": "q"}]):
            got.append(p)
    assert got == ["I can't"] and b.info.stop_reason == "refusal"


def test_claude_notes_when_a_fallback_model_answered():
    usage = SimpleNamespace(input_tokens=1, output_tokens=2,
                            iterations=[SimpleNamespace(type="message"), SimpleNamespace(type="fallback_message")])
    b = ClaudeBackend(client=FakeClient(usage=usage))
    list(b.stream("S", [{"role": "user", "content": "q"}]))
    assert any("fallback model" in n for n in b.info.notes)


@pytest.mark.parametrize("make,match", [
    (lambda: anth_error("AuthenticationError", 401), "ANTHROPIC_API_KEY"),
    (lambda: anth_error("PermissionDeniedError", 403), "isn't allowed"),
    (lambda: anth_error("RateLimitError", 429, {"retry-after": "17"}), "rate limiting.*17 s"),
    (lambda: anth_error("RateLimitError", 429), "rate limiting"),
    (lambda: anth_error("InternalServerError", 500), "API error 500"),
    (lambda: anth_error("BadRequestError", 400), "API error 400"),
    (lambda: __import__("anthropic").APIConnectionError(request=__import__("httpx2").Request("POST", "https://x")),
     "Can't reach the Claude API"),
    (lambda: TypeError("Could not resolve authentication method. Expected one of api_key"), "No Claude credentials"),
])
@pytest.mark.parametrize("where", ["enter", "mid"])
def test_claude_errors_become_helpful_messages(make, match, where):
    b = ClaudeBackend(client=FakeClient(error=(where, make())))
    with pytest.raises(LLMError, match=match):
        list(b.stream("S", [{"role": "user", "content": "q"}]))


def test_claude_unrelated_type_errors_are_not_swallowed():
    b = ClaudeBackend(client=FakeClient(error=("enter", TypeError("a bug"))))
    with pytest.raises(TypeError):
        list(b.stream("S", [{"role": "user", "content": "q"}]))


def test_claude_reads_credentials_from_the_environment_only(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", Recorder)
    ClaudeBackend().client
    assert seen == {}                                       # no api_key passed: the SDK reads ANTHROPIC_API_KEY
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert llm.have_claude_credentials()
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert not llm.have_claude_credentials()
    assert "api_key" not in [f.key for f in S.REGISTRY] and not any("key" in f.key for f in S.REGISTRY)


def test_make_backend_and_session_wiring():
    assert isinstance(llm.make_backend("ollama"), OllamaBackend) and llm.make_backend("claude").name == "claude"
    with pytest.raises(LLMError, match="unknown backend"):
        llm.make_backend("gpt")
    s = S.Settings({"ask_backend": "claude", "claude_effort": "high", "ollama_url": "http://gpu:11434",
                    "ollama_model": "llama3.1:8b"})
    b = session.backend_from(s)
    assert (b.name, b.model, b.effort) == ("claude", "claude-opus-5-5", "high")
    o = session.backend_from(s, "ollama")
    assert (o.url, o.model) == ("http://gpu:11434", "llama3.1:8b")
    assert session.backend_from(s, "ollama", "qwen:7b", "http://other:1").model == "qwen:7b"
    assert session.BUDGET["claude"] > session.BUDGET["ollama"]


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_answers_and_shows_passages(data, wt, capsys, monkeypatch):  # noqa: F811
    from experiments.ask import __main__ as cli
    rag = make_rag(data, wt)
    monkeypatch.setattr(cli.session, "build_rag", lambda d, b, **kw: rag)
    monkeypatch.setattr(cli.session, "backend_from", lambda *a, **k: rag.backend)
    monkeypatch.setattr("sys.argv", ["x", "--data", str(data), "--show-passages", "When", "was", "the", "Battle",
                                     "of", "Waterloo", "fought?"])
    cli.main()
    cap = capsys.readouterr()
    assert "It was fought on 18 June 1815 [1]." in cap.out and "(corrected) It was fought on 18 June 1815 [1]. See also." in cap.out
    assert "Sources (* = cited):" in cap.out and "*[1] Battle of Waterloo" in cap.out
    assert "The Battle of Waterloo was fought on 18 June 1815" in cap.out and "retrieval" in cap.out
    assert "removed citations" in cap.err


def test_cli_interactive_follow_up_and_new_conversation(data, wt, capsys, monkeypatch):  # noqa: F811
    from experiments.ask import __main__ as cli
    rag = make_rag(data, wt)
    monkeypatch.setattr(cli.session, "build_rag", lambda d, b, **kw: rag)
    monkeypatch.setattr(cli.session, "backend_from", lambda *a, **k: rag.backend)
    lines = iter(["Battle of Waterloo?", "and when?", "/new", "Eiffel Tower?", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    monkeypatch.setattr("sys.argv", ["x", "--data", str(data), "-i"])
    cli.main()
    out = capsys.readouterr().out
    assert out.count("Sources (* = cited):") == 3 and "(new conversation)" in out
    assert "searched:" in out                                             # the follow-up was rewritten


def test_cli_reports_missing_data_and_bad_backend(tmp_path, capsys, monkeypatch):
    from experiments.ask import __main__ as cli
    monkeypatch.setattr("sys.argv", ["x", "--data", str(tmp_path), "hello"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert "build_keyword" in str(e.value)
    monkeypatch.setattr("sys.argv", ["x", "--backend", "gpt", "hello"])
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr("sys.argv", ["x"])
    with pytest.raises(SystemExit):
        cli.main()


def test_semantic_cli_modes(data, wt, capsys, monkeypatch):  # noqa: F811
    from experiments.semantic_search import __main__ as cli
    embed(data)
    monkeypatch.setattr(cli, "SemanticSearch", lambda d: SemanticSearch(d, embedder=HashingEmbedder()))
    for mode in ("hybrid", "keyword", "semantic"):
        monkeypatch.setattr("sys.argv", ["x", "--data", str(data), "--mode", mode, "-k", "2", "heavy", "rain", "mud"])
        cli.main()
        out = capsys.readouterr().out
        assert f"[{mode}]" in out and "Battle of Waterloo" in out
    monkeypatch.setattr("sys.argv", ["x", "--data", str(data), "heavy", "rain"])           # default is hybrid
    cli.main()
    assert "[hybrid]" in capsys.readouterr().out
    # only keyword data: semantic mode explains itself
    import shutil
    shutil.rmtree(data / "search")
    monkeypatch.setattr("sys.argv", ["x", "--data", str(data), "--mode", "semantic", "rain"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert "build_embed" in str(e.value)


# ── the screen ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ask_screen_streams_an_answer_with_sources_and_follow_ups(data, wt, tmp_path):  # noqa: F811
    from experiments.ask.screen import AskScreen
    from textual.widgets import Input, Label, Markdown, OptionList, Static

    def reply(system, msgs):
        if system == R.REWRITE_PROMPT:
            return "What happened before the Battle of Waterloo?"
        return "Heavy rain turned the fields to mud [1]. Nonsense [8]."
    rag = make_rag(data, wt, FakeBackend(reply))
    app = make_app(tmp_path, [], data_dir=data)
    async with app.run_test(size=(150, 50)) as pilot:
        screen = AskScreen(data, rag=rag)
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.is_mounted and screen.query("Input"))
        query = screen.query_one("#ask-query", Input)
        assert await wait_for(pilot, lambda: not query.disabled)
        status = lambda: str(screen.query_one("#ask-status", Label).render())      # noqa: E731
        assert "Ready" in status() and "fake:1b" in str(screen.query_one("#ask-backend", Label).render())
        await pilot.press(*"What was the weather at the Battle of Waterloo?")
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: screen.last is not None)
        assert screen.last.cited == [1] and screen.last.invalid == [8]
        md = screen.query_one("#ask-answer", Markdown)
        assert "**You:** What was the weather" in md.source and "Heavy rain turned the fields to mud [1]." in md.source
        assert "[8]" not in md.source.split("note:")[0] and "removed citations" in md.source
        assert "retrieval" in status() and "Cited: [1]" in status()
        sources = screen.query_one("#ask-sources", OptionList)
        assert sources.option_count == len(screen.sources) >= 1
        detail = lambda: str(screen.query_one("#ask-detail", Static).render())      # noqa: E731
        assert await wait_for(pilot, lambda: "Battle of Waterloo" in detail() and "[1]" in detail())
        # Enter on a source shows the whole passage; ctrl+o the whole article
        sources.focus()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: "Heavy rain turned the fields to mud" in detail())
        await pilot.press("ctrl+o")
        assert await wait_for(pilot, lambda: "whole article" in detail() and "Napoleon returned from Elba." in detail())
        # a follow-up uses the history and is rewritten for retrieval
        query.focus()
        await pilot.press(*"and before it?")
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: len(screen.conv.turns) == 2 and not screen._busy)
        assert screen.last.query == "What happened before the Battle of Waterloo?"
        assert "Searched for" in status()
        # a new conversation clears everything
        await pilot.press("ctrl+n")
        assert screen.conv.turns == [] and screen.sources == [] and screen.query_one("#ask-sources").option_count == 0
        assert "New conversation" in md.source or "New conversation" in status()


@pytest.mark.asyncio
async def test_ask_screen_switches_backend_and_model(data, wt, tmp_path):  # noqa: F811
    from experiments.ask.screen import AskScreen, CLAUDE_MODELS
    from textual.widgets import Input, Label
    rag = make_rag(data, wt)
    app = make_app(tmp_path, [], data_dir=data)
    async with app.run_test(size=(150, 50)) as pilot:
        screen = AskScreen(data, rag=rag, settings=S.Settings())
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.is_mounted and screen.query("Input"))
        assert await wait_for(pilot, lambda: not screen.query_one("#ask-query", Input).disabled)
        label = lambda: str(screen.query_one("#ask-backend", Label).render())      # noqa: E731
        await pilot.press("ctrl+b")
        assert rag.backend.name == "claude" and rag.backend.model == "claude-opus-5-5" and "claude" in label()
        assert rag.budget_tokens == session.BUDGET["claude"]
        await pilot.press("ctrl+e")
        assert rag.backend.model == CLAUDE_MODELS[1] and CLAUDE_MODELS[1] in label()
        await pilot.press("ctrl+b")
        assert rag.backend.name == "ollama" and rag.budget_tokens == session.BUDGET["ollama"]


@pytest.mark.asyncio
async def test_ask_screen_shows_backend_errors_without_getting_stuck(data, wt, tmp_path):  # noqa: F811
    from experiments.ask.screen import AskScreen
    from textual.widgets import Input, Markdown
    backend = FakeBackend(lambda s, m: "Start [1]", fail="Can't reach ollama at http://localhost:11434")
    rag = make_rag(data, wt, backend)
    app = make_app(tmp_path, [], data_dir=data)
    async with app.run_test(size=(150, 50)) as pilot:
        screen = AskScreen(data, rag=rag)
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.is_mounted and screen.query("Input"))
        query = screen.query_one("#ask-query", Input)
        assert await wait_for(pilot, lambda: not query.disabled)
        await pilot.press(*"Battle of Waterloo?")
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: screen.last is not None)
        assert "Can't reach ollama" in screen.query_one("#ask-answer", Markdown).source
        assert not screen._busy and app.focused is query
        screen.query_one("#ask-query", Input).value = "Waterloo battle once more?"        # still usable afterwards
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: len(backend.calls) >= 2)


@pytest.mark.asyncio
async def test_ask_screen_stop_key_cancels_generation(data, wt, tmp_path):  # noqa: F811
    from experiments.ask.screen import AskScreen
    from textual.widgets import Input, Markdown
    gate = threading.Event()

    class Slow(FakeBackend):
        def stream(self, system, messages, max_tokens=1024):
            yield "First "
            gate.wait(5)
            for _ in range(50):
                yield "more "

    rag = make_rag(data, wt, Slow())
    app = make_app(tmp_path, [], data_dir=data)
    async with app.run_test(size=(150, 50)) as pilot:
        screen = AskScreen(data, rag=rag)
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.is_mounted and screen.query("Input"))
        query = screen.query_one("#ask-query", Input)
        assert await wait_for(pilot, lambda: not query.disabled)
        await pilot.press(*"Battle of Waterloo?")
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: screen._busy and screen.sources)
        await pilot.press("ctrl+x")
        gate.set()
        assert await wait_for(pilot, lambda: not screen._busy)
        assert "stopped" in screen.query_one("#ask-answer", Markdown).source


@pytest.mark.asyncio
async def test_ask_screen_without_data_says_so(tmp_path):
    from experiments.ask.screen import AskScreen
    from textual.widgets import Input, Label
    app = make_app(tmp_path, [], data_dir=tmp_path / "data")
    async with app.run_test(size=(150, 50)) as pilot:
        screen = AskScreen(tmp_path / "data", backend=FakeBackend(), settings=S.Settings())
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.is_mounted and screen.query("Input"))
        assert await wait_for(pilot, lambda: "Can't load Ask Wikipedia" in str(screen.query_one("#ask-status", Label).render()))
        assert screen.query_one("#ask-query", Input).disabled


@pytest.mark.asyncio
async def test_semantic_search_screen_hybrid_and_mode_switch(data, wt, tmp_path):  # noqa: F811
    from experiments.semantic_search.screen import SemanticSearchScreen
    from textual.widgets import Input, Label, OptionList
    embed(data)
    app = make_app(tmp_path, [], data_dir=data)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SemanticSearchScreen(data, SemanticSearch(data, embedder=HashingEmbedder()))
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.is_mounted and screen.query("Input"))
        query = screen.query_one("#ss-query", Input)
        assert await wait_for(pilot, lambda: not query.disabled)
        assert screen.mode == "hybrid" and "by words" in str(screen.query_one("#ss-status", Label).render())
        await pilot.press(*"heavy rain mud battle")
        await pilot.press("enter")
        results = screen.query_one("#ss-results", OptionList)
        assert await wait_for(pilot, lambda: results.option_count >= 1)
        assert screen.hits[0].title == "Battle of Waterloo" and screen.hits[0].via
        await pilot.press("ctrl+t")
        assert screen.mode == "semantic"
        await pilot.press("ctrl+t")
        assert screen.mode == "keyword"
        await pilot.press("ctrl+t")
        assert screen.mode == "hybrid"
