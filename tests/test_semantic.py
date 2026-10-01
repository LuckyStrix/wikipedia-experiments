"""The text stage, the embed stage and semantic search, on a tiny dump and a fake (hashing) embedder,
so no model is downloaded."""
import json
import sqlite3
import threading
import zlib

import numpy as np
import pytest

from pipeline import build_embed, build_text
from pipeline import settings as S
from pipeline import stages as ST
from tests.test_pipeline import fake_stage, run
from tests.test_tui import log_text, make_app, wait_for
from tests.test_wikitext import PAGES, WATERLOO, wt, dump  # noqa: F401  (fixtures)
from wikiexp import semantic
from wikiexp.semantic import SemanticSearch

PAGE_IDS = np.array([10, 12, 20, 21, 30, 31])          # the main-namespace articles of the tiny dump
RANK = np.array([1, 0, 2, 5, 4, 3])                   # popularity rank by idx: Photosynthesis is most popular


class HashingEmbedder:
    """Bag-of-words hashed into `dim` buckets: texts sharing words are close. Counts its calls."""
    dim = 64

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def encode(self, texts, query=False):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("GPU on fire")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(":", " ").replace(".", " ").split():
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1, norm)


def build_leads(tmp_path, wt, lead_chars=1200):
    out = tmp_path / "data" / "text"
    build_text.build(out, lead_chars, workers=1, wt=wt, page_ids=PAGE_IDS, rank=RANK, say=lambda *a: None)
    return tmp_path / "data"


# ── text stage ───────────────────────────────────────────────────────────────

def test_build_text_writes_leads_in_popularity_order(tmp_path, wt):
    data = build_leads(tmp_path, wt)
    db = sqlite3.connect(data / "text" / "leads.sqlite")
    rows = db.execute("SELECT rank, page_id, idx, title, lead, disambig FROM leads ORDER BY rank").fetchall()
    assert [r[3] for r in rows] == ["Photosynthesis", "Battle of Waterloo", "Eiffel Tower", "Heading first",
                                    "Empty page", "Mercury (disambiguation)"]
    assert [r[0] for r in rows] == list(range(6))                      # rank = row number
    by_title = {r[3]: r for r in rows}
    assert by_title["Battle of Waterloo"][4].startswith("The Battle of Waterloo was fought on 18 June 1815")
    assert "Napoleon" not in by_title["Battle of Waterloo"][4]            # only the intro
    assert by_title["Photosynthesis"][4] == "Photosynthesis is how plants make sugar from light & water."
    assert by_title["Heading first"][4] == "It started long ago in Rome."
    assert by_title["Empty page"][4] == ""                                 # kept, so ids stay complete
    assert [r[5] for r in rows] == [0, 0, 0, 0, 0, 1]                      # only the disambiguation page
    assert by_title["Battle of Waterloo"][2] == 0 and by_title["Battle of Waterloo"][1] == 10
    meta = dict(db.execute("SELECT key, value FROM meta"))
    assert meta["articles"] == "6" and meta["disambig"] == "1" and meta["lead_chars"] == "1200"
    assert not list((data / "text").glob("*.building"))                   # temp files cleaned up


def test_build_text_lead_length_setting(tmp_path, wt):
    data = build_leads(tmp_path, wt, lead_chars=100)
    lead, = sqlite3.connect(data / "text" / "leads.sqlite").execute(
        "SELECT lead FROM leads WHERE title = 'Battle of Waterloo'").fetchone()
    assert len(lead) <= 100 and lead.endswith(".")


def test_build_text_skips_pages_missing_from_the_core_database(tmp_path, wt):
    out = tmp_path / "text"
    msgs = []
    build_text.build(out, workers=1, wt=wt, page_ids=np.array([10, 12]), rank=np.array([0, 1]), say=lambda m, p=None: msgs.append(m))
    assert sqlite3.connect(out / "leads.sqlite").execute("SELECT count(*) FROM leads").fetchone()[0] == 2
    assert any("not in wiki.sqlite" in m for m in msgs)


def test_extract_survives_a_page_the_cleaner_chokes_on(monkeypatch):
    monkeypatch.setattr(build_text.W, "lead", lambda *a: 1 / 0)
    assert build_text.extract(build_text.W.Page(1, "T", 0, None, "x"), 100) == ("", 0)


def test_popularity_ranks(tmp_path):
    g = tmp_path / "graph"
    g.mkdir()
    np.save(g / "idx_to_page_id.npy", np.array([5, 6, 7]))
    np.save(g / "in_indptr.npy", np.array([0, 1, 4, 6]))          # in-degrees 1, 3, 2
    ids, rank = build_text.popularity_ranks(g)
    assert list(rank) == [2, 0, 1]


# ── embed stage + search ─────────────────────────────────────────────────────

def embed(data, embedder=None, limit=0, shard=2, model="fake", **kw):
    emb = embedder or HashingEmbedder()
    out = data / "search"
    total, state = build_embed.embed_shards(out, data / "text" / "leads.sqlite", emb, model, limit=limit,
                                            shard_size=shard, batch_size=4, say=lambda *a: None, **kw)
    meta = build_embed.build_index(out, state, total, model, limit=limit, say=lambda *a: None)
    return emb, meta


def test_embed_and_search(tmp_path, wt):
    data = build_leads(tmp_path, wt)
    emb, meta = embed(data)
    assert meta["count"] == 4 and meta["index"] == "flat" and meta["dim"] == 64 and meta["limit"] == 0
    # the disambiguation page and the empty page are not embedded
    ids = np.load(data / "search" / "page_ids.npy")
    assert sorted(ids) == [10, 12, 20, 31]
    assert np.load(data / "search" / "shards" / "emb_00000.npy").dtype == np.float16
    assert (data / "search" / "index.faiss").exists() and json.loads((data / "search" / "meta.json").read_text())["count"] == 4

    ss = SemanticSearch(data, embedder=HashingEmbedder())
    assert ss.available() and len(ss) == 4
    hits = ss.search("heavy rain mud and artillery in the 1815 battle", k=3)
    assert [h.title for h in hits][0] == "Battle of Waterloo"
    assert len(hits) == 3 and hits[0].score >= hits[1].score >= hits[2].score
    top = hits[0]
    assert top.page_id == 10 and top.lead.startswith("The Battle of Waterloo") and top.snippet
    assert ss.search("   ") == [] and ss.search("rain", k=0) == []
    assert len(ss.search("plants and light", k=50)) == 4            # k larger than the index
    assert ss.search("sugar from light", k=1)[0].title == "Photosynthesis"


def test_documents_are_embedded_as_title_colon_lead(tmp_path, wt):
    seen = []

    class Spy(HashingEmbedder):
        def encode(self, texts, query=False):
            seen.extend(texts)
            return super().encode(texts, query)
    embed(build_leads(tmp_path, wt), Spy())
    assert "Eiffel Tower: The Eiffel Tower is a lattice tower in Paris. It is 330 m tall." in seen


def test_resume_after_a_stop_continues_with_finished_shards(tmp_path, wt):
    data = build_leads(tmp_path, wt)
    with pytest.raises(RuntimeError):
        embed(data, HashingEmbedder(fail_after=1))                  # dies while embedding the 2nd shard
    state = json.loads((data / "search" / "progress.json").read_text())
    assert list(state["shards"]) == ["0"] and not (data / "search" / "meta.json").exists()

    emb, meta = embed(data)                                         # a fresh run picks up at shard 1
    assert emb.calls == 1 and meta["count"] == 4   # shards 1 and 2 remain; shard 2 has nothing to embed
    full = tmp_path / "other"
    (full / "text").mkdir(parents=True)
    (full / "text" / "leads.sqlite").write_bytes((data / "text" / "leads.sqlite").read_bytes())
    _, meta2 = embed(full)
    assert np.array_equal(np.load(data / "search" / "page_ids.npy"), np.load(full / "search" / "page_ids.npy"))
    assert np.array_equal(np.load(data / "search" / "shards" / "emb_00001.npy"),
                          np.load(full / "search" / "shards" / "emb_00001.npy"))


def test_limit_embeds_the_most_popular_and_a_bigger_limit_reuses_shards(tmp_path, wt):
    data = build_leads(tmp_path, wt)
    emb, meta = embed(data, limit=2)
    assert meta["limit"] == 2 and sorted(np.load(data / "search" / "page_ids.npy")) == [10, 12]  # ranks 0 and 1
    emb2, meta2 = embed(data, limit=3)       # shard 0 (ranks 0-1) is reused; the new shard holds rank 2
    assert emb2.calls == 1 and meta2["count"] == 3
    emb3, meta3 = embed(data, limit=0)       # rank 3 is "Heading first", ranks 4, 5 are empty/disambig
    assert meta3["count"] == 4
    emb4, _ = embed(data, limit=0)
    assert emb4.calls == 0                   # everything finished: nothing to embed


def test_changed_model_or_leads_start_over(tmp_path, wt):
    data = build_leads(tmp_path, wt)
    embed(data)
    emb, _ = embed(data, model="another-model")
    assert emb.calls == 2
    emb, _ = embed(data, model="another-model", restart=True)
    assert emb.calls == 2
    build_leads(tmp_path, wt, lead_chars=500)                      # leads rebuilt: shards no longer match
    emb, _ = embed(data, model="another-model")
    assert emb.calls == 2


def test_big_indexes_use_ivf_sq8(tmp_path, monkeypatch):
    monkeypatch.setattr(build_embed, "FLAT_MAX", 100)
    data = tmp_path / "data"
    (data / "text").mkdir(parents=True)
    db = sqlite3.connect(data / "text" / "leads.sqlite")
    db.executescript(build_text.SCHEMA)
    words = [f"w{i}" for i in range(300)]
    rng = np.random.default_rng(1)
    for r in range(600):
        lead = " ".join(rng.choice(words, 12)) + f" unique{r}"
        db.execute("INSERT INTO leads VALUES (?,?,?,?,?,0)", (r, 1000 + r, r, f"Article {r}", lead))
    db.executemany("INSERT INTO meta VALUES (?,?)", [("built", "x"), ("articles", "600"), ("lead_chars", "1200")])
    db.commit()
    db.close()
    emb, meta = embed(data, shard=250)
    assert meta["index"].startswith("IVF") and meta["index"].endswith("SQ8") and meta["nprobe"] == build_embed.NPROBE
    ss = SemanticSearch(data, embedder=HashingEmbedder(), nprobe=60)
    target = ss.leads([1337])[1337]
    hit = ss.search(target[1], k=3)[0]            # searching with a document's own text finds it
    assert (hit.page_id, hit.title) == (1337, "Article 337")


def test_search_is_thread_safe(tmp_path, wt):
    data = build_leads(tmp_path, wt)
    embed(data)
    ss = SemanticSearch(data, embedder=HashingEmbedder())
    errors, results = [], []

    def work(q):
        try:
            results.append(ss.search(q, 2)[0].title)
        except Exception as e:           # pragma: no cover
            errors.append(e)
    queries = ["rain mud battle", "plants light sugar", "tower Paris lattice", "Rome long ago"] * 6
    threads = [threading.Thread(target=work, args=(q,)) for q in queries]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and sorted(set(results)) == ["Battle of Waterloo", "Eiffel Tower", "Heading first", "Photosynthesis"]


def test_search_without_data_explains_what_to_run(tmp_path):
    ss = SemanticSearch(tmp_path, embedder=HashingEmbedder())
    assert not ss.available()
    with pytest.raises(FileNotFoundError, match="embed"):
        ss.search("anything")


def test_query_and_document_prefixes():
    q, d = semantic.prefixes("BAAI/bge-small-en-v1.5")
    assert q.startswith("Represent this sentence") and d == ""
    assert semantic.prefixes("intfloat/e5-small-v2") == ("query: ", "passage: ")
    assert semantic.prefixes("sentence-transformers/all-MiniLM-L6-v2") == ("", "")


def test_sentence_transformer_embedder_applies_the_query_prefix(monkeypatch):
    seen = []

    class FakeModel:
        max_seq_length = 512

        def encode(self, texts, **kw):
            seen.append(list(texts))
            return np.ones((len(texts), 3), dtype=np.float32)

        def get_sentence_embedding_dimension(self):
            return 3

    import sys
    import types
    fake = types.SimpleNamespace(SentenceTransformer=lambda name, device, **kw: FakeModel())
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    e = semantic.SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", device="cpu")
    e.encode(["where"], query=True)
    e.encode(["Title: doc"])
    assert seen[0][0].startswith("Represent this sentence for searching relevant passages: where")
    assert seen[1] == ["Title: doc"] and e.dim == 3 and e.device == "cpu"


# ── stages and settings ──────────────────────────────────────────────────────

def test_run_all_skips_manual_stages(tmp_path):
    marker = tmp_path / "ran.txt"
    auto = fake_stage("auto", f"open(r'{marker}', 'w')")
    manual = fake_stage("gpu", "raise SystemExit(5)")
    manual.manual = True
    ev = run([auto, manual], S.Settings({"data_dir": str(tmp_path)}), lambda r: r.run_all())
    assert marker.exists()
    assert ("state", "gpu", "running") not in ev
    assert any("Skipping manual" in e[1] for e in ev if e[0] == "log")
    # it still runs from its own card
    ev = run([auto, manual], S.Settings({"data_dir": str(tmp_path)}), lambda r: r.run_stage("gpu"))
    assert ("state", "gpu", "failed") in ev


def test_run_all_with_only_manual_stages_left(tmp_path):
    manual = fake_stage("gpu", "pass")
    manual.manual = True
    ev = run([manual], S.Settings({"data_dir": str(tmp_path)}), lambda r: r.run_all())
    assert not [e for e in ev if e[0] == "state"]
    assert any("automatic stage" in e[1] for e in ev if e[0] == "log")


def test_text_and_embed_stages(tmp_path, wt):
    s = S.Settings({"data_dir": str(tmp_path / "data"), "dumps_dir": str(tmp_path / "dumps")})
    text, emb = ST.BY_KEY["text"], ST.BY_KEY["embed"]
    assert emb.manual and not text.manual
    assert [st.key for st in ST.STAGES][-3:] == ["text", "keyword", "embed"]
    assert "pipeline.build_text" in text.command(s) and "--lead-chars" in text.command(s)
    cmd = emb.command(s)
    assert cmd[cmd.index("--model") + 1] == "BAAI/bge-small-en-v1.5" and cmd[cmd.index("--limit") + 1] == "0"
    assert "Needs" not in emb.hint(s) and "GPU" in emb.hint(s)
    assert any("pages-articles-multistream" in m for m in text.missing_inputs(s))

    data = build_leads(tmp_path, wt)
    assert text.is_done(s) and not emb.is_done(s)
    embed(data, model=s["embed_model"])
    assert emb.is_done(s)
    assert any("model:" in line for line in emb.summary(s))
    # data pulled from a GPU machine counts as done even if this machine's settings differ; the View says so
    s["embed_limit"] = "100"
    assert emb.is_done(s) and any("different model or article limit" in line for line in emb.summary(s))


def test_embed_hint_counts_finished_shards(tmp_path, wt):
    s = S.Settings({"data_dir": str(tmp_path / "data")})
    data = build_leads(tmp_path, wt)
    with pytest.raises(RuntimeError):
        embed(data, HashingEmbedder(fail_after=1))
    assert "1 shard(s)" in ST.BY_KEY["embed"].hint(s)
    assert any("unfinished run" in line for line in ST.BY_KEY["embed"].summary(s))


def test_new_settings_validate():
    st = S.BY_KEY
    assert S.valid(st["embed_device"], "cuda:1") and not S.valid(st["embed_device"], "tpu")
    assert S.valid(st["embed_limit"], "50000") and not S.valid(st["embed_limit"], "lots")
    assert not S.valid(st["text_lead_chars"], "5")
    assert S.Settings({"text_lead_chars": "800"}).invalid() == []


# ── the screen ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_semantic_search_screen(tmp_path, wt):
    from experiments.semantic_search.screen import SemanticSearchScreen
    from textual.widgets import Input, OptionList, Static
    data = build_leads(tmp_path, wt)
    embed(data)
    app = make_app(tmp_path, [], data_dir=data)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SemanticSearchScreen(data, SemanticSearch(data, embedder=HashingEmbedder()))
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.is_mounted and screen.query("#ss-query"))
        query = screen.query_one("#ss-query", Input)
        assert await wait_for(pilot, lambda: not query.disabled)
        assert "4 articles searchable" in str(screen.query_one("#ss-status").render())
        await pilot.press(*"heavy rain mud battle")
        await pilot.press("enter")
        results = screen.query_one("#ss-results", OptionList)
        assert await wait_for(pilot, lambda: results.option_count == 4)
        assert screen.hits[0].title == "Battle of Waterloo"
        detail = lambda: str(screen.query_one("#ss-detail", Static).render())
        assert await wait_for(pilot, lambda: "Heavy rain turned the fields" in detail())
        await pilot.press("down")
        assert await wait_for(pilot, lambda: screen.hits[1].title in detail())
        await pilot.press("ctrl+l")
        assert app.focused is query
        screen.query_one("#ss-query", Input).value = "zzz qqq"
        await pilot.press("enter")                      # nothing in common: still shows ranked results, no crash
        assert await wait_for(pilot, lambda: "results in" in str(screen.query_one("#ss-status").render()))


@pytest.mark.asyncio
async def test_semantic_search_screen_without_data_says_so(tmp_path):
    from experiments.semantic_search.screen import SemanticSearchScreen
    app = make_app(tmp_path, [], data_dir=tmp_path / "data")
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SemanticSearchScreen(tmp_path / "data", SemanticSearch(tmp_path / "data", embedder=HashingEmbedder()))
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.is_mounted and screen.query("#ss-status"))
        assert await wait_for(pilot, lambda: "Can't load search data" in str(screen.query_one("#ss-status").render()))
        assert screen.query_one("#ss-query").disabled


def test_cli_prints_ranked_results(tmp_path, wt, capsys, monkeypatch):
    from experiments.semantic_search import __main__ as cli
    data = build_leads(tmp_path, wt)
    embed(data)
    monkeypatch.setattr(cli, "SemanticSearch", lambda d: SemanticSearch(d, embedder=HashingEmbedder()))
    monkeypatch.setattr("sys.argv", ["x", "--data", str(data), "-k", "2", "heavy", "rain", "mud"])
    cli.main()
    out = capsys.readouterr().out
    assert "1." in out and "Battle of Waterloo" in out and "[10]" in out and out.count("\n  ") >= 2
    monkeypatch.setattr("sys.argv", ["x", "--data", str(tmp_path / "empty"), "rain"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert "pipeline.build_embed" in str(e.value)


@pytest.mark.asyncio
async def test_manual_embed_card_explains_itself(tmp_path, wt):
    import dataclasses       # no process detection: a real build may be running on this machine
    stages = [dataclasses.replace(st, signature="") for st in ST.STAGES]
    app = make_app(tmp_path, stages, data_dir=tmp_path / "data")
    async with app.run_test(size=(140, 60)) as pilot:
        assert "manual stage" in app.cards["embed"].status
        assert "Needs" in app.cards["text"].status
        build_leads(tmp_path, wt)
        app.refresh_status()
        assert "run on a GPU machine" in app.cards["embed"].status
        assert app.cards["text"].state == "done"


def test_model_loads_from_cache_first_and_downloads_only_if_missing(monkeypatch):
    import sys
    import types
    calls = []

    class FakeModel:
        max_seq_length = 512

    def factory(name, device, **kw):
        calls.append(kw)
        if kw.get("local_files_only"):
            raise OSError("not cached")
        return FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=factory))
    semantic.SentenceTransformerEmbedder("some/model", device="cpu").model
    assert calls == [{"local_files_only": True}, {}]
