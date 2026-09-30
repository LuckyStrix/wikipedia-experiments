import pytest

from wikiexp.titles import TitleIndex


@pytest.fixture
def index(tiny_data):
    return TitleIndex(tiny_data / "titles.sqlite")


def test_exact_any_case(index):
    assert index.resolve("kevin bacon").article == "Kevin Bacon"
    assert index.resolve("KEVIN_BACON").article == "Kevin Bacon"
    assert index.resolve("Kevin Baco") is None


def test_redirects_lead_to_articles(index):
    m = index.resolve("mitochondria")
    assert (m.title, m.article) == ("Mitochondria", "Mitochondrion")
    assert m.label == "Mitochondria → Mitochondrion"


def test_prefix_and_contains(index):
    assert index.search("Mito")[0].article == "Mitochondrion"
    assert {m.article for m in index.search("bacon")} >= {"Kevin Bacon", "Bacon"}
    assert index.search("Fi")[0].article == "Film"   # short queries use the case-insensitive index


def test_one_result_per_article(index):
    ids = [m.page_id for m in index.search("bacon")]
    assert len(ids) == len(set(ids))


def test_typos_suggest_similar(index):
    assert index.search("Mitocondrion")[0].article == "Mitochondrion"
    assert index.search("Kevn Bacon")[0].article == "Kevin Bacon"


def test_popular_first(index):
    # "olo" is inside Biology (2 incoming links) and Cell (biology) (1)
    assert [m.article for m in index.search("olo")] == ["Biology", "Cell (biology)"]
    # an exact title beats more popular prefix matches
    assert [m.article for m in index.search("bacon")][:2] == ["Bacon", "Kevin Bacon"]


def test_random_article(index):
    assert index.random_article(top=5).article
