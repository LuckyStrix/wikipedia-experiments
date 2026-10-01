"""Nearby: parsing, geometry, the places build (from fake dumps) and the ranked search, on tiny data."""
import gzip
import random
import sqlite3
import sys

import numpy as np
import pytest

from experiments.nearby import geometry as G
from experiments.nearby.finder import Location, NearbyFinder, notability
from experiments.nearby.parsing import (CoordinateError, format_distance, format_radius, parse_coordinates,
                                        parse_radius)
from pipeline import build_geo
from tests.test_tui import make_app, wait_for
from wikiexp import paths
from wikiexp.titles import Match

# page ids in tests/conftest.py: Kevin Bacon 100, Footloose 101, Film 102, Biology 103,
# Cell (biology) 104, Mitochondrion 105, Isolated article 106, Bacon 107
PLACES = [  # page_id, lat, lon, type, langs
    (102, 48.8584, 2.2945, "landmark", 50),    # "Film" is the Eiffel Tower here
    (103, 48.8620, 2.3000, None, 5),           # ~0.5 km away
    (104, 48.9000, 2.4000, "edu", 10),         # ~9 km away
    (105, -18.0, 179.9, "isle", 1),            # either side of the antimeridian
    (101, -17.9, -179.9, "city", 30),
    (100, 89.95, 10.0, "mountain", 2),         # either side of the North Pole
    (107, 89.95, -170.0, None, 0),
]


@pytest.fixture
def geo_data(tiny_data):
    from wikiexp import core
    g = core.Graph(tiny_data / "graph")
    db = sqlite3.connect(tiny_data / "geo.sqlite")
    db.executescript(build_geo.SCHEMA)
    for i, (pid, lat, lon, typ, langs) in enumerate(PLACES, 1):
        idx = g.idx(pid)
        title = core.connect(tiny_data / "wiki.sqlite").execute(
            "SELECT title FROM articles WHERE page_id = ?", (pid,)).fetchone()[0]
        db.execute("INSERT INTO places VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (i, pid, idx, title, lat, lon, typ, None, 1000, None, None, langs, len(g.in_links(idx))))
        db.execute("INSERT INTO places_rt VALUES (?,?,?,?,?)", (i, lat, lat, lon, lon))
    db.executemany("INSERT INTO meta VALUES (?,?)", [("places", str(len(PLACES))), ("langlinks_used", "True")])
    db.commit()
    db.close()
    return tiny_data


@pytest.fixture
def finder(geo_data):
    return NearbyFinder(geo_data)


# ── parsing ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("48.8584, 2.2945", (48.8584, 2.2945)),
    ("48.8584 2.2945", (48.8584, 2.2945)),
    ("40.7484,-73.9857", (40.7484, -73.9857)),
    ("-33.86 151.21", (-33.86, 151.21)),
    ("48.86N 2.29E", (48.86, 2.29)),
    ("33.86S, 151.21E", (-33.86, 151.21)),
    ("40.7N 74.0W", (40.7, -74.0)),
    ("N 48.86, E 2.29", (48.86, 2.29)),
    ("2.29E 48.86N", (48.86, 2.29)),                 # longitude first, but the letters say so
    ("48.86° N, 2.29° E", (48.86, 2.29)),
    ("48°51′29″N 2°17′40″E", (48 + 51 / 60 + 29 / 3600, 2 + 17 / 60 + 40 / 3600)),
    ("48°51'29\"N 2°17'40\"E", (48 + 51 / 60 + 29 / 3600, 2 + 17 / 60 + 40 / 3600)),
    ("48°51'29''N, 2°17'40''E", (48 + 51 / 60 + 29 / 3600, 2 + 17 / 60 + 40 / 3600)),
    ("33°51′S 151°12′E", (-(33 + 51 / 60), 151 + 12 / 60)),
    ("33° 51' 54\" S 151° 12' 40\" E", (-(33 + 51 / 60 + 54 / 3600), 151 + 12 / 60 + 40 / 3600)),
    ("0, 0", (0.0, 0.0)),
    ("90, 180", (90.0, 180.0)),
])
def test_coordinate_formats(text, expected):
    lat, lon = parse_coordinates(text)
    assert lat == pytest.approx(expected[0]) and lon == pytest.approx(expected[1])


@pytest.mark.parametrize("text", ["Eiffel Tower", "NYC", "Area 51", "7 Eleven", "1984", "", "48.8584", "Paris, France"])
def test_names_are_not_coordinates(text):
    assert parse_coordinates(text) is None


@pytest.mark.parametrize("text", ["100, 20", "10, 200", "48.86N 2.29N", "48.86 N, -2.29 E", "48°75'N 2°E"])
def test_invalid_coordinates_explain_themselves(text):
    with pytest.raises(CoordinateError):
        parse_coordinates(text)


@pytest.mark.parametrize("text, km", [
    ("500 m", 0.5), ("500m", 0.5), ("2km", 2), ("2 KM", 2), ("3 mi", 3 * 1.609344), ("3 miles", 3 * 1.609344),
    ("1.5 kilometres", 1.5), ("5", 5), (" 0.25 ", 0.25), ("1,500 m", 1.5), ("1,5 km", 1.5), ("1000 ft", 0.3048),
])
def test_radius_formats(text, km):
    assert parse_radius(text) == pytest.approx(km)


@pytest.mark.parametrize("text", ["", "km", "-5 km", "0", "2 furlongs", "abc", "50000 km"])
def test_bad_radius(text):
    with pytest.raises(ValueError):
        parse_radius(text)


def test_formatting():
    assert format_distance(0.35) == "350 m"
    assert format_distance(2.345) == "2.35 km"
    assert format_distance(48.24) == "48.2 km"
    assert format_distance(1240.4) == "1,240 km"
    assert format_radius(0.5) == "500 m" and format_radius(2.0) == "2 km" and format_radius(1.5) == "1.5 km"


# ── geometry ──────────────────────────────────────────────────────────────────

def test_haversine_known_distances():
    assert G.haversine_km(48.8566, 2.3522, 51.5074, -0.1278) == pytest.approx(343.5, abs=1)     # Paris-London
    assert G.haversine_km(0, 179.5, 0, -179.5) == pytest.approx(111.2, abs=0.5)                # across 180
    assert G.haversine_km(89.95, 10, 89.95, -170) == pytest.approx(11.1, abs=0.2)              # across the pole
    assert G.haversine_km(10, 20, 10, 20) == 0


def test_bearing_and_compass():
    assert G.bearing_deg(0, 0, 1, 0) == pytest.approx(0)
    assert G.bearing_deg(0, 0, 0, 1) == pytest.approx(90)
    assert G.bearing_deg(0, 0, -1, 0) == pytest.approx(180)
    assert G.bearing_deg(0, 0, 0, -1) == pytest.approx(270)
    assert [G.compass(b) for b in (0, 44, 90, 135, 200, 270, 315, 359.9)] == ["N", "NE", "E", "SE", "S", "W", "NW", "N"]
    assert G.compass(np.array([0.0, 180.0])) == ["N", "S"]


def test_boxes_cross_the_antimeridian_and_pole():
    assert len(G.bounding_boxes(-18, 179.9, 50)) == 2
    east, west = G.bounding_boxes(0, -179.9, 50)
    assert {east[3], west[2]} == {180.0, -180.0} or east[3] == 180.0 or west[2] == -180.0
    (box,) = G.bounding_boxes(89.9, 0, 50)           # reaches the pole: every longitude
    assert box[2:] == (-180.0, 180.0) and box[1] == 90.0
    (box,) = G.bounding_boxes(10, 20, 50)
    assert box[2] < 20 < box[3] and box[0] < 10 < box[1]


def test_boxes_always_contain_the_circle():
    rng = random.Random(1)
    for _ in range(300):
        lat, lon = rng.uniform(-90, 90), rng.uniform(-180, 180)
        radius = rng.choice([1, 20, 300, 2000, 9000])
        boxes = G.bounding_boxes(lat, lon, radius)
        for _ in range(60):     # random points inside the circle must land in some box
            bearing, dist = rng.uniform(0, 2 * np.pi), radius * rng.random() ** 0.5 * 0.999
            d = dist / G.EARTH_RADIUS_KM
            p1, l1 = np.radians(lat), np.radians(lon)
            p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(bearing))
            l2 = l1 + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
            plat, plon = np.degrees(p2), G.wrap_lon(np.degrees(l2))
            assert any(a <= plat <= b and c <= plon <= e for a, b, c, e in boxes), (lat, lon, radius, plat, plon)


# ── score ─────────────────────────────────────────────────────────────────────

def test_notability_formula():
    assert notability(0, 0, 1000, 100) == 0
    assert notability(1000, 100, 1000, 100) == pytest.approx(100)
    a, b = notability(999, 0, 1000, 100), notability(0, 99, 1000, 100)
    assert 69 < a < 70 and 29 < b < 30                    # 70% from the signal, 30% from languages
    assert notability(500, 0, 1000, 0) == pytest.approx(100 * np.log1p(500) / np.log1p(1000))   # no langlinks data
    assert notability(10, 5, 1000, 100) < notability(100, 5, 1000, 100) < notability(100, 50, 1000, 100)


# ── the build, from fake dumps ────────────────────────────────────────────────

def write_dump(path, table, columns, rows):
    cols = "\n".join(f"  `{c}` int(8) NOT NULL," for c in columns).rstrip(",")
    body = ",\n".join(rows)
    with gzip.open(path, "wt") as f:
        f.write(f"-- header\nDROP TABLE IF EXISTS `{table}`;\nCREATE TABLE `{table}` (\n{cols}\n) ENGINE=InnoDB;\n"
                f"INSERT INTO `{table}` VALUES\n{body};\n")


def test_build_geo_from_dumps(tiny_data, tmp_path, monkeypatch):
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    monkeypatch.setattr(paths, "DUMPS", dumps)
    monkeypatch.setattr(paths, "DUMP_DATE", "20260901")
    monkeypatch.setattr(build_geo, "log", lambda *a, **k: None)
    monkeypatch.setenv("WIKI_WORKERS", "1")
    cols = ["gt_id", "gt_page_id", "gt_globe", "gt_primary", "gt_lat", "gt_lon", "gt_dim", "gt_type",
            "gt_name", "gt_country", "gt_region"]
    geo_rows = [
        "(1,102,'earth',1,48.8584,2.2945,1000,'landmark',NULL,'FR','75')",
        "(2,103,'earth',1,48.862,2.3,NULL,'city(334,000',NULL,NULL,NULL)",
        "(3,104,'earth',0,48.9,2.4,NULL,NULL,NULL,NULL,NULL)",        # secondary only: skipped
        "(4,105,'earth',1,-18.0,179.9,NULL,'river:us-oh',NULL,NULL,NULL)",
        "(5,100,'moon',1,10.0,10.0,NULL,NULL,NULL,NULL,NULL)",        # not Earth
        "(6,999,'earth',1,10.0,10.0,NULL,NULL,NULL,NULL,NULL)",       # not an article here
        "(7,101,'earth',1,0.0,0.0,NULL,NULL,NULL,NULL,NULL)",         # (0, 0) placeholder
        "(8,107,'earth',1,95.0,10.0,NULL,NULL,NULL,NULL,NULL)",       # impossible latitude
        "(9,102,'earth',1,1.0,1.0,NULL,NULL,NULL,NULL,NULL)",         # second primary: first wins
    ]
    write_dump(dumps / "enwiki-20260901-geo_tags.sql.gz", "geo_tags", cols, geo_rows)
    write_dump(dumps / "enwiki-20260901-langlinks.sql.gz", "langlinks", ["ll_from", "ll_lang", "ll_title"],
               ["(102,'fr','Tour')", "(102,'de','Turm')", "(102,'es','Torre')", "(103,'fr','Biologie')",
                "(500,'fr','x')"])
    monkeypatch.setattr(sys, "argv", ["build_geo"])
    build_geo.main()

    db = sqlite3.connect(tiny_data / "geo.sqlite")
    rows = {r[1]: r for r in db.execute("SELECT page_id, title, lat, lon, type, pop, dim, country, region, langs, links "
                                        "FROM places")}
    assert set(rows) == {"Film", "Biology", "Mitochondrion"}
    film = rows["Film"]
    assert film[2:] == (48.8584, 2.2945, "landmark", None, 1000, "FR", "75", 3, 1)   # 3 langs; 1 incoming link
    assert rows["Biology"][4:6] == ("city", 334000) and rows["Biology"][9] == 1
    assert rows["Mitochondrion"][4] == "river" and rows["Mitochondrion"][9] == 0
    meta = dict(db.execute("SELECT key, value FROM meta"))
    assert meta["skipped_secondary_only"] == "1" and meta["skipped_other_bodies"] == "1"
    assert meta["skipped_not_articles"] == "1" and meta["skipped_bad_coordinates"] == "2"
    assert meta["repeated_primary_tags"] == "1" and meta["places"] == "3"
    # the R*Tree answers box queries
    n = db.execute("SELECT count(*) FROM places_rt WHERE min_lat >= 48 AND max_lat <= 49 AND min_lon >= 2 "
                   "AND max_lon <= 3").fetchone()[0]
    assert n == 2


def test_build_geo_without_langlinks(tiny_data, tmp_path, monkeypatch):
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    monkeypatch.setattr(paths, "DUMPS", dumps)
    monkeypatch.setattr(paths, "DUMP_DATE", "20260901")
    monkeypatch.setattr(build_geo, "log", lambda *a, **k: None)
    monkeypatch.setenv("WIKI_WORKERS", "1")
    cols = ["gt_id", "gt_page_id", "gt_globe", "gt_primary", "gt_lat", "gt_lon", "gt_dim", "gt_type",
            "gt_name", "gt_country", "gt_region"]
    write_dump(dumps / "enwiki-20260901-geo_tags.sql.gz", "geo_tags", cols,
               ["(1,102,'earth',1,48.8584,2.2945,NULL,NULL,NULL,NULL,NULL)"])
    monkeypatch.setattr(sys, "argv", ["build_geo"])
    build_geo.main()
    db = sqlite3.connect(tiny_data / "geo.sqlite")
    assert db.execute("SELECT langs FROM places").fetchall() == [(0,)]
    assert dict(db.execute("SELECT * FROM meta"))["langlinks_used"] == "False"


def test_split_type():
    assert build_geo.split_type(b"city(334,000") == ("city", 334000)
    assert build_geo.split_type(b"City(1000)") == ("city", 1000)
    assert build_geo.split_type(b"river:us-oh") == ("river", None)
    assert build_geo.split_type(None) == (None, None)


# ── search ────────────────────────────────────────────────────────────────────

def test_nearby_ranks_and_leaves_out_the_centre(finder):
    centre = finder.location_of_page(102)
    r = finder.nearby(centre, 1.0)
    assert [h.title for h in r.hits] == ["Biology"]
    assert r.excluded == "Film" and r.in_range == 1
    h = r.hits[0]
    assert 0.4 < h.distance_km < 0.7 and h.compass == "NE"
    # outside 1 km, inside 10 km: both, most notable first
    r = finder.nearby(centre, 10)
    assert [h.title for h in r.hits][0] in {"Biology", "Cell (biology)"} and r.total == 2
    # the same centre as bare coordinates keeps the article itself
    r = finder.nearby(Location(48.8584, 2.2945, "x"), 1.0)
    assert {h.title for h in r.hits} == {"Film", "Biology"} and r.excluded is None


def test_sort_and_type_filter(finder):
    centre = Location(48.8584, 2.2945, "here")
    by_distance = finder.nearby(centre, 20, sort="distance")
    assert [h.title for h in by_distance.hits] == ["Film", "Biology", "Cell (biology)"]
    assert by_distance.hits[0].distance_km < by_distance.hits[1].distance_km < by_distance.hits[2].distance_km
    by_score = finder.nearby(centre, 20, sort="notable")
    assert by_score.hits[0].title == "Biology" and by_score.hits[0].score > by_score.hits[-1].score   # 3 links beat 1
    only = finder.nearby(centre, 20, types=["EDU"])
    assert [h.title for h in only.hits] == ["Cell (biology)"] and only.in_range == 3 and only.total == 1
    assert only.type_counts["landmark"] == 1 and only.type_counts["untyped"] == 1
    assert finder.nearby(centre, 20, types=["untyped"]).hits[0].title == "Biology"
    none = finder.nearby(centre, 20, types=["mountain"])
    assert none.hits == [] and none.in_range == 3
    assert len(finder.nearby(centre, 20, limit=2).hits) == 2 and finder.nearby(centre, 20, limit=2).total == 3
    with pytest.raises(ValueError):
        finder.nearby(centre, 5, sort="loudest")


def test_antimeridian_and_poles(finder):
    r = finder.nearby(Location(-18.0, 180.0, "dateline"), 12)
    assert {h.title for h in r.hits} == {"Mitochondrion"}          # 10.6 km west; Footloose is 15 km away
    r = finder.nearby(Location(-17.95, 180.0, "dateline"), 30, sort="distance")   # both within 30 km
    assert {h.title for h in r.hits} == {"Mitochondrion", "Footloose"}
    east = next(h for h in r.hits if h.title == "Footloose")
    west = next(h for h in r.hits if h.title == "Mitochondrion")
    assert east.compass in {"E", "NE", "SE"} and west.compass in {"W", "NW", "SW"}
    r = finder.nearby(Location(90.0, 0.0, "pole"), 30, sort="distance")
    assert {h.title for h in r.hits} == {"Kevin Bacon", "Bacon"}
    r = finder.nearby(Location(89.0, 100.0, "near the pole"), 200)
    assert r.total == 2
    r = finder.nearby(Location(-89.0, 0.0, "south"), 500)
    assert r.total == 0


def test_empty_results_and_nearest(finder):
    centre = Location(-40.0, -140.0, "ocean")
    r = finder.nearby(centre, 100)
    assert r.hits == [] and r.in_range == 0
    near = finder.nearest(centre, 100)
    assert near is not None and near.distance_km > 100
    assert finder.nearest(Location(0, 0, "x"), 5).distance_km > 5


def test_score_uses_pagerank_when_present(geo_data):
    assert NearbyFinder(geo_data).signal_name == "incoming links"
    from wikiexp import core
    n = len(core.Graph(geo_data / "graph"))
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    pr[2] = 0.5                                       # Film's idx
    (geo_data / "centrality").mkdir()
    np.save(geo_data / "centrality" / "pagerank.npy", pr)
    f = NearbyFinder(geo_data)
    assert f.signal_name == "PageRank"
    r = f.nearby(Location(48.8584, 2.2945, "here"), 20)
    assert r.hits[0].title == "Film" and r.hits[0].score > 80
    np.save(geo_data / "centrality" / "pagerank.npy", np.zeros(3, dtype=np.float32))   # wrong size: ignored
    assert NearbyFinder(geo_data).signal_name == "incoming links"


def test_score_prefers_prose_pagerank(geo_data):
    from wikiexp import core
    n = len(core.Graph(geo_data / "graph"))
    (geo_data / "centrality").mkdir()
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    pr[2] = 0.5                                       # Film: the template-link favourite
    np.save(geo_data / "centrality" / "pagerank.npy", pr)
    prose = np.full(n, 1.0 / n, dtype=np.float32)
    prose[3] = 0.5                                    # Biology: what the article text favours
    np.save(geo_data / "centrality" / "prose_pagerank.npy", prose)
    f = NearbyFinder(geo_data)
    assert f.signal_name == "prose PageRank"
    r = f.nearby(Location(48.8584, 2.2945, "here"), 20)
    assert r.hits[0].title == "Biology"
    np.save(geo_data / "centrality" / "prose_pagerank.npy", np.zeros(3, dtype=np.float32))   # wrong size: skipped
    f = NearbyFinder(geo_data)
    assert f.signal_name == "PageRank" and f.nearby(Location(48.8584, 2.2945, "here"), 20).hits[0].title == "Film"


# ── resolving locations ───────────────────────────────────────────────────────

def test_resolve_names_redirects_and_coordinates(finder):
    res = finder.resolve_location("film")
    assert res.location.page_id == 102 and not res.ambiguous and "Film (48.8584, 2.2945)" in res.note
    res = finder.resolve_location("Movie")                          # redirect to Film
    assert res.location.title == "Film" and res.note.startswith("Movie → Film (48.8584")
    res = finder.resolve_location("Biolgy")                          # typo
    assert res.location.title == "Biology" and "closest match" in res.note
    res = finder.resolve_location("48.8584, 2.2945")
    assert res.location.page_id is None and res.location.lat == 48.8584 and "coordinates" in res.note
    with pytest.raises(CoordinateError):
        finder.resolve_location("120, 5")
    assert finder.resolve_location("  ").location is None
    assert finder.resolve_location("Zzzzqqq").location is None


def test_resolve_falls_back_to_titles_with_coordinates(finder):
    res = finder.resolve_location("Isolated article")               # exists, has no coordinates, nothing similar
    assert res.location is None and "no coordinates" in res.note


def test_resolve_falls_back_when_the_title_has_no_coordinates(finder, monkeypatch):
    m = lambda title, pid, how="prefix": Match(title, pid, title, 5, how)
    monkeypatch.setattr(finder.titles, "search", lambda q, limit=10: [m("Jazz", 106, "exact"), m("Film", 102)])
    res = finder.resolve_location("Jazz")
    assert res.location.title == "Film" and not res.ambiguous and len(res.candidates) == 1
    assert "'Jazz' has no coordinates" in res.note and "Film (48.8584, 2.2945)" in res.note
    # a stand-in must start like the query: a mere substring match is not offered
    monkeypatch.setattr(finder.titles, "search", lambda q, limit=10: [m("Jazz", 106, "exact"), m("Film", 102, "contains")])
    assert finder.resolve_location("Jazz").location is None


def test_resolve_ambiguous_names_return_candidates(finder, monkeypatch):
    m = lambda title, pid, how="prefix": Match(title, pid, title, 5, how)
    monkeypatch.setattr(finder.titles, "search", lambda q, limit=10: [
        m("Springfield", 106, "exact"), m("Film", 102), m("Biology", 103), m("Elsewhere", 999)])
    res = finder.resolve_location("Springfield")
    assert res.ambiguous and res.location.title == "Film"
    assert [c.match.article for c in res.candidates] == ["Film", "Biology"]
    assert "disambiguation" in res.note and "1 other places" in res.note
    assert res.candidates[1].location.page_id == 103


def test_suggest_only_offers_places(finder):
    assert [m.article for m in finder.suggest("Bio")] == ["Biology", "Cell (biology)"]
    assert finder.suggest("Isolated") == []


def test_missing_places_index_is_explained(tiny_data):
    with pytest.raises(FileNotFoundError, match="build_geo"):
        NearbyFinder(tiny_data)


# ── command line ──────────────────────────────────────────────────────────────

def test_cli_prints_places_and_handles_empty(geo_data, capsys, monkeypatch):
    from experiments.nearby.__main__ import main
    monkeypatch.setattr(paths, "DATA", geo_data)
    main(["Film", "-r", "10km", "--sort", "distance"])
    out = capsys.readouterr().out
    assert "Using Film (48.8584, 2.2945)" in out and "Biology" in out
    assert "leaving out Film" in out
    main(["-40, -140", "-r", "100km"])
    out = capsys.readouterr().out
    assert "Nothing in range" in out and "nearest place is" in out
    main(["Film", "-r", "20km", "--type", "mountain"])
    assert "No place of type mountain" in capsys.readouterr().out
    for bad in (["120, 5"], ["Film", "-r", "far"], ["Zzzzqqq"]):
        with pytest.raises(SystemExit):
            main(bad)


# ── the screen ────────────────────────────────────────────────────────────────

def table_titles(screen):
    t = screen.query_one("#nb-results")
    return [str(t.get_row_at(i)[1]) for i in range(t.row_count)]


@pytest.mark.asyncio
async def test_screen_search_recentre_and_coordinates(geo_data, tmp_path):
    from experiments.nearby.screen import CoordMatch, NearbyScreen
    from textual.widgets import Button, Label, Select
    app = make_app(tmp_path, [], data_dir=geo_data)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = NearbyScreen(geo_data)
        app.push_screen(screen)
        assert await wait_for(pilot, lambda: screen.finder is not None)
        where = screen.where
        status = lambda: str(screen.query_one("#nb-status", Label).render())

        # a name: pick the suggestion, the search runs by itself
        screen.query_one("#nb-radius").value = "10 km"
        where.input.focus()
        await pilot.press(*"film")
        assert await wait_for(pilot, lambda: where.query_one("OptionList").display)
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: screen.result is not None)
        assert table_titles(screen) == ["Biology", "Cell (biology)"]      # 3 incoming links beat 1
        assert "Film is the place you searched from" in str(screen.query_one("#nb-note", Label).render())

        # sort toggle re-runs the same search
        await pilot.press("ctrl+o")
        assert await wait_for(pilot, lambda: screen.result.sort == "distance")
        assert table_titles(screen) == ["Biology", "Cell (biology)"]
        assert str(screen.query_one("#nb-sort", Button).label) == "Sort: nearest"

        # Enter on a result re-centres on it
        await pilot.pause(0.1)
        table = screen.query_one("#nb-results")
        table.focus()
        await pilot.press("down", "enter")        # second row: Cell (biology)
        assert await wait_for(pilot, lambda: screen.result.centre.title == "Cell (biology)")
        assert where.input.value == "Cell (biology)" and "Film" in table_titles(screen)
        await pilot.press("ctrl+b")                # and back again
        assert await wait_for(pilot, lambda: screen.result.centre.title == "Film")

        # typed coordinates wait for Enter, and bad ones say why
        where.input.focus()
        where.input.value = ""
        await pilot.press(*"-18, 179.95")
        await pilot.pause(0.2)
        assert isinstance(where.match, CoordMatch) and screen.result.centre.title == "Film"
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: not screen.result.centre.is_article)
        assert table_titles(screen) == ["Mitochondrion"]
        where.input.focus()
        where.input.value = ""
        await pilot.press(*"120, 5")
        await pilot.pause(0.2)
        assert where.match is None and "-bad" in where.input.classes

        # an empty result says so and names the nearest place; a type filter narrows
        where.input.focus()
        where.input.value = ""
        await pilot.press(*"-40, -140")
        await pilot.press("enter")
        screen.query_one("#nb-radius").value = "100 km"
        await pilot.press("ctrl+f")
        assert await wait_for(pilot, lambda: screen.result.in_range == 0)
        assert "nearest place is" in status() and table_titles(screen) == []
        screen.query_one("#nb-radius").value = "xyz"
        await pilot.press("ctrl+f")
        await pilot.pause(0.2)
        assert "distance" in status()


def test_geo_stage_is_registered(tmp_path):
    from pipeline import settings as S, stages as ST
    from experiments import EXPERIMENTS
    s = S.Settings({"data_dir": str(tmp_path / "data"), "dumps_dir": str(tmp_path / "dumps")})
    st = ST.BY_KEY["geo"]
    assert st.command(s)[-2:] == ["-m", "pipeline.build_geo"] and not st.is_done(s)
    assert "enwiki-20260901-geo_tags.sql.gz" in st.missing_inputs(s)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "geo.sqlite").write_bytes(b"")
    assert st.is_done(s)
    assert "geo" in next(e for e in EXPERIMENTS if e.key == "nearby").requires
