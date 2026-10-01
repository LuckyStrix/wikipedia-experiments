""""What's notable near here?": resolve a location, then rank the geotagged articles around it.

UI-free. ``NearbyFinder`` loads data/geo.sqlite (see pipeline/build_geo.py) and the title index.

Notability score (0-100), the same for every place so results are comparable across queries:

    score = 100 * (0.7 * log(1 + signal) / log(1 + max_signal) + 0.3 * log(1 + langs) / log(1 + max_langs))

- signal is the article's PageRank (data/centrality/pagerank.npy, scaled by the number of
  articles so the average article is ~1) when that file exists, otherwise its incoming link count.
  PageRank counts a link from an important article for more than one from an obscure one, so it
  separates the Eiffel Tower from the hundreds of streets and schools around it better than raw
  links do.
- langs is the number of other-language Wikipedias that have the article: a cheap, independent
  vote on whether the place matters beyond the English-speaking world.
- Logs, because both quantities span five orders of magnitude and a plain ratio would give almost
  everything a score of zero. The maxima are taken over all places, so a score doesn't change with
  the radius. Without langlinks data the whole score comes from the signal.
"""
from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wikiexp import paths
from wikiexp.titles import Match, TitleIndex

from .geometry import bearing_deg, bounding_boxes, compass, haversine_km
from .parsing import MAX_RADIUS_KM, format_coords, parse_coordinates

SIGNAL_WEIGHT = 0.7
LANGS_WEIGHT = 0.3
SORTS = ("notable", "distance")
PLACE_COLS = "p.id, p.page_id, p.idx, p.title, p.lat, p.lon, p.type, p.pop, p.dim, p.country, p.region, p.langs, p.links"


def notability(signal, langs, signal_max: float, langs_max: float):
    """Score 0-100 (scalars or numpy arrays); see the module docstring for the formula."""
    a = np.log1p(signal) / math.log1p(signal_max) if signal_max > 0 else 0.0
    if langs_max <= 0:
        return 100.0 * np.clip(a, 0, 1)
    b = np.log1p(langs) / math.log1p(langs_max)
    return 100.0 * np.clip(SIGNAL_WEIGHT * a + LANGS_WEIGHT * b, 0, 1)


@dataclass(frozen=True)
class Location:
    lat: float
    lon: float
    label: str                       # what to show: "48.8584, 2.2945" or an article title
    page_id: int | None = None       # set when the location is an article's own coordinates
    title: str | None = None

    @property
    def is_article(self) -> bool:
        return self.page_id is not None

    @property
    def coords(self) -> str:
        return format_coords(self.lat, self.lon)


@dataclass(frozen=True)
class Candidate:
    match: Match
    lat: float
    lon: float
    type: str | None

    @property
    def location(self) -> Location:
        return Location(self.lat, self.lon, self.match.article, self.match.page_id, self.match.article)


@dataclass
class Resolution:
    """Outcome of resolving what the user typed."""
    query: str
    location: Location | None        # what will be used (None: nothing usable)
    note: str                        # one sentence saying what was used, always safe to show
    candidates: list[Candidate] = field(default_factory=list)   # alternatives, best first
    ambiguous: bool = False          # the name matched no single place; candidates are the options


@dataclass(frozen=True)
class Hit:
    page_id: int
    title: str
    lat: float
    lon: float
    distance_km: float
    bearing: float
    compass: str
    type: str | None
    score: float
    langs: int
    links: int
    dim: int | None = None
    country: str | None = None
    region: str | None = None
    pop: int | None = None


@dataclass
class NearbyResult:
    centre: Location
    radius_km: float
    sort: str
    types: list[str]                   # the type filter that was applied (empty = none)
    total: int                         # places in range that match the filter
    in_range: int                      # places in range before the type filter
    hits: list[Hit]
    type_counts: Counter               # types of all places in range (None shown as "untyped")
    excluded: str | None = None        # the centre article, left out of the list
    seconds: float = 0.0


class NearbyFinder:
    def __init__(self, data_dir: Path | None = None):
        data_dir = Path(data_dir or paths.DATA)
        geo = data_dir / "geo.sqlite"
        if not geo.exists():
            raise FileNotFoundError(f"{geo} not found - build the places index first "
                                    "(python -m pipeline.build_geo, or the app's Build places index)")
        self.db = sqlite3.connect(f"file:{geo}?mode=ro", uri=True, check_same_thread=False)
        self.lock = threading.Lock()
        self.titles = TitleIndex(data_dir / "titles.sqlite")
        self.meta = dict(self.db.execute("SELECT key, value FROM meta"))
        self.pagerank = _load_pagerank(data_dir / "centrality" / "pagerank.npy")
        self.langs_max = self._q("SELECT max(langs) FROM places")[0][0] or 0
        self.places_count = int(self.meta.get("places", 0))
        if self.pagerank is not None:
            idx = np.array([r[0] for r in self._q("SELECT idx FROM places")], dtype=np.int64)
            if len(idx) and idx.max() >= len(self.pagerank):
                self.pagerank = None        # built from a different graph; don't trust it
            else:
                self._pr_scale = float(len(self.pagerank))
                self.signal_max = float(self.pagerank[idx].max() * self._pr_scale) if len(idx) else 1.0
        if self.pagerank is None:
            self.signal_max = float(self._q("SELECT max(links) FROM places")[0][0] or 1)
        self.signal_name = "PageRank" if self.pagerank is not None else "incoming links"

    def _q(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    # ── locations ────────────────────────────────────────────────────────────
    def coordinates_of(self, page_ids) -> dict[int, tuple[float, float, str | None]]:
        """{page_id: (lat, lon, type)} for the given articles that have coordinates."""
        ids = list(dict.fromkeys(int(p) for p in page_ids))
        out = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            for pid, lat, lon, typ in self._q(
                    f"SELECT page_id, lat, lon, type FROM places WHERE page_id IN ({marks})", chunk):
                out[pid] = (lat, lon, typ)
        return out

    def suggest(self, text: str, limit: int = 8) -> list[Match]:
        """Title suggestions for a location field: only articles that have coordinates."""
        matches = self.titles.search(text, limit=limit * 5)
        have = self.coordinates_of(m.page_id for m in matches)
        return [m for m in matches if m.page_id in have][:limit]

    def location_of_page(self, page_id: int) -> Location | None:
        rows = self._q("SELECT lat, lon, title FROM places WHERE page_id = ?", (int(page_id),))
        return Location(rows[0][0], rows[0][1], rows[0][2], int(page_id), rows[0][2]) if rows else None

    def resolve_location(self, text: str, max_candidates: int = 8) -> Resolution:
        """Coordinates if `text` is a coordinate pair, otherwise the article it names.

        Names go through the title index, so redirects work ("NYC" -> New York City) and typos are
        forgiven. If the best title match has no coordinates (a disambiguation page like
        "Springfield", or a topic like "Jazz") the next suggestions that do have them are used; with
        two or more of those the result is flagged ambiguous and they are all in `candidates`.
        Raises CoordinateError for coordinates that are out of range.
        """
        text = text.strip()
        if not text:
            return Resolution(text, None, "Type a place name or coordinates.")
        coords = parse_coordinates(text)
        if coords is not None:
            loc = Location(coords[0], coords[1], format_coords(*coords))
            return Resolution(text, loc, f"Using coordinates {loc.coords}.")

        matches = self.titles.search(text, limit=60)
        if not matches:
            return Resolution(text, None, f"No article matches '{text}'.")
        have = self.coordinates_of(m.page_id for m in matches)
        first = matches[0]
        # when the best title has no coordinates, only titles that start like the query are
        # plausible stand-ins ("Springfield" -> "Springfield, Illinois", not "Hull Jazz Festival" for "Jazz")
        near = ("exact", "prefix") if first.how in ("exact", "prefix") and first.page_id not in have else None
        usable = [Candidate(m, *have[m.page_id]) for m in matches
                  if m.page_id in have and (near is None or m.how in near or m is first)]
        if not usable:
            return Resolution(text, None, f"'{first.article}' has no coordinates, and neither do similar titles.")
        best = usable[0]
        loc, m = best.location, best.match
        where = f"{m.article} ({format_coords(best.lat, best.lon)})"
        if best.match is first:
            if first.how == "exact":
                note = f"{m.title} → {where}" if m.title != m.article else f"Using {where}"
            else:
                note = f"No article is titled '{text}'; closest match with coordinates: {where}."
            return Resolution(text, loc, note, usable[:max_candidates])
        ambiguous = len(usable) >= 2
        why = f"'{first.article}' has no coordinates" + (" (it may be a disambiguation page)" if first.how == "exact" else "")
        note = f"{why}; using {where}" + (f"; {len(usable) - 1} other places match." if ambiguous else ".")
        return Resolution(text, loc, note, usable[:max_candidates], ambiguous)

    # ── search ───────────────────────────────────────────────────────────────
    def type_options(self, limit: int = 40) -> list[tuple[str, int]]:
        """The most common place types overall, [(type, count)], for a filter menu."""
        return [(t, n) for t, n in self._q(
            "SELECT type, count(*) FROM places WHERE type IS NOT NULL GROUP BY type "
            "ORDER BY count(*) DESC LIMIT ?", (limit,))]

    def _signal(self, idx: np.ndarray, links: np.ndarray) -> np.ndarray:
        if self.pagerank is not None:
            return np.asarray(self.pagerank[idx], dtype=np.float64) * self._pr_scale
        return links.astype(np.float64)

    def nearby(self, centre: Location, radius_km: float, sort: str = "notable", limit: int = 30,
               types: list[str] | tuple[str, ...] | None = None, exclude_page_id: int | None = None) -> NearbyResult:
        """Places within radius_km of centre: R*Tree box query, then an exact great-circle filter."""
        t0 = time.time()
        if sort not in SORTS:
            raise ValueError(f"sort must be one of {', '.join(SORTS)}")
        if exclude_page_id is None and centre.page_id is not None:
            exclude_page_id = centre.page_id
        rows = []
        for lo_lat, hi_lat, lo_lon, hi_lon in bounding_boxes(centre.lat, centre.lon, radius_km):
            rows += self._q(f"SELECT {PLACE_COLS} FROM places_rt r JOIN places p ON p.id = r.id "
                            "WHERE r.min_lat >= ? AND r.max_lat <= ? AND r.min_lon >= ? AND r.max_lon <= ?",
                            (lo_lat, hi_lat, lo_lon, hi_lon))
        want = [t.strip().lower() for t in (types or []) if t.strip()]
        excluded = None
        if not rows:
            return NearbyResult(centre, radius_km, sort, want, 0, 0, [], Counter(), None, time.time() - t0)

        n = len(rows)
        page_id = np.fromiter((r[1] for r in rows), dtype=np.int64, count=n)
        lat = np.fromiter((r[4] for r in rows), dtype=np.float64, count=n)
        lon = np.fromiter((r[5] for r in rows), dtype=np.float64, count=n)
        dist = haversine_km(centre.lat, centre.lon, lat, lon)
        keep = dist <= radius_km
        if exclude_page_id is not None:
            gone = keep & (page_id == exclude_page_id)
            if gone.any():
                excluded = rows[int(np.argmax(gone))][3]
                keep &= ~gone
        sel = np.flatnonzero(keep)
        kinds = [rows[i][6] for i in sel]
        type_counts = Counter(k or "untyped" for k in kinds)
        in_range = len(sel)
        if want:
            ok = np.array([(k or "untyped") in want for k in kinds], dtype=bool)
            sel = sel[ok]
        if len(sel) == 0:
            return NearbyResult(centre, radius_km, sort, want, 0, in_range, [], type_counts, excluded,
                                time.time() - t0)

        idx = np.fromiter((rows[i][2] for i in sel), dtype=np.int64, count=len(sel))
        links = np.fromiter((rows[i][12] for i in sel), dtype=np.int64, count=len(sel))
        langs = np.fromiter((rows[i][11] for i in sel), dtype=np.int64, count=len(sel))
        score = notability(self._signal(idx, links), langs, self.signal_max, self.langs_max)
        d = dist[sel]
        order = np.lexsort((d, -score)) if sort == "notable" else np.lexsort((-score, d))
        order = order[:limit]
        brg = bearing_deg(centre.lat, centre.lon, lat[sel][order], lon[sel][order])
        hits = []
        for k, o in enumerate(order):
            r = rows[sel[o]]
            hits.append(Hit(page_id=r[1], title=r[3], lat=r[4], lon=r[5], distance_km=float(d[o]),
                            bearing=float(brg[k]), compass=compass(float(brg[k])) if d[o] >= 0.001 else "", type=r[6],
                            score=float(score[o]), langs=r[11], links=r[12], dim=r[8], country=r[9],
                            region=r[10], pop=r[7]))
        return NearbyResult(centre, radius_km, sort, want, len(sel), in_range, hits, type_counts, excluded,
                            time.time() - t0)


    def nearest(self, centre: Location, after_km: float = 0.0) -> Hit | None:
        """The closest place beyond `after_km` (for "nothing in range": how far must I look?)."""
        r = max(after_km * 4, 10.0)
        while True:
            found = self.nearby(centre, min(r, MAX_RADIUS_KM), "distance", 1)
            if found.hits or r >= MAX_RADIUS_KM:
                return found.hits[0] if found.hits else None
            r *= 4


def _load_pagerank(path: Path) -> np.ndarray | None:
    """data/centrality/pagerank.npy (float32, indexed by graph idx) if another stage has built it."""
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    return arr if arr.ndim == 1 else None
