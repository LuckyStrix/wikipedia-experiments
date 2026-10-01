"""Nearby screen for the app: pick a place (or type coordinates), a radius, and see what's notable around it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Select

from tui.widgets import TitlePicker
from wikiexp.titles import Match

from .finder import Hit, Location, NearbyFinder, NearbyResult
from .parsing import format_coords, format_distance, format_radius, parse_coordinates, parse_radius

LIMIT = 100
SORT_LABELS = {"notable": "Sort: most notable", "distance": "Sort: nearest"}


@dataclass(frozen=True)
class CoordMatch:
    """What TitlePicker holds when the text is a coordinate pair rather than a title."""
    lat: float
    lon: float
    links: int = 0
    how: str = "coordinates"

    @property
    def article(self) -> str:
        return format_coords(self.lat, self.lon)

    @property
    def status(self) -> str:
        return f"✓ coordinates {self.article}  ·  press Enter to search"

    @property
    def location(self) -> Location:
        return Location(self.lat, self.lon, self.article)


def literal_coordinates(text: str) -> CoordMatch | None:
    """TitlePicker's `literal` hook: coordinates typed as such; ValueError if out of range."""
    parsed = parse_coordinates(text)
    return CoordMatch(*parsed) if parsed else None


class NearbyScreen(Screen):
    TITLE = "What's notable near here?"
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("ctrl+f", "search", "Search", priority=True),
        Binding("ctrl+o", "toggle_sort", "Sort", priority=True),
        Binding("ctrl+b", "back", "Previous centre", priority=True),
    ]
    DEFAULT_CSS = """
    NearbyScreen #nb-body { padding: 1 2; height: 1fr; }
    NearbyScreen .nb-intro { color: $text-muted; margin-bottom: 1; width: 100%; height: auto; }
    NearbyScreen .nb-row { height: auto; margin-bottom: 1; }
    NearbyScreen .nb-lbl { width: 8; color: $secondary; text-style: bold; }
    NearbyScreen #nb-radius { width: 16; }
    NearbyScreen #nb-radius-note { width: 22; margin-left: 1; color: $text-muted; }
    NearbyScreen #nb-radius-note.-bad { color: $warning; }
    NearbyScreen #nb-sort { margin-left: 2; }
    NearbyScreen #nb-type { width: 34; margin-left: 2; }
    NearbyScreen #nb-btns { height: auto; margin-bottom: 1; }
    NearbyScreen #nb-btns Button { margin-right: 2; }
    NearbyScreen #nb-status { height: auto; width: 100%; }
    NearbyScreen #nb-status.-bad { color: $warning; }
    NearbyScreen #nb-note { height: auto; width: 100%; color: $text-muted; margin-bottom: 1; }
    NearbyScreen #nb-results { height: 1fr; }
    NearbyScreen #nb-detail { height: auto; width: 100%; color: $text-muted; margin-top: 1; }
    """

    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = data_dir
        self.finder: NearbyFinder | None = None
        self.sort = "notable"
        self.history: list = []            # previous centres (Match or CoordMatch), most recent last
        self.current = None                # the centre the table shows
        self.hits: dict[str, Hit] = {}
        self.result: NearbyResult | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="nb-body"):
            yield Label("The most notable articles within a distance of any place. Type a place name and pick "
                        "from the suggestions, or type coordinates (48.8584, 2.2945 · 48.86N 2.29E · "
                        "48°51′29″N 2°17′40″E) and press Enter.", classes="nb-intro")
            with Horizontal(classes="nb-row"):
                yield Label("Where", classes="nb-lbl")
                yield TitlePicker("e.g. Eiffel Tower, or 48.8584, 2.2945", self._suggest, self._resolve,
                                  literal=literal_coordinates, id="nb-where")
            with Horizontal(classes="nb-row"):
                yield Label("Radius", classes="nb-lbl")
                yield Input("2 km", placeholder="500 m, 2 km, 3 mi", id="nb-radius", compact=True)
                yield Label("= 2 km", id="nb-radius-note")
                yield Button(SORT_LABELS["notable"], id="nb-sort", compact=True)
                yield Select([], prompt="Any type of place", id="nb-type", compact=True)
            with Horizontal(id="nb-btns"):
                yield Button("Search", id="nb-search", variant="primary", compact=True, disabled=True)
                yield Button("← Previous centre", id="nb-back", compact=True, disabled=True)
            yield Label("Loading the places index…", id="nb-status")
            yield Label("", id="nb-note")
            yield DataTable(id="nb-results", cursor_type="row", zebra_stripes=True)
            yield Label("Select a row and press Enter to search around that place instead.", id="nb-detail")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#nb-results", DataTable)
        table.add_columns("#", "Place", "Distance", "Dir", "Type", "Notability")
        self.where.disabled = True
        self._load()

    @property
    def where(self) -> TitlePicker:
        return self.query_one("#nb-where", TitlePicker)

    @work(thread=True)
    def _load(self) -> None:
        try:
            finder = NearbyFinder(self.data_dir)
        except Exception as e:
            self.app.call_from_thread(self._status, f"Can't load data: {e}", True)
            return
        self.app.call_from_thread(self._loaded, finder)

    def _loaded(self, finder: NearbyFinder) -> None:
        self.finder = finder
        self.where.disabled = False
        self.where.input.focus()
        self.query_one("#nb-search", Button).disabled = False
        select = self.query_one("#nb-type", Select)
        select.set_options([(f"{t} ({n:,})", t) for t, n in finder.type_options()] + [("untyped", "untyped")])
        self._status(f"Ready: {finder.places_count:,} places. Notability from {finder.signal_name}"
                     + (" and language editions." if finder.langs_max else "."))

    def _status(self, text: str, bad: bool = False) -> None:
        lbl = self.query_one("#nb-status", Label)
        lbl.update(text)
        lbl.set_class(bad, "-bad")

    # called from TitlePicker worker threads
    def _suggest(self, q: str):
        return self.finder.suggest(q, limit=8) if self.finder else []

    def _resolve(self, q: str):
        if not self.finder:
            return None
        m = self.finder.titles.exact(q)
        return m if m and self.finder.coordinates_of([m.page_id]) else None

    # ── inputs ───────────────────────────────────────────────────────────────
    @on(TitlePicker.Picked)
    def _picked(self, event: TitlePicker.Picked) -> None:
        match = self.where.match
        if match is None:
            return
        if isinstance(match, CoordMatch) and not event.submitted:
            return        # typed coordinates wait for Enter: they are valid after every keystroke
        self.action_search()

    @on(Input.Changed, "#nb-radius")
    def _radius_changed(self, event: Input.Changed) -> None:
        note = self.query_one("#nb-radius-note", Label)
        try:
            note.update("= " + format_radius(parse_radius(event.value)))
            note.remove_class("-bad")
        except ValueError as e:
            note.update(str(e))
            note.add_class("-bad")

    @on(Input.Submitted, "#nb-radius")
    def _radius_submitted(self) -> None:
        self.action_search()

    @on(Select.Changed, "#nb-type")
    def _type_changed(self) -> None:
        if self.current is not None:
            self.action_search(focus=False)

    @on(Button.Pressed, "#nb-search")
    def _search_pressed(self) -> None:
        self.action_search()

    @on(Button.Pressed, "#nb-sort")
    def action_toggle_sort(self) -> None:
        self.sort = "distance" if self.sort == "notable" else "notable"
        self.query_one("#nb-sort", Button).label = SORT_LABELS[self.sort]
        if self.current is not None:
            self.action_search(focus=False)

    # ── searching ────────────────────────────────────────────────────────────
    def action_search(self, focus: bool = True) -> None:
        if not self.finder:
            return
        match = self.where.match
        if match is None:
            self._status("Pick a place from the suggestions, or type coordinates, first.", True)
            return
        try:
            radius = parse_radius(self.query_one("#nb-radius", Input).value)
        except ValueError as e:
            self._status(str(e), True)
            return
        kind = self.query_one("#nb-type", Select).value
        types = [] if kind is Select.NULL else [str(kind)]
        if self.current is not None and self.current != match:
            self.history.append(self.current)
            self.query_one("#nb-back", Button).disabled = False
        self.current = match
        self._status("Searching…")
        self._query(match, radius, self.sort, types, focus)

    @work(thread=True, exclusive=True, group="nearby")
    def _query(self, match, radius: float, sort: str, types: list[str], focus: bool) -> None:
        loc = match.location if isinstance(match, CoordMatch) else self.finder.location_of_page(match.page_id)
        if loc is None:
            self.app.call_from_thread(self._status, f"{match.article} has no coordinates.", True)
            return
        result = self.finder.nearby(loc, radius, sort, LIMIT, types)
        nearest = self.finder.nearest(loc, radius) if result.in_range == 0 else None
        self.app.call_from_thread(self._show, result, nearest, focus)

    def _show(self, r: NearbyResult, nearest: Hit | None, focus: bool) -> None:
        self.result = r
        theme = self.app.current_theme
        table = self.query_one("#nb-results", DataTable)
        table.clear()
        self.hits = {}
        for i, h in enumerate(r.hits, 1):
            self.hits[str(h.page_id)] = h
            bar = "█" * max(1, round(h.score / 12.5))
            table.add_row(Text(str(i), style="dim"), Text(h.title, style="bold"),
                          Text(format_distance(h.distance_km), justify="right"), h.compass or "·",
                          Text(h.type or "·", style="dim"),
                          Text.assemble((f"{h.score:3.0f} ", ""), (bar, theme.primary)), key=str(h.page_id))
        c = r.centre
        where = f"{c.label} ({c.coords})" if c.is_article else c.coords
        head = f"Within {format_radius(r.radius_km)} of {where}: {r.in_range:,} place{'s' * (r.in_range != 1)}"
        if r.types:
            head += f", {r.total:,} of type {', '.join(r.types)}"
        self.query_one("#nb-note", Label).update(
            f"{r.excluded} is the place you searched from, so it is left out." if r.excluded else "")
        if not r.hits:
            if r.in_range == 0:
                self._status(head + ". Nothing in range."
                             + (f" The nearest place is {nearest.title}, {format_distance(nearest.distance_km)} "
                                f"{nearest.compass} of here." if nearest else ""), True)
            else:
                common = ", ".join(f"{t} ({n})" for t, n in r.type_counts.most_common(6))
                self._status(f"{head}. No place of that type here; there are: {common}.", True)
            self.query_one("#nb-detail", Label).update("")
            return
        by = "most notable first" if r.sort == "notable" else "nearest first"
        shown = f"showing {len(r.hits)}, {by}" if r.total > len(r.hits) else by
        self._status(f"{head} ({shown}). {r.seconds:.2f}s")
        if focus:
            table.focus()

    # ── results ──────────────────────────────────────────────────────────────
    @on(DataTable.RowHighlighted, "#nb-results")
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        h = self.hits.get(event.row_key.value) if event.row_key else None
        if h is None:
            return
        bits = [h.type or "untyped place", f"{format_distance(h.distance_km)} {h.compass}".strip(),
                format_coords(h.lat, h.lon)]
        if h.country:
            bits.append(h.country + (f"-{h.region}" if h.region else ""))
        bits.append(f"{h.links:,} incoming links")
        if self.finder and self.finder.langs_max:
            bits.append(f"{h.langs} languages")
        self.query_one("#nb-detail", Label).update(
            Text.assemble((h.title, "bold"), "  ·  " + "  ·  ".join(bits) + "   (Enter: search around it)"))

    @on(DataTable.RowSelected, "#nb-results")
    def _selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a result: make it the new centre."""
        h = self.hits.get(event.row_key.value) if event.row_key else None
        if h is not None:
            self.where.set_match(Match(h.title, h.page_id, h.title, h.links, "exact"))

    def action_back(self) -> None:
        if not self.history:
            return
        prev = self.history.pop()
        self.current = None     # so going back doesn't push the centre we are leaving onto the stack
        self.query_one("#nb-back", Button).disabled = not self.history
        self.where.set_match(prev)
        if isinstance(prev, CoordMatch):
            self.action_search()

    @on(Button.Pressed, "#nb-back")
    def _back_pressed(self) -> None:
        self.action_back()
