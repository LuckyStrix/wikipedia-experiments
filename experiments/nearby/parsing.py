"""Turn what a person types into numbers: coordinates ("48°51′29″N 2°17′40″E") and radii ("2 km").

UI-free. Both parsers are forgiving about spacing, punctuation and unicode look-alikes (′ ″ º −).
"""
from __future__ import annotations

import re

MAX_RADIUS_KM = 20_000.0

UNITS_KM = {
    "m": 0.001, "meter": 0.001, "meters": 0.001, "metre": 0.001, "metres": 0.001,
    "km": 1.0, "kilometer": 1.0, "kilometers": 1.0, "kilometre": 1.0, "kilometres": 1.0, "k": 1.0,
    "mi": 1.609344, "mile": 1.609344, "miles": 1.609344,
    "ft": 0.0003048, "foot": 0.0003048, "feet": 0.0003048,
    "yd": 0.0009144, "yard": 0.0009144, "yards": 0.0009144,
    "nmi": 1.852, "nm": 1.852,
}


class CoordinateError(ValueError):
    """The text looks like coordinates but they aren't valid (e.g. latitude 100)."""


# ── coordinates ───────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = (text.replace("′", "'").replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
            .replace("″", '"').replace("”", '"').replace("“", '"').replace("''", '"')
            .replace("º", "°").replace("˚", "°").replace("−", "-").replace("–", "-"))
    return re.sub(r"\s+", " ", text).strip()


def _number(n: str) -> str:
    # degrees, then optionally ° minutes ' seconds " (each part only when the previous mark is there)
    return (rf"(?P<d{n}>\d+(?:\.\d+)?)\s*(?:°\s*(?:(?P<m{n}>\d+(?:\.\d+)?)\s*'\s*"
            rf"(?:(?P<s{n}>\d+(?:\.\d+)?)\s*\"?)?)?)?")


def _suffix_comp(n: str) -> str:       # -48.86 N   |   48°51'29"N
    return rf"(?P<sign{n}>[-+])?\s*{_number(n)}\s*(?P<h{n}>[NSEWnsew])?"


def _prefix_comp(n: str) -> str:       # N 48.86   |   N48°51'29"
    return rf"(?P<h{n}>[NSEWnsew])\s*{_number(n)}"


SEP = r"(?:\s*[,;/]\s*|\s+)"
SUFFIX_FORM = re.compile(rf"^{_suffix_comp('1')}{SEP}{_suffix_comp('2')}$")
PREFIX_FORM = re.compile(rf"^{_prefix_comp('1')}{SEP}{_prefix_comp('2')}$")


def _value(m: re.Match, n: str) -> tuple[float, str | None]:
    deg = float(m[f"d{n}"])
    mins, secs = m[f"m{n}"], m[f"s{n}"]
    if mins is not None:
        if float(mins) >= 60:
            raise CoordinateError(f"Minutes must be below 60 (got {mins}).")
        deg += float(mins) / 60
    if secs is not None:
        if float(secs) >= 60:
            raise CoordinateError(f"Seconds must be below 60 (got {secs}).")
        deg += float(secs) / 3600
    letter = (m[f"h{n}"] or "").upper() or None
    sign = m.groupdict().get(f"sign{n}")
    if sign == "-":
        if letter:
            raise CoordinateError("Use either a minus sign or S/W, not both.")
        deg = -deg
    return deg, letter


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """(lat, lon) in degrees if `text` is a coordinate pair, None if it isn't (so it may be a place
    name), and CoordinateError if it is one but out of range or inconsistent.

    Accepted: "48.8584, 2.2945", "-33.86 151.21", "48.86N 2.29E", "N 48.86, E 2.29",
    "48.86° N, 2.29° E", 48°51′29″N 2°17′40″E (also with plain ' and "), "2.29E, 48.86N".
    Without hemisphere letters the order is latitude, longitude.
    """
    s = _normalise(text)
    m = PREFIX_FORM.match(s) or SUFFIX_FORM.match(s)
    if not m:
        return None
    (a, la), (b, lb) = _value(m, "1"), _value(m, "2")
    roles = {"N": "lat", "S": "lat", "E": "lon", "W": "lon"}
    r1, r2 = roles.get(la), roles.get(lb)
    if r1 and r2 and r1 == r2:
        raise CoordinateError("Give one latitude (N/S) and one longitude (E/W).")
    if r1 == "lon" or r2 == "lat":
        a, b, la, lb = b, a, lb, la     # longitude was typed first
    lat, lon = (-a if la == "S" else a), (-b if lb == "W" else b)
    if abs(lat) > 90:
        raise CoordinateError(f"Latitude {lat:g} is outside -90..90 (is the order latitude, longitude?).")
    if abs(lon) > 180:
        raise CoordinateError(f"Longitude {lon:g} is outside -180..180.")
    return lat, lon


def format_coords(lat: float, lon: float) -> str:
    return f"{lat:.4f}, {lon:.4f}"


# ── radius ────────────────────────────────────────────────────────────────────

_RADIUS = re.compile(r"^\s*(\d[\d,]*(?:\.\d+)?|\.\d+)\s*([a-zA-Z]*)\.?\s*$")


def parse_radius(text: str) -> float:
    """Radius in km from "500 m", "2km", "3 mi", "1.5 kilometres"; a plain number is km.
    ValueError (with a message fit to show the user) if it isn't a usable distance."""
    m = _RADIUS.match(text or "")
    if not m:
        raise ValueError("Give a distance like 500 m, 2 km or 3 mi.")
    num, unit = m.groups()
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", num):
        num = num.replace(",", "")
    elif "," in num:
        num = num.replace(",", ".")    # "1,5 km"
    try:
        value = float(num)
    except ValueError:
        raise ValueError("Give a distance like 500 m, 2 km or 3 mi.") from None
    factor = UNITS_KM.get(unit.lower()) if unit else 1.0
    if factor is None:
        raise ValueError(f"Unknown unit '{unit}'. Use m, km, mi, ft or yd.")
    km = value * factor
    if km <= 0:
        raise ValueError("The radius must be more than zero.")
    if km > MAX_RADIUS_KM:
        raise ValueError(f"The radius can be at most {MAX_RADIUS_KM:,.0f} km (half way round the Earth).")
    return km


def format_distance(km: float) -> str:
    """Short human distance: "350 m", "2.35 km", "48.2 km", "1,240 km"."""
    if km < 1:
        return f"{round(km * 1000):,} m" if km >= 0.0005 else "0 m"
    if km < 10:
        return f"{km:.2f} km"
    if km < 100:
        return f"{km:.1f} km"
    return f"{round(km):,} km"


def format_radius(km: float) -> str:
    """A radius for display: "500 m", "2 km", "1.5 km"."""
    if km < 1:
        return f"{km * 1000:g} m"
    return f"{km:.2f}".rstrip("0").rstrip(".") + " km"
