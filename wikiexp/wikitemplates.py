"""Rendering of the common inline MediaWiki templates as the plain text a reader sees.

``wikitext.clean`` used to drop every template it did not know, so ``{{convert|300|m}}`` or
``{{chem|UBr|3}}`` left a hole in the sentence ("a listed height of," / "include:,, and."). This
module turns the ones that carry prose into text; everything else (citations, infoboxes, hatnotes,
maintenance tags, pronunciation) is still dropped, which is what a reader of the article sees too.

    render("convert", ["300", "m"], {})        # -> "300 m (980 ft)"
    render("birth date", ["1950", "3", "7"], {})   # -> "March 7, 1950"
    render("sfn", ["Smith", "2000"], {})       # -> None (unknown/dropped)

``render`` returns None for a template it doesn't know and "" for one it knows should vanish. It
never raises: a template with odd arguments renders as "" (the caller also guards).
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
_MONTH_NUM = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MONTH_NUM.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)})

Handler = Callable[[list, dict], str]
_HANDLERS: dict[str, Handler] = {}


def template(*names: str):
    """Register a handler for template names (lowercase, single spaces)."""
    def deco(fn: Handler) -> Handler:
        for n in names:
            _HANDLERS[n] = fn
        return fn
    return deco


def _arg(pos, i, default=""):
    return pos[i] if len(pos) > i else default


def _first(pos, named, *keys):
    return pos[0] if pos else next((named[k] for k in keys if named.get(k)), "")


def _is_int(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", s.strip()))


# ── conversions ──────────────────────────────────────────────────────────────

# unit name -> (dimension, size in the dimension's base unit, how it is written, canonical name)
_UNITS: dict[str, tuple[str, float, str, str]] = {}


def _unit(names: str, dim: str, factor: float, shown: str) -> None:
    canon = names.split()[0]
    for n in names.split():
        _UNITS[n] = (dim, factor, shown, canon)


_unit("mm", "len", 0.001, "mm"); _unit("cm", "len", 0.01, "cm"); _unit("m metre meter metres meters", "len", 1, "m")
_unit("km kilometre kilometer kilometres kilometers", "len", 1000, "km")
_unit("in inch inches", "len", 0.0254, "in"); _unit("ft foot feet", "len", 0.3048, "ft")
_unit("yd yard yards", "len", 0.9144, "yd"); _unit("mi mile miles", "len", 1609.344, "mi")
_unit("nmi", "len", 1852, "nmi")
_unit("m2 sqm", "area", 1, "m²"); _unit("km2 sqkm", "area", 1e6, "km²"); _unit("cm2", "area", 1e-4, "cm²")
_unit("ha hectare hectares", "area", 1e4, "ha"); _unit("acre acres", "area", 4046.8564224, "acres")
_unit("sqmi mi2 sqmile", "area", 2589988.110336, "sq mi"); _unit("sqft ft2", "area", 0.09290304, "sq ft")
_unit("sqyd yd2", "area", 0.83612736, "sq yd")
_unit("g", "mass", 0.001, "g"); _unit("kg kilogram kilograms", "mass", 1, "kg")
_unit("t tonne tonnes", "mass", 1000, "t"); _unit("lb lbs pound pounds", "mass", 0.45359237, "lb")
_unit("oz ounce ounces", "mass", 0.028349523125, "oz"); _unit("st stone", "mass", 6.35029318, "st")
_unit("LT", "mass", 1016.0469088, "long tons"); _unit("ST", "mass", 907.18474, "short tons")
_unit("l L litre litres liter liters", "vol", 1, "L"); _unit("ml mL", "vol", 0.001, "mL")
_unit("m3", "vol", 1000, "m³"); _unit("cuft ft3", "vol", 28.316846592, "cu ft")
_unit("usgal galUS", "vol", 3.785411784, "US gal"); _unit("impgal galimp", "vol", 4.54609, "imp gal")
_unit("km/h kph km/hr", "speed", 1 / 3.6, "km/h"); _unit("mph mi/h", "speed", 0.44704, "mph")
_unit("m/s", "speed", 1, "m/s"); _unit("kn knot knots", "speed", 1852 / 3600, "kn")
_unit("C °C degC", "temp", 1, "°C"); _unit("F °F degF", "temp", 1, "°F"); _unit("K", "temp", 1, "K")

# what {{convert|x|unit}} converts to when the template names no target
_DEFAULT_TARGET = {"m": "ft", "cm": "in", "mm": "in", "km": "mi", "mi": "km", "ft": "m", "in": "cm",
                   "yd": "m", "nmi": "km", "m2": "sqft", "km2": "sqmi", "ha": "acre", "acre": "ha",
                   "sqmi": "km2", "sqft": "m2", "kg": "lb", "lb": "kg", "g": "oz", "oz": "g", "st": "kg",
                   "km/h": "mph", "mph": "km/h", "m/s": "mph", "kn": "km/h", "C": "F", "F": "C",
                   "l": "usgal", "cm2": "sqft"}
_NUM = re.compile(r"[-−+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_RANGE_WORDS = {"to": " to ", "and": " and ", "or": " or ", "by": " by ", "x": " × ", "×": " × ",
                "-": "–", "–": "–", "+": " + ", "±": " ± ", "to about": " to about ",
                "and about": " and about ", "+/-": " ± "}
_SCALE = {"3": "thousand", "6": "million", "9": "billion", "12": "trillion"}


def _num(s: str) -> float | None:
    s = s.strip()
    if not _NUM.fullmatch(s):
        return None
    try:
        return float(s.replace(",", "").replace("−", "-"))
    except ValueError:
        return None


def _sig_figs(s: str) -> int:
    """Significant figures the writer gave: '300' -> 1, '1,000' -> 1, '1.50' -> 3, '250' -> 2."""
    s = re.sub(r"[eE].*", "", s.replace(",", "").lstrip("-−+"))
    sig = s.replace(".", "").lstrip("0") if "." in s else s.lstrip("0").rstrip("0")
    return max(1, len(sig))


def _fmt(x: float, sf: int, places: int | None = None) -> str:
    """x to `sf` significant figures (or `places` decimals when the template asks) as plain text
    with thousands commas: 984.25 -> '980'."""
    if x == 0:
        return "0"
    if places is None:
        places = sf - 1 - math.floor(math.log10(abs(x)))
    x = round(x, places)       # a negative `places` rounds to tens, hundreds, ...
    return f"{x:,.{places}f}" if places > 0 else f"{int(x):,}"


def _to_base(value: float, unit: str) -> float:
    dim, factor, shown, _ = _UNITS[unit]
    if dim != "temp":
        return value * factor
    return {"°C": value, "°F": (value - 32) * 5 / 9, "K": value - 273.15}[shown]


def _from_base(value: float, unit: str) -> float:
    dim, factor, shown, _ = _UNITS[unit]
    if dim != "temp":
        return value / factor
    return {"°C": value, "°F": value * 9 / 5 + 32, "K": value + 273.15}[shown]


def _split_scale(unit: str) -> tuple[str, str]:
    """'e6km2' -> ('km2', 'million ')."""
    m = re.fullmatch(r"e(\d+)(.+)", unit)
    return (m.group(2), _SCALE[m.group(1)] + " ") if m and m.group(1) in _SCALE else (unit, "")


def _words(values: list[str], numbers: list[str] | None = None) -> str:
    """The values with their range words ("10 to 20", "5–7"); `numbers` replaces the numbers."""
    it = iter(numbers) if numbers is not None else None
    out = []
    for v in values:
        out.append(_RANGE_WORDS[v] if v in _RANGE_WORDS and _num(v) is None else (next(it) if it else v))
    return "".join(out)


def _unit_text(unit: str, values: list[str]) -> str:
    shown = _UNITS[unit][2]
    return "acre" if shown == "acres" and values == ["1"] else shown


@template("convert", "cvt")
def convert(pos, named):
    """{{convert|300|m}} -> "300 m (980 ft)". Handles ranges ("10 to 20 km"), compound input
    ("5 ft 6 in"), a chosen target unit, `e6` scaled units and disp=out/flip. A unit it doesn't
    know gives the value and unit as written (never nothing)."""
    toks = []
    for t in pos:
        parts = t.split()
        if len(parts) > 1 and all(_num(x) is not None or x in _RANGE_WORDS for x in parts):
            toks.extend(parts)                     # "31 and 33" in one argument
        elif t != "":
            toks.append(t)
    values, i = [], 0                              # numbers and range words, as written
    while i < len(toks) and (_num(toks[i]) is not None or (values and toks[i] in _RANGE_WORDS)):
        values.append(toks[i])
        i += 1
    if not values:
        return " ".join(toks[:2])                  # "5/8 hp", "4+1/2 mi": as written
    if i >= len(toks):
        return _words(values)
    unit, scale = _split_scale(toks[i])
    i += 1
    if unit not in _UNITS:
        return f"{_words(values)} {scale}{unit}"
    dim = _UNITS[unit][0]
    written = f"{_words(values)} {scale}{_unit_text(unit, values)}"
    extra = None                                   # the second half of "5 ft 6 in"
    if i + 1 < len(toks) and _num(toks[i]) is not None and _split_scale(toks[i + 1])[0] in _UNITS \
            and _UNITS[toks[i + 1]][0] == dim:
        extra = (toks[i], toks[i + 1])
        written += f" {toks[i]} {_UNITS[toks[i + 1]][2]}"
        i += 2
    target = _split_scale(toks[i])[0] if i < len(toks) else ""
    if target == "ftin" or (target in _UNITS and _UNITS[target][0] == dim):
        i += 1
    else:
        target = _DEFAULT_TARGET.get(_UNITS[unit][3], "")
    if not target:
        return written
    places = next((int(t) for t in toks[i:] if re.fullmatch(r"-?\d", t)), None)    # {{convert|1|km|mi|0}}
    base = [_to_base(_num(v), unit) for v in values if _num(v) is not None]
    if extra:
        base = [base[0] + _to_base(_num(extra[0]), extra[1])]
    if target == "ftin":
        feet, inches = divmod(round(base[0] / 0.0254), 12)
        out = f"{int(feet)} ft {int(inches)} in"
    else:
        sf = max([2] + [_sig_figs(v) for v in values if _num(v) is not None])
        out_nums = [_fmt(_from_base(b, target), sf, places) for b in base]
        out = f"{_words(values, out_nums)} {scale}{_unit_text(target, out_nums)}"
    disp = (named.get("disp") or named.get("order") or "").lower()
    if disp in ("out", "output only"):
        return out
    if disp == "flip":
        return f"{out} ({written})"
    return f"{written} ({out})"


# ── numbers, fractions, scientific notation ──────────────────────────────────

@template("frac", "sfrac", "fraction")
def frac(pos, named):
    pos = [p for p in pos if p != ""]
    if len(pos) >= 3:
        return f"{pos[0]} {pos[1]}/{pos[2]}"
    if len(pos) == 2:
        return f"{pos[0]}/{pos[1]}"
    return f"1/{pos[0]}" if pos else ""


@template("ordinal", "ord")
def ordinal(pos, named):
    n = _arg(pos, 0)
    if not _is_int(n):
        return n
    k = int(n)
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


@template("e")
def e_exp(pos, named):
    return f"×10^{pos[0]}" if pos else ""


@template("val")
def val(pos, named):
    number = _arg(pos, 0) or named.get("1", "")
    if _is_int(number) and len(number) > 4:
        number = f"{int(number):,}"
    exp = named.get("e", "")
    unit = named.get("u", "") or named.get("ul", "") or named.get("up", "")
    out = number + (f"×10^{exp}" if exp else "")
    return f"{out} {unit}".strip()


_SAFE_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return -_eval(node.operand) if isinstance(node.op, ast.USub) else _eval(node.operand)
    raise ValueError("unsupported")


def parser_function(name: str, arg: str) -> str | None:
    """`{{formatnum:1234567}}` and `{{#expr:1+2}}`: the only magic words that show up in prose."""
    name = name.lower()
    arg = arg.strip()
    if name == "formatnum":
        return f"{int(arg):,}" if re.fullmatch(r"\d{4,}", arg) else arg
    if name == "#expr" and re.fullmatch(r"[\d.+\-*/() ]+", arg):
        try:
            v = _eval(ast.parse(arg, mode="eval"))
        except (ValueError, SyntaxError, ZeroDivisionError):
            return ""
        return f"{int(v):,}" if float(v).is_integer() else f"{v:,.2f}".rstrip("0").rstrip(".")
    return None


# ── dates ────────────────────────────────────────────────────────────────────

def _month(s: str) -> int | None:
    if _is_int(s) and 1 <= int(s) <= 12:
        return int(s)
    return _MONTH_NUM.get(s.strip().lower().rstrip("."))


def format_date(y: str, m: int | None, d: str | None, dayfirst: bool) -> str:
    if m is None:
        return y
    if d and _is_int(d):
        return f"{int(d)} {MONTHS[m - 1]} {y}" if dayfirst else f"{MONTHS[m - 1]} {int(d)}, {y}"
    return f"{MONTHS[m - 1]} {y}"


def _dayfirst(named) -> bool:
    return named.get("df", "").lower() in ("y", "yes", "true", "1") or named.get("mf", "").lower() == "n"


def _ymd(pos) -> tuple[str, int | None, str | None]:
    pos = [p for p in pos if p != ""]
    y = pos[0] if pos else ""
    m = _month(pos[1]) if len(pos) > 1 else None
    d = pos[2] if len(pos) > 2 else None
    return y, m, d


def _date(pos, named):
    y, m, d = _ymd(pos)
    return format_date(y, m, d, _dayfirst(named)) if y else ""


for _n in ("birth date and age", "birth date", "death date and age", "death date", "bda", "dob", "birth-date",
           "start date", "end date", "start date and age", "end date and age", "start-date", "end-date",
           "birth date and given age", "death date and given age", "release date", "film date"):
    _HANDLERS[_n] = _date
_HANDLERS["birth year and age"] = _HANDLERS["death year and age"] = lambda p, n: _arg(p, 0)


@template("birth date and age2", "death date and age2")
def date_age2(pos, named):
    """{{birth date and age2|ref y|ref m|ref d|y|m|d}}: the date is the last three arguments."""
    return _date(pos[3:6], named)


@template("date")
def date_t(pos, named):
    """{{date|2023-05-09|mdy}}: a date string, reformatted when it is ISO."""
    s = _arg(pos, 0)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if not m:
        return s
    fmt = (_arg(pos, 1) or "").lower()
    return format_date(m.group(1), int(m.group(2)), m.group(3), fmt.startswith("dmy"))


@template("as of")
def as_of(pos, named):
    """{{as of|2020|5|3}} -> "As of 3 May 2020"; lc=y lower-cases, since=y says "Since"."""
    if named.get("alt"):
        return named["alt"]
    if not pos:
        return ""
    y, m, d = _ymd(pos)
    when = format_date(y, m, d, named.get("df", "").lower() != "us")
    word = "since" if named.get("since") == "y" else "as of"
    if named.get("lc") != "y":
        word = word.capitalize()
    return f"{word} {when}{named.get('post', '')}"


@template("oldstyledate")
def old_style_date(pos, named):
    return f"{_arg(pos, 0)} [O.S. {_arg(pos, 2)}] {_arg(pos, 1)}" if len(pos) > 2 else " ".join(pos[:2])


@template("circa", "c.", "ca.", "circa2")
def circa(pos, named):
    return ("c. " + "–".join(p for p in pos[:2] if p)) if pos else "c."


@template("fl")
def floruit(pos, named):
    return "fl. " + _arg(pos, 0)


@template("reign")
def reign(pos, named):
    era = f" {named['era']}" if named.get("era") else ""
    return "r. " + " – ".join(p for p in pos[:2] if p) + era


@template("bce")
def bce(pos, named):
    return f"{_arg(pos, 0)} BCE".strip()


@template("bc")
def bc(pos, named):
    return f"{_arg(pos, 0)} BC".strip()


@template("ce")
def ce(pos, named):
    return f"{_arg(pos, 0)} CE".strip()


@template("ad")
def ad(pos, named):
    return f"AD {_arg(pos, 0)}".strip()


@template("marriage")
def marriage(pos, named):
    """{{marriage|[[Spouse]]|1914|1915|end=div}} -> "Spouse (m. 1914; div. 1915)"."""
    name, start, end = _arg(pos, 0), _arg(pos, 1), _arg(pos, 2)
    reason = {"d": "died", "died": "died", "s": "sep.", "sep": "sep.", "separated": "sep.", "div": "div.",
              "divorced": "div.", "a": "annulled", "annulled": "annulled"}.get(named.get("end", "").lower(), "div.")
    if reason == "died":
        reason = "d."
    bits = [f"m. {start}"] if start else []
    if end:
        bits.append(f"{reason} {end}")
    return f"{name} ({'; '.join(bits)})" if bits else name


@template("nee", "née")
def nee(pos, named):
    return f"née {_arg(pos, 0)}".strip()


# ── languages and names ──────────────────────────────────────────────────────

@template("lang", "langx", "lang-x", "native name", "native phrase", "langnf")
def lang(pos, named):
    return _arg(pos, 1) or named.get("text", "")


@template("transl", "transliteration", "translit", "tlit", "transl-ru")
def transl(pos, named):
    return pos[-1] if pos else ""


@template("nihongo", "nihongo foot", "nihongo2", "nihongo3", "nihongo-s")
def nihongo(pos, named):
    """{{nihongo|English|kanji|romaji}} -> "English (kanji, romaji)", as Wikipedia shows it."""
    english, kanji, romaji = _arg(pos, 0), _arg(pos, 1), _arg(pos, 2)
    extra = ", ".join(p for p in (kanji, romaji) if p)
    out = english if english else extra
    if english and extra:
        out = f"{english} ({extra})"
    return out + named.get("post", "")


@template("zh", "chinese", "zho")
def zh(pos, named):
    """{{zh|c=北京|p=Běijīng}} -> "Chinese: 北京; pinyin: Běijīng" (labels=no drops the labels)."""
    labels = named.get("labels", "").lower() != "no"
    parts = []
    hanzi = named.get("c") or named.get("s") or named.get("t") or _arg(pos, 0)
    if hanzi:
        parts.append(f"Chinese: {hanzi}" if labels else hanzi)
    if named.get("p"):
        parts.append(f"pinyin: {named['p']}" if labels else named["p"])
    return "; ".join(parts)


@template("gloss", "translation", "tr")
def gloss(pos, named):
    t = _arg(pos, 0) or named.get("1", "")
    return f"‘{t}’" if t else ""


@template("lit", "literal", "literal translation")
def lit(pos, named):
    return f"lit. {_arg(pos, 0)}".strip()


@template("script")
def script(pos, named):
    return _arg(pos, 1)


@template("wikt-lang")
def wikt_lang(pos, named):
    return _arg(pos, 1)


@template("sortname")
def sortname(pos, named):
    return " ".join(pos[:2])


@template("sort")
def sort(pos, named):
    return _arg(pos, 1)


# ships: "{{HMS|Beagle}}" -> "HMS Beagle", "{{sclass|Acciaio|submarine}}" -> "Acciaio-class submarine"
_SHIP_PREFIX = {"uss": "USS", "hms": "HMS", "hmcs": "HMCS", "hmas": "HMAS", "hmnzs": "HMNZS", "ss": "SS",
                "rms": "RMS", "usns": "USNS", "uscgc": "USCGC", "sms": "SMS", "smu": "SM", "gs": "GS",
                "hnlms": "HNLMS", "hmt": "HMT", "hmy": "HMY", "mv": "MV", "ms": "MS", "ssn": "SSN",
                "ijn": "IJN", "ts": "TS", "rv": "RV", "ft": "FT"}
for _p, _shown_prefix in _SHIP_PREFIX.items():
    _HANDLERS[_p] = (lambda prefix: lambda pos, named: f"{prefix} {_arg(pos, 0)}".strip())(_shown_prefix)


@template("ship")
def ship(pos, named):
    return " ".join(p for p in (_arg(pos, 0), _arg(pos, 1)) if p)


@template("sclass", "sclass2", "sclass-")
def sclass(pos, named):
    return f"{_arg(pos, 0)}-class {_arg(pos, 1)}".strip()


# ── science, symbols, typography ─────────────────────────────────────────────

@template("chem", "chem2")
def chem(pos, named):
    """{{chem|UO|2|F|2}} -> "UO2F2"; {{chem2|[Fe(CN)6]^3-}} -> "[Fe(CN)6]3-"."""
    text = "".join(p for p in pos if p)
    text = text.replace("->", "→").replace("<=>", "⇌")
    return re.sub(r"[_^{}\\]", "", text)


@template("nuclide")
def nuclide(pos, named):
    return f"{_arg(pos, 0)}-{_arg(pos, 1)}" if len(pos) > 1 else _arg(pos, 0)


@template("angbr", "angle bracket", "angbr ipa")
def angbr(pos, named):
    return f"⟨{_arg(pos, 0)}⟩"


@template("bibleverse", "bibleref", "bibleref2")
def bibleverse(pos, named):
    return " ".join(p for p in pos[:2] if p)


@template("isbn")
def isbn(pos, named):
    return f"ISBN {_arg(pos, 0)}".strip()


@template("issn")
def issn(pos, named):
    return f"ISSN {_arg(pos, 0)}".strip()


_SIMPLE = {
    "pi": "π", "tau": "τ", "!": "|", "=": "=", "'": "'", "'s": "'s", "'\"": "'\"", "\"'": "\"'", "' \"": "'\"",
    "-\"": '"', "\"": '"', "-'": "'", "-": "", "snd": " – ", "spnd": " – ", "spaced ndash": " – ",
    "spaced en dash": " – ", "ndash": "–", "endash": "–", "mdash": "—", "emdash": "—", "em dash": "—",
    "dash": "–", "snds": "–", "nbnd": "–", "nbh": "-", "nbsp": " ", "nbs": " ", "spaces": " ", "space": " ",
    "times": "×", "dagger": "†", "extinct": "†", "double dagger": "‡", "euro": "€", "solar mass": "M☉",
    "co2": "CO2", "h2o": "H2O", "ellipsis": "…", "…": "…", "clear": "", "-": "",
}
for _k, _v in _SIMPLE.items():
    _HANDLERS[_k] = (lambda v: lambda pos, named: v)(_v)

_FIRST_ARG = ("nowrap", "nobr", "small", "smaller", "larger", "big", "huge", "tiny", "em", "strong", "sc",
              "smallcaps", "abbr", "sup", "sub", "lower", "upper", "math", "mvar", "var", "nobold", "noitalic",
              "no wrap", "bold", "italic", "italics", "bi", "b", "i", "u", "mono", "ill", "interlanguage link",
              "interlanguage link multi", "iw", "link-interwiki", "linktext", "lang-en", "unicode", "wikt",
              "sic", "not a typo", "typo", "tooltip", "ruby", "code", "kbd", "samp", "nwr", "nobreak", "awrap",
              "center", "midsize", "longitem", "poemquote", "iast", "abbrlink", "proper name", "visible anchor",
              "nts", "grc-transl", "bigmath", "mlby", "plainlink", "angle", "ipa-explicit", "no break",
              "lang-latn", "hidden", "block indent", "indent", "quote-block", "stnlnk", "stnlink", "stn", "metro")
for _k in _FIRST_ARG:
    _HANDLERS[_k] = lambda pos, named: _arg(pos, 0) or named.get("1", "")
_HANDLERS["resize"] = lambda pos, named: _arg(pos, 1)
_HANDLERS["quote"] = _HANDLERS["cquote"] = _HANDLERS["blockquote"] = _HANDLERS["quotation"] = \
    lambda pos, named: _first(pos, named, "text", "quote")
_HANDLERS["plainlist"] = _HANDLERS["flatlist"] = lambda pos, named: _arg(pos, 0)
_LISTS = ("unbulleted list", "ubl", "hlist", "bulleted list", "ordered list", "plain list", "collapsible list")
for _k in _LISTS:
    _HANDLERS[_k] = lambda pos, named: ", ".join(p for p in pos if p)
_HANDLERS["flag"] = lambda pos, named: "" if re.fullmatch(r"[A-Z]{2,3}", _arg(pos, 0)) else _arg(pos, 0)
_CURRENCY = {"us$": "US$", "usd": "US$", "gbp": "£", "£": "£", "eur": "€", "€": "€", "cad": "C$", "aud": "A$",
             "cny": "CN¥", "jpy": "¥", "¥": "¥", "inr": "₹", "chf": "CHF "}
for _k, _v in _CURRENCY.items():
    _HANDLERS[_k] = (lambda sym: lambda pos, named: sym + _arg(pos, 0))(_v)


# ── things that are pronunciation or presentation only: say "known, render nothing" ─────────────

def is_silent(key: str) -> bool:
    """Templates that carry no readable prose (pronunciations, audio): dropped on purpose."""
    return key.startswith(("ipa", "ipac", "respell", "pronunciation", "audio", "listen", "pron-"))


# ── entry point ──────────────────────────────────────────────────────────────

def render(key: str, pos: list[str], named: dict[str, str]) -> str | None:
    """Rendering of template `key` (lowercase, single spaces), or None if it isn't a known one."""
    h = _HANDLERS.get(key)
    if h is None:
        if key.startswith("lang-") and len(key) <= 9:        # {{lang-fr|text}}, {{lang-zh-hant|text}}
            return _arg(pos, 0) or named.get("text", "")
        if ":" in key and key.split(":", 1)[0] in ("formatnum", "#expr"):
            return parser_function(*key.split(":", 1))
        if is_silent(key):
            return ""
        return None
    try:
        return h(pos, named)
    except Exception:
        return ""
