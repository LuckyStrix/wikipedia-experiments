"""Registry of every app setting, plus load/save to config.toml.

One ``Setting`` per value the user can edit. The UI builds its forms from ``REGISTRY`` (in order,
grouped by ``tab`` then ``section``); stages read plain values out of a ``Settings`` mapping keyed
by ``Setting.key``.

Settings live in config.toml, the same per-machine file the command-line scripts read, at the
location given by ``Setting.toml`` (e.g. ``("paths", "data")`` is ``[paths] data = ...``). Saving
rewrites config.toml but keeps sections the app doesn't manage, such as ``[machines.*]``.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wikiexp.paths import ROOT

CONFIG_PATH = ROOT / "config.toml"

# Setting.type values
BOOL, STR, PATH = "bool", "str", "path"


@dataclass(frozen=True)
class Setting:
    key: str
    type: str
    default: Any
    tab: str
    section: str
    label: str
    toml: tuple[str, ...]
    hint: str = ""        # short grey text next to the field
    help: str = ""        # grey paragraph under the field
    pattern: str = ""     # STR only: regex the value must match (blank always allowed if default is blank)


REGISTRY: list[Setting] = [
    # ── Paths ──────────────────────────────────────────────────────────────────
    Setting("data_dir", PATH, "", "Paths", "Folders", "Data folder", ("paths", "data"),
            hint="blank = data/ in the repo",
            help="Where built files go (wiki.sqlite, graph/, titles.sqlite, runs/)."),
    Setting("dumps_dir", PATH, "", "Paths", "Folders", "Dumps folder", ("paths", "dumps"),
            hint="blank = dumps/ in the repo",
            help="Where downloaded Wikimedia dumps and the Kiwix .zim live."),
    Setting("dump_date", STR, "20260901", "Paths", "Wikimedia dump", "Dump date", ("dump_date",),
            hint="YYYYMMDD", pattern=r"\d{8}",
            help="Which dump from dumps.wikimedia.org/enwiki/ to download and build from. Dumps "
                 "older than a few months are removed there; see the README for mirrors."),

    # ── Download ───────────────────────────────────────────────────────────────
    Setting("dl_extras", BOOL, True, "Download", "Optional files", "Extra tables (~3.5 GB)",
            ("stages", "download", "extras"),
            help="Categories, coordinates, Wikidata IDs and language links, for the category, "
                 "map and notability projects."),
    Setting("dl_text", BOOL, True, "Download", "Optional files", "Article text (~25 GB)",
            ("stages", "download", "text"),
            help="Full wikitext of every article plus its index, for the text and search projects."),

    # ── Build ──────────────────────────────────────────────────────────────────
    Setting("verify", BOOL, True, "Build", "Core database", "Verify checksums first",
            ("stages", "core", "verify"),
            help="Checks the four input dumps against Wikimedia's sha1sums before building "
                 "(about a minute). Turn off only for test data."),
    Setting("titles_redirects", BOOL, True, "Build", "Title index", "Include redirects",
            ("stages", "titles", "redirects"),
            help="Lets searches for aliases like 'USA' or 'JFK' find the real article. Roughly "
                 "triples the index size."),
]

BY_KEY = {s.key: s for s in REGISTRY}


def valid(st: Setting, value: Any) -> bool:
    if st.type == STR and st.pattern and value != st.default:
        return bool(re.fullmatch(st.pattern, str(value)))
    return True


class Settings(dict):
    """Current values keyed by Setting.key; unknown keys are rejected."""

    def __init__(self, values: dict | None = None):
        super().__init__({s.key: s.default for s in REGISTRY})
        for k, v in (values or {}).items():
            self[k] = v

    def __setitem__(self, key, value):
        if key not in BY_KEY:
            raise KeyError(key)
        super().__setitem__(key, value)

    def text(self, key: str) -> str:
        return str(self[key]).strip()

    def invalid(self) -> list[str]:
        return [s.label for s in REGISTRY if not valid(s, self[s.key])]

    # ── config.toml ────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Settings":
        config = read_config(path)
        values = {}
        for st in REGISTRY:
            node = config
            for part in st.toml:
                if not isinstance(node, dict) or part not in node:
                    break
                node = node[part]
            else:
                values[st.key] = node
        return cls(values)

    def save(self, path: Path = CONFIG_PATH) -> None:
        config = read_config(path)
        for st in REGISTRY:
            node = config
            for part in st.toml[:-1]:
                node = node.setdefault(part, {})
            if self[st.key] == st.default and st.type != BOOL and not self[st.key]:
                node.pop(st.toml[-1], None)  # blank paths mean "use the default"; don't write them
            else:
                node[st.toml[-1]] = self[st.key]
        path.write_text(HEADER + dump_toml(config))


def env_for(s: Mapping) -> dict[str, str]:
    """Environment variables that make wikiexp.paths use these settings in a subprocess."""
    env = {"WIKI_DUMP_DATE": str(s["dump_date"]).strip()}
    if str(s["data_dir"]).strip():
        env["WIKI_DATA"] = str(s["data_dir"]).strip()
    if str(s["dumps_dir"]).strip():
        env["WIKI_DUMPS"] = str(s["dumps_dir"]).strip()
    return env


def read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


HEADER = "# This machine's settings (not in git). Edited by the app's ctrl+s; see config.example.toml.\n"


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"can't write {type(v).__name__} to TOML")


def dump_toml(data: dict, prefix: str = "") -> str:
    """Minimal TOML writer for config.toml: scalars, lists and nested tables."""
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    out = "".join(f"{k} = {_toml_value(v)}\n" for k, v in scalars.items())
    for k, v in tables.items():
        name = f"{prefix}.{k}" if prefix else k
        if any(not isinstance(x, dict) for x in v.values()) or not v:
            out += f"\n[{name}]\n"
        out += dump_toml(v, name)
    return out
