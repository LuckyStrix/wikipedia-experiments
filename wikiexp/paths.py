"""Where data lives on this machine.

Code is shared through git; data is not. Each machine can keep its data wherever it likes by
setting paths in config.toml (see config.example.toml) or the WIKI_DATA / WIKI_DUMPS environment
variables. The defaults are data/ and dumps/ inside the repo.
"""
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    path = ROOT / "config.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


CONFIG = load_config()
_paths = CONFIG.get("paths", {})

DATA = Path(os.environ.get("WIKI_DATA") or _paths.get("data") or ROOT / "data")
DUMPS = Path(os.environ.get("WIKI_DUMPS") or _paths.get("dumps") or ROOT / "dumps")
DUMP_DATE = CONFIG.get("dump_date", "20260901")

DB = DATA / "wiki.sqlite"
GRAPH = DATA / "graph"


def dump(name):
    """Path of a Wikimedia dump file, e.g. dump("page.sql.gz")."""
    return DUMPS / f"enwiki-{DUMP_DATE}-{name}"


def zim():
    """Path of the newest Kiwix .zim file in DUMPS."""
    found = sorted(DUMPS.glob("*.zim"))
    if not found:
        raise FileNotFoundError(f"no .zim file in {DUMPS}")
    return found[-1]
