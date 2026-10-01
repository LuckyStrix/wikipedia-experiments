"""Semantic search from the command line.

  python -m experiments.semantic_search "that battle where the weather decided everything" -k 10
  python -m experiments.semantic_search      # asks for queries until you press Enter on an empty line

Needs the `text` and `embed` stages (data/text/leads.sqlite and data/search/). The first query loads
the embedding model, which takes a few seconds; later ones are fast.
"""
import argparse
import sys
import textwrap
import time
from pathlib import Path

from wikiexp import paths
from wikiexp.semantic import SemanticSearch


def show(ss: SemanticSearch, query: str, k: int, full: bool) -> None:
    t0 = time.time()
    hits = ss.search(query, k)
    ms = (time.time() - t0) * 1000
    print(f"\n{query!r}: {len(hits)} results ({ms:.0f} ms)")
    for i, h in enumerate(hits, 1):
        print(f"{i:>3}. {h.score:.3f}  {h.title}   [{h.page_id}]")
        body = h.lead if full else h.snippet
        print(textwrap.indent(textwrap.fill(body, 100), "           "))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="what to look for (several words need no quotes)")
    ap.add_argument("-k", type=int, default=10, help="number of results (default 10)")
    ap.add_argument("--full", action="store_true", help="show the whole stored lead, not a snippet")
    ap.add_argument("--data", help="data folder (default: the configured one)")
    args = ap.parse_args()

    ss = SemanticSearch(Path(args.data) if args.data else paths.DATA)
    if not ss.available():
        sys.exit(f"No search data in {ss.dir}. Build it with the 'text' and 'embed' stages "
                 f"(python -m pipeline.build_text; python -m pipeline.build_embed); see "
                 f"experiments/semantic_search/README.md.")
    if args.query:
        show(ss, " ".join(args.query), args.k, args.full)
        return
    if not sys.stdin.isatty():
        ap.error("give a query")
    while True:
        try:
            q = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(ss, q, args.k, args.full)


if __name__ == "__main__":
    main()
