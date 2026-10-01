"""Search the articles from the command line: hybrid (default), semantic or keyword.

  python -m experiments.semantic_search "that battle where the weather decided everything" -k 10
  python -m experiments.semantic_search "Tacoma Narrows Bridge collapse" --mode keyword
  python -m experiments.semantic_search      # asks for queries until you press Enter on an empty line

--mode hybrid (default) fuses meaning (embeddings, if built), the question's words (keyword index) and
exact article names (title index); semantic and keyword use one retriever alone. Needs the `text`
stage plus the `keyword` and/or `embed` stages. The first query loads the embedding model, which takes
a few seconds; later ones are fast.
"""
import argparse
import sys
import textwrap
import time
from pathlib import Path

from wikiexp import paths
from wikiexp.retrieval import MODES, HybridSearch
from wikiexp.semantic import SemanticSearch

SHORT = {"semantic": "sem", "keyword": "kw", "title": "name"}


def show(hs: HybridSearch, query: str, k: int, full: bool, mode: str) -> None:
    t0 = time.time()
    hits = hs.search(query, k, mode)
    ms = (time.time() - t0) * 1000
    print(f"\n{query!r} [{mode}]: {len(hits)} results ({ms:.0f} ms)")
    for i, h in enumerate(hits, 1):
        score = f"{h.score:.3f}" if mode == "semantic" else f"{h.score:.1f}" if mode == "keyword" else \
            "+".join(SHORT[v] for v in h.via)
        print(f"{i:>3}. {score:<12} {h.title}   [{h.page_id}]")
        body = h.lead if full else h.snippet
        print(textwrap.indent(textwrap.fill(body, 100), "           "))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="what to look for (several words need no quotes)")
    ap.add_argument("-k", type=int, default=10, help="number of results (default 10)")
    ap.add_argument("--mode", choices=MODES, default="hybrid", help="hybrid (default), semantic or keyword")
    ap.add_argument("--full", action="store_true", help="show the whole stored lead, not a snippet")
    ap.add_argument("--data", help="data folder (default: the configured one)")
    args = ap.parse_args()

    data = Path(args.data) if args.data else paths.DATA
    hs = HybridSearch(data, semantic=SemanticSearch(data))
    modes = hs.modes()
    if not modes:
        sys.exit(f"No search data in {data}. Build the keyword index and/or the embeddings with the 'text', "
                 f"'keyword' and 'embed' stages (python -m pipeline.build_text; python -m pipeline.build_keyword; "
                 f"python -m pipeline.build_embed); see experiments/semantic_search/README.md.")
    if args.mode not in modes:
        need = "pipeline.build_embed" if args.mode == "semantic" else "pipeline.build_keyword"
        sys.exit(f"--mode {args.mode} needs data that isn't built here (python -m {need}); available: "
                 f"{', '.join(modes)}.")
    if args.query:
        show(hs, " ".join(args.query), args.k, args.full, args.mode)
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
        show(hs, q, args.k, args.full, args.mode)


if __name__ == "__main__":
    main()
