"""Which Wikipedia articles are the most central? From the command line:

  python -m experiments.centrality top 50 [--metric prose_pagerank|pagerank|indegree|reverse_pagerank] [--skip 50]
  python -m experiments.centrality "Albert Einstein"
  python -m experiments.centrality surprises [--top 100000] [-n 25]

`top` is a leaderboard (by prose PageRank when the prose link graph is built, else PageRank), a title
looks one article up (case-insensitive; redirects work), and `surprises` lists articles whose
PageRank rank and in-degree rank disagree the most. Prose PageRank counts only links written in
article text; plain PageRank also counts every link a template generates (ISBN, coordinates, ...).
The app (python app.py -> Centrality) has the same views with live autocomplete.
"""
import argparse
import sys

from wikiexp.centrality import METRICS

from .explorer import SHORT, Explorer, format_score


def pick(ex: Explorer, text: str):
    m = ex.titles.resolve(text)
    if m:
        return m
    options = ex.titles.search(text, limit=8)
    if not options:
        sys.exit(f"No article matches '{text}'.")
    print(f"No article is titled '{text}'. Did you mean:")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o.label}")
    if not sys.stdin.isatty():
        sys.exit(2)
    choice = input(f"Article [1-{len(options)}, Enter = 1, q = quit]: ").strip().lower()
    if choice == "q":
        sys.exit(0)
    try:
        return options[int(choice or 1) - 1]
    except (ValueError, IndexError):
        sys.exit("Not one of the options.")


def cmd_top(ex: Explorer, n: int, metric: str, skip: int) -> None:
    others = [m for m in ex.metrics if m != metric]
    print(f"{METRICS[metric]}: ranks {skip + 1}-{skip + n}")
    print(f"{'#':>6}  {'Article':<44} {SHORT[metric]:>10} {'in':>9} {'out':>7}" + "".join(
        f" {SHORT[m] + ' #':>11}" for m in others))
    for i, r in enumerate(ex.leaderboard(metric, n, skip), skip + 1):
        print(f"{i:>6}  {r.title[:44]:<44} {format_score(metric, r.score):>10} {r.in_links:>9,} "
              f"{r.out_links:>7,}" + "".join(f" {r.ranks[m]:>11,}" for m in others))


def cmd_lookup(ex: Explorer, text: str) -> None:
    m = pick(ex, text)
    p = ex.profile(ex.idx_of(m))
    print(p.title + (f"   (via {m.title})" if m.title != m.article else ""))
    print(f"  {p.in_links:,} incoming links, {p.out_links:,} outgoing links, of {len(ex.c):,} articles")
    if p.prose_in is not None:
        print(f"  in running text only: {p.prose_in:,} incoming, {p.prose_out:,} outgoing")
    print(f"  {'metric':<11} {'rank':>10} {'top %':>8} {'score':>10}")
    for metric in ex.metrics:
        print(f"  {SHORT[metric]:<11} {p.ranks[metric]:>10,} {100 - p.percentiles[metric]:>7.3f}% "
              f"{format_score(metric, p.scores[metric]):>10}")
    text_only = " in its text" if p.metric == "prose_pagerank" else ""
    for label, rows in ((f"Most notable articles linking here{text_only}", p.linked_from),
                        (f"Most notable articles it links to{text_only}", p.links_to)):
        if rows:
            print(f"  {label} (by {SHORT[p.metric]} rank): " + "; ".join(f"{t} (#{r:,})" for t, r in rows))


def cmd_surprises(ex: Explorer, top: int, n: int) -> None:
    up, down = ex.surprises(top, n)
    for title, rows in ((f"Punching above their links: PageRank rank far better than in-degree rank "
                         f"(among the top {top:,} by either)", up),
                        ("Many links, little weight: in-degree rank far better than PageRank rank", down)):
        print(title)
        print(f"  {'Article':<44} {'PageRank #':>11} {'In-links #':>11} {'in':>9} {'x':>8}")
        for s in rows:
            print(f"  {s.title[:44]:<44} {s.pagerank_rank:>11,} {s.indegree_rank:>11,} {s.in_links:>9,} "
                  f"{s.factor:>7.1f}x")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", metavar="top N | surprises | TITLE")
    ap.add_argument("--metric", choices=list(METRICS), help="top: default prose_pagerank if built, else pagerank")
    ap.add_argument("--skip", type=int, default=0, help="top: start after this many")
    ap.add_argument("--top", type=int, default=100_000, help="surprises: consider this many top articles")
    ap.add_argument("-n", type=int, default=25, help="surprises: rows per list")
    args = ap.parse_args()

    try:
        ex = Explorer()
    except FileNotFoundError as e:
        sys.exit(str(e))
    args.metric = args.metric or ex.default_metric
    if args.metric not in ex.metrics:
        sys.exit(f"{args.metric} wasn't built; run the Build prose link graph stage, then Compute centrality "
                 "(prose PageRank), or turn on reverse PageRank (gateway).")
    first, rest = args.what[0], args.what[1:]
    if first == "top" and len(rest) <= 1 and all(r.isdigit() for r in rest):
        cmd_top(ex, int(rest[0]) if rest else 25, args.metric, args.skip)
    elif first == "surprises" and not rest:
        cmd_surprises(ex, args.top, args.n)
    else:
        cmd_lookup(ex, " ".join(rest if first == "lookup" else args.what))


if __name__ == "__main__":
    main()
