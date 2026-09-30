"""Six Degrees of Wikipedia from the command line.

  python -m experiments.six_degrees "Kevin Bacon" "Mitochondria"
  python -m experiments.six_degrees --random

Titles are matched case-insensitively, and redirects work ("usa" -> United States). If a title
isn't an article you get suggestions, and in a terminal you can pick one by number.
The app (python app.py -> Six Degrees) has live autocomplete instead.
"""
import argparse
import sys

from .finder import PathFinder


def pick(finder: PathFinder, text: str, role: str):
    m = finder.titles.resolve(text)
    if m:
        return m
    options = finder.titles.search(text, limit=8)
    if not options:
        sys.exit(f"No article matches '{text}'.")
    print(f"No article is titled '{text}'. Did you mean:")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o.label}")
    if not sys.stdin.isatty():
        sys.exit(2)
    choice = input(f"{role} [1-{len(options)}, Enter = 1, q = quit]: ").strip().lower()
    if choice == "q":
        sys.exit(0)
    try:
        return options[int(choice or 1) - 1]
    except (ValueError, IndexError):
        sys.exit("Not one of the options.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", nargs="?")
    ap.add_argument("goal", nargs="?")
    ap.add_argument("--random", action="store_true", help="pick two random well-known articles")
    args = ap.parse_args()
    if not args.random and not (args.start and args.goal):
        ap.error("give two titles, or --random")

    finder = PathFinder()
    if args.random:
        start, goal = finder.titles.random_article(), finder.titles.random_article()
    else:
        start, goal = pick(finder, args.start, "Start"), pick(finder, args.goal, "Goal")

    result, titles = finder.find(start, goal)
    if result.path is None:
        print(f"No path from {start.article} to {goal.article} "
              f"(searched {result.visited:,} articles in {result.seconds:.2f}s).")
        sys.exit(1)
    print(f"{start.article} → {goal.article}: {result.degrees} click{'s' * (result.degrees != 1)}")
    for i, t in enumerate(titles):
        print(f"  {'  ' * min(i, 1)}{'→ ' if i else ''}{t}")
    print(f"(searched {result.visited:,} articles in {result.seconds:.2f}s)")


if __name__ == "__main__":
    main()
