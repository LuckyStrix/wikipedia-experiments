"""What's notable near a place, from the command line.

  python -m experiments.nearby "Eiffel Tower" --radius 2km
  python -m experiments.nearby "48.8584, 2.2945" --radius 500m --sort distance
  python -m experiments.nearby "48°51′29″N 2°17′40″E" -r 1mi --type landmark --limit 10
  python -m experiments.nearby "Mount Everest" -r 50 --types            # list the place types around it

The location is a title (redirects and typos work: "NYC", "Eifel Tower") or coordinates (decimal,
with N/S/E/W, or degrees-minutes-seconds). The radius is "500 m", "2km", "3 mi"; a bare number is km.
If a name is ambiguous ("Springfield") you pick from the candidates in a terminal; otherwise the
most-linked one is used and the others are listed. The app (python app.py -> Nearby) does the same
with live suggestions.
"""
import argparse
import math
import sys

from .finder import NearbyFinder, Resolution
from .parsing import format_distance, format_radius, parse_radius

DEFAULT_RADIUS = "2 km"


def choose(res: Resolution):
    """The Location to use, asking the user when the name was ambiguous and we're in a terminal."""
    if not res.ambiguous or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return res.location
    print(res.note)
    for i, c in enumerate(res.candidates, 1):
        print(f"  {i}. {c.match.label}  ({c.lat:.4f}, {c.lon:.4f})" + (f"  {c.type}" if c.type else ""))
    choice = input(f"Which one? [1-{len(res.candidates)}, Enter = 1, q = quit]: ").strip().lower()
    if choice == "q":
        sys.exit(0)
    try:
        return res.candidates[int(choice or 1) - 1].location
    except (ValueError, IndexError):
        sys.exit("Not one of the options.")


def table(result) -> str:
    head = ("#", "Place", "Distance", "Dir", "Type", "Score")
    lines = [[str(i), h.title, format_distance(h.distance_km), h.compass or "-", h.type or "-", f"{h.score:.0f}"]
             for i, h in enumerate(result.hits, 1)]
    widths = [max(len(head[c]), *(len(r[c]) for r in lines)) for c in range(len(head))]
    widths[1] = min(widths[1], 56)
    align = ["{:>%d}", "{:<%d}", "{:>%d}", "{:<%d}", "{:<%d}", "{:>%d}"]
    fmt = "  ".join(a % w for a, w in zip(align, widths))
    out = [fmt.format(*head), fmt.format(*("-" * w for w in widths))]
    for r in lines:
        r[1] = r[1] if len(r[1]) <= widths[1] else r[1][:widths[1] - 1] + "…"
        out.append(fmt.format(*r))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("location", help="a title or coordinates")
    ap.add_argument("-r", "--radius", default=DEFAULT_RADIUS, help=f"e.g. 500m, 2km, 3mi (default {DEFAULT_RADIUS})")
    ap.add_argument("--sort", choices=["notable", "distance"], default="notable")
    ap.add_argument("--limit", type=int, default=30, help="how many places to show (default 30)")
    ap.add_argument("--type", action="append", default=[], metavar="TYPE",
                    help="only this kind of place (landmark, city, mountain, railwaystation, ...); repeatable or comma-separated")
    ap.add_argument("--types", action="store_true", help="list the kinds of place in range instead of the places")
    args = ap.parse_args(argv)

    try:
        radius = parse_radius(args.radius)
        finder = NearbyFinder()
        res = finder.resolve_location(args.location)
    except (ValueError, FileNotFoundError) as e:   # CoordinateError is a ValueError
        sys.exit(str(e))
    if res.location is None:
        sys.exit(res.note)
    print(res.note)
    if res.ambiguous and not (sys.stdin.isatty() and sys.stdout.isatty()):
        for c in res.candidates[1:]:
            print(f"    also: {c.match.label} ({c.lat:.4f}, {c.lon:.4f})")
    loc = choose(res)
    types = [t for part in args.type for t in part.split(",")]

    r = finder.nearby(loc, radius, sort=args.sort, limit=args.limit, types=types)
    where = f"{loc.label} ({loc.coords})" if loc.is_article else loc.coords
    print(f"\nWithin {format_radius(radius)} of {where}: {r.in_range:,} place{'s' * (r.in_range != 1)}"
          + (f"; {r.total:,} of type {', '.join(r.types)}" if r.types else "")
          + f"  [{r.seconds:.2f}s, notability from {finder.signal_name}"
          + (" + languages" if finder.langs_max else "") + "]")
    if r.excluded:
        print(f"(leaving out {r.excluded}, the place you searched from)")
    if args.types:
        for t, n in r.type_counts.most_common():
            print(f"  {t:<20} {n:,}")
        return
    if not r.hits:
        if r.in_range == 0:
            print("Nothing in range.")
            near = finder.nearest(loc, radius)
            if near:
                print(f"The nearest place is {near.title}, {format_distance(near.distance_km)} {near.compass} "
                      f"of here; try --radius {format_radius(math.ceil(near.distance_km * 1.1))}.")
        else:
            common = ", ".join(f"{t} ({n})" for t, n in r.type_counts.most_common(8))
            print(f"No place of type {', '.join(r.types)} in range. Types here: {common}")
        return
    print(f"Top {len(r.hits)} by {'notability' if r.sort == 'notable' else 'distance'}:\n")
    print(table(r))
    if r.total > len(r.hits):
        print(f"\n… {r.total - len(r.hits):,} more (use --limit)")


if __name__ == "__main__":
    main()
