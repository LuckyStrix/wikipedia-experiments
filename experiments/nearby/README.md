# Nearby: what's notable near here?

Give it a place name or coordinates and a distance; it lists the most notable Wikipedia articles
within that distance, with how far away they are and in which direction.

**Needs:** the core database, title index and places index (pipeline stages 2, 3 and `geo`, which also
wants `geo_tags.sql.gz`; `langlinks.sql.gz` is used when present). **Machine:** any, CPU only. The
places index is 190 MB and queries take a few milliseconds. If `data/centrality/prose_pagerank.npy` exists
(float32, indexed by graph idx; PageRank of the links written in article text) notability uses it,
else `pagerank.npy`, otherwise incoming links.

**Run:** from the app (sidebar -> Nearby), or

```bash
python -m experiments.nearby "Eiffel Tower" --radius 2km
python -m experiments.nearby "48.8584, 2.2945" -r 500m --sort distance
python -m experiments.nearby "48°51′29″N 2°17′40″E" -r 1mi --type landmark --limit 10
python -m experiments.nearby "Mount Everest" -r 50 --types     # what kinds of place are around it
```

In the app: type a place name and pick a suggestion (only articles that have coordinates are offered),
or type coordinates and press Enter. The radius, the sort (`ctrl+o`: most notable / nearest) and the
type filter re-run the search as you change them. In the results, Enter on a row makes that place
the new centre and `ctrl+b` goes back to the previous one.

## Input

- **Location:** decimal `48.8584, 2.2945` (also `-33.86 151.21`), with hemisphere letters
  `48.86N 2.29E` / `N 48.86, E 2.29` (longitude first is fine if letters say so), or
  degrees-minutes-seconds `48°51′29″N 2°17′40″E` (plain `'` and `"` work too). Anything else is a
  place name, matched through the title index: any case, redirects (`NYC`), typos (`Eifel Tower`).
  If the best title has no coordinates (a disambiguation page like `Springfield`, or a topic like
  `Jazz`) the next suggestions that start with the same text and do have coordinates are tried, and
  the tool says which one it used. With two or more of those the name is *ambiguous*: the CLI asks
  you to choose (in a terminal; otherwise it uses the most-linked and lists the others), the app's
  dropdown shows them all.
- **Radius:** `500 m`, `2km`, `3 mi`, `1000 ft`, `1.5 kilometres`; a bare number is km. Up to 20,000 km.
- **Type filter:** the place type from the geotag (`landmark`, `city`, `mountain`, `edu`,
  `railwaystation`, `river`, `waterbody`, `adm1st`, `airport`, ...; `untyped` for the 56% of
  places that have none). `--types` lists what is in range.

When the location was an article, that article is left out of its own results (and says so).

## How it works

- `pipeline/build_geo.py` -> `data/geo.sqlite`: one row per article with a *primary Earth*
  coordinate (1,242,066 on the 2026-09 dump), plus an R*Tree over lat/lon. See the README at the
  top of the repo for the columns.
- `geometry.py` - haversine distance, initial bearing (8-point compass), and `bounding_boxes`:
  the lat/lon box of a circle, widened with latitude, split in two across the antimeridian, and
  turned into a full-longitude band when the circle reaches a pole. The R*Tree answers the box,
  haversine filters it exactly.
- `parsing.py` - coordinates and radii.
- `finder.py` - `NearbyFinder`: resolving names, the search, and the score.
- `screen.py`, `__main__.py` - the app screen (a `TitlePicker` with its `literal=` hook for
  coordinates) and the command line.

### Notability score

```
score = 100 * (0.7 * log(1 + signal) / log(1 + max_signal)  +  0.3 * log(1 + langs) / log(1 + max_langs))
```

`signal` is the article's prose PageRank (scaled by the article count, so an average article is ~1)
when `data/centrality/prose_pagerank.npy` exists, else its PageRank, else its incoming link count; `langs` is how many other
language editions have the article. Both are heavy-tailed, so logs; maxima are over all places, so a
score means the same in every query. Without langlinks the score is the signal alone.

### Which articles are in the index

Only primary coordinates count: an article's primary tag is its own location. 128,709 articles have
only *secondary* tags (a river's mouth, the stations of a railway, the sites of a campaign) and are
skipped: there is no single sensible point for them. Tags on other bodies (Moon, Mars, ...), on
non-articles, impossible values and `(0, 0)` placeholders are dropped too.

## Findings (2026-09 dump)

Build: 62 s, 2.1 GB peak (parent; 1.4 GB per parser process), 190 MB output.

```
$ python -m experiments.nearby "Eiffel Tower" --radius 1km --limit 5
Using Eiffel Tower (48.8582, 2.2945)
Within 1 km of Eiffel Tower (48.8582, 2.2945): 89 places  [0.00s]
(leaving out Eiffel Tower, the place you searched from)
 #  Place                                           Distance  Dir  Type      Score
 1  Organisation internationale de la Francophonie     676 m  E    -            60
 2  Exposition Universelle (1900)                      338 m  SE   -            58
 3  Champ de Mars                                      366 m  SE   -            56
 4  International Chamber of Commerce                  674 m  N    -            55
 5  Palais de Tokyo                                    679 m  N    landmark     54

$ python -m experiments.nearby "40.7484, -73.9857" -r 500m --limit 4        # the Empire State Building
 1  Empire State Building       14 m  SE   landmark     67
 2  Rockefeller Foundation     336 m  NE   -            62
 3  Morgan Library & Museum    373 m  E    landmark     58
 4  American Academy of Dramatic Arts  333 m  S  -     54

$ python -m experiments.nearby "Mount Everest" -r 50 --limit 4
 1  Himalayas        1.01 km  SW   -            75
 2  Lhotse           3.06 km  S    mountain     57
 3  Makalu           19.5 km  SE   mountain     57
 4  Cho Oyu          28.5 km  NW   mountain     55

$ python -m experiments.nearby "-40, -140" -r 100km                       # the South Pacific
Within 100 km of -40.0000, -140.0000: 0 places
Nothing in range.
The nearest place is Maria Theresa Reef, 457 km NE of here; try --radius 503km.

$ python -m experiments.nearby "-18, 179.9" -r 300km --limit 4            # just west of the antimeridian
 1  Fiji         95.2 km  W    country    80
 2  Suva          155 km  W    city       67
 3  Viti Levu     202 km  W    isle       55
 4  Lautoka       262 km  W    city       54

$ python -m experiments.nearby Springfield -r 10km --limit 3
'Springfield' has no coordinates (it may be a disambiguation page); using Springfield, Massachusetts
(42.1014, -72.5903); 6 other places match.
    also: Springfield, Illinois (39.7975, -89.6450) ...
```

- Everything near 180° and the poles works: the 300 km query above reaches both sides of the
  antimeridian (Suva is 155 km west of the query point, and 47 of the 275 places found are on the 179 W side, e.g. the Lau Islands 56 km away), and a
  search at 90 N finds the North Pole and the Arctic Ocean tags, 0 m away.
- 56% of places have no type, so the type filter is most useful in cities (`landmark`, `edu`,
  `railwaystation`) and on terrain (`mountain`, `river`, `waterbody`).
- Link-based notability favours articles that are linked from everywhere for reasons unrelated to
  the place itself, e.g. the Francophonie organisation's Paris office tops the Eiffel Tower's
  neighbours, and the Himalayas outrank Lhotse near Everest. PageRank (once built) and the language
  count soften this but don't remove it; type-based boosts or a pageview signal would be the next step.

### Before and after prose PageRank

Top results with PageRank over all links (before) versus PageRank over links written in article text
(after, `notability from prose PageRank + languages`):

| query | before | after |
|---|---|---|
| Eiffel Tower, 1 km | Francophonie (53), Exposition Universelle 1900, International Chamber of Commerce, Champ de Mars, Guimet Museum, Quai Branly, Death of Diana, Palais de Tokyo | Francophonie (52), Exposition Universelle 1900 (50), International Chamber of Commerce, Champ de Mars, Death of Diana, French School of the Far East, \u00c9cole Militaire, Palais de Tokyo |
| Times Square, 500 m | Nasdaq (62), Midtown Manhattan, NYT Building, One Astor Plaza, Bank of America Tower, Hess Corporation, Bryant Park | Nasdaq (64), Midtown Manhattan (56), Theater District (38), Port Authority Bus Terminal, NYT Building, Bryant Park, One Times Square, Palace Theatre |
| Mount Everest, 50 km | Himalayas (65), Lhotse, Makalu, Cho Oyu, Island Peak, Sagarmatha NP, Nuptse, Changtse | Himalayas (73), Lhotse, Makalu, Cho Oyu, Sagarmatha NP, Khumbu, Khumbu Glacier, Namche Bazaar |

Times Square improves most: corporate lobby buildings (One Astor Plaza, Hess Corporation) give way
to the Theater District, the bus terminal and the Palace Theatre. Everest loses the climbing-route
article Island Peak for the Khumbu region and Namche Bazaar. Paris barely changes: the Francophonie
office stays first because it is linked in prose from many articles about member states and the
language count (`langs`) adds to it, so the next step there is a type weight or pageviews, not a
different link graph.

## Ideas

- Pageviews as a notability signal; per-type weights ("landmark" over "railwaystation")
- Search along a line (a road trip) or inside a polygon rather than a circle
- Map output (SVG/GeoJSON) and secondary coordinates for linear features
