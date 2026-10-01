"""Registry of experiments the app can open. See experiments/README.md for conventions."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    key: str
    name: str
    description: str
    requires: tuple[str, ...]     # pipeline stage keys whose outputs it needs
    screen: str                   # "module:Class" of its Textual screen


EXPERIMENTS = [
    Experiment("six_degrees", "Six Degrees", "Shortest chain of links between any two articles.",
               ("core", "titles"), "experiments.six_degrees.screen:SixDegreesScreen"),
    Experiment("nearby", "Nearby", "What's notable within a distance of any place or coordinates.",
               ("core", "titles", "geo"), "experiments.nearby.screen:NearbyScreen"),
    Experiment("centrality", "Centrality", "PageRank leaderboard, article lookup and over/under-rated articles.",
               ("core", "titles", "centrality"), "experiments.centrality.screen:CentralityScreen"),
    Experiment("semantic_search", "Semantic search", "Find articles by meaning, words and names (hybrid search).",
               ("text", "keyword"), "experiments.semantic_search.screen:SemanticSearchScreen"),
    Experiment("ask", "Ask Wikipedia", "Ask a question; a local language model answers from Wikipedia, with citations.",
               ("core", "titles", "text", "keyword"), "experiments.ask.screen:AskScreen"),
]
