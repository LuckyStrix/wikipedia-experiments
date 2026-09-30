from collections import deque

import numpy as np

from experiments.six_degrees.solver import shortest_path
from wikiexp.core import Graph


class ArrayGraph:
    def __init__(self, n, edges):
        edges = sorted(set(edges))
        src = np.array([a for a, b in edges], dtype=np.int32)
        dst = np.array([b for a, b in edges], dtype=np.int32)
        self.n = n
        self.out_indptr = np.concatenate(([0], np.cumsum(np.bincount(src, minlength=n))))
        self.out_indices = dst
        order = np.argsort(dst, kind="stable")
        self.in_indptr = np.concatenate(([0], np.cumsum(np.bincount(dst, minlength=n))))
        self.in_indices = src[order]

    def __len__(self):
        return self.n


def bfs_distance(n, edges, s, t):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist.get(t)


def test_matches_plain_bfs_on_random_graphs():
    rng = np.random.default_rng(42)
    for trial in range(40):
        n = int(rng.integers(5, 60))
        m = int(rng.integers(0, n * 3))
        edges = list({(int(a), int(b)) for a, b in rng.integers(0, n, (m, 2)) if a != b})
        g = ArrayGraph(n, edges)
        edge_set = set(edges)
        for _ in range(10):
            s, t = (int(x) for x in rng.integers(0, n, 2))
            r = shortest_path(g, s, t)
            expected = bfs_distance(n, edges, s, t)
            assert r.degrees == expected, (trial, s, t)
            if r.path:
                assert r.path[0] == s and r.path[-1] == t
                assert all((a, b) in edge_set for a, b in zip(r.path, r.path[1:]))


def test_same_article_is_zero_degrees():
    g = ArrayGraph(3, [(0, 1)])
    assert shortest_path(g, 2, 2).degrees == 0


def test_on_tiny_wikipedia(tiny_data):
    from tests.conftest import ARTICLES
    g = Graph(tiny_data / "graph")
    r = shortest_path(g, ARTICLES.index("Kevin Bacon"), ARTICLES.index("Mitochondrion"))
    assert [ARTICLES[i] for i in r.path] == ["Kevin Bacon", "Footloose", "Film", "Biology", "Mitochondrion"]
    assert shortest_path(g, ARTICLES.index("Kevin Bacon"), ARTICLES.index("Isolated article")).path is None
