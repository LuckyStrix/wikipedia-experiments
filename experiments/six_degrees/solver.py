"""Shortest link path between two articles: bidirectional breadth-first search on the CSR graph.

UI-free. Searches forward along outgoing links from the start and backward along incoming links
from the goal, always expanding whichever side has fewer links to follow, until they meet. Each
level is expanded with numpy in one go, so even million-article frontiers take well under a second.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    path: list[int] | None      # article idx values from start to goal, or None if unreachable
    visited: int                 # articles touched by the search
    seconds: float

    @property
    def degrees(self) -> int | None:
        return None if self.path is None else len(self.path) - 1


def _neighbors(indptr, indices, frontier):
    """All (neighbor, from) pairs for the frontier nodes, vectorised."""
    starts = np.asarray(indptr[frontier], dtype=np.int64)
    counts = np.asarray(indptr[frontier + 1], dtype=np.int64) - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    # position of each edge = its node's start + its offset within that node's run
    offsets = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    positions = np.repeat(starts, counts) + offsets
    return np.asarray(indices[positions]), np.repeat(frontier, counts).astype(np.int32)


def _edge_count(indptr, frontier) -> int:
    return int((np.asarray(indptr[frontier + 1], np.int64) - np.asarray(indptr[frontier], np.int64)).sum())


def shortest_path(graph, start: int, goal: int, max_degrees: int = 20) -> Result:
    """graph: wikiexp.core.Graph (or anything with out_/in_ indptr/indices and __len__)."""
    t0 = time.perf_counter()
    if start == goal:
        return Result([start], 1, time.perf_counter() - t0)
    n = len(graph)
    # parent pointers: forward side points toward start, backward side toward goal; -1 = unvisited
    parent = [np.full(n, -1, np.int32), np.full(n, -1, np.int32)]
    dist = [np.full(n, -1, np.int16), np.full(n, -1, np.int16)]
    parent[0][start], dist[0][start] = start, 0
    parent[1][goal], dist[1][goal] = goal, 0
    frontier = [np.array([start], np.int32), np.array([goal], np.int32)]
    csr = [(graph.out_indptr, graph.out_indices), (graph.in_indptr, graph.in_indices)]
    visited = 2

    for _ in range(max_degrees):
        if len(frontier[0]) == 0 or len(frontier[1]) == 0:
            break
        side = 0 if _edge_count(csr[0][0], frontier[0]) <= _edge_count(csr[1][0], frontier[1]) else 1
        nbrs, froms = _neighbors(*csr[side], frontier[side])
        fresh = parent[side][nbrs] == -1
        nbrs, froms = nbrs[fresh], froms[fresh]
        nbrs, first = np.unique(nbrs, return_index=True)   # keep one parent per new node
        froms = froms[first]
        level = dist[side][frontier[side][0]] + 1
        parent[side][nbrs] = froms
        dist[side][nbrs] = level
        visited += len(nbrs)
        frontier[side] = nbrs
        met = nbrs[parent[1 - side][nbrs] != -1]
        if len(met):
            # every meeting node is `level` from this side; pick the one closest to the other side
            best = int(met[np.argmin(dist[1 - side][met])])
            return Result(_path(parent, best, start, goal), visited, time.perf_counter() - t0)
    return Result(None, visited, time.perf_counter() - t0)


def _path(parent, meet: int, start: int, goal: int) -> list[int]:
    fwd, node = [meet], meet
    while node != start:
        node = int(parent[0][node])
        fwd.append(node)
    fwd.reverse()
    node = meet
    while node != goal:
        node = int(parent[1][node])
        fwd.append(node)
    return fwd
