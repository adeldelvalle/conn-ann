"""Approximate nearest-neighbour search by greedy navigation of an LSH graph.

Two stages, in the spirit of IVF and HNSW but splitting the work differently:

- *Entry*: the query is hashed and the rarest buckets it lands in supply a
  handful of seed nodes.  This is the job HNSW gives to its upper layers and
  IVF to its coarse quantiser; here the LSH tables already answer it, and the
  smallest bucket is the most informative one to ask (the same
  ``log(n / m_b)`` argument that drives the surprisal weighting).
- *Descent*: from those seeds, walk the neighbour graph, computing real
  distances, always expanding the closest unexpanded node, until no unvisited
  neighbour improves on the current result set.

Unlike the graph *construction*, which never computes a distance, search is
metric: the collision graph only proposes where to look.
"""
from __future__ import annotations

import heapq

import numpy as np


class GraphSearcher(object):
    """
    Nearest-neighbour search over a neighbour graph built by
    `NeighborGraphBuilder`.

    :param index: the `LSHIndex` the graph was built from, used to hash queries
        and find entry points.
    :param points: the indexed points, (N, d).
    :param edges: the graph as ``(i, j, weight)`` tuples; weights are ignored,
        only the topology is used.
    :param adjacency: pre-built ``(indptr, indices)`` CSR pair, instead of
        `edges`.
    """

    def __init__(self, index, points, edges=None, adjacency=None):
        self.index = index
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self._sq = (self.points ** 2).sum(axis=1)
        n = self.points.shape[0]
        if adjacency is None:
            if edges is None:
                raise ValueError("pass either edges or adjacency")
            adjacency = self.adjacency_from_edges(edges, n)
        self.indptr, self.indices = adjacency
        self.n_distance_calls = 0

    @staticmethod
    def adjacency_from_edges(edges, n):
        """
        CSR adjacency of the undirected graph.

        Each edge is stored in both directions so a walk can leave a node by
        any incident edge, whichever endpoint the builder emitted it from.
        """
        if len(edges) == 0:
            return np.zeros(n + 1, dtype=np.int64), np.empty(0, dtype=np.int32)
        e = np.asarray([(i, j) for i, j, _ in edges], dtype=np.int64)
        src = np.concatenate([e[:, 0], e[:, 1]])
        dst = np.concatenate([e[:, 1], e[:, 0]])
        order = np.argsort(src, kind="stable")
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
        return indptr, dst[order].astype(np.int32)

    @property
    def degrees(self):
        return np.diff(self.indptr)

    # ------------------------------------------------------------------ #

    def _distances(self, idx, query, qq):
        """Squared euclidean distances from `query` to `points[idx]`."""
        self.n_distance_calls += int(idx.size)
        return self._sq[idx] - 2.0 * (self.points[idx] @ query) + qq

    def _entry_points(self, code_row, n_entry):
        """
        Seed nodes for the descent: members of the rarest buckets the query
        falls into, smallest bucket first, until `n_entry` are collected.
        """
        buckets = []
        for table, code in zip(self.index.tables, code_row):
            bucket = table.get(code)
            if bucket is not None and bucket.size:
                buckets.append(bucket)
        if not buckets:
            return np.empty(0, dtype=np.int64)
        buckets.sort(key=lambda b: b.size)
        taken, total = [], 0
        for bucket in buckets:
            taken.append(bucket)
            total += bucket.size
            if total >= n_entry:
                break
        seeds = np.unique(np.concatenate(taken))
        if seeds.size > n_entry:
            seeds = seeds[:n_entry]
        return seeds

    def search_one(self, query, code_row, k=10, ef=32, n_entry=32):
        """
        Nearest neighbours of one query.

        :param ef: beam width - how many best-so-far nodes are kept.  ``ef=1``
            is plain greedy descent ("move to the closer one and repeat"),
            which is cheap but gets stuck in the first local minimum; larger
            `ef` keeps alternatives alive and is what makes graph search
            competitive.  Must be >= k to return k results reliably.
        :param n_entry: how many seed nodes to take from the buckets.
        :returns: ``(indices, squared_distances)`` sorted nearest first.
        """
        query = np.asarray(query, dtype=np.float32)
        qq = float(query @ query)

        seeds = self._entry_points(code_row, n_entry)
        if seeds.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        seed_d = self._distances(seeds, query, qq)
        visited = set(seeds.tolist())

        # `frontier` is a min-heap on distance (what to expand next);
        # `best` is a max-heap on distance (the ef closest found so far).
        frontier = list(zip(seed_d.tolist(), seeds.tolist()))
        heapq.heapify(frontier)
        best = [(-d, i) for d, i in frontier]
        heapq.heapify(best)
        while len(best) > ef:
            heapq.heappop(best)

        indptr, indices = self.indptr, self.indices
        while frontier:
            dist_c, node = heapq.heappop(frontier)
            if len(best) >= ef and dist_c > -best[0][0]:
                break  # nothing left that could improve the beam

            neighbours = indices[indptr[node]:indptr[node + 1]]
            fresh = [int(v) for v in neighbours.tolist() if v not in visited]
            if not fresh:
                continue
            visited.update(fresh)

            fresh_d = self._distances(np.asarray(fresh, dtype=np.int64), query, qq)
            worst = -best[0][0] if best else float("inf")
            for d, i in zip(fresh_d.tolist(), fresh):
                if len(best) < ef or d < worst:
                    heapq.heappush(best, (-d, i))
                    heapq.heappush(frontier, (d, i))
                    if len(best) > ef:
                        heapq.heappop(best)
                    worst = -best[0][0]

        top = heapq.nlargest(min(k, len(best)), best)
        idx = np.asarray([i for _, i in top], dtype=np.int64)
        dst = np.asarray([-d for d, _ in top], dtype=np.float32)
        return idx, dst

    def search(self, queries, k=10, ef=32, n_entry=32):
        """
        Batch version of `search_one`.

        :returns: ``(indices, distances)`` of shape (n_queries, k), padded with
            -1 / inf where a query found fewer than k results.
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        codes = self.index.encode(queries).tolist()

        idx = np.full((queries.shape[0], k), -1, dtype=np.int64)
        dst = np.full((queries.shape[0], k), np.inf, dtype=np.float32)
        for row, (query, code_row) in enumerate(zip(queries, codes)):
            i, d = self.search_one(query, code_row, k=k, ef=ef, n_entry=n_entry)
            idx[row, :i.size] = i
            dst[row, :d.size] = d
        return idx, dst
