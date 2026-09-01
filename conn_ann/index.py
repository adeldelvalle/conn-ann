"""Bucket index: hash tables over a point set, and their collision statistics."""
from __future__ import annotations

import math

import numpy as np

from .hashing import MAX_PACKED_BITS, RandomProjectionHasher


class LSHIndex(object):
    """
    `num_tables` hash tables over one point set.

    Cost model (the claim a timing figure has to defend)
    ----------------------------------------------------
    Indexing is O(N L b) for the projections plus O(N L) for the bucket
    assignment.  A neighbour search scans, for every query point `i` and every
    table `l`, the whole bucket that `i` falls into, so the total candidate
    work is

        sum_i sum_l m_{b_l(i)}  =  L * sum_b m_b^2  =  L * N^2 * sum_b p_b^2

    i.e. it is governed by the second moment of the bucket-occupancy
    distribution.  At a *fixed* code length `b` the occupancies are fixed, so
    this is Theta(N^2): the build is subquadratic only if the code length grows
    with N.  With ``b ~ log2(N / c)`` the expected occupancy is `c` and the
    total work collapses to O(N L c), linear in N.  That is what
    ``hash_size="auto"`` does.  `collision_stats` reports the realised second
    moment, so the claim can be measured rather than assumed.

    :param hash_size: bits per table, or `None` / `"auto"` to derive it from
        the dataset size at index time as ``ceil(log2(N / bucket_size))``.
    :param input_dim: dimension of the input vectors.
    :param num_tables: number of independent tables.
    :param bucket_size: target expected bucket occupancy used when `hash_size`
        is automatic.  Keep it comfortably above the `k` asked of the graph
        builder, or a single table cannot supply k candidates.
    :param hasher: pre-built `RandomProjectionHasher` to use instead of
        constructing one (mainly for tests that pin the planes).
    :param hasher_kwargs: forwarded to `RandomProjectionHasher`.
    """

    def __init__(self, hash_size, input_dim, num_tables=1, bucket_size=64,
                 hasher=None, **hasher_kwargs):
        self.auto_hash_size = hash_size is None or hash_size == "auto"
        self.hash_size = None if self.auto_hash_size else int(hash_size)
        self.input_dim = int(input_dim)
        self.num_tables = int(num_tables)
        self.bucket_size = int(bucket_size)
        self._hasher_kwargs = hasher_kwargs

        self.tables = [{} for _ in range(self.num_tables)]

        # Populated by `build`.
        self.codes = None            # (N, L) codes of the indexed set
        self.points = None           # reference to the array that was indexed
        self.n_points = 0

        self.hasher = hasher
        if self.hasher is None and not self.auto_hash_size:
            self.hasher = self._make_hasher()

    def _make_hasher(self):
        return RandomProjectionHasher(self.hash_size, self.input_dim,
                                      self.num_tables, **self._hasher_kwargs)

    # ------------------------------------------------------------------ #
    # code length
    # ------------------------------------------------------------------ #

    def resolve_hash_size(self, n):
        """
        Code length keeping the expected bucket occupancy at `bucket_size` for
        `n` points: ``b = ceil(log2(n / c))``.

        This is the ``b ~ log n`` growth that turns the ``L * sum_b m_b^2``
        candidate work into O(n L c) instead of Theta(n^2).
        """
        b = int(math.ceil(math.log2(max(2, n) / max(1.0, float(self.bucket_size)))))
        return max(1, min(b, self.input_dim, MAX_PACKED_BITS))

    # ------------------------------------------------------------------ #
    # building
    # ------------------------------------------------------------------ #

    def build(self, points):
        """
        Hash `points` and group them into buckets.

        The codes are kept on the index so that nothing downstream has to
        re-hash the indexed set.

        :param points: (N, input_dim)
        :returns: (N, num_tables) array of bucket codes
        """
        points = np.asarray(points)
        n = points.shape[0]

        if self.auto_hash_size:
            b = self.resolve_hash_size(n)
            if b != self.hash_size or self.hasher is None:
                self.hash_size = b
                self.hasher = self._make_hasher()
        elif self.hasher is None:
            self.hasher = self._make_hasher()

        codes = self.hasher.codes(points)
        self.tables = [self._group(codes[:, i], n) for i in range(self.num_tables)]

        self.codes = codes
        self.points = points
        self.n_points = n
        return codes

    @staticmethod
    def _group(column, n):
        """
        Group point indices by bucket code with one sort.

        A stable sort leaves each bucket's member list in ascending index
        order, which the candidate scan relies on.
        """
        if n == 0:
            return {}
        order = np.argsort(column, kind="stable")
        ordered = column[order]
        cuts = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [n]))
        keys = ordered[starts].tolist()
        return {key: order[s:e] for key, s, e in zip(keys, starts, ends)}

    def encode(self, points):
        """Bucket codes for arbitrary points, without touching the tables."""
        if self.hasher is None:
            raise RuntimeError("No projection planes yet - call index_batch first.")
        return self.hasher.codes(points)

    def codes_for(self, points):
        """
        Codes for `points`, reusing the cached ones when `points` is the very
        array that was indexed (the common case) so nothing is re-hashed.
        """
        if self.codes is None:
            raise RuntimeError("Call index_batch(...) before building the graph.")
        if points is None or points is self.points:
            return self.codes
        return self.encode(np.asarray(points))

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #

    def collision_stats(self):
        """
        Realised candidate work of the index.

        `candidates_per_query` is ``(1/N) * L * sum_b m_b^2``, the quantity
        that decides whether a build is linear or quadratic in N; compare it
        across N to check that it stays flat (it does with an automatic hash
        size, it grows like N at a fixed one).
        """
        n = self.n_points
        second_moments = []
        bucket_counts = []
        for table in self.tables:
            sizes = np.fromiter((b.size for b in table.values()),
                                dtype=np.int64, count=len(table))
            second_moments.append(int((sizes.astype(np.int64) ** 2).sum()))
            bucket_counts.append(len(table))
        total = float(sum(second_moments))
        return {
            "n": n,
            "hash_size": self.hash_size,
            "num_hashtables": self.num_tables,
            "buckets_per_table": bucket_counts,
            "mean_bucket_size": (n / bucket_counts[0]) if bucket_counts and bucket_counts[0] else 0.0,
            "second_moment_per_table": second_moments,
            "candidates_per_query": (total / n) if n else 0.0,
            "total_candidate_work": total,
        }
