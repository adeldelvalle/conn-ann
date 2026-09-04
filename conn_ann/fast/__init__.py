"""Compiled multi-probe LSH retrieval.

The pipeline is: hash the query into L tables, probe each table's bucket and a
few near-miss buckets, count how many tables put each candidate in a probed
bucket, keep the top C by that count, and rank those C by true distance.

Only the counting stage is compiled - it is ~91% of query time in the numpy
version, because it is L x n_probes interpreter iterations of binary search plus
a scattered increment.  Projection and the final rerank stay in numpy, where
they are already BLAS calls.

    index = FastLSH(hash_size=10, num_tables=160, pca_dim=32).fit(X)
    idx, dist = index.search(queries, k=10, n_candidates=256)
"""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np

from ._vote import vote, vote_direct

__all__ = ["FastLSH", "submask_table", "principal_axes"]


def principal_axes(Xc, n_components, sample=6000):
    """Top `n_components` principal directions of already-centred data.

    Split out so a sweep can compute it once and pass it to many `fit` calls -
    the eigendecomposition is O(d^3) and dominates fitting at large d.
    """
    cov = np.cov(np.asarray(Xc[:min(sample, len(Xc))], dtype=np.float64).T)
    w, V = np.linalg.eigh(cov)
    return np.ascontiguousarray(V[:, np.argsort(-w)[:n_components]], dtype=np.float32)


def submask_table(p, r):
    """Probe patterns: every subset, up to size `r`, of the `p` nearest bits.

    Returned as bitmasks over rank positions, so the kernel can XOR them against
    whichever bits a given query ranked closest to their hyperplane.  Position 0
    is the bit whose hyperplane the query sits nearest, so these are ordered by
    how likely the bit was to have flipped.
    """
    masks = [0]
    for size in range(1, r + 1):
        for comb in combinations(range(p), size):
            m = 0
            for j in comb:
                m |= 1 << j
            masks.append(m)
    return np.asarray(masks, dtype=np.int32)


class FastLSH(object):
    """
    Multi-probe random-projection LSH with cross-table vote ranking.

    :param hash_size: bits per table (`b`).  Must be <= 62.
    :param num_tables: number of independent tables (`L`).  More tables mean a
        finer vote, at a cost linear in query time.
    :param pca_dim: project onto this many principal directions before hashing.
        Random hyperplanes otherwise spend bits on directions the data does not
        occupy; set it near the intrinsic dimension (a small multiple of it).
        None disables the projection.
    :param probe_bits: how many of the nearest-hyperplane bits are candidates
        for flipping (`p`).
    :param probe_radius: how many of them may flip at once (`r`).
    :param random_state: seed for the projection planes.
    :param direct_buckets: build a 2**hash_size prefix-sum table per table so a
        bucket lookup is an array index instead of a binary search.  Costs
        ``2**hash_size * num_tables * 4`` bytes; disabled automatically past
        `max_table_bytes`.
    :param max_table_bytes: memory ceiling for that table.
    :param rerank_pca: shortlist with cheap PCA-space distances before computing
        full-dimension ones.  An orthogonal projection can only shorten a
        difference vector, so the PCA distance is a true *lower bound* on the
        real one - but a bound is only useful when it is tight.  It is free when
        `pca_dim` retains most of the variance (on 3072-d images, 32 components
        hold ~80% and cost no recall at all); it is not free when the projection
        is aggressive relative to the data's own dimensionality.  Set it to
        roughly half `n_candidates` and check recall before going lower.  0
        disables it.
    """

    def __init__(self, hash_size=10, num_tables=160, pca_dim=32,
                 probe_bits=4, probe_radius=3, random_state=0,
                 direct_buckets=True, max_table_bytes=64 << 20, rerank_pca=0):
        if hash_size > 62:
            raise ValueError("hash_size must be <= 62 to pack into an int64")
        self.hash_size = int(hash_size)
        self.num_tables = int(num_tables)
        self.pca_dim = pca_dim
        self.probe_bits = int(probe_bits)
        self.probe_radius = int(probe_radius)
        self.random_state = random_state
        self.direct_buckets = direct_buckets
        self.max_table_bytes = int(max_table_bytes)
        self.rerank_pca = int(rerank_pca)
        self._submasks = submask_table(self.probe_bits, self.probe_radius)

    # ------------------------------------------------------------------ #

    @property
    def n_probes(self):
        """Buckets consulted per table, including the exact one."""
        return int(self._submasks.shape[0])

    def fit(self, X, projection=None):
        """Center, project, hash, and group each table's codes.

        :param projection: a precomputed (d, pca_dim) column-orthonormal matrix,
            reused instead of running the eigendecomposition.  Fitting many
            configurations over the same data otherwise repeats an O(d^3)
            decomposition per fit, which dominates everything else at large d.
            `principal_axes` computes one you can share.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.n_, d = X.shape
        self.mean_ = X.mean(0)
        Xc = X - self.mean_

        if projection is not None:
            self.projection_ = np.ascontiguousarray(projection, dtype=np.float32)
            self.pca_dim = self.projection_.shape[1]
            H = Xc @ self.projection_
        elif self.pca_dim:
            self.projection_ = principal_axes(Xc, self.pca_dim)
            H = Xc @ self.projection_
        else:
            self.projection_ = None
            H = Xc
        self.dim_ = H.shape[1]

        rng = np.random.RandomState(self.random_state)
        planes = rng.randn(self.num_tables * self.hash_size, self.dim_)
        planes /= np.linalg.norm(planes, axis=1, keepdims=True)
        self.planes_ = np.ascontiguousarray(planes, dtype=np.float32)

        H = np.ascontiguousarray(H, dtype=np.float32)
        codes = self._codes(H)
        self.codes_ = codes
        self.offsets_ = None
        self._acc = None
        self._codes_buf = codes
        self._hpoints_buf = H
        self._hsqnorm_buf = (H ** 2).sum(1)
        self._points_buf = np.ascontiguousarray(Xc, dtype=np.float32)
        self._sqnorm_buf = (self._points_buf ** 2).sum(1)
        self._rebind()
        self._group()

        return self

    def _sort_dtype(self):
        """Narrowest unsigned type that holds a code.

        numpy picks a radix sort for stable sorts of narrow integers and a
        comparison sort otherwise, so grouping 160 tables of int64 codes costs
        ~157 ms where the same values as uint16 cost ~11 ms. The codes only span
        2**hash_size, so the cast is lossless and the permutation identical.
        """
        n_codes = 1 << self.hash_size
        if n_codes <= (1 << 16):
            return np.uint16
        if n_codes <= (1 << 32):
            return np.uint32
        return np.int64

    def _group(self):
        """(Re)build every table from `codes_`.

        Grouping is a stable sort per table plus a prefix sum over codes, so
        `add` and `remove` regroup wholesale rather than splicing buckets.
        """
        n = self.n_
        n_codes = 1 << self.hash_size
        sdt = self._sort_dtype()
        self.order_ = np.empty((self.num_tables, n), dtype=np.int32)
        use_direct = self.direct_buckets and n_codes * self.num_tables * 4 <= self.max_table_bytes
        self.offsets_ = np.empty((self.num_tables, n_codes + 1), dtype=np.int32) if use_direct else None
        if not use_direct:
            self.scodes_ = np.empty((self.num_tables, n), dtype=np.int64)
        for l in range(self.num_tables):
            col = self.codes_[:n, l]
            o = np.argsort(col.astype(sdt, copy=False), kind="stable")
            self.order_[l] = o.astype(np.int32)
            if self.offsets_ is not None:
                counts = np.bincount(col.astype(np.int64, copy=False), minlength=n_codes)
                self.offsets_[l, 0] = 0
                np.cumsum(counts, out=self.offsets_[l, 1:])
            else:
                self.scodes_[l] = col[o]
        if self._acc is None or self._acc.shape[0] != n:
            self._acc = np.zeros(n, dtype=np.int16)
        self._hist = np.zeros(self.num_tables + 2, dtype=np.int32)

    def add(self, X):
        """Insert points into an existing index.

        Hash the newcomers and regroup. The projection and the hyperplanes are
        *not* recomputed - doing so would invalidate every code already stored -
        so an index whose data drifts far from its fitted basis should be refitted.

        :returns: the ids assigned to the new points.
        """
        X = np.atleast_2d(np.ascontiguousarray(X, dtype=np.float32))
        Xc = X - self.mean_
        H = np.ascontiguousarray(Xc @ self.projection_ if self.projection_ is not None else Xc,
                                 dtype=np.float32)
        first = self.n_
        m = X.shape[0]
        self._reserve(first + m)
        self._codes_buf[first:first + m] = self._codes(H)
        self._points_buf[first:first + m] = Xc
        self._sqnorm_buf[first:first + m] = (Xc ** 2).sum(1)
        self._hpoints_buf[first:first + m] = H
        self._hsqnorm_buf[first:first + m] = (H ** 2).sum(1)
        self.n_ = first + m
        self._rebind()
        self._group()
        return np.arange(first, self.n_)

    def _reserve(self, need):
        """Grow the backing arrays geometrically, so repeated `add` is amortised
        O(1) copies rather than one full copy of the point matrix per call."""
        cap = self._points_buf.shape[0]
        if need <= cap:
            return
        new = max(need, cap * 2, 1024)
        def grow(a, fill_shape):
            b = np.zeros(fill_shape, dtype=a.dtype)
            b[:self.n_] = a[:self.n_]
            return b
        d = self._points_buf.shape[1]
        hd = self._hpoints_buf.shape[1]
        self._codes_buf = grow(self._codes_buf, (new, self.num_tables))
        self._points_buf = grow(self._points_buf, (new, d))
        self._sqnorm_buf = grow(self._sqnorm_buf, (new,))
        self._hpoints_buf = grow(self._hpoints_buf, (new, hd))
        self._hsqnorm_buf = grow(self._hsqnorm_buf, (new,))
        self._acc = None

    def _rebind(self):
        """Public views over the live prefix of the backing arrays."""
        n = self.n_
        self.codes_ = self._codes_buf[:n]
        self.points_ = self._points_buf[:n]
        self.sqnorm_ = self._sqnorm_buf[:n]
        self.hpoints_ = self._hpoints_buf[:n]
        self.hsqnorm_ = self._hsqnorm_buf[:n]

    def remove(self, ids):
        """Delete points by id.

        Nothing is tombstoned: the points leave the tables entirely, so queries
        never spend a distance on a deleted point and recall does not decay with
        churn. Remaining ids are renumbered - `add` returns ids, so hold onto
        your own mapping if you need stable external keys.

        :returns: the number of points removed.
        """
        ids = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        keep = np.ones(self.n_, dtype=bool)
        keep[ids] = False
        removed = int(self.n_ - keep.sum())
        n = self.n_ - removed
        for name in ("_codes_buf", "_points_buf", "_sqnorm_buf", "_hpoints_buf", "_hsqnorm_buf"):
            buf = getattr(self, name)
            buf[:n] = buf[:self.n_][keep]
        self.n_ = n
        self._acc = None
        self._rebind()
        self._group()
        return removed

    def _project(self, X):
        Xc = np.ascontiguousarray(X, dtype=np.float32) - self.mean_
        return np.ascontiguousarray(Xc @ self.projection_ if self.projection_ is not None else Xc,
                                    dtype=np.float32)

    def _codes(self, H):
        bits = (H @ self.planes_.T) > 0
        w = (1 << np.arange(self.hash_size, dtype=np.int64))
        return (bits.reshape(H.shape[0], self.num_tables, self.hash_size).astype(np.int64) @ w)

    # ------------------------------------------------------------------ #

    def candidates(self, q, n_candidates=256):
        """
        Top-`n_candidates` by cross-table vote, without computing any distance.

        :returns: (indices, vote counts), highest-voted first is *not* guaranteed
            - the selection is by vote level, ties broken by index.
        """
        h = self._project(q.reshape(1, -1))[0]
        z = np.ascontiguousarray((self.planes_ @ h).reshape(self.num_tables, self.hash_size))
        out_idx = np.empty(n_candidates, dtype=np.int32)
        out_cnt = np.empty(n_candidates, dtype=np.int16)
        if self.offsets_ is not None:
            n = vote_direct(z, self.offsets_, self.order_, self._submasks, self.probe_bits,
                            self._acc, self._hist, out_idx, out_cnt, n_candidates)
        else:
            n = vote(z, self.scodes_, self.order_, self._submasks, self.probe_bits,
                     self._acc, self._hist, out_idx, out_cnt, n_candidates)
        return out_idx[:n], out_cnt[:n]

    def _scratch(self):
        return (np.zeros(self.n_, dtype=np.int16),
                np.zeros(self.num_tables + 2, dtype=np.int32))

    def _search_range(self, queries, Qc, Hq, lo, hi, k, n_candidates, idx, dst, scratch):
        """Search queries[lo:hi] using a private accumulator.

        The kernel runs with the GIL released, so worker threads genuinely
        overlap; each needs its own accumulator because voting scatters writes
        across the whole array.
        """
        acc, hist = scratch
        out_idx = np.empty(n_candidates, dtype=np.int32)
        out_cnt = np.empty(n_candidates, dtype=np.int16)
        for r in range(lo, hi):
            h = (queries[r] - self.mean_) @ self.projection_ if self.projection_ is not None \
                else (queries[r] - self.mean_)
            z = np.ascontiguousarray(
                (self.planes_ @ np.ascontiguousarray(h, dtype=np.float32)).reshape(
                    self.num_tables, self.hash_size))
            if self.offsets_ is not None:
                n = vote_direct(z, self.offsets_, self.order_, self._submasks,
                                self.probe_bits, acc, hist, out_idx, out_cnt, n_candidates)
            else:
                n = vote(z, self.scodes_, self.order_, self._submasks,
                         self.probe_bits, acc, hist, out_idx, out_cnt, n_candidates)
            if n == 0:
                continue
            c = out_idx[:n].astype(np.int64)
            if self.rerank_pca and c.size > self.rerank_pca:
                lo_d = self.hsqnorm_[c] - 2.0 * (self.hpoints_[c] @ Hq[r]) + float(Hq[r] @ Hq[r])
                c = c[np.argpartition(lo_d, self.rerank_pca - 1)[:self.rerank_pca]]
            d = self.sqnorm_[c] - 2.0 * (self.points_[c] @ Qc[r]) + float(Qc[r] @ Qc[r])
            take = min(k, c.size)
            sel = np.argpartition(d, take - 1)[:take] if c.size > take else np.arange(c.size)
            sel = sel[np.argsort(d[sel])]
            idx[r, :take] = c[sel]
            dst[r, :take] = d[sel]

    def search_parallel(self, queries, k=10, n_candidates=256, n_threads=4):
        """Thread-parallel batch search.  Queries are independent, so this scales
        with cores up to memory bandwidth."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        nq = queries.shape[0]
        idx = np.full((nq, k), -1, dtype=np.int64)
        dst = np.full((nq, k), np.inf, dtype=np.float32)
        Qc = np.ascontiguousarray(queries - self.mean_, dtype=np.float32)
        Hq = Qc @ self.projection_ if self.projection_ is not None else Qc
        bounds = np.linspace(0, nq, n_threads + 1).astype(int)
        pool = [self._scratch() for _ in range(n_threads)]
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(lambda t: self._search_range(queries, Qc, Hq, bounds[t], bounds[t + 1],
                                                     k, n_candidates, idx, dst, pool[t]),
                        range(n_threads)))
        return idx, dst

    def search(self, queries, k=10, n_candidates=256):
        """
        Nearest neighbours by exact distance over the voted shortlist.

        :returns: (indices, squared distances), each (n_queries, k), padded with
            -1 / inf when a query turned up fewer than k candidates.
        """
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        nq = queries.shape[0]
        idx = np.full((nq, k), -1, dtype=np.int64)
        dst = np.full((nq, k), np.inf, dtype=np.float32)
        Qc = np.ascontiguousarray(queries - self.mean_, dtype=np.float32)
        Hq = Qc @ self.projection_ if self.projection_ is not None else Qc
        for r in range(nq):
            cand, _ = self.candidates(queries[r], n_candidates)
            if cand.size == 0:
                continue
            c = cand.astype(np.int64)
            if self.rerank_pca and c.size > self.rerank_pca:
                # PCA distance under-estimates the true distance, so ranking by it
                # and keeping a margin is a sound way to shrink the exact pass
                lo = self.hsqnorm_[c] - 2.0 * (self.hpoints_[c] @ Hq[r]) + float(Hq[r] @ Hq[r])
                c = c[np.argpartition(lo, self.rerank_pca - 1)[:self.rerank_pca]]
            d = self.sqnorm_[c] - 2.0 * (self.points_[c] @ Qc[r]) + float(Qc[r] @ Qc[r])
            take = min(k, c.size)
            sel = np.argpartition(d, take - 1)[:take] if c.size > take else np.arange(c.size)
            sel = sel[np.argsort(d[sel])]
            idx[r, :take] = c[sel]
            dst[r, :take] = d[sel]
        return idx, dst
