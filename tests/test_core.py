"""Invariants that must not regress.

The compiled voting kernel is the risky part: it reimplements in C what numpy
does with `searchsorted` + `bincount`, and a subtle selection bug there is
invisible except as slightly worse recall.  These tests pin it down.
"""
from itertools import combinations

import numpy as np
import pytest

from spectral_lsh import LSHIndex, NeighborGraphBuilder, WEIGHTINGS, get_weighting
from spectral_lsh_fast import FastLSH, submask_table


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(12, 40)) * 3
    labels = rng.integers(0, 12, 3000)
    return (centres[labels] + rng.normal(size=(3000, 40))).astype(np.float32)


# --------------------------------------------------------------------- #
# compiled kernel
# --------------------------------------------------------------------- #

def _numpy_votes(f, q):
    """Reference implementation of the voting stage, in numpy."""
    h = (q - f.mean_) @ f.projection_ if f.projection_ is not None else (q - f.mean_)
    z = (f.planes_ @ h.astype(np.float32)).reshape(f.num_tables, f.hash_size)
    acc = np.zeros(f.n_, np.int32)
    weights = 1 << np.arange(f.hash_size, dtype=np.int64)
    for l in range(f.num_tables):
        c0 = int((z[l] > 0).astype(np.int64) @ weights)
        rank = np.argsort(np.abs(z[l]))[:f.probe_bits]
        for mask in f._submasks:
            code = c0
            for k in range(f.probe_bits):
                if mask & (1 << k):
                    code ^= 1 << int(rank[k])
            col = f.order_[l]
            if f.offsets_ is not None:
                lo, hi = f.offsets_[l, code], f.offsets_[l, code + 1]
            else:
                lo = np.searchsorted(f.scodes_[l], code, "left")
                hi = np.searchsorted(f.scodes_[l], code, "right")
            if hi > lo:
                acc[col[lo:hi]] += 1
    return acc


@pytest.mark.parametrize("direct", [True, False])
def test_compiled_votes_match_numpy(blobs, direct):
    f = FastLSH(hash_size=8, num_tables=24, pca_dim=16, direct_buckets=direct).fit(blobs)
    for q in blobs[:5]:
        ref = _numpy_votes(f, q)
        idx, cnt = f.candidates(q, n_candidates=f.n_)
        got = np.zeros(f.n_, np.int32)
        got[idx] = cnt
        assert np.array_equal(ref, got)


def test_binary_search_and_direct_table_agree(blobs):
    a = FastLSH(hash_size=8, num_tables=24, pca_dim=16, direct_buckets=True).fit(blobs)
    b = FastLSH(hash_size=8, num_tables=24, pca_dim=16, direct_buckets=False).fit(blobs)
    assert a.offsets_ is not None and b.offsets_ is None
    for q in blobs[:5]:
        ia, ca = a.candidates(q, 128)
        ib, cb = b.candidates(q, 128)
        assert np.array_equal(np.sort(ia), np.sort(ib))
        assert ca.sum() == cb.sum()


def test_top_c_keeps_every_strictly_higher_vote(blobs):
    """The selection must not drop a high-voted candidate to make room for a tie.

    A single "vote >= threshold, first C by index" pass silently does exactly
    that, and costs real recall.
    """
    f = FastLSH(hash_size=8, num_tables=24, pca_dim=16).fit(blobs)
    for q in blobs[:5]:
        full = _numpy_votes(f, q)
        idx, cnt = f.candidates(q, n_candidates=64)
        assert idx.size <= 64
        floor = cnt.min()
        above = set(np.flatnonzero(full > floor).tolist())
        assert above <= set(idx.tolist())


def test_accumulator_is_left_clean(blobs):
    f = FastLSH(hash_size=8, num_tables=24, pca_dim=16).fit(blobs)
    for q in blobs[:20]:
        f.candidates(q, 64)
    assert f._acc.sum() == 0


def test_submask_table_shape():
    assert len(submask_table(4, 3)) == 1 + 4 + 6 + 4
    assert submask_table(0, 0).tolist() == [0]


# --------------------------------------------------------------------- #
# retrieval quality
# --------------------------------------------------------------------- #

def _recall(got, truth, k):
    return np.mean([len(set(a[a >= 0].tolist()) & set(b.tolist())) / k
                    for a, b in zip(got, truth)])


def test_search_finds_true_neighbours(blobs):
    base, queries = blobs[:2800], blobs[2800:]
    sq = (base ** 2).sum(1)
    truth = np.stack([np.argsort(sq - 2 * base @ q + q @ q)[:10] for q in queries])
    f = FastLSH(hash_size=8, num_tables=64, pca_dim=16, rerank_pca=0).fit(base)
    idx, dist = f.search(queries, k=10, n_candidates=256)
    assert _recall(idx, truth, 10) > 0.9
    assert np.all(np.diff(dist, axis=1) >= -1e-4)          # distances sorted


def test_pca_prefilter_is_cheap_when_the_bound_is_tight(blobs):
    """The PCA distance lower-bounds the true one, but only usefully when the
    retained subspace holds most of the variance.

    Truncating hard makes the bound loose, and a loose bound evicts real
    neighbours - so this asserts the safe regime, and the next test pins the
    unsafe one so the trade-off cannot be forgotten.
    """
    base, queries = blobs[:2800], blobs[2800:]
    sq = (base ** 2).sum(1)
    truth = np.stack([np.argsort(sq - 2 * base @ q + q @ q)[:10] for q in queries])
    plain = FastLSH(hash_size=8, num_tables=64, pca_dim=32).fit(base)
    pre = FastLSH(hash_size=8, num_tables=64, pca_dim=32, rerank_pca=64).fit(base)
    r0 = _recall(plain.search(queries, 10, 256)[0], truth, 10)
    r1 = _recall(pre.search(queries, 10, 256)[0], truth, 10)
    assert r1 >= r0 - 0.02


def test_aggressive_pca_prefilter_costs_recall(blobs):
    """Documents the failure mode: 16 of 40 dimensions is too tight a bound to
    prefilter 256 candidates down to 64 safely."""
    base, queries = blobs[:2800], blobs[2800:]
    sq = (base ** 2).sum(1)
    truth = np.stack([np.argsort(sq - 2 * base @ q + q @ q)[:10] for q in queries])
    tight = FastLSH(hash_size=8, num_tables=64, pca_dim=16, rerank_pca=64).fit(base)
    loose = FastLSH(hash_size=8, num_tables=64, pca_dim=16, rerank_pca=128).fit(base)
    assert _recall(tight.search(queries, 10, 256)[0], truth, 10) < 0.95
    assert _recall(loose.search(queries, 10, 256)[0], truth, 10) > 0.95


def test_parallel_matches_serial(blobs):
    base, queries = blobs[:2800], blobs[2800:]
    f = FastLSH(hash_size=8, num_tables=64, pca_dim=16).fit(base)
    a, _ = f.search(queries, k=10, n_candidates=256)
    b, _ = f.search_parallel(queries, k=10, n_candidates=256, n_threads=4)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------- #
# graph construction
# --------------------------------------------------------------------- #

def test_auto_hash_size_beats_a_fixed_code_length(blobs):
    """The point of hash_size="auto".

    Candidate work is ``L * sum_b m_b^2``, so at a fixed code length it grows
    linearly with N and construction is quadratic.  Tying the code length to
    log2(N) should make it grow far more slowly.  Asserted as a comparison
    rather than an absolute bound, because the second moment depends on how
    evenly the data happens to fill its buckets.
    """
    def growth(hash_size, **kw):
        seen = []
        for n in (750, 1500, 3000):                       # N grows 4x
            idx = LSHIndex(hash_size, blobs.shape[1], num_tables=8, **kw)
            idx.build(blobs[:n])
            seen.append(idx.collision_stats()["candidates_per_query"])
        return seen[-1] / seen[0]

    auto = growth("auto", bucket_size=32)
    fixed = growth(6, bucket_size=64)
    assert fixed > 3.0                    # a fixed code length tracks N
    assert auto < fixed / 1.5             # auto is markedly flatter


@pytest.mark.parametrize("weighting", WEIGHTINGS)
def test_every_weighting_builds_a_graph(blobs, weighting):
    idx = LSHIndex(8, blobs.shape[1], num_tables=12)
    idx.build(blobs)
    edges = NeighborGraphBuilder(idx, weighting=weighting).build(k=15)
    assert edges and all(i != j for i, j, _ in edges)
    assert len({frozenset((i, j)) for i, j, _ in edges}) == len(edges)   # one row per pair


def test_mutual_edges_are_a_subset_of_the_union(blobs):
    idx = LSHIndex(8, blobs.shape[1], num_tables=12)
    idx.build(blobs)
    b = NeighborGraphBuilder(idx)
    mutual = {(i, j) for i, j, _ in b.build(k=15, mutual=True)}
    union = {(i, j) for i, j, _ in b.build(k=15, mutual=False)}
    assert mutual <= union


def test_unknown_weighting_is_rejected(blobs):
    with pytest.raises(ValueError):
        get_weighting("nope")
