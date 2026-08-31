# CoNN - Co-ocurrence Nearest Neighbours

Locality-sensitive hashing that treats bucket collisions as **votes**: two points that
keep landing in the same bucket across many independent hash tables are probably
neighbours, and how *surprising* each collision was says how much to believe it.

Two things are built on that idea:

- **`spectral_lsh`** — a weighted neighbour graph for community detection
  (Louvain / Leiden). Construction computes no distances at all.
- **`spectral_lsh_fast`** — a compiled multi-probe retrieval path: vote, shortlist,
  then rank the shortlist by true distance.

```python
from spectral_lsh import LSHIndex, NeighborGraphBuilder

index = LSHIndex("auto", input_dim=784, num_tables=40, bucket_size=64).build(X)
edges = NeighborGraphBuilder(index).build(k=60)      # [(i, j, weight), ...]
```

```python
from spectral_lsh_fast import FastLSH

index = FastLSH(num_tables=160, pca_dim=32).fit(X)
neighbours, distances = index.search(queries, k=10, n_candidates=512)
```

---

## Install

```bash
git clone <your-remote> spectral-lsh
cd spectral-lsh
pip install -e .                     # builds the Cython extension
pip install -e ".[clustering,bench,dev]"   # igraph/leidenalg, faiss, pytest
```

Requires a C compiler and Cython (both pulled in by the build). Core runtime
dependency is numpy alone. For development without installing:

```bash
python setup.py build_ext --inplace
python -m pytest tests/
```

---

## Should you use this?

Be honest with yourself about the answer. Measured against `faiss` HNSW, both
single-threaded, k=10, on five datasets:

| dataset | d | intrinsic dim | ours | HNSW | verdict |
|---|---|---|---|---|---|
| **LFW faces** | 2914 | 20.2 | 0.988 @ 0.34 ms | 0.986 @ 0.26 ms | **competitive** |
| CIFAR-10 | 3072 | 19.5 | 0.933 @ 0.33 ms | 0.935 @ 0.24 ms | close |
| Fashion-MNIST | 784 | 10.7 | 0.995 @ 0.34 ms | 1.000 @ 0.07 ms | 5× behind |
| GloVe-6B-100d | 100 | 22.1 | 0.993 @ 1.05 ms | 0.991 @ 0.23 ms | 4.5× behind |
| covtype | 54 | 2.8 | 1.000 @ 0.23 ms | 1.000 @ 0.01 ms | 40× behind |

**Use it when a single distance evaluation is expensive** — vectors in the
thousands of dimensions, costly metrics, or distances fetched over a network.
The voting stage has a fixed cost (`L × probes × bucket size` memory touches)
that it pays before computing a single distance, and that only pays for itself
when the distances it avoids are dear. On 100-d embeddings it never is.

Two things it wins at regardless:

- **Build time.** 2 s versus 46 s for HNSW on GloVe-100k — 23× faster, consistently.
- **Clustering.** The graph is the point; retrieval latency is irrelevant there.

For general-purpose in-memory ANN on 100–1536-d embeddings, use HNSW.

---

## The graph: `spectral_lsh`

Every point throws a *star* — it looks up its bucket in each of `L` tables, tallies a
weighted vote for everything it collides with, and keeps its top `k`. The stars are
then reconciled into an undirected graph.

**Votes.** Under the null that a point lands uniformly at random among `n` points, it
falls in bucket `b` with probability `p_b = m_b / n`, so a collision there carries
`log(n / m_b)` nats. Independent tables mean evidence adds, so the accumulated sum is a
log-likelihood ratio against the null — and it *is* the weight. It is already on a log
scale; exponentiating it (as an earlier version did) collapses the dynamic range and
modularity optimisers then read the graph as unweighted.

**Mutuality.** By default an edge survives only if *both* points put each other in
their top `k`. In high dimensions a point in a dense region lands in everyone's top-k
while its own top-k goes elsewhere; those one-sided hub edges smear communities
together. On MNIST-25k the worst hub had degree 249 in the union graph and 60 under
mutuality.

### Weighting schemes

Selectable per call, so ablations are one loop. MNIST-25k, `b=10`, `L=40`, `k=60`,
Leiden/modularity, **centred** data:

| `weighting` | ARI | NMI |
|---|---|---|
| `surprisal` — `log(n / m_b)`, the default | 0.642 | 0.747 |
| `inv_log` — `1.5 / log(m_b + 1)` | 0.640 | 0.746 |
| `legacy` — that kernel through `exp(v/k)`, kept for reproducing old numbers | 0.636 | 0.741 |
| `uniform` — `1.0`, a plain count of shared tables | 0.642 | 0.746 |

**On centred data the weighting scheme does not matter** — the spread is 0.006, which
is noise. It only appeared to matter on *un-centred* data (spread 0.041), where bucket
occupancies vary ~8× and `log(n / m_b)` has something to discriminate on. Centre your
data and the surprisal weight degenerates into a scaled collision count.

**Mutuality does matter, and more so after centring**: 0.642 vs 0.494 for `mutual=False`
(+0.148, against +0.064 un-centred).

Register your own scheme with a `VoteWeighting` subclass and `register(...)`; nothing
in `graph.py` changes.

### Choosing parameters

**Centre your data.** This is not optional. Random-hyperplane LSH is a *directional*
hash, so an off-origin mean dominates every projection. On raw CIFAR only **138 of 1024**
buckets were ever occupied; after centring, **1008**. Every result above assumes it.

**`hash_size="auto"`** ties the code length to `ceil(log2(N / bucket_size))`. Candidate
work is `L · Σ_b m_b²`, so at a fixed code length it is Θ(N²) and only a code length
growing like `log N` holds it to `O(N·L·c)`. Verified: as N grows 4×, candidate work
grows 1.9× under `"auto"` against 4.0× at a fixed code length.

**`bucket_size`** — target occupancy under `"auto"`. Keep it above `k`.

**`mutual`** — `True` for clustering, `False` for search. They want different graphs:
the union graph clusters worse (ARI 0.583 vs 0.647) but searches better, because higher
degree makes it more navigable.

Use `index.collision_stats()["candidates_per_query"]` to check the real second moment on
your data rather than assuming uniform occupancy.

---

## Retrieval: `spectral_lsh_fast`

```python
from spectral_lsh_fast import FastLSH

index = FastLSH(
    hash_size=10,        # bits per table
    num_tables=160,      # more tables -> finer vote, linear cost
    pca_dim=32,          # project before hashing; None to disable
    probe_bits=4,        # how many near-hyperplane bits may flip
    probe_radius=3,      # how many may flip at once  -> 15 buckets/table
    rerank_pca=128,      # cheap PCA shortlist before exact distances
).fit(X)

idx, dist = index.search(queries, k=10, n_candidates=512)
idx, dist = index.search_parallel(queries, k=10, n_candidates=512, n_threads=4)
cand, votes = index.candidates(q, n_candidates=256)   # no distances computed
```

**Multi-probe.** Each table contributes at most one vote, no matter how many of its
buckets you open — a point holds one code per table and the probe codes are distinct.
Probing raises the *chance* of a vote per table; it cannot multiply votes within one.
Probes are query-directed: the bits flipped are those whose hyperplane the query sits
nearest, ordered by `|w·x|`.

**Optimisations, and what each was worth** (CIFAR, single-query):

| | ms | note |
|---|---|---|
| numpy reference | 8.89 | ~91% of it in the voting loop |
| Cython kernel | 1.29 | 6.9× — removes 2,400 interpreter iterations per query |
| + direct bucket table | 1.27 | 1.08× only; the sorted columns were already cache-hot |
| + PCA prefilter → 128 | 0.49 | **2.8×**, at zero recall cost on this data |
| 4 threads | 0.21 | 2.15×; both we and HNSW plateau here — memory bandwidth |

`direct_buckets` replaces the binary search with a `2**hash_size` prefix-sum table
(costs `2**hash_size · num_tables · 4` bytes, auto-disabled past `max_table_bytes`).

`rerank_pca` ranks candidates by PCA-space distance first. That distance *lower-bounds*
the true one, since an orthogonal projection can only shorten a difference vector — but
a bound must also be **tight** to be safe. Free when `pca_dim` retains most of the
variance; on 40-d data truncated to 16 it costs 0.16 recall. Set it near half
`n_candidates` and check.

### A caveat on `pca_dim`

An earlier version of this document suggested `pca_dim ≈ 2 × intrinsic_dim`. **That rule
does not generalise** — it was fitted on CIFAR and MNIST, where ambient dimension vastly
exceeds intrinsic. On covtype (54-d) it gives 0.904 recall where `pca_dim=32` gives
1.000; on GloVe-100 it gives 0.977 where no truncation gives 0.993. Truncation helps in
proportion to how much variance your data hides in directions that carry no
neighbourhood structure — large for raw pixels, near zero for a trained embedding.
Sweep it.

---

## Layout

```
spectral_lsh/          pure Python
  hashing.py           RandomProjectionHasher — points -> bucket codes
  index.py             LSHIndex — codes -> tables, collision statistics
  weighting.py         VoteWeighting + four schemes, registry
  graph.py             NeighborGraphBuilder, EdgeStats
  search.py            GraphSearcher — beam search over the built graph
  lshash.py            LSHash, a flat backwards-compatible facade
spectral_lsh_fast/     compiled
  _vote.pyx            voting kernel: codes, probes, bucket lookup, top-C
  __init__.py          FastLSH
tests/test_core.py     17 tests, including numpy-vs-C equivalence
```

---

## Status and limitations

Research code; the API is not stable. Known bounds:

- **Not a k-NN graph approximation.** Only ~21% of graph edges are true Euclidean
  60-NN pairs, though 76.5% are same-class (true 60-NN: 86.7%). Good enough for
  community detection, which is the design target.
- **Not incremental.** `build()` / `fit()` rebuild from scratch; there is no insert.
- **`mutual=True` returns nothing for un-indexed queries** — nothing reciprocates a
  stranger. Use `mutual=False` for out-of-index query sets.
- **Collision count shortlists well and ranks badly.** True neighbours score far above
  the crowd (32–57 votes against a median of 6) but not above each other — the
  highest-voted candidate for one CIFAR query was its 50th nearest. Always rerank by
  distance.
