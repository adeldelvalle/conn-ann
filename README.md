# spectral-lsh

Locality-sensitive hashing that builds a **weighted neighbour graph from collision
statistics**, for community detection (Louvain / Leiden) and approximate
nearest-neighbour search.

The idea in one line: two points that keep landing in the same hash bucket are
probably neighbours, and *how surprising* each of those collisions was tells you
how much to believe it. Graph construction never computes a distance — only
search does.

```python
from spectral_lsh import LSHIndex, NeighborGraphBuilder

index = LSHIndex("auto", input_dim=784, num_tables=40, bucket_size=64)
index.build(X)                                    # (N, 784) float array

edges = NeighborGraphBuilder(index).build(k=60)   # [(i, j, weight), ...]
```

Feed `edges` straight to `networkx.Graph.add_weighted_edges_from` or
`igraph`/`leidenalg`.

---

## Install

Requires **numpy** and nothing else. From source:

```bash
git clone <your-remote> spectral-lsh
cd spectral-lsh
pip install numpy
```

There is no `pyproject.toml` yet, so `pip install -e .` and `pip install
spectral-lsh` do not work — import the package from the repository root for now.
The examples below additionally use `python-igraph` + `leidenalg` (clustering)
and `faiss` (benchmarking), neither of which the library itself needs.

---

## How it works

Every point throws a *star*: it looks up its bucket in each of the `L` tables,
tallies a weighted vote for everything it collides with, and keeps its top `k`
candidates. The stars are then reconciled into an undirected graph.

**1. Votes.** Each shared bucket contributes the *surprisal* of that collision.
Under the null model that a point lands uniformly at random among `n` indexed
points, it falls in bucket `b` with probability `p_b = m_b / n`, so a collision
there carries `-log p_b = log(n / m_b)` nats. Independent tables mean the
evidence adds, so the accumulated sum is a log-likelihood ratio for "these two
are neighbours". Rare buckets are the informative ones.

**2. Reconciliation.** By default (`mutual=True`) an edge survives only if *both*
points put each other in their top `k` — a mutual k-NN graph. This matters more
than the weighting: in high dimensions a point in a dense region lands in
everyone's top-k while its own top-k goes elsewhere, and those one-sided hub
edges smear communities together. On MNIST-25k the worst hub had degree 249 in
the union graph and 60 under mutuality.

**3. Weight.** The accumulated vote *is* the weight — it is already on a log
scale, so exponentiating it (as an earlier scheme did) collapses the dynamic
range and modularity optimisers read the graph as unweighted.

---

## API

Four layers; use whichever you need.

| module | class | role |
|---|---|---|
| `spectral_lsh.hashing` | `RandomProjectionHasher` | points → bucket codes |
| `spectral_lsh.index` | `LSHIndex` | codes → hash tables, collision statistics |
| `spectral_lsh.weighting` | `VoteWeighting` + schemes | what one shared bucket is worth |
| `spectral_lsh.graph` | `NeighborGraphBuilder`, `EdgeStats` | tables → weighted edges |
| `spectral_lsh.search` | `GraphSearcher` | ANN search over the built graph |

### Clustering

```python
import igraph as ig, leidenalg
from spectral_lsh import LSHIndex, NeighborGraphBuilder

index = LSHIndex("auto", X.shape[1], num_tables=40, bucket_size=64)
index.build(X)

builder = NeighborGraphBuilder(index, weighting="surprisal", mutual=True)
edges = builder.build(k=60)

g = ig.Graph(n=len(X), edges=[(i, j) for i, j, _ in edges])
g.es["weight"] = [w for _, _, w in edges]
labels = leidenalg.find_partition(
    g, leidenalg.ModularityVertexPartition, weights=g.es["weight"]
).membership
```

### Search

```python
from spectral_lsh import GraphSearcher

searcher = GraphSearcher(index, X, edges)
neighbours, sq_distances = searcher.search(queries, k=10, ef=32)
```

Search is two-stage: the query's **rarest** buckets supply seed nodes, then a
best-first walk over the graph refines them with real distances. `ef` is the beam
width — `ef=1` is plain greedy descent and performs badly (see below); use
`ef >= 10`.

### Diagnostics

```python
index.collision_stats()      # candidates_per_query, bucket second moments
builder.diagnostics()        # weight distribution + a sample of edges
searcher.n_distance_calls    # distance computations since last reset
```

---

## Weighting schemes

Selectable per call, so ablations are one loop. Measured on MNIST-25k,
`hash_size=10`, `L=40`, `k=60`, Leiden/modularity:

| `weighting` | `mutual` | edges | weight range | ARI | NMI |
|---|---|---|---|---|---|
| `surprisal` (default) | True | 465,794 | [11.0, 151.4] | **0.647** | **0.751** |
| `inv_log` | True | 467,774 | [0.72, 9.72] | 0.642 | 0.745 |
| `legacy` | True | 467,774 | [1.024, 1.383] | 0.623 | 0.735 |
| `uniform` | True | 378,326 | [2, 28] | 0.606 | 0.721 |
| `surprisal` | False | 1,034,206 | [8.86, 151.4] | 0.583 | 0.688 |
| `uniform` | False | 1,121,674 | [2, 28] | 0.526 | 0.644 |

- `surprisal` — `log(n / m_b)`, the default.
- `inv_log` — `1.5 / log(m_b + 1)`, additive.
- `legacy` — the same kernel through the historical `exp(v / k)` transform,
  squared on reciprocation. Kept for reproducing older numbers; note the
  collapsed weight range.
- `uniform` — `1.0`, a plain count of shared tables. The ablation that isolates
  topology from weighting.

Decomposition of the gains at `mutual=True`: the mutual filter is worth **+0.06
ARI**, dropping `exp` **+0.019**, any kernel over none **+0.036**, and the
surprisal kernel over `inv_log` **+0.005**. Register your own scheme with a
subclass of `VoteWeighting` and `register(...)`; nothing in `graph.py` changes.

---

## Choosing parameters

**`hash_size="auto"`** ties the code length to `ceil(log2(N / bucket_size))`.
This is what keeps construction subquadratic: the candidate work is
`L · Σ_b m_b²`, so at a *fixed* code length it is Θ(N²), and only a code length
growing like `log N` holds it to `O(N·L·c)`. Measured `candidates_per_query`
as N goes 2k → 16k:

| | 2,000 | 4,000 | 8,000 | 16,000 |
|---|---|---|---|---|
| fixed `hash_size=10` | 67 | 114 | 206 | 394 |
| `"auto"`, `bucket_size=128` | 2,579 | 2,644 | 2,676 | 2,740 |

Use `index.collision_stats()["candidates_per_query"]` to check this on your own
data rather than assuming uniform occupancy.

**`bucket_size`** — target expected occupancy under `"auto"`. Keep it
comfortably above `k`, or one table cannot supply `k` candidates.

**`num_tables` (L)** — more tables, more consensus, linear cost.

**`mutual`** — `True` for clustering, `False` for search. They genuinely want
different graphs: the union graph clusters worse (ARI 0.583 vs 0.647) but
searches better (recall 0.993 vs 0.971 at `ef=32`), because higher degree makes
it more navigable. You can build both from one index.

---

## Search benchmarks

MNIST, 784-d, 24k indexed, recall@10, union graph:

| ef | recall@10 | distances/query |
|---|---|---|
| 1 (plain greedy) | 0.094 | 211 |
| 10 | 0.953 | 638 |
| 32 | 0.993 | 1,292 |
| 128 | 1.000 | 2,994 |
| *rerank every bucket candidate* | 0.986 | 9,206 |
| *brute force* | 1.000 | 24,000 |

Graph navigation beats reranking the full candidate set on both axes — and
exceeds its recall ceiling, because the walk reaches points that never collided
with the query in any table.

Against **faiss HNSW** at recall ≥ 0.95, measuring distance computations
(implementation-independent):

| N | ours | HNSW |
|---|---|---|
| 2,000 | 409 | 222 |
| 8,000 | 492 | 226 |
| 24,000 | 773 | 347 |
| **log-log slope** | **0.225** | **0.212** |

Same scaling; a ~2.2× constant factor apart. HNSW gets its logarithmic search
from a hierarchy supplying long-range links — here the LSH bucket lookup plays
that role, teleporting the query into the right neighbourhood so the graph only
does local refinement.

---

## What this is not

- **Not a k-NN graph approximation.** Only ~21% of edges are true Euclidean
  60-NN. They are, however, 76.5% same-class (true 60-NN: 86.7%) — good enough
  for community detection, which is the design target.
- **Not incremental.** `index.build()` rebuilds from scratch; there is no insert.
- **Not C-speed.** Search is a Python heap loop at ~1 ms/query; `hnswlib` is
  20–50× faster in wall clock at equal recall. The *complexity* is competitive,
  the constant is not.
- **`mutual=True` yields no edges for un-indexed queries** — nothing reciprocates
  a stranger. Use `mutual=False` for out-of-index query sets.

---

## Compatibility

The old flat API still works:

```python
from spectral_lsh import LSHash

lsh = LSHash(10, 784, 40)
lsh.index_batch(X)
edges = lsh.find_topk_neighbors_with_weights(k=60)
```

Verified bit-identical to the pre-refactor implementation across 384
configurations (2 datasets × 4 hash sizes × 2 table counts × 3 values of k × 4
weightings × 2 mutuality settings), comparing planes, codes, tables, collision
statistics, edge lists under exact float equality, and diagnostics.

## Status

Research code, API not yet stable. Missing before this is packageable: a
`pyproject.toml`, a test suite (the verification harness lives outside the repo),
and CI.
