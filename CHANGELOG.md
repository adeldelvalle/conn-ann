# Changelog

## 0.1.0 — preliminary release

First public version.

**Added**
- `conn_ann.fast` — compiled multi-probe retrieval. Cython voting kernel
  (bucket codes, query-directed probe enumeration, bucket lookup, counting-based
  top-C selection), `FastLSH` wrapper, thread-parallel batch search.
- `conn_ann.search.GraphSearcher` — beam search over a constructed graph.
- `principal_axes()` and `FastLSH.fit(projection=...)`, so a sweep can share one
  eigendecomposition instead of repeating an O(d^3) decomposition per fit.
- `direct_buckets` — 2**hash_size prefix-sum table replacing the binary search.
- `rerank_pca` — PCA-space lower-bound prefilter ahead of exact distances.
- `FastLSH.add()` / `.remove()` — insert and delete against a live index. Deletion
  is real rather than tombstoned, so recall does not decay with churn; faiss HNSW
  cannot delete at all. CIFAR 20k x 3072: insert 1 point 26 ms vs HNSW's 138 ms,
  1,000 points 34 ms vs 1,165 ms; delete 100 points 82 ms.
- Test suite (22 tests), including numpy-vs-Cython equivalence of the vote counts
  and the losslessness of the narrow-dtype grouping.

**Changed**
- Graph construction rewritten as two passes (directed scores, then reconciliation),
  removing an order dependence in the edge weights.
- Vote weight is the accumulated surprisal itself; the previous `exp(v/k)` transform
  squeezed weights into a ~5% band that modularity optimisers read as unweighted.
- `mutual=True` is the default: an edge survives only if both endpoints select each
  other. Worth +0.148 ARI on centred MNIST.
- Removed the quasi-orthogonality rejection sampler from plane generation. It never
  fires above d≈32 (measured: zero rejections at d=64) and only cost time.

**Performance**
- Table grouping casts codes to the narrowest unsigned type that holds them, which
  makes numpy choose a radix sort over a comparison sort: 157 ms -> 11 ms for 160
  tables of 20,000 points. Speeds up `fit`, `add` and `remove` alike.
- Backing arrays grow geometrically, so repeated `add` no longer copies the whole
  point matrix per call.

**Known issues**
- Hyperparameters do not transfer between datasets; sweep them. Reusing one dataset's
  settings on another cost up to 5x in measured latency.
- `conn_ann.LSHIndex` (the graph side) still has no incremental insertion; only
  `FastLSH` can be mutated in place.
- `add` does not refit the projection or hyperplanes, and `remove` renumbers ids.
- `mutual=True` returns no edges for queries outside the indexed set.
