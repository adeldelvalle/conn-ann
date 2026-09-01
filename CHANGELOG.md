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
- Test suite (17 tests), including numpy-vs-Cython equivalence of the vote counts.

**Changed**
- Graph construction rewritten as two passes (directed scores, then reconciliation),
  removing an order dependence in the edge weights.
- Vote weight is the accumulated surprisal itself; the previous `exp(v/k)` transform
  squeezed weights into a ~5% band that modularity optimisers read as unweighted.
- `mutual=True` is the default: an edge survives only if both endpoints select each
  other. Worth +0.148 ARI on centred MNIST.
- Removed the quasi-orthogonality rejection sampler from plane generation. It never
  fires above d≈32 (measured: zero rejections at d=64) and only cost time.

**Known issues**
- Hyperparameters do not transfer between datasets; sweep them. Reusing one dataset's
  settings on another cost up to 5x in measured latency.
- No incremental insertion; `build()`/`fit()` rebuild from scratch.
- `mutual=True` returns no edges for queries outside the indexed set.
