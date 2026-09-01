"""Building a weighted neighbour graph out of LSH bucket collisions."""
from __future__ import annotations

import numpy as np

from .weighting import get_weighting

#: Rows of the code matrix converted to Python lists at a time.  Converting the
#: whole matrix at once costs a lot of memory for large N; converting row by row
#: costs a call per point.
CODE_BLOCK = 4096


class EdgeStats(object):
    """
    Per-edge record of one graph build, and the summary drawn from it.

    :ivar src, dst: endpoints, one row per undirected edge.
    :ivar weight: the emitted weight.
    :ivar shared_tables: how many tables the pair collided in.
    :ivar surprisal: summed ``log(n / m_b)`` over those collisions, recorded
        under every weighting so the ablations stay comparable.
    :ivar mutual: whether both directions had each other in their top-k.
    """

    __slots__ = ("src", "dst", "weight", "shared_tables", "surprisal",
                 "mutual", "weighting", "mutual_only", "k")

    def __init__(self, src, dst, weight, shared_tables, surprisal, mutual,
                 weighting, mutual_only, k):
        self.src = src
        self.dst = dst
        self.weight = weight
        self.shared_tables = shared_tables
        self.surprisal = surprisal
        self.mutual = mutual
        self.weighting = weighting
        self.mutual_only = mutual_only
        self.k = k

    def __len__(self):
        return int(self.weight.size)

    def edges(self):
        """The build's return value: a list of ``(i, j, weight)`` tuples."""
        return list(zip(self.src.tolist(), self.dst.tolist(), self.weight.tolist()))

    def summary(self, sample=10, random_state=None):
        """
        Weight distribution plus a random sample of edges.

        Reports min/max/mean/std with median, 1st/99th percentiles and the
        max/min and p99/p01 ratios, so a scheme that collapses into a narrow
        band is visible at a glance rather than only in the clustering scores.
        """
        w = self.weight
        summary = {
            "n_edges": int(w.size),
            "weighting": self.weighting,
            "mutual_only": self.mutual_only,
            "k": self.k,
        }
        if w.size == 0:
            summary.update({"weight": {}, "shared_tables": {}, "surprisal": {},
                            "mutual_fraction": 0.0, "sample": []})
            return summary

        lo, hi = float(w.min()), float(w.max())
        p01, p99 = (float(x) for x in np.percentile(w, [1, 99]))
        summary["mutual_fraction"] = float(self.mutual.mean())
        summary["weight"] = {
            "min": lo, "max": hi,
            "mean": float(w.mean()), "std": float(w.std()),
            "median": float(np.median(w)), "p01": p01, "p99": p99,
            "max_over_min": (hi / lo) if lo > 0 else float("inf"),
            "p99_over_p01": (p99 / p01) if p01 > 0 else float("inf"),
            "n_zero": int((w == 0).sum()),
        }
        for field in ("shared_tables", "surprisal"):
            v = getattr(self, field)
            summary[field] = {"min": float(v.min()), "max": float(v.max()),
                              "mean": float(v.mean()), "std": float(v.std())}

        rng = np.random.default_rng(random_state)
        picks = np.sort(rng.choice(w.size, size=int(min(sample, w.size)), replace=False))
        summary["sample"] = [
            {"i": int(self.src[p]), "j": int(self.dst[p]),
             "shared_tables": int(self.shared_tables[p]),
             "surprisal": float(self.surprisal[p]),
             "weight": float(w[p]),
             "mutual": bool(self.mutual[p])}
            for p in picks
        ]
        return summary


class NeighborGraphBuilder(object):
    """
    Turns an `LSHIndex` into a weighted k-nearest-neighbour-by-consensus graph.

    Two passes.  The first accumulates *directed* scores - for each point, the
    weighted count of tables in which each candidate collided with it - and
    keeps that point's top `k`.  The second pairs each direction with its
    reverse and emits one undirected edge, so the result never depends on which
    direction happened to be processed first.

    The total candidate work is ``L * sum_b m_b^2`` (see `LSHIndex`): linear in
    N only when the code length grows like ``log N``.

    :param index: a built `LSHIndex`.
    :param weighting: scheme name or `VoteWeighting` instance.
    :param mutual: if True an edge survives only when i and j are in each
        other's top-k (a mutual k-NN graph); if False, one direction suffices.
    """

    def __init__(self, index, weighting="surprisal", mutual=True):
        self.index = index
        self.weighting = get_weighting(weighting).name
        self.mutual = bool(mutual)
        self.stats = None

    def build(self, k=70, points=None, weighting=None, mutual=None,
              collect_surprisal=True):
        """
        Build the graph and return its edges as ``(i, j, weight)`` tuples.

        :param k: neighbours retained per point, by score.
        :param points: only needed to query a set other than the indexed one;
            the codes cached by the index are reused otherwise.
        :param weighting: override the builder's scheme for this call.
        :param mutual: override the builder's mutuality requirement.
        :param collect_surprisal: accumulate the surprisal alongside the score
            so `EdgeStats` can report it under every scheme.  Turning it off
            saves one pass per point under `"inv_log"`, `"legacy"` and
            `"uniform"`, at the cost of the diagnostic.
        """
        index = self.index
        scheme = get_weighting(self.weighting if weighting is None else weighting)
        mutual = self.mutual if mutual is None else bool(mutual)

        codes = index.codes_for(points)
        n_query = codes.shape[0]
        n = index.n_points
        tables = [table.get for table in index.tables]
        n_tables = len(tables)

        increment = scheme.increment(n)
        want_surprisal = collect_surprisal and scheme.name != "surprisal"
        surprisal = get_weighting("surprisal").increment(n) if want_surprisal else None
        score_is_count = scheme.score_is_count

        counts_per_point = []
        dst_parts, score_parts, shared_parts, surp_parts = [], [], [], []
        empty = np.empty(0, dtype=np.int64)

        # ---- pass 1: directed scores, top-k per point -------------------- #
        for block_start in range(0, n_query, CODE_BLOCK):
            rows = codes[block_start:block_start + CODE_BLOCK].tolist()

            for offset, row in enumerate(rows):
                point_idx = block_start + offset

                buckets, sizes, incs, surps = [], [], [], []
                for table_idx in range(n_tables):
                    bucket = tables[table_idx](row[table_idx])
                    if bucket is None or bucket.size == 0:
                        continue
                    m = bucket.size
                    buckets.append(bucket)
                    sizes.append(m)
                    if not score_is_count:
                        incs.append(increment(m))
                    if want_surprisal:
                        surps.append(surprisal(m))

                if not buckets:
                    counts_per_point.append(0)
                    continue

                candidates = np.concatenate(buckets)
                sizes_arr = np.asarray(sizes, dtype=np.int64)
                uniq, inverse, shared = np.unique(candidates, return_inverse=True,
                                                  return_counts=True)

                if score_is_count:
                    # Every collision is worth 1.0, so the collision counts are
                    # already the score - exactly, not approximately.
                    score = shared.astype(np.float64)
                else:
                    score = np.bincount(
                        inverse,
                        weights=np.repeat(np.asarray(incs, dtype=np.float64), sizes_arr),
                        minlength=uniq.size)
                if want_surprisal:
                    surp = np.bincount(
                        inverse,
                        weights=np.repeat(np.asarray(surps, dtype=np.float64), sizes_arr),
                        minlength=uniq.size)
                else:
                    surp = score  # the surprisal *is* the score under "surprisal"

                aliased = surp is score

                pos = np.searchsorted(uniq, point_idx)
                if pos < uniq.size and uniq[pos] == point_idx:
                    sl, sr = slice(0, pos), slice(pos + 1, None)
                    uniq = np.concatenate((uniq[sl], uniq[sr]))
                    shared = np.concatenate((shared[sl], shared[sr]))
                    surp = None if aliased else np.concatenate((surp[sl], surp[sr]))
                    score = np.concatenate((score[sl], score[sr]))
                    if aliased:
                        surp = score

                if k <= 0 or uniq.size == 0:
                    counts_per_point.append(0)
                    continue

                if uniq.size > k:
                    head = np.argpartition(-score, k - 1)[:k]
                    sel = head[np.argsort(-score[head], kind="stable")]
                    uniq, shared = uniq[sel], shared[sel]
                    surp = None if aliased else surp[sel]
                    score = score[sel]
                    if aliased:
                        surp = score

                counts_per_point.append(uniq.size)
                dst_parts.append(uniq)
                score_parts.append(score)
                shared_parts.append(shared)
                surp_parts.append(surp)

        if not dst_parts:
            self.stats = EdgeStats(empty, empty, np.empty(0), empty, np.empty(0),
                                   np.empty(0, bool), scheme.name, mutual, k)
            return []

        src = np.repeat(np.arange(len(counts_per_point), dtype=np.int64),
                        np.asarray(counts_per_point, dtype=np.int64))
        dst = np.concatenate(dst_parts).astype(np.int64, copy=False)
        score = np.concatenate(score_parts)
        shared = np.concatenate(shared_parts).astype(np.int64, copy=False)
        surp = np.concatenate(surp_parts)
        n_directed = src.size

        # ---- pass 2: pair the two directions, emit undirected edges ------ #
        base = np.int64(max(n, n_query) + 1)
        key = src * base + dst
        reverse_key = dst * base + src
        order = np.argsort(key, kind="stable")
        ordered_key = key[order]
        pos = np.clip(np.searchsorted(ordered_key, reverse_key), 0, n_directed - 1)
        has_reverse = ordered_key[pos] == reverse_key
        reverse_idx = np.where(has_reverse, order[pos], np.arange(n_directed))
        reverse_score = score[reverse_idx]

        if mutual:
            # Both directions required; emit once, from the lower-index side.
            keep = has_reverse & (src < dst)
        else:
            # Either direction suffices; a reciprocated pair is still emitted
            # once, from the lower-index side.
            keep = (src < dst) | ~has_reverse

        weight = scheme.combine(score, reverse_score, has_reverse, mutual, k)

        self.stats = EdgeStats(src[keep], dst[keep], weight[keep], shared[keep],
                               surp[keep], has_reverse[keep], scheme.name, mutual, k)
        return self.stats.edges()

    def diagnostics(self, sample=10, random_state=None):
        """Summary of the last build; see `EdgeStats.summary`."""
        if self.stats is None:
            raise RuntimeError("Call find_topk_neighbors_with_weights(...) first.")
        return self.stats.summary(sample=sample, random_state=random_state)
