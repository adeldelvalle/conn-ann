# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""Hot path of multi-probe LSH voting.

The Python/numpy version spends ~91% of query time here: for every one of L
tables it derives a bucket code, enumerates probe codes, binary-searches the
sorted code column and increments a counter for each member.  That is 2,400
interpreter iterations per query for L=160 with 15 probes.  This does the whole
thing in one C loop.
"""
import numpy as np
cimport numpy as cnp
from libc.string cimport memset

cnp.import_array()

ctypedef cnp.int64_t i64
ctypedef cnp.int32_t i32
ctypedef cnp.int16_t i16


cdef inline Py_ssize_t lower_bound(const i64* a, Py_ssize_t n, i64 key) noexcept nogil:
    cdef Py_ssize_t lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < key: lo = mid + 1
        else: hi = mid
    return lo


cdef inline Py_ssize_t upper_bound(const i64* a, Py_ssize_t n, i64 key) noexcept nogil:
    cdef Py_ssize_t lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] <= key: lo = mid + 1
        else: hi = mid
    return lo


def vote_direct(const float[:, ::1] z,   # (L, b) query projections onto each table's planes
                const i32[:, ::1] offsets,       # (L, 2**b + 1) prefix sums over codes
                const i32[:, ::1] order,         # (L, N) point ids grouped by code
                const i32[::1] submasks,
                int p,
                i16[::1] acc,
                i32[::1] hist,
                i32[::1] out_idx,
                i16[::1] out_cnt,
                int C):
    """Same as `vote`, with O(1) bucket lookup.

    With b bits there are only 2**b possible codes, so a prefix-sum table over
    codes replaces the binary search - trading 2**b * L * 4 bytes of memory for
    ~log2(N) fewer cache lines touched per probe.
    """
    cdef Py_ssize_t L = z.shape[0], b = z.shape[1], N = order.shape[1]
    cdef Py_ssize_t nsub = submasks.shape[0]
    cdef Py_ssize_t l, j, k, s, lo, hi, i
    cdef i64 c0, code
    cdef int rank[64]
    cdef float mag[64]
    cdef float tmp
    cdef int ti, mask
    cdef const i32* orow
    cdef Py_ssize_t thr, cum, n
    cdef i16 v

    with nogil:
        for l in range(L):
            c0 = 0
            for j in range(b):
                if z[l, j] > 0: c0 |= (<i64>1) << j
                mag[j] = z[l, j] if z[l, j] >= 0 else -z[l, j]
                rank[j] = <int>j
            for k in range(p if p < b else b):
                for j in range(k + 1, b):
                    if mag[j] < mag[k]:
                        tmp = mag[k]; mag[k] = mag[j]; mag[j] = tmp
                        ti = rank[k]; rank[k] = rank[j]; rank[j] = ti
            orow = &order[l, 0]
            for s in range(nsub):
                mask = submasks[s]
                code = c0
                for k in range(p):
                    if mask & (1 << k):
                        code ^= (<i64>1) << rank[k]
                lo = offsets[l, code]
                hi = offsets[l, code + 1]
                for i in range(lo, hi):
                    acc[orow[i]] += 1

        memset(&hist[0], 0, (L + 2) * sizeof(i32))
        for i in range(N):
            hist[acc[i]] += 1
        thr = L
        cum = 0
        while thr > 1:
            if cum + hist[thr] >= C: break
            cum += hist[thr]
            thr -= 1
        n = 0
        for i in range(N):
            v = acc[i]
            if v > thr and n < C:
                out_idx[n] = <i32>i; out_cnt[n] = v; n += 1
        for i in range(N):
            v = acc[i]
            acc[i] = 0
            if v == thr and n < C:
                out_idx[n] = <i32>i; out_cnt[n] = v; n += 1
    return n


def vote(const float[:, ::1] z,          # (L, b) query projections onto each table's planes
         const i64[:, ::1] scodes,       # (L, N) each row sorted ascending
         const i32[:, ::1] order,        # (L, N) point ids in that sorted order
         const i32[::1] submasks,        # probe patterns, as bitmasks over the p ranked bits
         int p,
         i16[::1] acc,                   # (N,) scratch, must be all-zero on entry
         i32[::1] hist,                  # (L+2,) scratch
         i32[::1] out_idx,               # (C,) output
         i16[::1] out_cnt,               # (C,) output
         int C):
    """Accumulate cross-table votes and return the top-C candidates.

    A point holds one code per table, and a table's probe codes are distinct, so
    each table contributes at most one vote per candidate: the count is the
    number of tables that put the candidate in a probed bucket.

    `acc` is left zeroed on return so it can be reused across queries.
    """
    cdef Py_ssize_t L = z.shape[0], b = z.shape[1], N = scodes.shape[1]
    cdef Py_ssize_t nsub = submasks.shape[0]
    cdef Py_ssize_t l, j, k, s, lo, hi, i
    cdef i64 c0, code
    cdef int rank[64]
    cdef float mag[64]
    cdef float av, tmp
    cdef int ti, mask
    cdef const i64* row
    cdef const i32* orow
    cdef Py_ssize_t thr, cum, n
    cdef i16 v

    with nogil:
        for l in range(L):
            # bucket code, and |projection| per bit
            c0 = 0
            for j in range(b):
                if z[l, j] > 0: c0 |= (<i64>1) << j
                mag[j] = z[l, j] if z[l, j] >= 0 else -z[l, j]
                rank[j] = <int>j
            # partial selection sort: the p bits whose hyperplane is nearest
            for k in range(p if p < b else b):
                for j in range(k + 1, b):
                    if mag[j] < mag[k]:
                        tmp = mag[k]; mag[k] = mag[j]; mag[j] = tmp
                        ti = rank[k]; rank[k] = rank[j]; rank[j] = ti
            row = &scodes[l, 0]
            orow = &order[l, 0]
            for s in range(nsub):
                mask = submasks[s]
                code = c0
                for k in range(p):
                    if mask & (1 << k):
                        code ^= (<i64>1) << rank[k]
                lo = lower_bound(row, N, code)
                hi = upper_bound(row, N, code)
                for i in range(lo, hi):
                    acc[orow[i]] += 1

        # counting selection: find the vote level at which >= C candidates survive
        memset(&hist[0], 0, (L + 2) * sizeof(i32))
        for i in range(N):
            hist[acc[i]] += 1
        thr = L
        cum = 0
        while thr > 1:
            if cum + hist[thr] >= C: break
            cum += hist[thr]
            thr -= 1

        # Emit strictly-above-threshold first, THEN fill with ties.  A single
        # "v >= thr, first C by index" pass is not a top-C: it can hit the cap on
        # tied candidates and drop a higher-voted one sitting at a larger index.
        n = 0
        for i in range(N):
            v = acc[i]
            if v > thr and n < C:
                out_idx[n] = <i32>i
                out_cnt[n] = v
                n += 1
        for i in range(N):
            v = acc[i]
            acc[i] = 0
            if v == thr and n < C:
                out_idx[n] = <i32>i
                out_cnt[n] = v
                n += 1
    return n
