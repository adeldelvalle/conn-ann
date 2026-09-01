"""Backwards-compatible facade over `LSHIndex` + `NeighborGraphBuilder`.

`LSHash` keeps the original flat API - construct, `index_batch`,
`find_topk_neighbors_with_weights` - while delegating to the components.  New
code is better served talking to `LSHIndex` and `NeighborGraphBuilder`
directly; this exists so existing scripts keep running unchanged.
"""
from __future__ import annotations

import numpy as np

from .graph import NeighborGraphBuilder
from .hashing import MAX_PACKED_BITS, RandomProjectionHasher
from .index import LSHIndex
from .weighting import MIN_SURPRISAL, WEIGHTINGS, get_weighting


class LSHash(object):
    """
    Locality-sensitive hashing by random projection, plus the consensus
    neighbour graph built from the resulting collisions.

    :param hash_size: bits per table, or `None` / `"auto"` to tie the code
        length to ``log2(N)`` at index time so that the expected bucket
        occupancy stays at `bucket_size` as N grows.  See `LSHIndex` for why
        that is what makes the build subquadratic.
    :param input_dim: dimension of the input vectors.
    :param num_hashtables: number of independent tables.
    :param bucket_size: target expected bucket occupancy when `hash_size` is
        automatic.
    :param weighting: default vote-weighting scheme, one of `WEIGHTINGS`.
    :param mutual: default mutuality requirement for edges.

    `storage_config`, `matrices_filename`, `hashtable_filename` and `overwrite`
    are accepted for signature compatibility and unused.
    """

    _MAX_PACKED_BITS = MAX_PACKED_BITS

    def __init__(self, hash_size, input_dim, num_hashtables=1,
                 storage_config=None, matrices_filename=None, hashtable_filename=None,
                 overwrite=False, bucket_size=64,
                 weighting="surprisal", mutual=True):

        self.weighting = get_weighting(weighting).name
        self.mutual = bool(mutual)
        self.overwrite = overwrite

        self.index = LSHIndex(hash_size, input_dim, num_hashtables,
                              bucket_size=bucket_size)
        self.builder = NeighborGraphBuilder(self.index, self.weighting, self.mutual)

    # ------------------------------------------------------------------ #
    # attributes that used to live directly on LSHash
    # ------------------------------------------------------------------ #

    @property
    def hash_size(self):
        return self.index.hash_size

    @hash_size.setter
    def hash_size(self, value):
        self.index.hash_size = value

    @property
    def input_dim(self):
        return self.index.input_dim

    @property
    def num_hashtables(self):
        return self.index.num_tables

    @property
    def bucket_size(self):
        return self.index.bucket_size

    @property
    def auto_hash_size(self):
        return self.index.auto_hash_size

    @property
    def n_points(self):
        return self.index.n_points

    @property
    def hash_tables(self):
        """List of dicts mapping a bucket code to its member indices."""
        return self.index.tables

    @hash_tables.setter
    def hash_tables(self, value):
        self.index.tables = value

    @property
    def uniform_planes(self):
        """List of `num_hashtables` arrays of shape (hash_size, input_dim)."""
        return None if self.index.hasher is None else self.index.hasher.planes

    @uniform_planes.setter
    def uniform_planes(self, value):
        if self.index.hasher is None:
            self.index.hasher = RandomProjectionHasher(
                self.index.hash_size or len(value[0]), self.index.input_dim,
                self.index.num_tables)
        self.index.hasher.planes = value

    @property
    def _codes(self):
        return self.index.codes

    @property
    def _indexed_points(self):
        return self.index.points

    @property
    def _edge_stats(self):
        return self.builder.stats

    # ------------------------------------------------------------------ #
    # hashing helpers (kept for compatibility)
    # ------------------------------------------------------------------ #

    def _init_uniform_planes(self):
        self.index.hasher = self.index._make_hasher()

    def _init_hashtables(self):
        self.index.tables = [{} for _ in range(self.index.num_tables)]

    def _resolve_hash_size(self, n):
        return self.index.resolve_hash_size(n)

    def _generate_uniform_planes(self, similarity_threshold=None, max_attempts=None):
        """Planes for one table.  `similarity_threshold` and `max_attempts` are
        accepted for signature compatibility and ignored - see
        `RandomProjectionHasher._generate_planes` for why the orthogonality
        rejection sampler was removed."""
        return RandomProjectionHasher(self.index.hash_size, self.index.input_dim, 1).planes[0]

    def _bits_batch(self, planes, input_points):
        """Sign of the projection of each point onto each plane, (N, b) bool."""
        return RandomProjectionHasher.signs(planes, input_points)

    @classmethod
    def _pack_bits(cls, bits):
        """Pack a (N, b) boolean matrix into N hashable bucket codes."""
        return RandomProjectionHasher.pack(bits)

    def _hash_batch(self, planes, input_points):
        """Binary hash strings for one plane set (human-readable)."""
        bits = RandomProjectionHasher.signs(planes, input_points).astype(int)
        return ["".join(map(str, row)) for row in bits]

    def _code_batch(self, input_points):
        return self.index.encode(input_points)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def index_batch(self, input_points):
        """
        Index a batch of points at once.

        :returns: (N, num_hashtables) array of bucket codes
        """
        return self.index.build(input_points)

    def collision_stats(self):
        """Realised candidate work of the index; see `LSHIndex.collision_stats`."""
        return self.index.collision_stats()

    def find_topk_neighbors_with_weights(self, points=None, k=70,
                                         weighting=None, mutual=None):
        """
        Weighted edges of the consensus neighbour graph, as ``(i, j, weight)``.

        See `NeighborGraphBuilder.build` for the two-pass construction and
        `conn_ann.weighting` for the schemes.
        """
        return self.builder.build(k=k, points=points, weighting=weighting,
                                  mutual=mutual)

    def edge_diagnostics(self, sample=10, random_state=None):
        """Summary of the weights produced by the last build."""
        return self.builder.diagnostics(sample=sample, random_state=random_state)
