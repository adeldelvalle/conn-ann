"""Random-hyperplane hashing for locality-sensitive indexing.

The hasher owns the projection planes and turns points into *bucket codes*: one
hashable value per (point, table).  Nothing here knows about hash tables,
neighbours or graphs.
"""
from __future__ import annotations

import numpy as np

#: Codes wider than this cannot be packed into an int64 and fall back to
#: (still vectorised) byte-string keys.
MAX_PACKED_BITS = 62

#: Rows hashed per matrix product.  Bounds the peak size of the intermediate
#: sign matrix without measurably hurting BLAS throughput.
DEFAULT_BLOCK_SIZE = 8192


class RandomProjectionHasher(object):
    """
    `num_tables` independent random-hyperplane hashes of width `hash_size`.

    Each table draws `hash_size` quasi-orthogonal unit vectors; a point's code
    in that table is the sign pattern of its projections onto them, packed into
    a single hashable value.  Two points collide in a table exactly when they
    fall on the same side of all of its hyperplanes, which happens with
    probability ``(1 - theta / pi) ** hash_size`` for an angle `theta` between
    them - the locality-sensitive property the index relies on.

    :param hash_size: bits per table.
    :param input_dim: dimension of the input vectors.
    :param num_tables: number of independent tables.
    :param similarity_threshold: reject a candidate plane whose absolute cosine
        with an accepted one exceeds this, to keep the bits of a table from
        measuring the same direction twice.
    :param max_attempts: give up drawing diverse planes after this many tries.
    :param block_size: rows per projection matrix product.
    :param random_state: object exposing `randn`, e.g. `numpy.random` (the
        default) or a `numpy.random.RandomState`.
    """

    def __init__(self, hash_size, input_dim, num_tables=1,
                 similarity_threshold=0.8, max_attempts=5000,
                 block_size=DEFAULT_BLOCK_SIZE, random_state=None):
        self.hash_size = int(hash_size)
        self.input_dim = int(input_dim)
        self.num_tables = int(num_tables)
        self.similarity_threshold = similarity_threshold
        self.max_attempts = max_attempts
        self.block_size = int(block_size)
        self.random_state = np.random if random_state is None else random_state

        self.planes = [self._generate_planes() for _ in range(self.num_tables)]

    # ------------------------------------------------------------------ #
    # planes
    # ------------------------------------------------------------------ #

    @property
    def planes(self):
        """List of `num_tables` arrays of shape (hash_size, input_dim)."""
        return self._planes

    @planes.setter
    def planes(self, value):
        self._planes = list(value)
        self._stacked = None  # invalidate the fused projection matrix

    def _generate_planes(self):
        """
        Draw `hash_size` normalised, pairwise quasi-orthogonal row vectors.

        Rejection sampling: a candidate is kept only if its absolute cosine
        with every accepted plane stays below `similarity_threshold`.  In high
        dimension random unit vectors are nearly orthogonal already, so this
        almost never rejects; it matters for small `input_dim`.
        """
        planes = []
        attempts = 0

        while len(planes) < self.hash_size and attempts < self.max_attempts:
            v = self.random_state.randn(self.input_dim)
            v /= np.linalg.norm(v)

            is_similar = any(
                abs(np.dot(v, u)) > self.similarity_threshold for u in planes
            )

            if not is_similar:
                planes.append(v)

            attempts += 1

        if len(planes) < self.hash_size:
            print(f"⚠️ Only generated {len(planes)} diverse planes out of requested {self.hash_size}")

        return np.array(planes)

    @property
    def bits_per_table(self):
        """Actual plane count per table (may fall short of `hash_size`)."""
        return [p.shape[0] for p in self._planes]

    def _stack(self):
        """
        Fuse the per-table planes into one (num_tables * b, input_dim) matrix so
        that a batch can be projected with a single matrix product instead of
        `num_tables` of them.  Returns None when the tables came out ragged, in
        which case the caller falls back to the per-table path.
        """
        if self._stacked is None:
            widths = self.bits_per_table
            if len(set(widths)) == 1 and widths[0] > 0:
                self._stacked = np.concatenate(self._planes, axis=0)
            else:
                self._stacked = False  # ragged: no fused path
        return self._stacked if self._stacked is not False else None

    # ------------------------------------------------------------------ #
    # hashing
    # ------------------------------------------------------------------ #

    @staticmethod
    def signs(planes, points):
        """
        Sign of the projection of each point onto each plane.

        :param planes: (b, input_dim)
        :param points: (N, input_dim), or a single (input_dim,) vector
        :returns: boolean array of shape (N, b)
        """
        try:
            points = np.asarray(points)
            if points.ndim == 1:
                points = points[None, :]
            projections = np.dot(points, planes.T)
        except TypeError:
            print("The input points must be an array-like object with numbers only.")
            raise
        except ValueError as e:
            print("Dimension mismatch between input points and planes.", e)
            raise
        else:
            return projections > 0

    @staticmethod
    def pack(bits):
        """
        Pack a (N, b) boolean matrix into N hashable codes.

        Integers while they fit in an int64 - cheaper to hash and to group than
        the binary strings they replace - and packed byte strings beyond that.
        """
        b = bits.shape[1]
        if b <= MAX_PACKED_BITS:
            weights = np.left_shift(np.int64(1), np.arange(b, dtype=np.int64))
            return np.dot(bits.astype(np.int64), weights)
        packed = np.packbits(bits, axis=1)
        return np.array([row.tobytes() for row in packed], dtype=object)

    def codes(self, points):
        """
        Bucket codes of a batch for every table.

        One fused matrix product per block of rows covers all tables at once;
        the per-table path is used only for ragged plane sets.

        :param points: (N, input_dim)
        :returns: (N, num_tables) array of hashable codes
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points[None, :]

        stacked = self._stack()
        if stacked is None:
            columns = [self.pack(self.signs(planes, points)) for planes in self._planes]
            return np.stack(columns, axis=1)

        n = points.shape[0]
        width = self.bits_per_table[0]
        out = None
        for start in range(0, n, self.block_size):
            block = points[start:start + self.block_size]
            bits = self.signs(stacked, block).reshape(block.shape[0], self.num_tables, width)
            columns = [self.pack(bits[:, i, :]) for i in range(self.num_tables)]
            chunk = np.stack(columns, axis=1)
            if out is None:
                out = np.empty((n, self.num_tables), dtype=chunk.dtype)
            out[start:start + block.shape[0]] = chunk
        if out is None:  # empty input
            out = np.empty((0, self.num_tables), dtype=np.int64)
        return out

    def hash_strings(self, table_index, points):
        """Binary hash strings for one table (human-readable; not used for lookup)."""
        bits = self.signs(self._planes[table_index], points).astype(int)
        return ["".join(map(str, row)) for row in bits]
