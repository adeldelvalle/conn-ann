"""Spectral-LSH: locality-sensitive hashing and consensus neighbour graphs.

Layers, bottom up:

- `spectral_lsh.hashing`   - `RandomProjectionHasher`: points -> bucket codes.
- `spectral_lsh.index`     - `LSHIndex`: codes -> hash tables, plus the
                             collision statistics that decide whether a build
                             is linear or quadratic in N.
- `spectral_lsh.weighting` - what one shared bucket is worth, and how two
                             directed scores become one undirected weight.
- `spectral_lsh.graph`     - `NeighborGraphBuilder`: tables -> weighted edges.
- `spectral_lsh.lshash`    - `LSHash`, the flat backwards-compatible facade.

Typical use::

    index = LSHIndex("auto", d, num_tables=40, bucket_size=64)
    index.build(X)
    edges = NeighborGraphBuilder(index, weighting="surprisal").build(k=60)
"""
from .graph import EdgeStats, NeighborGraphBuilder
from .hashing import MAX_PACKED_BITS, RandomProjectionHasher
from .index import LSHIndex
from .search import GraphSearcher
from .lshash import LSHash
from .weighting import (MIN_SURPRISAL, WEIGHTINGS, InverseLogWeighting,
                        LegacyWeighting, SurprisalWeighting, UniformWeighting,
                        VoteWeighting, get_weighting, register)

__version__ = "0.3.0"

__all__ = [
    "LSHash", "LSHIndex", "RandomProjectionHasher", "GraphSearcher",
    "NeighborGraphBuilder", "EdgeStats",
    "VoteWeighting", "SurprisalWeighting", "InverseLogWeighting",
    "LegacyWeighting", "UniformWeighting",
    "WEIGHTINGS", "get_weighting", "register",
    "MIN_SURPRISAL", "MAX_PACKED_BITS",
]
