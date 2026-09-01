"""CoNN - Co-occurrence Nearest Neighbours.

Neighbours are decided by how often two points land in the same hash bucket
across many independent tables, not by measuring the distance between them.

Layers, bottom up:

- `conn_ann.hashing`   - `RandomProjectionHasher`: points -> bucket codes.
- `conn_ann.index`     - `LSHIndex`: codes -> hash tables, plus the
                             collision statistics that decide whether a build
                             is linear or quadratic in N.
- `conn_ann.weighting` - what one shared bucket is worth, and how two
                             directed scores become one undirected weight.
- `conn_ann.graph`     - `NeighborGraphBuilder`: tables -> weighted edges.
- `conn_ann.lshash`    - `LSHash`, the flat backwards-compatible facade.

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

__version__ = "0.1.0"

__all__ = [
    "LSHash", "LSHIndex", "RandomProjectionHasher", "GraphSearcher",
    "NeighborGraphBuilder", "EdgeStats",
    "VoteWeighting", "SurprisalWeighting", "InverseLogWeighting",
    "LegacyWeighting", "UniformWeighting",
    "WEIGHTINGS", "get_weighting", "register",
    "MIN_SURPRISAL", "MAX_PACKED_BITS",
]
