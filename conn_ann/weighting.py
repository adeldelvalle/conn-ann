"""Vote-weighting schemes for LSH neighbour graphs.

A scheme answers two questions: what one shared bucket is worth (`increment`),
and how the two directed scores of a pair become one undirected weight
(`combine`).  Everything else about graph construction is scheme-independent,
so adding a scheme means adding a class here and registering it.
"""
from __future__ import annotations

import math

import numpy as np

#: Floor for the surprisal of a collision.  ``log(n / m_b)`` is 0 for a bucket
#: holding the whole dataset: such a collision carries no information, but a
#: literal 0 would erase the edge (and zero out any geometric mean it takes
#: part in), so it is floored to a negligible-but-positive weight instead.
MIN_SURPRISAL = 1e-12


class VoteWeighting(object):
    """
    Base class.  Subclasses set `name` and implement `increment`.

    :cvar score_is_count: True when every collision is worth exactly 1.0, which
        lets the builder reuse the integer collision counts as the score.
    """

    name = None
    score_is_count = False

    def increment(self, n_points):
        """
        Return a callable ``m -> float``: what one shared bucket of size `m` is
        worth, given `n_points` indexed points.  Results are memoised on `m`
        because bucket sizes repeat heavily across tables.
        """
        raise NotImplementedError

    def combine(self, score, reverse_score, has_reverse, mutual, k):
        """
        Fold the two directed scores of each pair into one undirected weight.

        `score` is the score of the emitted direction and `reverse_score` that
        of the opposite one (equal to `score` where `has_reverse` is False).

        - `mutual`: geometric mean, the scale-correct way to average two
          scores that are already on a log scale.
        - otherwise: the larger of the two, or the only one there is.
        """
        if mutual:
            return np.sqrt(score * reverse_score)
        return np.where(has_reverse, np.maximum(score, reverse_score), score)

    def __repr__(self):
        return "%s(name=%r)" % (type(self).__name__, self.name)


class SurprisalWeighting(VoteWeighting):
    """
    ``log(n / m_b)`` - the information content of the collision.

    Null model: a point is placed uniformly at random over the `n` indexed
    points, so it lands in bucket `b` with probability ``p_b = m_b / n``.
    Observing that i and j share bucket `b` is then an event of probability
    `p_b` carrying ``-log p_b = log(n / m_b)`` nats.  The `L` tables use
    independent random projections, so evidence from the tables in which i and
    j collide adds, and the accumulated sum is the log-likelihood ratio of "i
    and j are neighbours" against the null.  That sum *is* the weight: it is
    already on a log scale, so exponentiating it would undo the additivity.

    Rare (small) buckets are the informative ones, and this grows like
    ``log n`` for a bucket of fixed size - the asymptotics
    ``1 / log(m + 1)`` fails to reproduce, since that saturates at
    ``1 / log 2`` and so treats the rarest collisions much like commonplace
    ones.
    """

    name = "surprisal"

    def increment(self, n_points):
        log = math.log
        cache = {}

        def increment(m):
            v = cache.get(m)
            if v is None:
                v = cache[m] = max(log(n_points / m), MIN_SURPRISAL) if m else MIN_SURPRISAL
            return v

        return increment


class InverseLogWeighting(VoteWeighting):
    """
    ``1.5 / log(m + 1)`` - the kernel of the earlier working-tree scheme,
    accumulated additively and combined by the ordinary mutuality rule.

    Contrasting this with `LegacyWeighting` separates the damage done by the
    kernel from the damage done by the ``exp(v / k)`` transform, since the two
    share a kernel and differ only in the transform.  The leading 1.5 is a
    global rescale and so cannot change a modularity-based partition; it is
    kept only to match the constant that scheme used.
    """

    name = "inv_log"

    def increment(self, n_points):
        log = math.log
        cache = {}

        def increment(m):
            v = cache.get(m)
            if v is None:
                v = cache[m] = 1.5 / log(m + 1)
            return v

        return increment


class LegacyWeighting(InverseLogWeighting):
    """
    The historical scheme: the `InverseLogWeighting` kernel pushed through
    ``exp(v / k)``, squared when the pair is reciprocated.  Kept to reproduce
    prior numbers.

    Both defects are visible in `combine`: dividing by `k` is unjustified (`k`
    is the neighbour count, unrelated to the scale of the votes) and the
    exponential squeezes an already-log-scale score into a narrow band that
    modularity optimisers read as an unweighted graph.
    """

    name = "legacy"

    def combine(self, score, reverse_score, has_reverse, mutual, k):
        directed = np.exp(score / k)
        return np.where(has_reverse, directed * directed, directed)


class UniformWeighting(VoteWeighting):
    """
    ``1.0`` - a plain count of shared tables.

    The ablation that isolates the consensus topology from the weighting.
    """

    name = "uniform"
    score_is_count = True

    def increment(self, n_points):
        def increment(m):
            return 1.0

        return increment


_REGISTRY = {}


def register(weighting):
    """Register a `VoteWeighting` instance under its `name`."""
    _REGISTRY[weighting.name] = weighting
    return weighting


for _scheme in (SurprisalWeighting(), InverseLogWeighting(),
                LegacyWeighting(), UniformWeighting()):
    register(_scheme)

#: Names accepted wherever a `weighting` argument is taken.
WEIGHTINGS = tuple(_REGISTRY)


def get_weighting(weighting):
    """
    Resolve a scheme name (or a `VoteWeighting` instance) to an instance.

    :raises ValueError: if the name is not registered.
    """
    if isinstance(weighting, VoteWeighting):
        return weighting
    try:
        return _REGISTRY[weighting]
    except KeyError:
        raise ValueError("weighting must be one of %r, got %r" % (WEIGHTINGS, weighting))
