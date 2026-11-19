"""Central-interval arithmetic shared by the ensemble and Gaussian metric families.

Both families score the SAME interval and differ only in how its bounds are
produced -- empirical quantiles over ensemble members, or ``mu +/- z sigma`` from
a parametric head. The arithmetic lives here so that a change to the Winkler
penalty or the coverage indicator cannot land on one family and silently miss the
other, which is precisely how they came to be written twice.

⚠ This module deliberately depends on ``torch`` and the shape aliases and NOTHING
else in this package: the heavy metric modules import THIS one, never the reverse.
"""

from __future__ import annotations

import torch

from autocast.types import Tensor


def z_for_central_interval(coverage_level: float) -> float:
    """Half-width of the symmetric standard-normal interval at ``coverage_level``.

    For ``coverage_level = 0.9`` this returns ``1.6449...``, i.e. the ``z_{0.05}``
    upper-tail quantile.
    """
    p = 0.5 + coverage_level / 2.0
    return torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(p)).item()


def winkler_penalty(
    lower: Tensor, upper: Tensor, target: Tensor, alpha: float
) -> Tensor:
    """Pointwise Winkler interval score: width, plus a penalty for falling outside.

    ``alpha`` is the total tail mass, so a central ``1 - alpha`` interval. Lower is
    better: narrow intervals are rewarded and misses are charged in proportion to
    their distance outside the interval, at rate ``2 / alpha``.

    References:
    Winkler, R. L. (1972). A Decision-Theoretic Approach to Interval Estimation.
    Journal of the American Statistical Association, 67(337), 187-191.
    """
    width = upper - lower
    below = (2.0 / alpha) * torch.clamp(lower - target, min=0.0)
    above = (2.0 / alpha) * torch.clamp(target - upper, min=0.0)
    return width + below + above


def interval_indicator(lower: Tensor, upper: Tensor, target: Tensor) -> Tensor:
    """1 where ``target`` lies inside the CLOSED interval ``[lower, upper]``, else 0.

    Closed on both ends, so a target exactly on a bound counts as covered -- the
    convention both families already used, kept explicit here because an
    inclusive/exclusive split between them would bias coverage by a boundary case
    that nothing downstream could detect.
    """
    return ((target >= lower) & (target <= upper)).to(target.dtype)
