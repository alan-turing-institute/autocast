"""Lag-1 trace gauge for residual bias-vs-noise diagnosis.

A single scalar ``rho_hat`` that tracks whether a model's one-step residual
looks like fresh per-frame noise or a stale mean-error bias that persists across
a step. Intended as a per-epoch monitor on held-out residuals: as the mean
converges, a well-behaved head's ``rho_hat`` falls toward zero.
"""

from __future__ import annotations

import torch
from torch import Tensor


def lag_trace_rho(residual: Tensor, traj_id: Tensor) -> float:
    """Lag-1 trace gauge ``rho_hat = S_bar1 / (S0 - S_bar1)`` over time-ordered blocks.

    Within each trajectory (its frames assumed time-consecutive),
    ``S0 = mean_t <e_t, e_t>`` and ``S_bar1 = mean_t <e_t, e_{t+1}>`` (inner
    products summed over all non-time dims). A structured mean-error bias
    persists across a step (a near-unit lag-1 inner product on the residual)
    while fresh per-frame noise does not (its lag-1 inner product averages to
    zero), so the ratio tracks the bias/noise ratio.

    Parameters
    ----------
    residual
        ``(F, N)`` residual ``e = target - mu``, flattened to ``F`` frames over
        ``N = spatial * C``, with frames grouped by trajectory in ``traj_id``.
    traj_id
        ``(F,)`` integer trajectory id per frame; frames sharing an id must be in
        time order.

    Returns:
    -------
    float
        ``rho_hat``; ``nan`` if the denominator ``S0 - S_bar1`` is (near) zero.
    """
    s0_sum, s1_sum, c0, c1 = 0.0, 0.0, 0, 0
    for t in torch.unique(traj_id):
        blk = residual[traj_id == t]  # (L, N) frames in time order
        s0_sum += float((blk * blk).sum())
        c0 += blk.shape[0]
        if blk.shape[0] >= 2:
            s1_sum += float((blk[:-1] * blk[1:]).sum())
            c1 += blk.shape[0] - 1
    s0 = s0_sum / max(c0, 1)
    s1 = s1_sum / max(c1, 1)
    denom = s0 - s1
    return float(s1 / denom) if abs(denom) > 1e-30 else float("nan")
