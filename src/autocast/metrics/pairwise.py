"""The one pairwise ``|x_i - x_j|^p`` kernel behind every variogram in this package.

Three copies of this arithmetic existed before (the VariogramScore metric, the
VariogramScoreLoss free function and the dependence rank histogram's pre-rank),
and two had drifted -- one guarded the gradient singularity at exact zeros, one
did not. This module is the single source; the callers keep their own shapes
and reductions and delegate the pair sum here.

Exactness, not approximation. The variogram summand
``(mean_m |x_mi - x_mj|^p - |y_i - y_j|^p)^2`` is symmetric in ``(i, j)`` and
zero on the diagonal, so the full ``D x D`` sum equals the sum over ``i < j``
with each pair carrying the weight of both orientations, ``w_ij + w_ji``. That
halves the work and changes nothing but floating-point summation order.

Members are accumulated one at a time, so the largest live tensor is
``(rows, chunk, D - start - 1)`` -- the ``M`` axis is never materialised.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import torch

from autocast.metrics.base import (
    DEFAULT_MAX_PAIR_ELEMENTS as DEFAULT_PAIR_BLOCK_ELEMENTS,
)
from autocast.types import Tensor

P_SQRT: float = 0.5  # the variogram default; |t|^0.5 is a square root
P_ABS: float = 1.0  # |t|^1 is the absolute value itself
P_SQUARE: float = 2.0  # |t|^2 is a square

# DEFAULT_PAIR_BLOCK_ELEMENTS (2**27, one constant, owned by metrics/base.py) is
# the elements of one pairwise slab held at once. A FLOOR, valid on every device
# (CPU, XPU, a 40 GB cluster GPU); resolve_pair_block_elements may raise it per
# device but never lower it, so behaviour off the DGX is unchanged.
# ⚠ metrics/base.py must not import this module (base -> pairwise -> base cycle).


def safe_abs_pow(diff: Tensor, p: float) -> Tensor:
    """``|diff|^p``, out of place, safe for autograd where ``diff`` is exactly zero.

    ``d(|t|^p)/dt`` diverges as ``t -> 0`` for ``p < 1``; on a pairwise grid's
    diagonal ``t`` is exactly ``x - x = 0``, and chaining that with an
    upstream-zero derivative gives a ``0 * inf`` NaN although the true gradient
    is 0. Substitutes 1 before ``pow`` at exact zeros, then restores the exact
    zero in the forward value. Use on any path that needs gradients.
    """
    is_zero = diff == 0
    safe = torch.where(is_zero, torch.ones_like(diff), diff).abs()
    if p == P_SQRT:
        powered = safe.sqrt()
    elif p == P_ABS:
        powered = safe
    elif p == P_SQUARE:
        powered = safe * safe
    else:
        powered = safe.pow(p)
    return torch.where(is_zero, torch.zeros_like(diff), powered)


def abs_pow_(x: Tensor, p: float) -> Tensor:
    """``|x|^p`` IN PLACE -- forward-only.

    Never call on a path that needs gradients: the in-place ``abs_``/``sqrt_``
    destroy the inputs their backward needs.
    """
    x = x.abs_()
    if p == P_SQRT:
        return x.sqrt_()
    if p == P_ABS:
        return x
    if p == P_SQUARE:
        return x.mul_(x)
    return x.pow_(p)


def _member_term(p_rows: Tensor, p_cols: Tensor, p: float) -> Tensor:
    """``|p_rows - p_cols|^p`` for ONE member, out of place so it can fuse.

    Broadcasts ``(rows, chunk, 1)`` against ``(rows, 1, D')``. Identical
    arithmetic to :func:`abs_pow_` on the same difference, in the same order, so
    the values are bit-identical -- what differs is the memory traffic. The
    in-place path allocates the difference and then makes three passes over that
    slab (subtract, ``abs_``, ``sqrt_``); a fused kernel reads the two small
    inputs and writes the slab once. On a 273 GB/s box that is the whole point.

    ⚠ Out of place ON PURPOSE. In-place ops cannot be fused into the producer,
    so "optimising" this back to ``abs_pow_`` would silently remove the effect
    the compiled path exists to get.
    """
    diff = (p_rows - p_cols).abs()
    if p == P_SQRT:
        return diff.sqrt()
    if p == P_ABS:
        return diff
    if p == P_SQUARE:
        return diff * diff
    return diff.pow(p)


# Switchable because inductor is not available everywhere (XPU has no backend
# today) and because the gain is hardware-specific: this machine's own note
# measures `torch.compile` as a LOSS on small models, where the bottleneck is
# kernel launch. This kernel is the opposite regime -- multi-GB elementwise
# slabs, bandwidth-bound -- which is why it was worth measuring rather than
# assuming in either direction. The default below is set FROM that measurement;
# see
# `notes/design/training_infrastructure/eval_optimization/artefacts/bench_step4_2026-09-05.txt`.  # noqa: E501
#
# MEASURED 2026-09-05, idle GB10, cns64's production shape (B=32, T=4, S=4096,
# C=3, M=8), 3 timed reps after an untimed warm-up:
#     current in-place kernel   48.278 s/batch   1.000x
#     this function, eager      48.107 s/batch   1.004x  (the refactor is free)
#     this function, compiled   30.055 s/batch   1.606x  <- kept
# 1.606x clears the plan's 1.3x keep rule, and clears it on BOTH rows
# independently (spatial 1.600x, spatiotemporal 1.608x). It still clears it with
# the compile warm-up charged in (1.575x), so it pays even on a single cold call.
_COMPILE_DEFAULT = "1"
_COMPILE = (
    os.environ.get("AUTOCAST_COMPILE_PAIRWISE", _COMPILE_DEFAULT) == "1"
    and torch.cuda.is_available()
)
# `dynamic=True`: the last block of a sweep is narrower than the rest, and a
# static compile would recompile per distinct width. There are ~2,000 blocks per
# batch, so a recompile storm would be far slower than staying eager.
_member_term_fast = (
    torch.compile(_member_term, dynamic=True) if _COMPILE else _member_term
)


def upper_pair_blocks(
    d: int, rows: int, max_elements: int, device: torch.device
) -> Iterator[tuple[int, int, int, Tensor]]:
    """Yield ``(start, end, col0, mask)`` blocks covering every pair ``i < j``.

    Rows ``i`` in ``[start, end)`` only need columns ``j`` in ``[col0, D)`` with
    ``col0 = start + 1``; inside that block ``j > i`` is the mask
    ``jj >= ii`` (a small triangular corner). The last row ``D - 1`` has no
    ``j > i`` and is never visited. The block is sized so that
    ``rows * (end - start) * (D - col0) <= max_elements``.
    """
    chunk = max(1, min(d, max_elements // max(1, rows * d)))
    for start in range(0, d - 1, chunk):
        end = min(d - 1, start + chunk)
        col0 = start + 1
        ii = torch.arange(end - start, device=device)
        jj = torch.arange(d - col0, device=device)
        yield start, end, col0, jj[None, :] >= ii[:, None]


def variogram_pair_sum(
    pred: Tensor,  # (rows, D, M)
    true: Tensor,  # (rows, D)
    p: float,
    weights: Tensor | None,  # (D, D) non-negative, any symmetry, or None
    max_elements: int = DEFAULT_PAIR_BLOCK_ELEMENTS,
) -> Tensor:  # (rows,)
    """``sum_ij w_ij (mean_m |x_mi - x_mj|^p - |y_i - y_j|^p)^2`` per row, exactly.

    Uniform weights (``None``) are the full ``D x D`` sum, i.e. twice the
    ``i < j`` sum. Given weights carry ``w_ij + w_ji`` on the upper triangle,
    which is exact for asymmetric ``w`` too.
    """
    rows, d, m = pred.shape
    needs_grad = torch.is_grad_enabled() and (pred.requires_grad or true.requires_grad)
    out = pred.new_zeros(rows)
    tri_w = None if weights is None else weights + weights.T
    for start, end, col0, upper in upper_pair_blocks(
        d, rows, max_elements, pred.device
    ):
        obs_diff = true[:, start:end, None] - true[:, None, col0:]
        if needs_grad:
            obs = safe_abs_pow(obs_diff, p)
            ens = torch.zeros_like(obs)
            for k in range(m):
                ens = ens + safe_abs_pow(
                    pred[:, start:end, None, k] - pred[:, None, col0:, k], p
                )
            ens = ens / m
            diff_sq = (ens - obs) ** 2
            if tri_w is not None:
                diff_sq = diff_sq * tri_w[start:end, col0:]
            out = out + (diff_sq * upper).sum(dim=(-2, -1))
        else:
            obs = abs_pow_(obs_diff, p)
            ens = torch.zeros_like(obs)
            for k in range(m):
                ens.add_(
                    _member_term_fast(
                        pred[:, start:end, None, k], pred[:, None, col0:, k], p
                    )
                )
            ens.div_(m)
            diff_sq = ens.sub_(obs).square_()
            if tri_w is not None:
                diff_sq.mul_(tri_w[start:end, col0:])
            diff_sq.mul_(upper)
            out += diff_sq.sum(dim=(-2, -1))
    if tri_w is None:
        out = out * 2.0
    return out


def weighted_pair_sum(
    fields: Tensor,  # (N, D)
    weights: Tensor,  # (D, D), zero diagonal by the caller's contract
    p: float,
    max_elements: int = DEFAULT_PAIR_BLOCK_ELEMENTS,
) -> Tensor:  # (N,)
    """``sum_ij w_ij |x_i - x_j|^p`` per field, over ``i < j`` with ``w_ij + w_ji``.

    Exact. The dependence rank histogram's pre-rank. Forward-only (float64 caller).
    Derived from ``variogram_pair_sum``'s observation term; Step 5's golden test
    is its verification.
    """
    n, d = fields.shape
    out = fields.new_zeros(n)
    tri_w = weights + weights.T
    for start, end, col0, upper in upper_pair_blocks(d, n, max_elements, fields.device):
        block = abs_pow_(fields[:, start:end, None] - fields[:, None, col0:], p)
        block.mul_(tri_w[start:end, col0:]).mul_(upper)
        out += block.sum(dim=(-2, -1))
    return out


LIVE_SLABS: int = 4  # peak live set in one block: diff, ens, obs, mask


def resolve_pair_block_elements(
    device: torch.device, itemsize: int, floor: int = DEFAULT_PAIR_BLOCK_ELEMENTS
) -> int:
    """Elements per pairwise slab for ``device``.

    The floor everywhere, raised on CUDA so that ``LIVE_SLABS`` slabs of
    ``itemsize``-byte elements fit in a quarter of the FREE memory. Never
    below the floor, so a small or shared GPU behaves exactly as before, and
    CPU/XPU are untouched.
    """
    if device.type != "cuda":
        return floor
    free, _total = torch.cuda.mem_get_info(device)
    return max(floor, int(free // 4 // (LIVE_SLABS * itemsize)))
