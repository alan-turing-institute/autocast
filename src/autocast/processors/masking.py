"""Composable temporal context sampling and forecast-only loss reduction."""

import torch
from einops import reduce
from torch import Tensor


def random_context_mask(
    x: Tensor, lmbda: float = 1.0, rho: float = 1.0, atleast: int = 0
) -> Tensor:
    """Sample LOLA's prefix/suffix observation mask for ``(B,T,...,C)``.

    Match the pinned implementation's Poisson draw and modulo wrapping,
    rather than truncating the distribution. Always leave a forecast frame.
    """
    if atleast < 0:
        msg = f"`atleast` must be non-negative, got {atleast}."
        raise ValueError(msg)
    batch_size, total_steps = x.shape[:2]
    atleast = min(atleast, total_steps - 1)
    rate = torch.full((batch_size, 1), fill_value=lmbda, device=x.device)
    context = torch.poisson(rate).long() % (total_steps - atleast) + atleast
    index = torch.arange(total_steps, device=x.device)
    if rho <= 0.0:
        mask = index >= total_steps - context
    elif rho >= 1.0:
        mask = index < context
    else:
        choose_prefix = torch.rand((batch_size, 1), device=x.device) < rho
        mask = torch.where(
            choose_prefix, index < context, index >= total_steps - context
        )
    return mask.reshape(batch_size, total_steps, *([1] * (x.ndim - 2))).expand_as(x)


def masked_mean(score: Tensor, mask: Tensor) -> Tensor:
    """Average selected elements within each example, then average examples.

    A longer forecast must not give an example more weight. An empty
    selection contributes zero rather than NaN.
    """
    # Half-precision sums/counts can overflow even for a 64x64 forecast.
    if score.dtype in (torch.float16, torch.bfloat16):
        score = score.float()
    mask = mask.expand_as(score)
    count = reduce(mask, "b ... -> b", "sum").clamp_min(1)
    total = reduce(torch.where(mask, score, 0), "b ... -> b", "sum")
    return (total / count).mean()
