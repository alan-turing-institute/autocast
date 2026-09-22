"""Shared synthetic-tensor fixtures for the conformal-package tests."""

from __future__ import annotations

import torch


def make_synthetic_dump(
    *,
    b_total: int = 30,
    n_frames: int = 8,
    height: int = 4,
    width: int = 4,
    n_channels: int = 2,
    n_members: int = 5,
    seed: int = 0,
) -> dict:
    """Fabricate an eval-dump-shaped ``{preds, trues, constant_scalars, meta}`` dict.

    Truth is Gaussian noise; the ensemble is truth plus independent per-member
    Gaussian noise, so the data is exchangeable and well-calibrated by
    construction -- useful for coverage-near-nominal tests.
    """
    generator = torch.Generator().manual_seed(seed)
    trues = torch.randn(
        b_total, n_frames, height, width, n_channels, generator=generator
    )
    noise = torch.randn(
        b_total, n_frames, height, width, n_channels, n_members, generator=generator
    )
    preds = trues.unsqueeze(-1) + noise
    return {
        "preds": preds.float(),
        "trues": trues.float(),
        "constant_scalars": None,
        "meta": {"n_members": n_members},
    }


def make_grouped_dump(
    *,
    n_groups: int = 6,
    n_per_group: int = 25,
    n_frames: int = 4,
    height: int = 3,
    width: int = 3,
    n_channels: int = 1,
    n_members: int = 4,
    seed: int = 0,
) -> dict:
    """Fabricate a Gray-Scott-shaped dump: ``n_groups`` distinct scalar rows."""
    generator = torch.Generator().manual_seed(seed)
    b_total = n_groups * n_per_group
    trues = torch.randn(
        b_total, n_frames, height, width, n_channels, generator=generator
    )
    noise = torch.randn(
        b_total, n_frames, height, width, n_channels, n_members, generator=generator
    )
    preds = trues.unsqueeze(-1) + noise
    group_ids = torch.arange(n_groups).repeat_interleave(n_per_group)
    constant_scalars = group_ids.float().unsqueeze(-1)
    return {
        "preds": preds.float(),
        "trues": trues.float(),
        "constant_scalars": constant_scalars,
        "meta": {"n_members": n_members},
    }


def save_dump(dump: dict, path) -> None:
    """Save a synthetic dump dict to ``path`` with ``torch.save``."""
    torch.save(dump, path)
