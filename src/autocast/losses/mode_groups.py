"""Joint Energy Scores on 2D Fourier modes from sampled closed triads."""

from collections.abc import Sequence
from functools import lru_cache
from math import isfinite, sqrt
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from autocast.losses.ensemble import JointCoefficientEnergyScore


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def _mode_grid(nx: int, ny: int) -> tuple[Tensor, Tensor]:
    """Positive-half-plane representatives and flat indices outside the retained set."""
    # Integer wavenumbers in full FFT storage order. The strict two-thirds
    # mask on each axis retains -21..21 on the SWE64 grid.
    axes = []
    for size in (nx, ny):
        index = torch.arange(size, device="cpu")
        axes.append(torch.where(index <= (size - 1) // 2, index, index - size))
    modes = torch.cartesian_prod(*axes)
    x, y = modes.unbind(-1)
    retained = (3 * x.abs() < nx) & (3 * y.abs() < ny) & ((x != 0) | (y != 0))
    half_plane = (x > 0) | ((x == 0) & (y > 0))
    return modes[retained & half_plane], (~retained).nonzero().flatten()


def _sample_triads(nx: int, ny: int, count: int, generator: torch.Generator) -> Tensor:
    """Uniform draws with replacement, modulo permutations and complex conjugation."""
    if min(nx, ny) < 4:
        msg = "Both spatial dimensions must be at least four for closed triads."
        raise ValueError(msg)
    modes, _ = _mode_grid(nx, ny)
    samples = []
    remaining = count
    while remaining:
        i, j = torch.randint(
            len(modes), (2, max(16, 3 * remaining)), generator=generator, device="cpu"
        )
        k, q = modes[i], modes[j]
        r = k + q
        # The positive half-plane is closed under addition. Each unordered
        # nondegenerate closed triad has two ordered representations (k,q).
        valid = (i != j) & (3 * r[:, 0].abs() < nx) & (3 * r[:, 1].abs() < ny)
        accepted = torch.stack((k[valid], q[valid], r[valid]), dim=1)[:remaining]
        samples.append(accepted)
        remaining -= len(accepted)
    coords = torch.cat(samples)
    return coords[..., 0].remainder(nx) * ny + coords[..., 1].remainder(ny)


class SampledModeEnergyScore(nn.Module):
    """Score singletons, pairs or closed triads on the same sampled modes.

    Inputs are full complex FFTs in (B,T,X,Y,C[,M]) order for real fields.
    A draw is shared across all cases, variables and members. Singleton mode
    scores each vertex separately, retaining repetitions in the sampled triads.
    Pair mode averages the three edges of each triad. With group_norm="rms",
    distances within a group are divided by the square root of its mode count.
    In pair mode, singleton_weight can allocate part of the retained-mode
    score to the vertices of the same sampled triads.
    The complement, including DC, keeps its one-mode score and its original
    share of the full FFT mean. Within the retained set mode exposure follows
    uniform triad sampling. Distances use the same channel scaling in every mode.
    """

    def __init__(
        self,
        group: Literal["singletons", "pairs", "triads"],
        triads_per_draw: int = 64,
        seed: int = 153164,
        alpha: float = 1.0,
        channel_multipliers: Sequence[float] | None = None,
        group_norm: Literal["raw", "rms"] = "raw",
        singleton_weight: float = 0.0,
    ) -> None:
        super().__init__()
        group_sizes = {"singletons": 1, "pairs": 2, "triads": 3}
        if (
            group not in group_sizes
            or type(triads_per_draw) is not int
            or triads_per_draw < 1
        ):
            msg = "Choose singletons, pairs or triads and a positive triads_per_draw."
            raise ValueError(msg)
        if group_norm not in ("raw", "rms"):
            msg = "group_norm must be 'raw' or 'rms'."
            raise ValueError(msg)
        if not isfinite(singleton_weight) or not 0 <= singleton_weight <= 1:
            msg = "singleton_weight must be finite and in [0, 1]."
            raise ValueError(msg)
        if singleton_weight and group != "pairs":
            msg = "singleton_weight is only supported for pairs."
            raise ValueError(msg)
        self.group = group
        self.singleton_weight = singleton_weight
        self.group_norm = group_norm
        self.triads_per_draw = triads_per_draw
        self.seed = seed
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self.single = JointCoefficientEnergyScore(
            alpha, channel_multipliers=channel_multipliers
        )
        size = group_sizes[group]
        self._group_scale = sqrt(size) if group_norm == "rms" else 1.0
        scales = (
            None if channel_multipliers is None else tuple(channel_multipliers) * size
        )
        self.grouped = JointCoefficientEnergyScore(alpha, channel_multipliers=scales)

    def get_extra_state(self) -> Tensor:
        """Checkpoint the independent sampling stream without changing model noise."""
        return self._generator.get_state()

    def set_extra_state(self, state: Tensor) -> None:
        self._generator.set_state(state.cpu())

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        if targets.ndim != 5 or preds.ndim != 6 or preds.shape[:-1] != targets.shape:
            msg = "Expected (B,T,X,Y,C,M) predictions and (B,T,X,Y,C) targets."
            raise ValueError(msg)
        if not preds.is_complex() or not targets.is_complex():
            msg = "Sampled mode scores require complex Fourier coefficients."
            raise TypeError(msg)
        nx, ny = targets.shape[2:4]
        generator = (
            self._generator
            if self.training
            else torch.Generator().manual_seed(self.seed)
        )
        triads = _sample_triads(nx, ny, self.triads_per_draw, generator)
        groups = triads
        if self.group == "pairs":
            groups = groups[:, ((0, 1), (0, 2), (1, 2))].reshape(-1, 2)
        elif self.group == "singletons":
            groups = groups.reshape(-1, 1)
        groups = groups.to(preds.device)
        _, outside_cpu = _mode_grid(nx, ny)
        outside = outside_cpu.to(preds.device)
        pred = rearrange(preds, "b t x y c m -> b t (x y) c m")
        target = rearrange(targets, "b t x y c -> b t (x y) c")
        joint = (
            self.grouped(
                rearrange(pred[:, :, groups], "b t n g c m -> b t n (g c) m"),
                rearrange(target[:, :, groups], "b t n g c -> b t n (g c)"),
            )
            / self._group_scale
        )
        if self.singleton_weight:
            vertices = triads.flatten().to(preds.device)
            singleton = self.single(pred[:, :, vertices], target[:, :, vertices])
            joint = torch.lerp(joint, singleton, self.singleton_weight)
        single = self.single(pred[:, :, outside], target[:, :, outside])
        outside_weight = len(outside_cpu) / (nx * ny)
        return (1 - outside_weight) * joint + outside_weight * single
