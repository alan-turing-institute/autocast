"""Online spectral spread/error calibration without conditional-truth redraws."""

import math
from collections.abc import Sequence

import torch
from einops import einsum, reduce
from torch import nn

from autocast.losses.decomposition import _align_targets, _spatial_dims
from autocast.losses.spectral import (
    FourierDecomposition,
    RadialFourierDecomposition,
    _radial_frequency,
)
from autocast.types import FieldTensor, Tensor, TensorInexactBTSC


class SpectralSpreadErrorLoss(nn.Module):
    """Match batch-mean radial anomaly power to observed residual power.

    Predictions are decoded real fields ``(B, T, *S, C, M)`` and targets are
    ``(B, T, *S, C)``, with at least two independent members on a regular
    periodic grid. Each prediction
    is centred on its ensemble mean. Observed residuals use that same mean
    with stop-gradient, so the auxiliary cannot directly move its target by
    changing the common predicted mean. The prediction-side mean remains
    differentiable: spread gradients sum to zero over members.

    ``power_spectra`` returns unbiased sample-variance power ``S`` (divisor
    ``M-1``) and observed residual power ``E``. At a calibrated, unbiased model,
    their expectations obey ``E[E] = (1 + 1/M) E[S]`` because the ensemble
    mean is itself sampled. The loss is the mean squared discrepancy between
    ``(1 + 1/M) S`` and ``E``, divided by a fixed power scale before squaring.
    Power is averaged over batch and modes BEFORE squaring, separately for
    each lead and channel, then discrepancies are averaged over bands, leads
    and channels. DC is a separate band; empty radial bands are omitted.

    This is a heuristic calibration regularizer, not a proper scoring rule or
    an exact conditional anomaly target. Mean bias still enters observed
    residual power. The moment correction does not remove finite-batch loss
    bias, identify phase or ensure conditional calibration. Compose it with
    the primary forecast score; do not use it as the sole training objective.
    Batch pooling is local to each process in distributed training; no running
    statistics, cross-process collectives or target caches are introduced.

    Args:
        edges: Radial upper edges in cycles per grid cell, following
            RadialFourierDecomposition. Internal upper edges are excluded and
            the final upper edge is included. DC is excluded from these bands.
            For a square N-cell grid, edges ``(j+0.5)/N`` give integer-radius
            shells; a final edge >= sqrt(n_spatial_dims)/2 covers all modes.
        include_dc: Also score the spatially uniform anomaly and residual power
            as a separate band, independent of the supplied radial edges.
        power_scale: Fixed finite positive divisor in input-squared units.
            No per-mode inverse-power weighting or normalization is fitted.
            Loss units are fourth power of the input units when this is one.
    """

    requires_ambient_predictions = True

    def __init__(
        self,
        edges: Sequence[int | float],
        *,
        include_dc: bool = True,
        power_scale: int | float = 1.0,
    ) -> None:
        super().__init__()
        # Reuse the existing radial API's edge validation and FFT convention.
        self.edges = RadialFourierDecomposition(edges).edges
        if not math.isfinite(power_scale) or power_scale <= 0:
            raise ValueError(
                f"power_scale must be finite and positive, got {power_scale}."
            )
        self.include_dc = include_dc
        self.power_scale = float(power_scale)
        self.fourier = FourierDecomposition(norm="ortho")

    def _band_weights(self, targets: Tensor, spatial: tuple[int, ...]) -> Tensor:
        radius = _radial_frequency(targets, spatial).flatten()
        masks = [radius == 0] if self.include_dc else []
        lower = 0.0
        for index, upper in enumerate(self.edges):
            mask = (radius > 0) & (radius >= lower)
            mask &= radius <= upper if index == len(self.edges) - 1 else radius < upper
            if mask.any():
                masks.append(mask)
            lower = upper
        if not masks:
            msg = "Configured spectral bands contain no nonempty scored modes."
            raise ValueError(msg)
        # Dense pooling on batch-mean power avoids full-sized member tensors
        # per band and dynamic-shape boolean indexing/synchronization on GPUs.
        weights = torch.stack(masks).to(dtype=torch.float64)
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights.to(device=targets.device, dtype=targets.dtype)

    @staticmethod
    def _pool(power: Tensor, weights: Tensor) -> Tensor:
        batch_mean = reduce(power, "b t ... c -> t (...) c", "mean")
        return einsum(batch_mean, weights, "t mode c, band mode -> t band c")

    def power_spectra(
        self, preds: FieldTensor, targets: TensorInexactBTSC
    ) -> tuple[Tensor, Tensor]:
        """Return uncorrected spread and error powers as ``(T, bands, C)``.

        Both are in input-squared units, with batch and within-band mode means.
        The spread uses divisor M-1; the error uses a detached ensemble mean.
        Half/bfloat16 inputs are promoted before centring and power formation.
        """
        _align_targets(preds, targets)
        if preds.ndim != targets.ndim + 1 or preds.shape[-1] < 2:
            msg = "SpectralSpreadErrorLoss requires at least two ensemble members."
            raise ValueError(msg)
        n_spatial_dims = targets.ndim - 3
        spatial = _spatial_dims(targets, n_spatial_dims)
        if not preds.is_floating_point() or not targets.is_floating_point():
            msg = "Spectral spread/error inputs must be real floating fields."
            raise ValueError(msg)
        if preds.device != targets.device:
            msg = "Spectral spread/error inputs must share a device."
            raise ValueError(msg)
        if any(size == 0 for size in targets.shape):
            msg = "Spectral spread/error fields must be nonempty."
            raise ValueError(msg)
        dtype = torch.promote_types(preds.dtype, targets.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        preds, targets = preds.to(dtype=dtype), targets.to(dtype=dtype)
        weights = self._band_weights(targets, spatial)

        mean = preds.mean(dim=-1)
        anomalies = preds - mean.unsqueeze(-1)
        residuals = targets - mean.detach()
        member_coefficients = self.fourier(anomalies, n_spatial_dims=n_spatial_dims)[
            "fourier"
        ]
        residual_coefficients = self.fourier(residuals, n_spatial_dims=n_spatial_dims)[
            "fourier"
        ]
        spread = member_coefficients.abs().square().sum(dim=-1) / (preds.shape[-1] - 1)
        error = residual_coefficients.abs().square()
        return self._pool(spread, weights), self._pool(error, weights)

    def forward(self, preds: FieldTensor, targets: TensorInexactBTSC) -> Tensor:
        spread, error = self.power_spectra(preds, targets)
        discrepancy = (1 + 1 / preds.shape[-1]) * spread - error
        return (discrepancy / self.power_scale).square().mean()
