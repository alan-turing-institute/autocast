"""Deterministic metrics.

Power-spectrum RMSE utilities in this module based on the implementation from:
- Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics
Emulation (Rozet et al., 2024), https://arxiv.org/abs/2507.02608,
https://github.com/PolymathicAI/lola
- Specific code from:
    - https://github.com/PolymathicAI/lola/blob/main/lola/fourier.py
    - https://github.com/PolymathicAI/lola/blob/main/experiments/eval.py
"""

import math
from functools import cache

import numpy as np
import torch

from autocast.metrics.base import BaseMetric
from autocast.types import Tensor, TensorBTC, TensorBTSC
from autocast.types.types import ArrayLike, TensorBTSCM


class BTSCMetric(BaseMetric[TensorBTSC | TensorBTSCM, TensorBTSC]):
    """
    Base class for metrics that operate on spatial tensors.

    Checks input types and shapes and converts to Tensor.

    Args:
        reduce_all: If True, return scalar by averaging over all non-batch dims
        dist_sync_on_step: Synchronize metric state across processes at each forward()
    """

    def _check_input(
        self, y_pred: ArrayLike, y_true: ArrayLike
    ) -> tuple[TensorBTSC, TensorBTSC]:
        """
        Check types and shapes and converts inputs to Tensor.

        Args:
            y_pred: Predictions of shape (B, T, S, C)
            y_true: Ground truth of shape (B, T, S, C)

        Returns
        -------
            Tuple of (y_pred, y_true) as Tensors
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)

        if not isinstance(y_pred, Tensor):
            raise TypeError(
                f"y_pred must be a Tensor or np.ndarray, got {type(y_pred)}"
            )
        if not isinstance(y_true, Tensor):
            raise TypeError(
                f"y_true must be a Tensor or np.ndarray, got {type(y_true)}"
            )

        if y_pred.ndim == y_true.ndim and y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred and y_true must have the same shape, "
                f"got {y_pred.shape} and {y_true.shape}"
            )

        if y_pred.ndim == y_true.ndim + 1 and y_pred.shape[:-1] != y_true.shape:
            raise ValueError(
                f"y_pred (ensemble) and y_true must have the same shape along "
                f"non-ensemble dims, got {y_pred.shape} and {y_true.shape}"
            )

        if y_pred.ndim < 4:
            raise ValueError(
                f"y_pred has {y_pred.ndim} dimensions, should be at least 4, "
                f"following the pattern(B, T, S, C)"
            )

        # Handle ensemble dimension if present
        if y_pred.ndim == y_true.ndim + 1:
            y_pred = self._ensemble_aggregation(y_pred)

        return y_pred, y_true

    def _ensemble_aggregation(self, y_pred: TensorBTSCM) -> TensorBTSC:
        """Aggregate ensemble dimension.

        Parameters
        ----------
        y_pred
            Predictions with ensemble dim, shape (B, T, C, M)

        Returns
        -------
            Aggregated predictions, shape (B, T, S, C)
        """
        return y_pred.mean(dim=-1)

    def _score(
        self, y_pred: TensorBTSC | TensorBTSCM, y_true: TensorBTSC | TensorBTSCM
    ) -> TensorBTC:
        """
        Compute metric reduced over spatial dims only.

        Expected input shape: (B, T, S, C)
        Expected output shape: (B, T, C)

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class MSE(BTSCMetric):
    """Mean Squared Error over spatial dims."""

    name: str = "mse"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims)


class MAE(BTSCMetric):
    """Mean Absolute Error over spatial dims."""

    name: str = "mae"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.mean((y_pred - y_true).abs(), dim=spatial_dims)


class NMAE(BTSCMetric):
    """Normalized Mean Absolute Error over spatial dims."""

    name: str = "nmae"

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm = torch.mean(torch.abs(y_true), dim=spatial_dims)
        return torch.mean((y_pred - y_true).abs(), dim=spatial_dims) / (norm + self.eps)


class NMSE(BTSCMetric):
    """Normalized Mean Squared Error over spatial dims."""

    name: str = "nmse"

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm = torch.mean(y_true**2, dim=spatial_dims)
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims) / (norm + self.eps)


class RMSE(BTSCMetric):
    """Root Mean Squared Error over spatial dims."""

    name: str = "rmse"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=spatial_dims))


class NRMSE(BTSCMetric):
    """Normalized Root Mean Squared Error over spatial dims."""

    name: str = "nrmse"

    def __init__(
        self,
        eps: float = 1e-7,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm = torch.mean(y_true**2, dim=spatial_dims)
        return torch.sqrt(
            torch.mean((y_pred - y_true) ** 2, dim=spatial_dims) / (norm + self.eps)
        )


class VMSE(BTSCMetric):
    """Variance Scaled Mean Squared Error over spatial dims."""

    name: str = "vmse"

    def __init__(
        self,
        eps: float = 1e-7,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        norm_var = torch.std(y_true, dim=spatial_dims) ** 2
        return torch.mean((y_pred - y_true) ** 2, dim=spatial_dims) / (
            norm_var + self.eps
        )


class VRMSE(BTSCMetric):
    """Variance-Scaled Root Mean Squared Error over spatial dims.

    Computes VRMSE = RMSE / std(y_true), where std is computed over spatial dims.
    """

    name: str = "vrmse"

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))

        norm_std = torch.std(y_true, dim=spatial_dims)

        return torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=spatial_dims)) / (
            norm_std + self.eps
        )


class LInfinity(BTSCMetric):
    """L-Infinity Norm over spatial dims."""

    name: str = "l_infinity"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        self.n_spatial_dims = self._infer_n_spatial_dims(y_pred)
        spatial_dims = tuple(range(-self.n_spatial_dims - 1, -1))
        return torch.max(
            torch.abs(y_pred - y_true).flatten(start_dim=spatial_dims[0], end_dim=-2),
            dim=-2,
        ).values


@cache
def _isotropic_binning(
    shape: tuple[int, ...],
    bins: int | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Isotropic frequency binning over FFT domain (Lola-style)."""
    k = []
    for s in shape:
        k_i = torch.fft.fftfreq(s, device=device)
        k.append(k_i)

    k2 = map(torch.square, k)
    k2_grid = torch.meshgrid(*k2, indexing="ij")
    k2_iso = torch.zeros_like(k2_grid[0])
    for component in k2_grid:
        k2_iso = k2_iso + component
    k_iso = torch.sqrt(k2_iso)

    if bins is None:
        bins = math.floor(math.sqrt(k_iso.ndim) * min(k_iso.shape) / 2)

    edges = torch.linspace(0, k_iso.max(), bins + 1, device=device)
    indices = torch.bucketize(k_iso.flatten(), edges)
    counts = torch.bincount(indices, minlength=bins + 1)

    return edges, counts, indices


def _isotropic_power_spectrum(
    y: TensorBTSC, n_spatial_dims: int
) -> tuple[Tensor, Tensor]:
    """Compute isotropic power spectrum with bucketized radial bins."""
    y_btc = y.movedim(-1, 2)  # (B, T, C, S...)
    spatial_shape = tuple(y_btc.shape[-n_spatial_dims:])

    edges, counts, indices = _isotropic_binning(spatial_shape, device=y.device)

    fft_dims = tuple(range(-n_spatial_dims, 0))
    spectrum = torch.fft.fftn(y_btc, dim=fft_dims, norm="ortho")
    power = torch.abs(spectrum).square().flatten(start_dim=-n_spatial_dims)

    power_iso = torch.zeros(
        (*power.shape[:-1], edges.numel()), dtype=y.dtype, device=y.device
    )
    power_iso = power_iso.scatter_add(dim=-1, index=indices.expand_as(power), src=power)

    counts = torch.clamp(counts, min=1).to(dtype=power_iso.dtype)
    counts = counts.view(*([1] * (power_iso.ndim - 1)), -1)
    power_iso = power_iso / counts

    # Drop DC component to match common isotropic spectrum conventions.
    return power_iso[..., 1:], edges[1:]


def _lola_eval_power_band_masks(
    freq_bins: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the four power-spectrum masks used in Lola eval.py."""
    # bins = torch.logspace(k[0].log2(), -1.0, steps=4, base=2)
    bins = torch.logspace(
        freq_bins[0].log2(),
        -1.0,
        steps=4,
        base=2,
        device=freq_bins.device,
    )

    m0 = torch.logical_and(bins[0] <= freq_bins, freq_bins <= bins[1])
    m1 = torch.logical_and(bins[1] <= freq_bins, freq_bins <= bins[2])
    m2 = torch.logical_and(bins[2] <= freq_bins, freq_bins <= bins[3])
    m3 = bins[3] <= freq_bins
    return m0, m1, m2, m3


def _power_spectrum_rmse_bands(
    y_pred: TensorBTSC,
    y_true: TensorBTSC,
    eps: float,
) -> tuple[TensorBTC, TensorBTC, TensorBTC, TensorBTC]:
    """Compute Lola-style per-band RMSE of relative isotropic power spectra."""
    n_spatial_dims = y_true.ndim - 3
    pred_spec, freq_bins = _isotropic_power_spectrum(y_pred, n_spatial_dims)
    true_spec, _ = _isotropic_power_spectrum(y_true, n_spatial_dims)

    m0, m1, m2, m3 = _lola_eval_power_band_masks(freq_bins)

    def _band_rmse(mask: Tensor) -> TensorBTC:
        # Small spatial grids can produce empty spectral bands.
        # Define RMSE over an empty band as zero to avoid NaNs in downstream
        # deterministic/stateful reductions.
        if not torch.any(mask):
            return torch.zeros(
                pred_spec.shape[:-1], dtype=pred_spec.dtype, device=pred_spec.device
            )
        # se_p = (1 - (p_v + eps) / (p_u + eps))^2
        se_p = torch.square(
            1.0 - (pred_spec[..., mask] + eps) / (true_spec[..., mask] + eps)
        )
        return torch.sqrt(torch.mean(se_p, dim=-1))

    return _band_rmse(m0), _band_rmse(m1), _band_rmse(m2), _band_rmse(m3)


class PowerSpectrumRMSE(BTSCMetric):
    """Average power spectrum RMSE across first three Lola eval bands."""

    name: str = "psrmse"

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__(
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.eps = eps

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        low, mid, high, _tail = _power_spectrum_rmse_bands(y_pred, y_true, eps=self.eps)
        return (low + mid + high) / 3.0


class PowerSpectrumRMSELow(PowerSpectrumRMSE):
    """Power spectrum RMSE in the low-frequency band."""

    name: str = "psrmse_low"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        low, _, _, _ = _power_spectrum_rmse_bands(y_pred, y_true, eps=self.eps)
        return low


class PowerSpectrumRMSEMid(PowerSpectrumRMSE):
    """Power spectrum RMSE in the mid-frequency band."""

    name: str = "psrmse_mid"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        _, mid, _, _ = _power_spectrum_rmse_bands(y_pred, y_true, eps=self.eps)
        return mid


class PowerSpectrumRMSEHigh(PowerSpectrumRMSE):
    """Power spectrum RMSE in the high-frequency band."""

    name: str = "psrmse_high"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        _, _, high, _ = _power_spectrum_rmse_bands(y_pred, y_true, eps=self.eps)
        return high


class PowerSpectrumRMSETail(PowerSpectrumRMSE):
    """Power spectrum RMSE in the Lola high-frequency tail band."""

    name: str = "psrmse_tail"

    def _score(self, y_pred: TensorBTSC, y_true: TensorBTSC) -> TensorBTC:
        _, _, _, tail = _power_spectrum_rmse_bands(y_pred, y_true, eps=self.eps)
        return tail
