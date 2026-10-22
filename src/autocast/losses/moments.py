"""Finite-ensemble-corrected scores for spectral mean and raw second moment."""

import torch
from torch import nn

from autocast.losses.decomposition import _align_targets
from autocast.losses.spectral import FourierDecomposition, FourierNorm
from autocast.types import FieldTensor, Tensor, TensorInexactBTSC


class SpectralMomentLoss(nn.Module):
    r"""Jointly score the mean Fourier coefficient and mean member power.

    For independent members ``z_m`` and an observation ``y`` at each Fourier
    coefficient, the two scores are

    .. math::
        |\bar z-y|^2-S_z/M,\qquad
        (\overline{|z|^2}-|y|^2)^2-S_{|z|^2}/M,

    where both sample variances use divisor ``M-1``. The subtractions remove
    the sampling variance of each estimated ensemble moment. Scores may be
    negative and are not clipped. Neither predicted mean is detached.

    Average scores over inputs, leads, modes and channels AFTER forming each
    per-input score. All full-FFT modes, including DC, receive equal weight.
    There is no radial pooling, fitted scale or conditional reference cache.
    This identifies the conditional mean and raw second moment in expectation,
    hence anomaly power when the mean is correct, not the full joint law.
    Squared power still gives weak restoring gradients at very small power.

    ``forward`` adds the two component scores with unit coefficients. They
    have different physical units: use consistently normalized fields and
    calibrate the outer weight when composing with the primary forecast loss.
    Half precision fields are promoted before the FFT and fourth moments.

    Args:
        norm: Full spatial FFT normalization; defaults to ``ortho``.
    """

    requires_ambient_predictions = True

    def __init__(self, norm: FourierNorm = "ortho") -> None:
        super().__init__()
        self.fourier = FourierDecomposition(norm=norm)

    def components(
        self, preds: FieldTensor, targets: TensorInexactBTSC
    ) -> tuple[Tensor, Tensor]:
        """Return scalar corrected mean and raw-power scores separately."""
        _align_targets(preds, targets)
        if targets.ndim < 4 or preds.ndim != targets.ndim + 1:
            msg = "Spectral moments require BT...SC targets and a final member axis."
            raise ValueError(msg)
        if preds.shape[:-1] != targets.shape or preds.shape[-1] < 2:
            msg = "Spectral moments require matching fields and at least 2 members."
            raise ValueError(msg)
        if not preds.is_floating_point() or not targets.is_floating_point():
            msg = "Spectral moment inputs must be real floating fields."
            raise ValueError(msg)
        if preds.device != targets.device:
            msg = "Spectral moment fields must share a device."
            raise ValueError(msg)
        if any(size == 0 for size in targets.shape):
            msg = "Spectral moment fields must be nonempty."
            raise ValueError(msg)
        dtype = torch.promote_types(preds.dtype, targets.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        n_spatial_dims = targets.ndim - 3
        z = self.fourier(preds.to(dtype=dtype), n_spatial_dims=n_spatial_dims)[
            "fourier"
        ]
        y = self.fourier(targets.to(dtype=dtype), n_spatial_dims=n_spatial_dims)[
            "fourier"
        ]
        members = preds.shape[-1]
        mean = z.mean(dim=-1)
        spread = (z - mean.unsqueeze(-1)).abs().square().sum(dim=-1) / (members - 1)
        powers = z.abs().square()
        mean_power = powers.mean(dim=-1)
        power_spread = (powers - mean_power.unsqueeze(-1)).square().sum(dim=-1) / (
            members - 1
        )
        mean_score = (mean - y).abs().square() - spread / members
        power_score = (mean_power - y.abs().square()).square() - power_spread / members
        return mean_score.mean(), power_score.mean()

    def forward(self, preds: FieldTensor, targets: TensorInexactBTSC) -> Tensor:
        """Return the sum of the corrected mean and raw-power scores."""
        mean_score, power_score = self.components(preds, targets)
        return mean_score + power_score
