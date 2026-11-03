"""Local velocity-vector scores on periodic two-dimensional fields."""

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from autocast.losses.decomposition import FieldComponents, FieldDecomposition
from autocast.losses.ensemble import CoefficientCRPSLoss
from autocast.types import FieldTensor, Tensor, TensorInexactBTSC


class HelmholtzDecomposition(FieldDecomposition):
    """Split a real periodic 2D velocity into solenoidal and divergent parts.

    Input shape is ``(B, T, X, Y, 2[, M])``. The two channels are velocity
    along the first and second spatial axes respectively, in common physical
    units. Each nonzero wavevector has the orthogonal projector
    ``P_div = k k.T / |k|**2``; the solenoidal projector is ``I - P_div``.

    On even grids the Nyquist frequency of each axis is set to zero, following
    the real pseudospectral derivative convention. This preserves conjugate
    symmetry and real, orthogonal projections. Joint derivative-null modes
    (DC, plus checkerboard modes on even grids) are assigned wholly to the
    solenoidal part. Thus that part includes harmonic modes and is not solely
    nonzero-vorticity flow. No velocity modes are discarded. Divergence/curl
    diagnostics must use the same derivative convention to verify the split.

    Args:
        domain_lengths: Positive physical periods along X and Y. Their common
            multiplier does not affect the projectors, but their ratio does.
    """

    def __init__(self, domain_lengths: Sequence[int | float] = (1.0, 1.0)) -> None:
        super().__init__()
        if len(domain_lengths) != 2 or any(
            not math.isfinite(length) or length <= 0 for length in domain_lengths
        ):
            msg = "domain_lengths must contain two finite positive periods."
            raise ValueError(msg)
        self.domain_lengths = tuple(float(length) for length in domain_lengths)

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        if n_spatial_dims != 2 or tensor.ndim not in (5, 6) or tensor.shape[4] != 2:
            msg = "HelmholtzDecomposition requires (B, T, X, Y, 2[, M]) fields."
            raise ValueError(msg)
        if not tensor.is_floating_point():
            msg = "HelmholtzDecomposition requires real floating-point fields."
            raise TypeError(msg)
        if tensor.shape[2] == 0 or tensor.shape[3] == 0:
            msg = "Spatial dimensions must be nonempty."
            raise ValueError(msg)
        tensor = tensor.to(torch.promote_types(tensor.dtype, torch.float32))
        coefficients = torch.fft.fftn(tensor, dim=(2, 3), norm="ortho")
        # Removing the channel axis leaves (B,T,X,Y[,M]); wavevectors broadcast
        # over the batch, lead and optional member axes, never over each other.
        wavevectors = []
        for axis, length in zip((2, 3), self.domain_lengths, strict=True):
            size = tensor.shape[axis]
            frequency = torch.fft.fftfreq(
                size, d=length / size, dtype=tensor.dtype, device=tensor.device
            )
            if size % 2 == 0:
                frequency[size // 2] = 0
            shape = [1] * (tensor.ndim - 1)
            shape[axis] = size
            wavevectors.append(frequency.reshape(shape))
        kx, ky = wavevectors
        squared_norm = kx.square() + ky.square()
        nonzero = squared_norm > 0
        inverse_norm = torch.where(nonzero, squared_norm, 1.0).reciprocal()
        inverse_norm = torch.where(nonzero, inverse_norm, 0.0)
        potential = (
            kx * coefficients.select(4, 0) + ky * coefficients.select(4, 1)
        ) * inverse_norm
        divergent_hat = torch.stack((kx * potential, ky * potential), dim=4)
        divergent = torch.fft.ifftn(divergent_hat, dim=(2, 3), norm="ortho").real
        return {"solenoidal": tensor - divergent, "divergent": divergent}


class VelocityScoreLoss(nn.Module):
    """Score local joint velocities, optionally after Helmholtz decomposition.

    Predictions have shape ``(B, T, X, Y, C, M)``, targets omit M. Channels
    selected by ``channels`` must represent velocity along X and Y. For
    normalized inputs, ``channel_stds`` restores their physical scales before
    projection; the common ``velocity_scale`` then supplies score units.
    Fixed channel means need not be restored because they cancel in all
    observation/member and member/member distances after a linear transform.

    ``mode='joint'`` scores the full local two-vector with the ordinary
    bivariate almost-fair Energy Score. ``mode='helmholtz'`` averages that
    score separately on the solenoidal and divergent velocity fields. The
    solenoidal component includes derivative-null harmonic modes as documented
    by :class:`HelmholtzDecomposition`; these therefore carry half the joint
    score's weight in the decomposed objective. Components are not separately
    standardized. Neither mode subtracts an ensemble mean or scores powers.

    Complex packing ``u + i*v`` reuses the existing unsmoothed coefficient
    score: its modulus is the Euclidean distance of this local two-vector.
    This does not join different grid points into a single feature vector.

    Args:
        channel_stds: Two positive fixed physical scales from training-data
            normalization. Supply ones for already physical input fields.
        channels: Indices of the X and Y velocity channels, in that order.
        velocity_scale: Positive common scale, defaulting to the RMS of the
            two channel standard deviations. Use the same value in both modes.
        mode: Full joint velocity or the mean of the two decomposed scores.
        domain_lengths: Physical periods along X and Y.
        alpha: Almost-fair coefficient-score parameter in [0, 1].
        reduction: ``mean``, ``sum`` or ``none``. Unreduced scores have shape
            ``(B, T, X, Y, 1)`` and retain spatial locations.
    """

    requires_ambient_predictions = True

    def __init__(
        self,
        channel_stds: Sequence[int | float] = (1.0, 1.0),
        *,
        channels: Sequence[int] = (0, 1),
        velocity_scale: int | float | None = None,
        mode: Literal["joint", "helmholtz"] = "joint",
        domain_lengths: Sequence[int | float] = (1.0, 1.0),
        alpha: int | float = 0.95,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if (
            len(channels) != 2
            or len(set(channels)) != 2
            or any(
                not isinstance(channel, int) or isinstance(channel, bool) or channel < 0
                for channel in channels
            )
        ):
            msg = "channels must contain two distinct non-negative integer indices."
            raise ValueError(msg)
        if len(channel_stds) != 2 or any(
            not math.isfinite(std) or std <= 0 for std in channel_stds
        ):
            msg = "channel_stds must contain two finite positive training scales."
            raise ValueError(msg)
        if velocity_scale is None:
            velocity_scale = math.hypot(*channel_stds) / math.sqrt(2)
        if not math.isfinite(velocity_scale) or velocity_scale <= 0:
            msg = "velocity_scale must be finite and positive."
            raise ValueError(msg)
        if any(not math.isfinite(std / velocity_scale) for std in channel_stds):
            msg = "channel_stds / velocity_scale must remain finite."
            raise ValueError(msg)
        if mode not in ("joint", "helmholtz"):
            raise ValueError(f"Unknown velocity score mode {mode!r}.")
        self.channels = tuple(channels)
        self.channel_stds = tuple(float(std) for std in channel_stds)
        self.velocity_scale = float(velocity_scale)
        self.mode = mode
        self.decomposition = HelmholtzDecomposition(domain_lengths)
        self.score = CoefficientCRPSLoss(alpha=alpha, reduction=reduction)

    def _velocity(self, tensor: Tensor) -> Tensor:
        tensor = tensor.to(torch.promote_types(tensor.dtype, torch.float32))
        return torch.stack(
            [
                tensor.select(4, channel) * (std / self.velocity_scale)
                for channel, std in zip(self.channels, self.channel_stds, strict=True)
            ],
            dim=4,
        )

    @staticmethod
    def _pack(velocity: Tensor) -> Tensor:
        return torch.complex(velocity.narrow(4, 0, 1), velocity.narrow(4, 1, 1))

    def forward(self, preds: FieldTensor, targets: TensorInexactBTSC) -> Tensor:
        if targets.ndim != 5 or preds.ndim != 6 or preds.shape[:-1] != targets.shape:
            msg = "VelocityScoreLoss requires (B,T,X,Y,C,M) and (B,T,X,Y,C) fields."
            raise ValueError(msg)
        if max(self.channels) >= targets.shape[4]:
            msg = "Velocity channel index exceeds the available field channels."
            raise ValueError(msg)
        if not preds.is_floating_point() or not targets.is_floating_point():
            msg = "VelocityScoreLoss requires real floating-point fields."
            raise TypeError(msg)
        preds, targets = self._velocity(preds), self._velocity(targets)
        if self.mode == "joint":
            return self.score(self._pack(preds), self._pack(targets))
        pred_parts = self.decomposition(preds, n_spatial_dims=2)
        target_parts = self.decomposition(targets, n_spatial_dims=2)
        scores = [
            self.score(self._pack(pred_parts[name]), self._pack(target_parts[name]))
            for name in ("solenoidal", "divergent")
        ]
        return torch.stack(scores).mean(dim=0)
