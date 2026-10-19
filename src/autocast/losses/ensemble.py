from typing import Literal, cast

import torch
from torch import nn

from autocast.metrics.ensemble import _alpha_fair_crps_score, _common_crps_score
from autocast.types import (
    Tensor,
    TensorBTSC,
    TensorBTSCM,
    TensorInexactBTSC,
    TensorInexactBTSCM,
)

Reduction = Literal["mean", "sum", "none"]


class EnsembleLoss(nn.Module):
    """Base class for ensemble losses."""

    reduction: Reduction

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction {reduction!r}.")
        self.reduction = cast(Reduction, reduction)

    def forward(self, preds: TensorBTSCM, targets: TensorBTSC) -> Tensor:
        """Compute the loss.

        Args:
            preds: Predictions of shape (B, ..., M)
            targets: Targets of shape (B, ...)

        Returns:
            Scalar loss (or tensor if reduction is 'none')
        """
        # Ensure targets do not have the ensemble dimension
        if targets.ndim == preds.ndim:
            msg = (
                "Targets should not have the ensemble dimension. "
                f"Got preds shape {preds.shape} and targets shape {targets.shape}."
            )
            raise ValueError(msg)

        score = self._compute_score(preds, targets)

        if self.reduction == "mean":
            return torch.mean(score)
        if self.reduction == "sum":
            return torch.sum(score)
        return score

    def _compute_score(self, preds: TensorBTSCM, targets: TensorBTSC) -> Tensor:
        raise NotImplementedError


class CRPSLoss(EnsembleLoss):
    """Continuous Ranked Probability Score (CRPS) Loss."""

    def _compute_score(self, preds: TensorBTSCM, targets: TensorBTSC) -> Tensor:
        # _common_crps_score returns (B, T, S, C)
        return _common_crps_score(preds, targets, adjustment_factor=1.0)


class FairCRPSLoss(EnsembleLoss):
    """Fair Continuous Ranked Probability Score (fCRPS) Loss."""

    def _compute_score(self, preds: TensorBTSCM, targets: TensorBTSC) -> Tensor:
        n_ensemble = preds.shape[-1]
        if n_ensemble <= 1:
            raise ValueError(
                "FairCRPSLoss requires at least 2 ensemble members "
                f"to compute the spread adjustment term. Got {n_ensemble}."
            )

        adjustment_factor = n_ensemble / (n_ensemble - 1)
        return _common_crps_score(preds, targets, adjustment_factor=adjustment_factor)


class AlphaFairCRPSLoss(EnsembleLoss):
    """Alpha-Fair Continuous Ranked Probability Score (afCRPS) Loss."""

    def __init__(self, alpha: int | float = 0.95, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.alpha = float(alpha)

    def _compute_score(self, preds: TensorBTSCM, targets: TensorBTSC) -> Tensor:
        return _alpha_fair_crps_score(preds, targets, alpha=self.alpha)


class EnsembleMAELoss(EnsembleLoss):
    """Mean absolute error computed from the ensemble mean forecast."""

    def _compute_score(self, preds: TensorBTSCM, targets: TensorBTSC) -> Tensor:
        ensemble_mean = preds.mean(dim=-1)
        return (ensemble_mean - targets).abs()


class CoefficientCRPSLoss(EnsembleLoss):
    r"""Score real or complex coefficients independently by absolute distance.

    For ``M >= 2`` and ``epsilon = (1 - alpha) / M``, each coefficient's
    score is

    .. math::
        \frac{1}{M}\sum_i |z_i-y|
        - \frac{1-\epsilon}{2M(M-1)}\sum_{i\ne j}|z_i-z_j|.

    Complex absolute values are moduli, retaining phase and magnitude. This
    is the coefficient-wise two-dimensional Energy Score called spectral
    CRPS in https://arxiv.org/html/2511.23043v2#S2.SS2. For real coefficients
    it agrees with CRPS values, but may select different subgradients at ties
    from a sort-based implementation. No axes are joined into feature vectors.

    Args:
        alpha: Fairness in [0, 1]. Zero gives the plug-in estimator and one
            gives the fair estimator. A single member is allowed only at zero,
            returning its absolute error.
        reduction: ``none`` preserves the target shape; ``mean`` or ``sum``
            reduces all batch, time, spatial/frequency and channel axes.

    Notes:
        Predictions have shape ``(B, T, S1, ..., Sd, C, M)`` and targets omit
        the final member axis. Inputs must be floating point or complex and
        on the same device. Computation uses their common PyTorch dtype with
        at least float32 precision (complex64 for complex inputs), including
        under autocast. Scores use the corresponding real dtype; float64 and
        complex128 precision is preserved. Gradients retain each input's dtype.
        Pairwise work and storage grow quadratically with ensemble size.
    """

    def __init__(self, alpha: int | float = 0.0, reduction: str = "mean") -> None:
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be finite and in [0, 1], got {alpha}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction {reduction!r}.")
        super().__init__(reduction=reduction)
        self.alpha = float(alpha)

    def forward(self, preds: TensorInexactBTSCM, targets: TensorInexactBTSC) -> Tensor:
        """Return independent coefficient scores with the requested reduction."""
        if preds.ndim != targets.ndim + 1 or preds.shape[:-1] != targets.shape:
            msg = (
                "Predictions must match targets with a final ensemble axis. "
                f"Got {tuple(preds.shape)} and {tuple(targets.shape)}."
            )
            raise ValueError(msg)
        if not all(t.is_floating_point() or t.is_complex() for t in (preds, targets)):
            msg = "Predictions and targets must be floating point or complex tensors."
            raise TypeError(msg)
        if preds.device != targets.device:
            msg = "Predictions and targets must be on the same device."
            raise ValueError(msg)
        n_ensemble = preds.shape[-1]
        if n_ensemble == 0:
            msg = "CoefficientCRPSLoss requires at least one ensemble member."
            raise ValueError(msg)
        if n_ensemble == 1 and self.alpha != 0:
            msg = "A fair spread term requires at least 2 members; use alpha=0 for one."
            raise ValueError(msg)

        score = self._compute_score(preds, targets)
        if self.reduction == "mean":
            return score.mean()
        if self.reduction == "sum":
            return score.sum()
        return score

    def _compute_score(self, preds: Tensor, targets: Tensor) -> Tensor:
        # Promote before subtraction too: a wider sum cannot repair an
        # already-overflowed half-precision distance.
        dtype = torch.promote_types(
            torch.promote_types(preds.dtype, targets.dtype), torch.float32
        )
        preds, targets = preds.to(dtype), targets.to(dtype)
        n_ensemble = preds.shape[-1]
        truth = (preds - targets.unsqueeze(-1)).abs().mean(dim=-1)
        if n_ensemble == 1:
            return truth

        # Each distinct pair contributes once. Direct abs also chooses a finite
        # zero subgradient at coincidences, without smoothing the objective.
        first, second = torch.triu_indices(
            n_ensemble, n_ensemble, offset=1, device=preds.device
        )
        spread = (preds[..., first] - preds[..., second]).abs().sum(dim=-1)
        coefficient = (1.0 - self.alpha) / n_ensemble**2 + self.alpha / (
            n_ensemble * (n_ensemble - 1)
        )
        return truth - coefficient * spread
