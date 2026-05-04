import torch
from torch import nn

from autocast.metrics.ensemble import _alpha_fair_crps_score, _common_crps_score
from autocast.types import Tensor, TensorBTSC, TensorBTSCM


class EnsembleLoss(nn.Module):
    """Base class for ensemble losses."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: TensorBTSCM, targets: TensorBTSC) -> Tensor:
        """Compute the loss.

        Args:
            preds: Predictions of shape (B, ..., M)
            targets: Targets of shape (B, ...)

        Returns
        -------
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

    def __init__(self, alpha: float = 0.95, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.alpha = alpha

    def _compute_score(self, preds: TensorBTSCM, targets: TensorBTSC) -> Tensor:
        return _alpha_fair_crps_score(preds, targets, alpha=self.alpha)


class EnsembleMAELoss(EnsembleLoss):
    """Mean absolute error computed from the ensemble mean forecast."""

    def _compute_score(self, preds: TensorBTSCM, targets: TensorBTSC) -> Tensor:
        ensemble_mean = preds.mean(dim=-1)
        return (ensemble_mean - targets).abs()
