import numpy as np
import torch
from einops import rearrange, repeat

from autocast.metrics.base import BaseMetric
from autocast.types import ArrayLike, Tensor, TensorBTC, TensorBTSC, TensorBTSCM


class BTSCMMetric(BaseMetric[TensorBTSCM, TensorBTSC]):
    """
    Base class for ensemble metrics that operate on spatial tensors.

    Checks input types and shapes and converts to Tensor.

    Args:
        reduce_all: If True, return scalar by averaging over all non-batch dims
        dist_sync_on_step: Synchronize metric state across processes at each forward()
    """

    name: str = "ensemble_base_metric"

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

        if y_pred.ndim < 5:
            raise ValueError(
                f"y_pred has {y_pred.ndim} dimensions, should be at least 5, "
                f"following the pattern (B, T, S, C, M)"
            )

        if y_pred.shape[:-1] != y_true.shape:
            raise ValueError(
                f"y_pred and y_true must have the same shape except for the last "
                f"dimension (ensemble members). Got {y_pred.shape} and {y_true.shape}"
            )

        return y_pred, y_true


def _common_crps_score(
    y_pred: TensorBTSCM, y_true: TensorBTSC, adjustment_factor: float
) -> TensorBTSC:
    """
    Compute CRPS reduced over spatial dims only.

    Expected input shape: (B, T, S, C, M)
    Expected output shape: (B, T, S, C)

    Args:
        y_pred: Predictions of shape (B, T, S, C, M)
        y_true: Ground truth of shape (B, T, S, C)
        adjustment_factor: Factor to adjust the second term in CRPS calculation

    Returns
    -------
        Tensor of shape (B, T, S, C) with CRPS scores
    """
    # Expand y_true to match ensemble dimension
    n_ensemble = y_pred.shape[-1]
    y_true_expanded = repeat(y_true, "... -> ... m", m=n_ensemble)  # (B, T, S, C, M)

    # Compute CRPS using the formula
    term1: TensorBTSC = torch.mean(torch.abs(y_pred - y_true_expanded), dim=-1)
    term2: TensorBTSC = (
        0.5
        * torch.mean(
            torch.abs(
                rearrange(y_pred, "... m -> ... 1 m")  # (B, T, S, C, 1, M)
                - rearrange(y_pred, "... m -> ... m 1")  # (B, T, S, C, M, 1)
            ),  # (B, T, S, C, M, M)
            dim=(-2, -1),  # (B, T, S, C)
        )
        * adjustment_factor  # e.g. for FairCRPS this is M / (M-1)
    )

    crps: TensorBTSC = term1 - term2

    return crps


class CRPS(BTSCMMetric):
    """
    Continuous Ranked Probability Score (CRPS) for ensemble forecasts.

    References
    ----------
    Hersbach, H., 2000: Decomposition of the Continuous Ranked Probability Score for
    Ensemble Prediction Systems. Wea. Forecasting, 15, 559-570,
    https://doi.org/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2.
    """

    name: str = "crps"

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTC:
        """
        Compute CRPS reduced over spatial dims only.

        Expected input shape: (B, T, S, C, M)
        Expected output shape: (B, T, C)

        Args:
            y_pred: Predictions of shape (B, T, S, C, M)
            y_true: Ground truth of shape (B, T, S, C)

        Returns
        -------
            Tensor of shape (B, T, C) with CRPS scores
        """
        crps = _common_crps_score(y_pred, y_true, adjustment_factor=1.0)
        # Reduce over spatial dimensions: (B, T, S, C) -> (B, T, C)
        n_spatial_dims = self._infer_n_spatial_dims(crps)
        crps_reduced = crps.mean(dim=tuple(range(2, 2 + n_spatial_dims)))

        return crps_reduced


class FairCRPS(BTSCMMetric):
    """
    Fair Continuous Ranked Probability Score (fCRPS) for ensemble forecasts.

    References
    ----------
    Ferro, C.A.T. (2014), Fair scores for ensemble forecasts. Q.J.R. Meteorol. Soc.,
    140: 1917-1923. https://doi.org/10.1002/qj.2270
    """

    name: str = "fcrps"

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTC:
        """
        Compute CRPS reduced over spatial dims only.

        Expected input shape: (B, T, S, C, M)
        Expected output shape: (B, T, C)

        Args:
            y_pred: Predictions of shape (B, T, S, C, M)
            y_true: Ground truth of shape (B, T, S, C)

        Returns
        -------
            Tensor of shape (B, T, C) with CRPS scores
        """
        # Expand y_true to match ensemble dimension
        n_ensemble = y_pred.shape[-1]
        crps = _common_crps_score(
            y_pred, y_true, adjustment_factor=n_ensemble / (n_ensemble - 1)
        )

        # Reduce over spatial dimensions: (B, T, S, C) -> (B, T, C)
        n_spatial_dims = self._infer_n_spatial_dims(crps)
        crps_reduced = crps.mean(dim=tuple(range(2, 2 + n_spatial_dims)))

        return crps_reduced


class AlphaFairCRPS(BTSCMMetric):
    r"""
    Almost Fair Continuous Ranked Probability Score (afCRPS) (stable form).

    Notes
    -----
    Definition:
    .. math::
        \text{afCRPS}_\alpha := \alpha \text{fCRPS} + (1-\alpha) \text{CRPS}

    Implementation follows eq. (4) in the AIFS-CRPS paper: rearranged sum of positive
    terms to avoid instability.

    References
    ----------
    Lang, S., Alexe, M., Clare, M. C., Roberts, C., Adewoyin, R., Bouallègue, Z. B.,
    ... & Leutbecher, M. (2024).
    AIFS-CRPS: ensemble forecasting using a model trained with a loss function based on
    the continuous ranked probability score. arXiv preprint arXiv:2412.15832.
    """

    name: str = "afcrps"

    def __init__(self, alpha: float = 0.95):
        super().__init__()
        # alpha close to 1 is close to fair CRPS, lower values tend toward standard CRPS
        assert 0 < alpha <= 1, "alpha must be in (0,1]"
        self.alpha = alpha

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTC:
        """
        Compute afCRPS reduced over spatial dims only.

        Args:
            y_pred: (B, T, S, C, M)
            y_true: (B, T, S, C)

        Returns
        -------
            afCRPS: (B, T, C)
        """
        # Expand y_true to match ensemble dimension
        n_ensemble = y_pred.shape[-1]
        y_true_m = repeat(y_true, "... -> ... m", m=n_ensemble)

        eps = (1.0 - self.alpha) / n_ensemble

        abs_diff_ens = torch.abs(
            rearrange(y_pred, "... m -> ... 1 m")
            - rearrange(y_pred, "... m -> ... m 1")
        )  # (B, T, S, C, M, M)

        abs_diff_truth = torch.abs(y_pred - y_true_m)  # (B, T, S, C, M)

        # build the stable sum over pairwise terms
        # zero the diagonal (j == k) since afCRPS sums only off-diagonal pairs.
        mask = ~torch.eye(n_ensemble, dtype=torch.bool, device=y_pred.device)  # (M, M)
        # (M, M) -> (1, 1, 1, 1, M, M)
        mask = mask.view(*([1] * (abs_diff_ens.ndim - 2)), n_ensemble, n_ensemble)

        # pairwise sum term: |x_j - y| + |x_k - y| - (1 - eps) * |x_j - x_k|
        term_pair = (
            rearrange(abs_diff_truth, "... m -> ... m 1")
            + rearrange(abs_diff_truth, "... m -> ... 1 m")
            - (1.0 - eps) * abs_diff_ens  # (..., M, M)
        )  # (B, T, S, C, M, M)

        # apply mask to set j == k terms to zero
        term_pair = term_pair.masked_fill(~mask, 0.0)  # (B, T, S, C, M, M)

        # sum over all off-diagonal pairs
        sum_pair = term_pair.sum(dim=(-1, -2))  # (B, T, S, C)

        # normalization factor = 2 M (M - 1)
        norm = 2.0 * n_ensemble * (n_ensemble - 1)

        afcrps = sum_pair / norm

        # Reduce over spatial dimensions: (B, T, S, C) -> (B, T, C)
        n_spatial_dims = self._infer_n_spatial_dims(afcrps)
        afcrps_reduced = afcrps.mean(dim=tuple(range(2, 2 + n_spatial_dims)))

        return afcrps_reduced
