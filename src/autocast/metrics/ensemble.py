import numpy as np
import torch
from einops import rearrange, repeat
from torchmetrics import Metric

from autocast.types import TensorBTC, TensorBTSC
from autocast.types.types import TensorBTSCM


class EnsembleBaseMetric(Metric):
    """
    Base class for ensemble metrics that operate on spatial tensors.

    Checks input types and shapes and converts to torch.Tensor.

    Args:
        reduce_all: If True, return scalar by averaging over all non-batch dims
        dist_sync_on_step: Synchronize metric state across processes at each forward()
    """

    name: str = "base_metric"

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.reduce_all = reduce_all

        # States shared by all derived metrics
        self.add_state("sum_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        # Internal flag to set shape of sum_score
        self._initialized = False

    def _check_input(
        self,
        y_pred: torch.Tensor | np.ndarray,
        y_true: torch.Tensor | np.ndarray,
    ) -> tuple[TensorBTSC, TensorBTSC]:
        """
        Check types and shapes and converts inputs to torch.Tensor.

        Args:
            y_pred: Predictions of shape (B, T, S, C)
            y_true: Ground truth of shape (B, T, S, C)

        Returns
        -------
            Tuple of (y_pred, y_true) as torch.Tensors
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)

        assert isinstance(y_pred, torch.Tensor), (
            f"y_pred must be a torch.Tensor or np.ndarray, got {type(y_pred)}"
        )
        assert isinstance(y_true, torch.Tensor), (
            f"y_true must be a torch.Tensor or np.ndarray, got {type(y_true)}"
        )

        assert y_pred.shape == y_true.shape, (
            f"y_pred and y_true must have the same shape, "
            f"got {y_pred.shape} and {y_true.shape}"
        )

        assert y_pred.ndim >= 4, (
            f"y_pred has {y_pred.ndim} dimensions, should be at least 4, "
            f"following the pattern(B, T, S, C)"
        )

        return y_pred, y_true

    def score(
        self,
        y_pred: TensorBTSCM,
        y_true: TensorBTSC,
    ) -> TensorBTC:
        """
        Compute metric reduced over spatial dims only.

        Expected input shape: (B, T, S, C, M)
        Expected output shape: (B, T, C)

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def update(
        self,
        y_pred: TensorBTSCM | np.ndarray,
        y_true: TensorBTSC | np.ndarray,
    ) -> None:
        """
        Update metric state with a batch of predictions and targets.

        Args:
            y_pred: Predictions of shape (B, T, S, C)
            y_true: Ground truth of shape (B, T, S, C)
        """
        y_pred, y_true = self._check_input(y_pred, y_true)

        # (B, T, S, C) -> (B, T, C)
        score_spatial = self.score(y_pred, y_true)

        if score_spatial.ndim != 3:
            raise ValueError(
                f"score must return shape (B, T, C), got {score_spatial.shape}"
            )

        batch_size = score_spatial.shape[0]

        # Sum over batch dimension: (B, T, C) -> (T, C)
        score_summed = torch.sum(score_spatial, dim=0)

        # Lazily set correct shape for sum_score on first batch
        if not self._initialized:
            self.sum_score = torch.zeros_like(score_summed)
            self._initialized = True

        self.sum_score += score_summed
        self.total_samples += batch_size

    def compute(self) -> torch.Tensor:
        """
        Compute final metric value.

        Returns
        -------
            Tensor of shape (T, C) or scalar if reduce_all=True
        """
        if self.total_samples == 0:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)

        score = self.sum_score / self.total_samples

        if self.reduce_all:
            # Average over time and channels
            return score.mean()

        return score

    def reset(self) -> None:
        """Reset metric state and initialization flag."""
        super().reset()
        self._initialized = False

    def _infer_n_spatial_dims(self, tensor: TensorBTSC) -> int:
        """Infer number of spatial dimensions from tensor shape."""
        return tensor.ndim - 3  # Subtract B, T, C


def _common_crps_score(
    y_pred: TensorBTSCM, y_true: TensorBTSC, adjustment_factor: float
) -> TensorBTC:
    """
    Compute CRPS reduced over spatial dims only.

    Expected input shape: (B, T, S, C, M)
    Expected output shape: (B, T, C)

    Args:
        y_pred: Predictions of shape (B, T, S, C, M)
        y_true: Ground truth of shape (B, T, S, C)
        adjustment_factor: Factor to adjust the second term in CRPS calculation

    Returns
    -------
        Tensor of shape (B, T, C) with CRPS scores
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


class CRPS(EnsembleBaseMetric):
    """
    Continuous Ranked Probability Score (CRPS) for ensemble forecasts.

    References
    ----------
    Hersbach, H., 2000: Decomposition of the Continuous Ranked Probability Score for
    Ensemble Prediction Systems. Wea. Forecasting, 15, 559-570,
    https://doi.org/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2.
    """

    name: str = "crps"

    def score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTC:
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


class FairCRPS(EnsembleBaseMetric):
    """
    Fair Continuous Ranked Probability Score (fCRPS) for ensemble forecasts.

    References
    ----------
    Ferro, C.A.T. (2014), Fair scores for ensemble forecasts. Q.J.R. Meteorol. Soc.,
    140: 1917-1923. https://doi.org/10.1002/qj.2270
    """

    name: str = "fcrps"

    def score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTC:
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


class AlphaFairCRPS(EnsembleBaseMetric):
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

    def score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTC:
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
