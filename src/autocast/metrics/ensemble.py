import abc
from typing import Literal

import numpy as np
import torch
from einops import rearrange

from autocast.metrics.base import BaseMetric
from autocast.types import (
    ArrayLike,
    Tensor,
    TensorBSC,
    TensorBTC,
    TensorBTSC,
    TensorBTSCM,
)

VectorDimsOption = Literal[
    "spatial",
    "temporal",
    "spatial_temporal",
    "spatial_temporal_channels",
]


def _normalize_vector_dims(vector_dims: str) -> VectorDimsOption:
    valid_options = (
        "spatial",
        "temporal",
        "spatial_temporal",
        "spatial_temporal_channels",
    )
    if vector_dims not in valid_options:
        raise ValueError(
            "vector_dims must be one of "
            "('spatial', 'temporal', 'spatial_temporal', "
            f"'spatial_temporal_channels'), got {vector_dims!r}"
        )
    return vector_dims


def _vectorize_selected_dims(
    y_pred: TensorBTSCM,
    y_true: TensorBTSC,
    vector_dims: VectorDimsOption,
) -> tuple[Tensor, Tensor]:
    if vector_dims == "spatial":
        y_true_vector = rearrange(y_true, "b t ... c -> b t c (...)")
        y_pred_vector = rearrange(y_pred, "b t ... c m -> b t c (...) m")
        return y_pred_vector, y_true_vector

    if vector_dims == "temporal":
        y_true_vector = rearrange(y_true, "b t ... c -> b ... c t")
        y_pred_vector = rearrange(y_pred, "b t ... c m -> b ... c t m")
        return y_pred_vector, y_true_vector

    if vector_dims == "spatial_temporal":
        y_true_vector = rearrange(y_true, "b t ... c -> b c (t ...)")
        y_pred_vector = rearrange(y_pred, "b t ... c m -> b c (t ...) m")
        return y_pred_vector, y_true_vector

    y_true_vector = rearrange(y_true, "b t ... c -> b (t ... c)")
    y_pred_vector = rearrange(y_pred, "b t ... c m -> b (t ... c) m")
    return y_pred_vector, y_true_vector


def _restore_vector_dims_singletons(
    score: Tensor,
    y_true: TensorBTSC,
    vector_dims: VectorDimsOption,
) -> Tensor:
    n_spatial_dims = y_true.ndim - 3

    if vector_dims == "spatial":
        for _ in range(n_spatial_dims):
            score = rearrange(score, "b t ... -> b t 1 ...")
        return score

    if vector_dims == "temporal":
        return rearrange(score, "b ... -> b 1 ...")

    if vector_dims == "spatial_temporal":
        for _ in range(n_spatial_dims + 1):
            score = rearrange(score, "b ... -> b 1 ...")
        return score

    for _ in range(n_spatial_dims + 2):
        score = rearrange(score, "b ... -> b 1 ...")

    return score


class BTSCMMetric(BaseMetric[TensorBTSCM, TensorBTSC]):
    """
    Base class for ensemble metrics that operate on spatial tensors.

    Checks input types and shapes and converts to Tensor.

    Args:
        score_dims: Which dimension to compute the score.
            'spatial': average over spatial dims → (B, T, C)
            'temporal': average over temporal dim → (B, S, C)
            None: no reduction → (B, T, S, C)
            These are the respective dimensions after calling .score()
            This works in conjunction with the reduce_all parameter that is
            applied in the compute() method to determine the final output shape
        reduce_all: If True, return scalar by averaging over all non-batch dims
        dist_sync_on_step: Synchronize metric state across processes at each forward()
    """

    def __init__(
        self,
        score_dims: Literal["spatial", "temporal"] | None = "spatial",
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(reduce_all=reduce_all, dist_sync_on_step=dist_sync_on_step)
        if score_dims not in ("spatial", "temporal", None):
            raise ValueError(
                f"score_dims must be 'spatial', 'temporal', or None, got {score_dims!r}"
            )
        self.score_dims = score_dims

    @abc.abstractmethod
    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC: ...

    def score(
        self, y_pred: ArrayLike, y_true: ArrayLike
    ) -> TensorBTC | TensorBSC | TensorBTSC:
        """Compute metric score, then reduce according to self.score_dims.

        Args:
            y_pred: Predictions of shape (B, T, S, C, M)
            y_true: Ground truth of shape (B, T, S, C)

        Returns
        -------
            Tensor of shape (B, T, C) if reduce_over='spatial',
            (B, S, C) if reduce_over='temporal', or (B, T, S, C) if None.
        """
        y_pred_tensor, y_true_tensor = self._check_input(y_pred, y_true)
        result = self._score(y_pred_tensor, y_true_tensor)  # (B, T, S, C)

        if self.score_dims == "spatial":
            n_spatial_dims = self._infer_n_spatial_dims(result)
            spatial_dims = tuple(range(2, 2 + n_spatial_dims))
            return result.mean(dim=spatial_dims)  # (B, T, C)
        if self.score_dims == "temporal":
            return result.mean(dim=1)  # (B, S, C)
        return result  # (B, T, S, C)

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


def _sorted_pairwise_abs_weighted_sum(y_pred: TensorBTSCM) -> TensorBTSC:
    r"""
    Compute :math:`\sum_{k=1}^{M} (2k - M - 1)\, x_{(k)}` along the ensemble dim.

    This is the order-statistic identity that lets the pairwise absolute
    difference sum be evaluated in O(M log M) time and O(M) memory instead of
    the O(M^2) / O(M^2)-memory broadcasting approach:

    .. math::
        \sum_{j,k=1}^{M} |x_j - x_k| = 2 \sum_{k=1}^{M} (2k - M - 1)\, x_{(k)},

    where :math:`x_{(k)}` is the k-th order statistic.

    Args:
        y_pred: Predictions of shape (..., M)

    Returns
    -------
        Tensor of shape (...) containing the weighted sum of sorted values.
    """
    n_ensemble = y_pred.shape[-1]
    y_sorted = torch.sort(y_pred, dim=-1).values
    k = torch.arange(1, n_ensemble + 1, device=y_pred.device, dtype=y_pred.dtype)
    weights = 2.0 * k - (n_ensemble + 1)
    return (y_sorted * weights).sum(dim=-1)


def _require_pairwise_ensemble_size(n_ensemble: int, metric_name: str) -> None:
    if n_ensemble <= 1:
        raise ValueError(
            f"{metric_name} requires at least 2 ensemble members "
            f"to compute the pairwise spread term. Got {n_ensemble}."
        )


def _crps_mae_term(y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
    """Return the mean absolute distance-to-truth term used by CRPS variants."""
    return torch.mean(torch.abs(y_pred - y_true.unsqueeze(-1)), dim=-1)


def _crps_pairwise_spread_term(
    y_pred: TensorBTSCM, pairwise_coefficient: float
) -> TensorBTSC:
    """Return the pairwise ensemble-distance term used by CRPS variants."""
    pairwise_weighted_sum = _sorted_pairwise_abs_weighted_sum(y_pred)
    return pairwise_weighted_sum * pairwise_coefficient


def _common_crps_terms(
    y_pred: TensorBTSCM, y_true: TensorBTSC, adjustment_factor: float
) -> tuple[TensorBTSC, TensorBTSC]:
    """Return the MAE-like and pairwise spread terms for CRPS/fCRPS."""
    n_ensemble = y_pred.shape[-1]
    term1 = _crps_mae_term(y_pred, y_true)
    term2 = _crps_pairwise_spread_term(
        y_pred, pairwise_coefficient=adjustment_factor / (n_ensemble**2)
    )
    return term1, term2


def _common_crps_spread_term(
    y_pred: TensorBTSCM, adjustment_factor: float
) -> TensorBTSC:
    n_ensemble = y_pred.shape[-1]
    return _crps_pairwise_spread_term(
        y_pred, pairwise_coefficient=adjustment_factor / (n_ensemble**2)
    )


def _common_crps_score(
    y_pred: TensorBTSCM, y_true: TensorBTSC, adjustment_factor: float
) -> TensorBTSC:
    """
    Compute CRPS reduced over spatial dims only.

    Expected input shape: (B, T, S, C, M)
    Expected output shape: (B, T, S, C)

    Uses the sort-based identity for the pairwise ensemble term to avoid
    materialising the (..., M, M) tensor that naive broadcasting would produce,
    reducing peak activation memory from O(M^2) to O(M).

    Args:
        y_pred: Predictions of shape (B, T, S, C, M)
        y_true: Ground truth of shape (B, T, S, C)
        adjustment_factor: Factor to adjust the second term in CRPS calculation

    Returns
    -------
        Tensor of shape (B, T, S, C) with CRPS scores
    """
    term1, term2 = _common_crps_terms(y_pred, y_true, adjustment_factor)
    return term1 - term2


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

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        """
        Compute CRPS score.

        Expected input shape: (B, T, S, C, M)
        Expected output shape: (B, T, S, C)

        Args:
            y_pred: Predictions of shape (B, T, S, C, M)
            y_true: Ground truth of shape (B, T, S, C)

        Returns
        -------
            Tensor of shape (B, T, S, C) with CRPS scores
        """
        return _common_crps_score(y_pred, y_true, adjustment_factor=1.0)


class CRPSMAETerm(BTSCMMetric):
    r"""
    Mean-absolute-error term in the CRPS decomposition.

    Notes
    -----
    This is the first CRPS term,

    .. math::
        \frac{1}{M}\sum_{m=1}^{M} |x_m - y|,

    so it is MAE-like, but it is **not** the deterministic MAE of the ensemble mean.
    """

    name: str = "crps_mae_term"

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        term1, _ = _common_crps_terms(y_pred, y_true, adjustment_factor=1.0)
        return term1


class CRPSSpreadTerm(BTSCMMetric):
    r"""
    Pairwise spread term in the CRPS decomposition.

    Notes
    -----
    This is the second CRPS term,

    .. math::
        \frac{1}{2M^2}\sum_{j=1}^{M}\sum_{k=1}^{M}|x_j - x_k|,

    represented via the sort-based identity used elsewhere in this module.
    """

    name: str = "crps_spread_term"

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        del y_true
        return _common_crps_spread_term(y_pred, adjustment_factor=1.0)


class FairCRPS(BTSCMMetric):
    """
    Fair Continuous Ranked Probability Score (fCRPS) for ensemble forecasts.

    References
    ----------
    Ferro, C.A.T. (2014), Fair scores for ensemble forecasts. Q.J.R. Meteorol. Soc.,
    140: 1917-1923. https://doi.org/10.1002/qj.2270
    """

    name: str = "fcrps"

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        """
        Compute fCRPS score.

        Expected input shape: (B, T, S, C, M)
        Expected output shape: (B, T, S, C)

        Args:
            y_pred: Predictions of shape (B, T, S, C, M)
            y_true: Ground truth of shape (B, T, S, C)

        Returns
        -------
            Tensor of shape (B, T, S, C) with fCRPS scores
        """
        n_ensemble = y_pred.shape[-1]
        _require_pairwise_ensemble_size(n_ensemble, "FairCRPS")
        return _common_crps_score(
            y_pred, y_true, adjustment_factor=n_ensemble / (n_ensemble - 1)
        )


class FairCRPSMAETerm(BTSCMMetric):
    """Mean-absolute-error term in the fCRPS decomposition."""

    name: str = "fcrps_mae_term"

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        n_ensemble = y_pred.shape[-1]
        _require_pairwise_ensemble_size(n_ensemble, "FairCRPSMAETerm")
        term1, _ = _common_crps_terms(
            y_pred, y_true, adjustment_factor=n_ensemble / (n_ensemble - 1)
        )
        return term1


class FairCRPSSpreadTerm(BTSCMMetric):
    """Pairwise spread term in the fCRPS decomposition."""

    name: str = "fcrps_spread_term"

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        del y_true
        n_ensemble = y_pred.shape[-1]
        _require_pairwise_ensemble_size(n_ensemble, "FairCRPSSpreadTerm")
        return _common_crps_spread_term(
            y_pred, adjustment_factor=n_ensemble / (n_ensemble - 1)
        )


def _alpha_fair_crps_terms(
    y_pred: TensorBTSCM, y_true: TensorBTSC, alpha: float
) -> tuple[TensorBTSC, TensorBTSC]:
    """Return the MAE-like and pairwise spread terms for afCRPS."""
    n_ensemble = y_pred.shape[-1]
    _require_pairwise_ensemble_size(n_ensemble, "AlphaFairCRPS")
    eps = (1.0 - alpha) / n_ensemble

    term1 = _crps_mae_term(y_pred, y_true)
    term2 = _crps_pairwise_spread_term(
        y_pred,
        pairwise_coefficient=(1.0 - eps) / (n_ensemble * (n_ensemble - 1)),
    )
    return term1, term2


def _alpha_fair_crps_spread_term(y_pred: TensorBTSCM, alpha: float) -> TensorBTSC:
    n_ensemble = y_pred.shape[-1]
    _require_pairwise_ensemble_size(n_ensemble, "AlphaFairCRPS")
    eps = (1.0 - alpha) / n_ensemble
    return _crps_pairwise_spread_term(
        y_pred,
        pairwise_coefficient=(1.0 - eps) / (n_ensemble * (n_ensemble - 1)),
    )


def _alpha_fair_crps_score(
    y_pred: TensorBTSCM, y_true: TensorBTSC, alpha: float
) -> TensorBTSC:
    r"""
    Compute afCRPS reduced over spatial dims only.

    Uses the order-statistic identity to avoid the (..., M, M) activation:

    .. math::
        \text{afCRPS}
        = \frac{1}{M}\sum_j |x_j - y|
          - \frac{1-\varepsilon}{M(M-1)} \sum_{k=1}^{M} (2k - M - 1) x_{(k)},

    with :math:`\varepsilon = (1-\alpha)/M`. This is algebraically identical to
    the off-diagonal pair sum form but needs only O(M) memory.

    Args:
        y_pred: (B, T, S, C, M)
        y_true: (B, T, S, C)
        alpha: Smoothing parameter

    Returns
    -------
        afCRPS: (B, T, S, C)
    """
    term1, term2 = _alpha_fair_crps_terms(y_pred, y_true, alpha)
    return term1 - term2


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

    def __init__(
        self,
        alpha: float = 0.95,
        *,
        score_dims: Literal["spatial", "temporal"] | None = "spatial",
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            score_dims=score_dims,
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        # alpha close to 1 is close to fair CRPS, lower values tend toward standard CRPS
        assert 0 < alpha <= 1, "alpha must be in (0,1]"
        self.alpha = alpha

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        """
        Compute afCRPS score.

        Args:
            y_pred: (B, T, S, C, M)
            y_true: (B, T, S, C)

        Returns
        -------
            afCRPS: (B, T, S, C)
        """
        return _alpha_fair_crps_score(y_pred, y_true, self.alpha)


class AlphaFairCRPSMAETerm(BTSCMMetric):
    """
    Mean-absolute-error term paired with afCRPS monitoring.

    The MAE-like term itself does not depend on ``alpha``, but this class accepts
    it so experiment configs can keep the afCRPS diagnostic bundle parameterized
    consistently.
    """

    name: str = "afcrps_mae_term"

    def __init__(
        self,
        alpha: float = 0.95,
        *,
        score_dims: Literal["spatial", "temporal"] | None = "spatial",
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            score_dims=score_dims,
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        assert 0 < alpha <= 1, "alpha must be in (0,1]"
        self.alpha = alpha

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        term1, _ = _alpha_fair_crps_terms(y_pred, y_true, self.alpha)
        return term1


class AlphaFairCRPSSpreadTerm(BTSCMMetric):
    """Pairwise spread term in the afCRPS decomposition."""

    name: str = "afcrps_spread_term"

    def __init__(
        self,
        alpha: float = 0.95,
        *,
        score_dims: Literal["spatial", "temporal"] | None = "spatial",
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            score_dims=score_dims,
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        assert 0 < alpha <= 1, "alpha must be in (0,1]"
        self.alpha = alpha

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        del y_true
        return _alpha_fair_crps_spread_term(y_pred, self.alpha)


class EnergyScore(BTSCMMetric):
    r"""
    Energy score (multivariate CRPS) for ensemble forecasts.

    For a vector-valued forecast with ensemble members :math:`x_m \in \mathbb{R}^d`
    and observation :math:`y \in \mathbb{R}^d`, this computes

    .. math::
        ES_\alpha(F, y) = \frac{1}{M} \sum_{m=1}^M \lVert x_m - y \rVert_2^\alpha
        - \frac{1}{2M^2} \sum_{m=1}^M \sum_{j=1}^M \lVert x_m - x_j \rVert_2^\alpha,

    with :math:`\alpha \in (0, 2)`.

    Notes
    -----
    The ``vector_dims`` argument controls which dimensions define the multivariate
    vector used in the norm.
    """

    name: str = "energy"
    vector_dims: VectorDimsOption

    def __init__(
        self,
        alpha: float = 1.0,
        vector_dims: VectorDimsOption = "spatial_temporal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not (0 < alpha < 2):
            raise ValueError(f"alpha must be in (0, 2), got {alpha}")
        self.alpha = alpha
        normalized_vector_dims: VectorDimsOption = _normalize_vector_dims(vector_dims)
        self.vector_dims = normalized_vector_dims

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        y_pred_vector, y_true_vector = _vectorize_selected_dims(
            y_pred, y_true, self.vector_dims
        )

        # y_pred: (..., M), y_true: (...)
        y_true_expanded = rearrange(y_true_vector, "... d -> ... d 1")

        # (..., M): ||x_m - y||_2^alpha over selected vector dimensions
        dist_truth = torch.linalg.vector_norm(
            y_pred_vector - y_true_expanded,
            ord=2,
            dim=-2,
        )
        term1 = dist_truth.pow(self.alpha).mean(dim=-1)

        # (..., M, M): ||x_m - x_j||_2^alpha over selected vector dimensions
        pairwise = rearrange(y_pred_vector, "... d m -> ... d m 1") - rearrange(
            y_pred_vector, "... d m -> ... d 1 m"
        )
        dist_pairwise = torch.linalg.vector_norm(pairwise, ord=2, dim=-3)
        term2 = 0.5 * dist_pairwise.pow(self.alpha).mean(dim=(-2, -1))

        # Reinsert selected vector dimensions as singleton axes.
        score = term1 - term2
        return _restore_vector_dims_singletons(score, y_true, self.vector_dims)


class VariogramScore(BTSCMMetric):
    r"""
    Variogram score for multivariate ensemble forecasts.

    For vector-valued forecast members :math:`x_m \in \mathbb{R}^d` and
    observation :math:`y \in \mathbb{R}^d`, this computes

    .. math::
        VS_p(F, y) = \sum_{i,j=1}^d w_{ij}
        \left(\frac{1}{M}\sum_{m=1}^M |x_{m,i} - x_{m,j}|^p - |y_i - y_j|^p\right)^2,

    with :math:`p > 0` and non-negative weights :math:`w_{ij}`.

    Notes
    -----
    The ``vector_dims`` argument controls which dimensions define the multivariate
    vector used by the variogram transformation.
    """

    name: str = "variogram"
    vector_dims: VectorDimsOption

    def __init__(
        self,
        p: float = 0.5,
        weights: Tensor | np.ndarray | None = None,
        vector_dims: VectorDimsOption = "spatial_temporal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if p <= 0:
            raise ValueError(f"p must be > 0, got {p}")
        self.p = p
        normalized_vector_dims: VectorDimsOption = _normalize_vector_dims(vector_dims)
        self.vector_dims = normalized_vector_dims

        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights)
            if not isinstance(weights, Tensor):
                raise TypeError(
                    "weights must be a Tensor, np.ndarray, or None, "
                    f"got {type(weights)}"
                )
            if (weights < 0).any().item():
                msg = "weights must be non-negative (w_ij >= 0)"
                raise ValueError(msg)
        self.weights = weights

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        y_pred_vector, y_true_vector = _vectorize_selected_dims(
            y_pred, y_true, self.vector_dims
        )
        vector_size = y_true_vector.shape[-1]

        if self.weights is None:
            weights = torch.ones(
                (vector_size, vector_size),
                device=y_pred.device,
                dtype=y_pred.dtype,
            )
        else:
            if self.weights.ndim != 2 or self.weights.shape != (
                vector_size,
                vector_size,
            ):
                raise ValueError(
                    "weights must have shape (D, D) matching the selected vector "
                    f"dimensions; got {tuple(self.weights.shape)} with D={vector_size}"
                )
            weights = self.weights.to(device=y_pred.device, dtype=y_pred.dtype)

        # Observation variogram term: (..., D, D)
        obs_pairwise = torch.abs(
            rearrange(y_true_vector, "... d -> ... d 1")
            - rearrange(y_true_vector, "... d -> ... 1 d")
        ).pow(self.p)

        # Ensemble variogram expectation: (..., D, D)
        ens_pairwise = torch.abs(
            rearrange(y_pred_vector, "... d m -> ... d 1 m")
            - rearrange(y_pred_vector, "... d m -> ... 1 d m")
        ).pow(self.p)
        ens_expected = ens_pairwise.mean(dim=-1)

        diff_sq = (ens_expected - obs_pairwise).pow(2)  # e.g. B T C D D if spatial
        score = (diff_sq * weights).sum(dim=(-2, -1))  # e.g. B T C if spatial

        # Reinsert selected vector dimensions as singleton axes.
        return _restore_vector_dims_singletons(
            score, y_true, self.vector_dims
        )  # B T S C


class SpreadSkillRatio(BTSCMMetric):
    r"""
    Corrected spread-to-skill ratio (SSR) for ensemble forecasts.

    Notes
    -----
    Uses the corrected finite-ensemble form:
    .. math::
        \text{SSR}_{\text{corrected}} = \frac{\text{Spread}}{\text{Skill}}
        \sqrt{\frac{M + 1}{M}},
    where skill is the pointwise RMSE of the ensemble mean and spread is the
    pointwise ensemble standard deviation. Spatial/temporal reductions are then
    handled by the base class according to score_dims.

    """

    name: str = "ssr"

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        if eps <= 0:
            msg = "eps must be > 0"
            raise ValueError(msg)
        self.eps = eps

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        """Not used directly, as we override score() to change the reduction order."""
        msg = "SpreadSkillRatio overrides score() directly."
        raise NotImplementedError(msg)

    def score(
        self, y_pred: ArrayLike, y_true: ArrayLike
    ) -> TensorBTC | TensorBSC | TensorBTSC:
        """
        Compute corrected spread-to-skill ratio.

        Reductions (spatial/temporal) are applied to the variance and MSE before
        taking the square root and computing the ratio (i.e., reduce variance/MSE
        first, then sqrt, then divide).

        Args:
            y_pred: (B, T, S, C, M)
            y_true: (B, T, S, C)

        Returns
        -------
            SSR: (B, T, C) if score_dims='spatial', (B, S, C) if temporal,
                 or (B, T, S, C) if None.
        """
        y_pred_tensor, y_true_tensor = self._check_input(y_pred, y_true)

        n_ensemble = y_pred_tensor.shape[-1]
        if n_ensemble < 2:
            raise ValueError(
                "SpreadSkillRatio requires at least 2 ensemble members "
                f"(got {n_ensemble})."
            )

        # Pointwise squared errors and variances
        ensemble_mean = y_pred_tensor.mean(dim=-1)
        skill_sq = (ensemble_mean - y_true_tensor) ** 2
        spread_var = y_pred_tensor.var(dim=-1, unbiased=True)

        # Reduce the variances/SSE before sqrt and division
        if self.score_dims == "spatial":
            n_spatial_dims = self._infer_n_spatial_dims(y_true_tensor)
            spatial_dims = tuple(range(2, 2 + n_spatial_dims))
            skill_sq = skill_sq.mean(dim=spatial_dims)
            spread_var = spread_var.mean(dim=spatial_dims)
        elif self.score_dims == "temporal":
            skill_sq = skill_sq.mean(dim=1)
            spread_var = spread_var.mean(dim=1)

        # Reduce to spread, skill, and take ratio
        skill = torch.sqrt(skill_sq)
        spread = torch.sqrt(spread_var)

        correction = float(np.sqrt((n_ensemble + 1) / n_ensemble))
        ssr = (spread / torch.clamp(skill, min=self.eps)) * correction

        return ssr


class EnsembleSpread(BTSCMMetric):
    r"""
    Ensemble spread for probabilistic forecasts.

    Notes
    -----
    By default, returns a **finite-ensemble corrected spread**:

    .. math::
        \text{Spread}_{\text{corr}} =
            \sqrt{\left\langle \mathrm{Var}_{m,\text{unbiased}}(x_m)\right\rangle}
            \sqrt{\frac{M + 1}{M}}.

    This correction is commonly used so that spread and skill are comparable for
    finite ensemble sizes when using unbiased sample variance. It matches the
    form used in LoLA/paper evaluations (Appendix "Spread / Skill") where:
    ``spread = sqrt((M+1)/(M-1) * mean((x_m - mean_m)^2))``, since
    ``Var_unbiased = (M/(M-1)) * mean((x_m - mean_m)^2)``.

    If ``corrected=False``, returns the uncorrected macroscopic ensemble standard
    deviation computed from the unbiased variance estimator:

    .. math::
        \sqrt{\left\langle \mathrm{Var}_{m,\text{unbiased}}(x_m)\right\rangle}.
    """

    name: str = "spread"

    def __init__(
        self,
        *,
        corrected: bool = True,
        score_dims: Literal["spatial", "temporal"] | None = "spatial",
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            score_dims=score_dims,
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.corrected = corrected

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        """Not used directly; we override score() to change reduction order."""
        msg = "EnsembleSpread overrides score() directly."
        raise NotImplementedError(msg)

    def score(
        self, y_pred: ArrayLike, y_true: ArrayLike
    ) -> TensorBTC | TensorBSC | TensorBTSC:
        y_pred_tensor, y_true_tensor = self._check_input(y_pred, y_true)

        n_ensemble = y_pred_tensor.shape[-1]
        if n_ensemble < 2:
            raise ValueError(
                "EnsembleSpread requires at least 2 ensemble members "
                f"(got {n_ensemble})."
            )

        spread_var = y_pred_tensor.var(dim=-1, unbiased=True)  # (B, T, S..., C)

        # Reduce variance before sqrt (macroscopic approach)
        if self.score_dims == "spatial":
            n_spatial_dims = self._infer_n_spatial_dims(y_true_tensor)
            spatial_dims = tuple(range(2, 2 + n_spatial_dims))
            spread_var = spread_var.mean(dim=spatial_dims)
        elif self.score_dims == "temporal":
            spread_var = spread_var.mean(dim=1)

        spread = torch.sqrt(spread_var)

        if self.corrected:
            correction = float(np.sqrt((n_ensemble + 1) / n_ensemble))
            spread = spread * correction

        return spread


class EnsembleSkill(BTSCMMetric):
    r"""
    Ensemble skill defined as RMSE of the ensemble mean.

    Notes
    -----
    Skill is defined as the RMSE of the ensemble mean:

    .. math::
        \text{Skill} = \sqrt{\left\langle (\bar{x} - y)^2 \right\rangle},

    where :math:`\langle \cdot \rangle` denotes the spatial mean.

    This metric reduces the squared error over spatial/temporal dimensions *before*
    taking the square root (macroscopic RMSE), as is commonly done in ensemble
    forecast evaluation (and in LoLA/paper appendices).

    In the default spatial-reduction evaluation path, this is numerically
    equivalent to the deterministic ``RMSE`` metric applied to an ensemble
    prediction tensor, because ``RMSE`` first averages over the ensemble
    dimension and then computes RMSE.
    """

    name: str = "skill"

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        """Not used directly; we override score() to change reduction order."""
        msg = "EnsembleSkill overrides score() directly."
        raise NotImplementedError(msg)

    def score(
        self, y_pred: ArrayLike, y_true: ArrayLike
    ) -> TensorBTC | TensorBSC | TensorBTSC:
        y_pred_tensor, y_true_tensor = self._check_input(y_pred, y_true)

        ensemble_mean = y_pred_tensor.mean(dim=-1)
        skill_sq = (ensemble_mean - y_true_tensor) ** 2

        # Reduce MSE before sqrt (macroscopic approach)
        if self.score_dims == "spatial":
            n_spatial_dims = self._infer_n_spatial_dims(y_true_tensor)
            spatial_dims = tuple(range(2, 2 + n_spatial_dims))
            skill_sq = skill_sq.mean(dim=spatial_dims)
        elif self.score_dims == "temporal":
            skill_sq = skill_sq.mean(dim=1)

        return torch.sqrt(skill_sq)


class WinklerScore(BTSCMMetric):
    r"""
    Winkler interval score for central prediction intervals.

    For significance level :math:`\alpha \in (0, 1)`, this metric computes
    central :math:`(1-\alpha)` prediction intervals from ensemble quantiles and
    returns the per-point interval score

    .. math::
        W_\alpha = (u - l)
        + \frac{2}{\alpha}(l - y)\mathbf{1}(y < l)
        + \frac{2}{\alpha}(y - u)\mathbf{1}(y > u),

        where :math:`l` and :math:`u` are the lower/upper interval bounds.

        Shape conventions
        -----------------
        - Input prediction tensor: y_pred has shape (B, T, S..., C, M)
        - Input truth tensor: y_true has shape (B, T, S..., C)
        - Quantiles are computed along ensemble dim M:
            - l (lower): ``q_{alpha/2}``, shape (B, T, S..., C)
            - u (upper): ``q_{1-alpha/2}``, shape (B, T, S..., C)
            - y corresponds to y_true, same shape (B, T, S..., C).

        The internal ``_score`` returns pointwise Winkler scores with shape
        ``(B, T, S..., C)``. The base class then applies ``score_dims`` and
        ``reduce_all`` reductions:
        - ``score_dims='spatial'`` (default) -> ``(B, T, C)``
        - ``score_dims='temporal'`` -> ``(B, S..., C)``
        - ``score_dims=None`` -> ``(B, T, S..., C)``
        - if ``reduce_all=True`` (default), ``compute()`` returns a scalar.

    Lower values are better: narrow intervals are rewarded, and misses are
    penalized in proportion to their distance outside the interval.
    """

    name: str = "winkler"

    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        """Compute pointwise Winkler score before spatial/temporal reductions."""
        q_low = self.alpha / 2.0
        q_high = 1.0 - self.alpha / 2.0

        q_tensor = torch.tensor(
            [q_low, q_high], device=y_pred.device, dtype=y_pred.dtype
        )
        quantiles = torch.quantile(y_pred, q_tensor, dim=-1)
        lower = quantiles[0]
        upper = quantiles[1]

        width = upper - lower
        below_penalty = (2.0 / self.alpha) * torch.clamp(lower - y_true, min=0.0)
        above_penalty = (2.0 / self.alpha) * torch.clamp(y_true - upper, min=0.0)

        return width + below_penalty + above_penalty
