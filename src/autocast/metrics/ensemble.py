import abc
from typing import Literal

import numpy as np
import torch
from einops import rearrange, repeat

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
        return _common_crps_score(
            y_pred, y_true, adjustment_factor=n_ensemble / (n_ensemble - 1)
        )


def _alpha_fair_crps_score(
    y_pred: TensorBTSCM, y_true: TensorBTSC, alpha: float
) -> TensorBTSC:
    """
    Compute afCRPS reduced over spatial dims only.

    Args:
        y_pred: (B, T, S, C, M)
        y_true: (B, T, S, C)
        alpha: Smoothing parameter

    Returns
    -------
        afCRPS: (B, T, S, C)
    """
    # Expand y_true to match ensemble dimension
    n_ensemble = y_pred.shape[-1]
    y_true_m = repeat(y_true, "... -> ... m", m=n_ensemble)

    eps = (1.0 - alpha) / n_ensemble

    abs_diff_ens = torch.abs(
        rearrange(y_pred, "... m -> ... 1 m") - rearrange(y_pred, "... m -> ... m 1")
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

    return afcrps


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

    def __init__(self, eps: float = 1e-6):
        super().__init__()
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
        taking the square root and computing the ratio, matching macroscopic
        approaches like Lola's.

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

        # Compute macroscopic spread, skill, and ratio
        skill = torch.sqrt(skill_sq)
        spread = torch.sqrt(spread_var)

        correction = float(np.sqrt((n_ensemble + 1) / n_ensemble))
        ssr = (spread / torch.clamp(skill, min=self.eps)) * correction

        return ssr
