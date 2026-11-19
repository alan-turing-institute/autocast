"""Closed-form Gaussian probabilistic metrics for heteroscedastic forecasts.

The public ``update`` signature takes packed predictions ``(B, T, ..., 2*C)``
and a target ``(B, T, ..., C)``: the first ``C`` channels of the prediction
are the predicted mean, the next ``C`` are the predicted log-variance. This
matches the parent ``EncoderProcessorDecoder``'s metric flow (which calls
``metric.update(y_pred, y_true)``), so heteroscedastic metrics integrate
through the standard ``MetricsMixin._update_and_log_metrics`` path. The
metric class does not touch any normalisation; that responsibility lives in
the upstream Lightning module so these metrics stay general-purpose.
"""

import math
from abc import abstractmethod

import torch
from torchmetrics import Metric

from autocast.metrics.intervals import (
    interval_indicator,
    winkler_penalty,
)
from autocast.metrics.intervals import (
    z_for_central_interval as _z_for_central_interval,
)
from autocast.types import Tensor, TensorBTSC, TensorBTSC2C


def _split_packed(packed: TensorBTSC2C) -> tuple[TensorBTSC, TensorBTSC]:
    n_channels = packed.shape[-1] // 2
    return packed[..., :n_channels], packed[..., n_channels:]


class BaseGaussianMetric(Metric):
    """Shared template for closed-form Gaussian metrics.

    Mirrors ``autocast.metrics.base.BaseMetric``: state is a lazily-shaped
    ``sum_score`` accumulated over the batch axis plus a ``total_samples``
    counter. Subclasses implement ``_pointwise(mu, log_var, target)``, an
    elementwise score on inputs of any one shape (``update`` passes
    ``(B, T, ..., C)``; the closed-form predictive metrics reuse it on their own
    per-site shapes). ``compute`` returns the
    scalar mean over all non-batch axes when ``reduce_all=True`` (default),
    or the per-axis tensor otherwise.
    """

    name: str

    def __init__(
        self,
        eps: float = 1e-6,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if eps <= 0:
            msg = f"eps must be > 0, got {eps}"
            raise ValueError(msg)
        self.eps = eps
        self.reduce_all = reduce_all
        self.add_state("sum_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self._initialized = False

    @abstractmethod
    def _pointwise(self, mu: Tensor, log_var: Tensor, target: Tensor) -> Tensor: ...

    def update(self, y_pred: TensorBTSC2C, y_true: TensorBTSC) -> None:
        """Accumulate one batch into the running sum.

        Args:
            y_pred: Packed (mu, log_var) predictions of shape (B, T, ..., 2*C).
            y_true: Targets of shape (B, T, ..., C).
        """
        if y_pred.ndim != y_true.ndim or y_pred.shape[-1] != 2 * y_true.shape[-1]:
            msg = (
                "Gaussian metric expects y_pred to pack (mu, log_var) along "
                "the channel axis with the same ndim as y_true; got "
                f"y_pred.shape={tuple(y_pred.shape)} and "
                f"y_true.shape={tuple(y_true.shape)}."
            )
            raise ValueError(msg)
        mu, log_var = _split_packed(y_pred)
        score = self._pointwise(mu, log_var, y_true)
        batch_size = score.shape[0]
        score_summed = torch.sum(score, dim=0)
        if not self._initialized:
            self.sum_score = torch.zeros_like(score_summed)
            self._initialized = True
        self.sum_score += score_summed
        self.total_samples += batch_size

    def compute(self) -> Tensor:
        """Return the mean score (scalar if ``reduce_all``, else per-axis)."""
        if self.total_samples == 0:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)
        score = self.sum_score / self.total_samples
        if self.reduce_all:
            return score.mean()
        return score

    def reset(self) -> None:
        super().reset()
        self._initialized = False

    def _sigma(self, log_var: Tensor) -> Tensor:
        return (0.5 * log_var).exp().clamp(min=self.eps)


class GaussianCRPS(BaseGaussianMetric):
    r"""Closed-form Gaussian Continuous Ranked Probability Score.

    For a predictive Gaussian :math:`\mathcal{N}(\mu, \sigma^2)` and
    observation :math:`y`,

    .. math::
        \mathrm{CRPS}(\mathcal{N}(\mu, \sigma^2), y)
        = \sigma \left[ z (2 \Phi(z) - 1) + 2 \varphi(z)
                        - \tfrac{1}{\sqrt{\pi}} \right],

    where :math:`z = (y - \mu) / \sigma`, :math:`\Phi` is the standard normal
    CDF and :math:`\varphi` is the standard normal PDF.
    """

    name: str = "gaussian_crps"

    def _pointwise(self, mu: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
        sigma = self._sigma(log_var)
        z = (target - mu) / sigma
        phi = (-0.5 * z.square()).exp() / math.sqrt(2.0 * math.pi)
        # 2*Phi(z) - 1 == erf(z/sqrt(2)); use the direct form to avoid the
        # 1+erf cancellation that loses fp32 precision for large |z|.
        erf_z = torch.erf(z / math.sqrt(2.0))
        return sigma * (z * erf_z + 2.0 * phi - 1.0 / math.sqrt(math.pi))


class GaussianCoverage(BaseGaussianMetric):
    r"""Fraction of targets inside the predicted symmetric central interval.

    For a coverage level :math:`\alpha \in (0, 1)`, this checks whether
    :math:`|y - \mu| \le z_{\alpha/2}\, \sigma` element-wise and reports
    the mean indicator. A well-calibrated model returns approximately
    :math:`\alpha`.
    """

    name: str = "gaussian_coverage"

    def __init__(self, coverage_level: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        if not (0.0 < coverage_level < 1.0):
            msg = f"coverage_level must be in (0, 1), got {coverage_level}"
            raise ValueError(msg)
        self.coverage_level = coverage_level
        self.z_half_width = _z_for_central_interval(coverage_level)

    def _pointwise(self, mu: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
        sigma = self._sigma(log_var)
        half_width = self.z_half_width * sigma
        return interval_indicator(mu - half_width, mu + half_width, target)


class GaussianWinkler(BaseGaussianMetric):
    r"""Winkler interval score for the predicted central interval.

    For coverage level :math:`\alpha = 1 - \mathrm{coverage\_level}`,

    .. math::
        W_\alpha = (u - l)
                 + \tfrac{2}{\alpha}(l - y) \mathbf{1}(y < l)
                 + \tfrac{2}{\alpha}(y - u) \mathbf{1}(y > u),

    with :math:`l = \mu - z_{\alpha/2} \sigma` and
    :math:`u = \mu + z_{\alpha/2} \sigma`. Lower is better.
    """

    name: str = "gaussian_winkler"

    def __init__(self, coverage_level: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        if not (0.0 < coverage_level < 1.0):
            msg = f"coverage_level must be in (0, 1), got {coverage_level}"
            raise ValueError(msg)
        self.coverage_level = coverage_level
        self.alpha = 1.0 - coverage_level
        self.z_half_width = _z_for_central_interval(coverage_level)

    def _pointwise(self, mu: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
        sigma = self._sigma(log_var)
        half_width = self.z_half_width * sigma
        return winkler_penalty(mu - half_width, mu + half_width, target, self.alpha)
