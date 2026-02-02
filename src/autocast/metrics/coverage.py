from __future__ import annotations

from typing import Any, cast

import torch
from torchmetrics import Metric

from autocast.metrics.ensemble import BTSCMMetric
from autocast.types import Tensor, TensorBTC, TensorBTSC, TensorBTSCM


class Coverage(BTSCMMetric):
    """
    Coverage probability for a fixed coverage level.

    Calculates the proportion of true values that fall within the symmetric
    prediction interval defined by the coverage level.
    """

    name: str = "coverage"

    def __init__(self, coverage_level: float = 0.95):
        """Initialize Coverage metric.

        Args:
            coverage_level: nominal coverage probability (e.g. 0.95 for 95% interval).
                Must be between 0 and 1.
        """
        super().__init__()
        if not (0 < coverage_level < 1):
            raise ValueError(f"coverage_level must be in (0, 1), got {coverage_level}")
        self.coverage_level = coverage_level

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTC:
        """
        Compute coverage reduced over spatial dims.

        Args:
            y_pred: (B, T, S, C, M)
            y_true: (B, T, S, C)

        Returns
        -------
            coverage: (B, T, C)
        """
        # Calculate quantiles of the ensemble distribution
        # e.g. coverage_level=0.95 -> 0.025 and 0.975 quantiles
        q_low = 0.5 - self.coverage_level / 2
        q_high = 0.5 + self.coverage_level / 2

        # Calculate quantiles
        q_tensor = torch.tensor(
            [q_low, q_high], device=y_pred.device, dtype=y_pred.dtype
        )
        quantiles = torch.quantile(y_pred, q_tensor, dim=-1)
        # quantiles shape: (2, B, T, S, C)

        lower_q = quantiles[0]
        upper_q = quantiles[1]

        # Calculate coverage (1 if inside, 0 otherwise)
        is_covered = ((y_true >= lower_q) & (y_true <= upper_q)).float()

        # Reduce over spatial dimensions: (B, T, S, C) -> (B, T, C)
        n_spatial_dims = self._infer_n_spatial_dims(is_covered)
        spatial_dims = tuple(range(2, 2 + n_spatial_dims))
        coverage_reduced = is_covered.mean(dim=spatial_dims)

        return coverage_reduced


class MultiCoverage(Metric):
    """
    Computes coverage for multiple coverage levels at once.

    This is a wrapper around multiple Coverage metrics. It inherits from Metric
    to integrate with PyTorch Lightning and TorchMetrics.
    """

    def __init__(self, coverage_levels: list[float] | None = None):
        super().__init__()
        if coverage_levels is None:
            coverage_levels = [
                round(x, 2) for x in torch.linspace(0.05, 0.95, steps=19).tolist()
            ]

        self.coverage_levels = coverage_levels
        # Create a list of Coverage metrics
        self.metrics = torch.nn.ModuleList(
            [Coverage(coverage_level=cl) for cl in coverage_levels]
        )

    def _cast_coverage_metrics(self) -> list[Coverage]:
        return cast(list[Coverage], self.metrics)

    def update(self, y_pred, y_true):
        for metric in self._cast_coverage_metrics():
            metric.update(y_pred, y_true)

    def compute(self) -> dict[str, Tensor]:
        """Return a dict of results, keys formatted as 'coverage_{coverage_level}'."""
        return {
            f"coverage_{cl}": metric.compute()
            for cl, metric in zip(
                self.coverage_levels, self._cast_coverage_metrics(), strict=True
            )
        }

    def plot(self) -> Any:
        try:
            import wandb  # noqa: PLC0415 since optional dependency
        except ImportError:
            return None

        # Gather computed values from sub-metrics
        results = [metric.compute().item() for metric in self._cast_coverage_metrics()]

        # Create a table for the calibration curve
        table = wandb.Table(
            data=[
                [exp, obs]
                for exp, obs in zip(self.coverage_levels, results, strict=True)
            ],
            columns=["expected_coverage", "observed_coverage"],
        )

        # Create a custom plot (Expected vs Observed)
        plot = wandb.plot.line(
            table,
            "expected_coverage",
            "observed_coverage",
            title="Reliability Diagram (Coverage)",
        )
        return plot

    def reset(self):
        # Reset all sub-metrics
        for metric in self._cast_coverage_metrics():
            metric.reset()
