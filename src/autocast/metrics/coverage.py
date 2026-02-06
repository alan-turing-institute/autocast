from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch.nn import ModuleList
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

    def __init__(self, coverage_level: float = 0.95, **kwargs):
        """Initialize Coverage metric.

        Args:
            coverage_level: nominal coverage probability (e.g. 0.95 for 95% interval).
                Must be between 0 and 1.
            **kwargs: Additional arguments passed to BaseMetric.
        """
        super().__init__(**kwargs)
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
        quantiles = torch.quantile(y_pred, q_tensor, dim=-1)  # (2, B, T, S, C)

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
        # List of Coverage metrics with reduce_all=False to allow per-channel analysis
        self.metrics = ModuleList(
            [Coverage(coverage_level=cl, reduce_all=False) for cl in coverage_levels]
        )

    def update(self, y_pred, y_true):
        for metric in self.metrics:
            assert isinstance(metric, Coverage)
            metric.update(y_pred, y_true)

    def compute(self) -> Tensor:
        """Compute the Average Calibration Error."""
        errors = []
        for cl, metric in zip(self.coverage_levels, self.metrics, strict=True):
            assert isinstance(metric, Coverage)
            # Calibration error: |observed - expected|
            errors.append(torch.abs(metric.compute() - cl))

        return torch.stack(errors).mean()

    def _compute_levels_and_values(self) -> tuple[list[float], list[float]]:
        """Get coverage levels and observed values for plotting."""
        levels, observed = [], []
        for cl, metric in zip(self.coverage_levels, self.metrics, strict=True):
            assert isinstance(metric, Coverage)
            levels.append(cl)
            observed.append(metric.compute().mean().item())
        return levels, observed

    def compute_detailed(self) -> dict[str, float]:
        """Return a dict of results, keys formatted as 'coverage_{coverage_level}'."""
        return {
            f"coverage_{level}": value
            for level, value in zip(*self._compute_levels_and_values(), strict=True)
        }

    def plot(
        self,
        save_path: Path | str | None = None,
        title: str = "Coverage Plot",
        cmap_str: str = "viridis",
    ):
        """
        Plot reliability diagram showing expected vs observed coverage.

        Parameters
        ----------
        save_path: str, optional
            Path to save the plot.
        title: str
            Plot title.
        cmap_str: str
            Color map string from matplotlib.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # Prepare data structure: levels -> [val_c1, val_c2, ...]
        levels = self.coverage_levels
        observed_means = []
        observed_channels = []  # shape (L, C)

        # Loop over metrics
        for metric in self.metrics:
            assert isinstance(metric, Coverage)
            val = metric.compute()
            val_c = val.mean(dim=0).cpu().numpy()  # (C,)
            observed_channels.append(val_c)
            observed_means.append(val_c.mean())

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Optimal line (y=x)
        ax.plot([0, 1], [0, 1], "k:", label="Expected", linewidth=2)

        # Plot channels
        observed_arr = np.stack(observed_channels)  # (L, C)
        cmap = plt.get_cmap(cmap_str)  # cmap for each channel
        n_channels = observed_channels[0].shape[0]
        for c in range(n_channels):
            color = cmap(c / n_channels) if n_channels > 1 else "blue"
            label = f"Ch {c}" if n_channels <= 10 else None
            ax.plot(
                levels,
                observed_arr[:, c],
                color=color,
                alpha=0.3,
                linewidth=1,
                label=label,
            )

        # Plot mean coverage in bold
        ax.plot(levels, observed_means, "k-", linewidth=3, label="Mean")
        ax.set_xlabel(r"Coverage level, $\alpha$")
        ax.set_ylabel("Observed Coverage")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Only show legend if not cluttered
        if n_channels > 10:
            # Add manual legend for trace
            custom_lines = [
                Line2D([0], [0], color="k", lw=3),
                Line2D([0], [0], color="grey", lw=1, alpha=0.5),
            ]
            ax.legend(custom_lines, ["Mean", "Individual Channels"])
        else:
            ax.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.close(fig)
        return fig

    def reset(self):
        # Reset all sub-metrics
        for metric in self.metrics:
            assert isinstance(metric, Coverage)
            metric.reset()
