"""Mixin for handling metrics in Lightning modules."""

from __future__ import annotations

from collections.abc import Sequence

import lightning as L
from torchmetrics import Metric, MetricCollection

from autocast.metrics import ALL_DETERMINISTIC_METRICS, ALL_ENSEMBLE_METRICS, VRMSE
from autocast.types.types import TensorBTSC, TensorBTSCM


class MetricsMixin:
    """Mixin for building and managing metrics in Lightning modules."""

    @staticmethod
    def _build_metrics(
        metrics: Sequence[Metric] | None,
        prefix: str,
    ) -> MetricCollection | None:
        # If no metrics provided, default to a single MSE
        metrics_list = [VRMSE()] if metrics is None else metrics

        metric_dict: dict[str, Metric | MetricCollection] = {}

        for metric in metrics_list:
            if not isinstance(metric, ALL_DETERMINISTIC_METRICS + ALL_ENSEMBLE_METRICS):
                allowed = ", ".join(
                    cls.__name__
                    for cls in ALL_DETERMINISTIC_METRICS + ALL_ENSEMBLE_METRICS
                )
                raise ValueError(
                    f"Invalid metric '{metric}'. Allowed metrics: {allowed}"
                )

            # Determine metric name
            name = getattr(metric, "name", None)
            if not isinstance(name, str):
                name = metric.__class__.__name__.lower()

            if name in metric_dict:
                raise ValueError(f"Duplicate metric name '{name}'")

            metric_dict[name] = metric

        if not metric_dict:
            return None

        return MetricCollection(metric_dict).clone(prefix=prefix)

    @staticmethod
    def _update_and_log_metrics(
        model: L.LightningModule,
        metrics: MetricCollection | None,
        y_pred: TensorBTSC | TensorBTSCM,
        y_true: TensorBTSC,
        batch_size: int,
    ) -> None:
        """Update metrics and log them.

        Parameters
        ----------
        model: L.LightningModule
            The Lightning module to log metrics to.
        metric_collection : MetricCollection | None
            The metric collection to update and log. If None, this method does nothing.
        y_pred : TensorBTSC | TensorBTSCM
            Model predictions.
        y_true : Tensor
            Ground truth targets.
        batch_size : int
            Batch size.
        """
        if metrics is not None:
            metrics.update(y_pred, y_true)
            model.log_dict(
                metrics,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
