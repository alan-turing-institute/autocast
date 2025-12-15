"""Mixin for handling metrics in Lightning modules."""

from __future__ import annotations

from typing import Literal

from torchmetrics import Metric, MetricCollection

from autocast.metrics import ALL_METRICS, MSE
from autocast.types import Tensor


class MetricsMixin:
    """Mixin for building and managing metrics in Lightning modules."""

    @staticmethod
    def _build_metrics(
        metrics: list[Metric] | None,
        prefix: str,
    ) -> MetricCollection | None:
        # If no metrics provided, default to a single MSE
        metrics_list = [MSE()] if metrics is None else metrics

        metric_dict: dict[str, Metric | MetricCollection] = {}

        for metric in metrics_list:
            if not isinstance(metric, ALL_METRICS):
                allowed = ", ".join(cls.__name__ for cls in ALL_METRICS)
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
