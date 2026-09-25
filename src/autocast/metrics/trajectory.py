"""Trajectory-level aggregation for windowed and rollout evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import torch
from torchmetrics import Metric

from autocast.metrics.coverage import Coverage, MultiCoverage
from autocast.types import TensorBTSC, TensorBTSCM

Window = tuple[int, int] | None
CsvValue = float | int | str


class TrajectoryTask(StrEnum):
    """Kinds of trajectory statistics written by evaluation."""

    SINGLE_STEP = "single_step"
    ROLLOUT_WINDOW = "rollout_window"
    ROLLOUT_LEAD = "rollout_lead"


@dataclass
class _MetricTotals:
    """Raw additive components for one trajectory and time selection."""

    metrics: dict[str, Metric]
    n_samples: int = 0
    vrmse_count: int = 0
    coverage_count: int = 0
    n_timesteps: int = 0
    window_indices: set[int] = field(default_factory=set)


@dataclass
class _PerTimestepResult:
    """Materialized scalar metrics for one trajectory and rollout lead."""

    metric_values: dict[str, float]
    vrmse_count: int
    coverage_count: int


def _window_label(window: Window) -> str:
    if window is None:
        return "all"
    return f"[{window[0]}:{window[1]})"


def _coverage_column(level: float) -> str:
    return f"coverage_{level:.2f}"


def build_trajectory_metadata(dataset: Any) -> dict[int, dict[str, CsvValue]]:
    """Export raw per-trajectory conditioning scalars with generic names."""
    constant_scalars = getattr(dataset, "constant_scalars", None)
    if constant_scalars is None:
        return {}

    values = torch.as_tensor(constant_scalars).detach().cpu()
    if values.ndim != 2:
        msg = (
            "Expected dataset.constant_scalars to have shape (trajectory, scalar), "
            f"got {tuple(values.shape)}."
        )
        raise ValueError(msg)

    scalar_names = [f"cs{index}" for index in range(int(values.shape[1]))]
    rows: dict[int, dict[str, CsvValue]] = {}

    for trajectory_idx, scalar_values in enumerate(values):
        row: dict[str, CsvValue] = {
            name: float(value.item())
            for name, value in zip(scalar_names, scalar_values, strict=True)
        }
        rows[trajectory_idx] = row

    return rows


class TrajectoryMetricAccumulator:
    """Stream prediction batches into trajectory-level metric components."""

    def __init__(
        self,
        *,
        sample_trajectory_indices: list[int],
        sample_window_indices: list[int],
        windows: list[Window] | None,
        metric_fns: dict[str, Callable[[], Metric]],
        include_per_timestep: bool = False,
        trajectory_metadata: dict[int, dict[str, CsvValue]] | None = None,
        row_context: dict[str, CsvValue] | None = None,
    ) -> None:
        if len(sample_trajectory_indices) != len(sample_window_indices):
            msg = (
                "Trajectory and window metadata must have the same length, got "
                f"{len(sample_trajectory_indices)} and {len(sample_window_indices)}."
            )
            raise ValueError(msg)
        if not sample_trajectory_indices:
            msg = "Trajectory metric aggregation requires at least one sample."
            raise ValueError(msg)
        if not metric_fns:
            msg = "Trajectory metric aggregation requires metric factories."
            raise ValueError(msg)

        normalized_windows = [None] if windows is None else list(windows)
        if not normalized_windows:
            msg = "Trajectory metric aggregation requires at least one window."
            raise ValueError(msg)
        for window in normalized_windows:
            if window is not None and window[0] >= window[1]:
                msg = f"Metric windows must satisfy start < end, got {window}."
                raise ValueError(msg)

        self.sample_trajectory_indices = list(sample_trajectory_indices)
        self.sample_window_indices = list(sample_window_indices)
        self.windows = normalized_windows
        self.metric_fns = dict(metric_fns)
        self.include_per_timestep = include_per_timestep
        self.trajectory_metadata = trajectory_metadata or {}
        self.row_context = row_context or {}
        self._next_sample = 0
        self._window_totals: dict[tuple[int, Window], _MetricTotals] = {}
        self._per_timestep_results: dict[tuple[int, int], _PerTimestepResult] = {}

        if self.include_per_timestep:
            trajectory_counts: dict[int, int] = {}
            for trajectory_idx in self.sample_trajectory_indices:
                trajectory_counts[trajectory_idx] = (
                    trajectory_counts.get(trajectory_idx, 0) + 1
                )
            repeated = {
                trajectory_idx: count
                for trajectory_idx, count in trajectory_counts.items()
                if count != 1
            }
            if repeated:
                msg = (
                    "Per-timestep trajectory metrics require one full-trajectory "
                    f"sample per trajectory, got repeated counts {repeated}."
                )
                raise ValueError(msg)

    def _new_totals(self, device: torch.device) -> _MetricTotals:
        metrics = {
            name: factory().to(device) for name, factory in self.metric_fns.items()
        }
        return _MetricTotals(metrics=metrics)

    def _new_time_series_metrics(self, device: torch.device) -> dict[str, Metric]:
        metrics = self._new_totals(device).metrics
        for metric in metrics.values():
            if hasattr(metric, "reduce_all"):
                metric.reduce_all = False
        return metrics

    @staticmethod
    def _time_values(value: Any, n_timesteps: int) -> torch.Tensor | None:
        if not isinstance(value, torch.Tensor):
            return None
        if value.ndim == 0 or int(value.shape[0]) != n_timesteps:
            return None
        return value.reshape(n_timesteps, -1).mean(dim=-1)

    @staticmethod
    def _metric_preserves_time(metric: Metric) -> bool:
        vector_dims = getattr(metric, "vector_dims", None)
        if vector_dims in (
            "temporal",
            "spatial_temporal",
            "spatial_temporal_channels",
        ):
            return False
        return getattr(metric, "score_dims", None) != "temporal"

    @staticmethod
    def _fallback_time_values(
        factory: Callable[[], Metric],
        preds: torch.Tensor,
        trues: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        for lead_time in range(int(trues.shape[1])):
            metric = factory().to(trues.device)
            metric.update(
                preds[:, lead_time : lead_time + 1],
                trues[:, lead_time : lead_time + 1],
            )
            value = metric.compute()
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=trues.device)
            values.append(value.mean())
        return torch.stack(values)

    def _compute_per_timestep_values(
        self,
        preds: torch.Tensor,
        trues: torch.Tensor,
    ) -> list[dict[str, float]]:
        """Compute all lead-wise scalars without retaining one metric per lead."""
        n_timesteps = int(trues.shape[1])
        per_lead: list[dict[str, float]] = [{} for _ in range(n_timesteps)]
        metrics = self._new_time_series_metrics(trues.device)
        time_preserving = {
            name
            for name, metric in metrics.items()
            if self._metric_preserves_time(metric)
        }
        for name in time_preserving:
            metrics[name].update(preds, trues)

        for name, metric in metrics.items():
            if isinstance(metric, MultiCoverage):
                observed_series = []
                for level, coverage_metric in zip(
                    metric.coverage_levels, metric.metrics, strict=True
                ):
                    coverage_metric = cast(Coverage, coverage_metric)
                    values = self._time_values(coverage_metric.compute(), n_timesteps)
                    if values is None:
                        msg = (
                            "Coverage metric did not preserve the rollout time "
                            f"dimension for {n_timesteps} lead times."
                        )
                        raise RuntimeError(msg)
                    observed_series.append(values)
                    for lead_time, value in enumerate(values.detach().cpu().tolist()):
                        per_lead[lead_time][_coverage_column(level)] = float(value)

                observed = torch.stack(observed_series, dim=-1)
                nominal = torch.as_tensor(
                    metric.coverage_levels,
                    device=observed.device,
                    dtype=observed.dtype,
                )
                coverage_mae = (observed - nominal).abs().mean(dim=-1)
                for lead_time, value in enumerate(coverage_mae.detach().cpu().tolist()):
                    per_lead[lead_time][name] = float(value)
                    per_lead[lead_time]["coverage_mae"] = float(value)
                continue

            values = (
                self._time_values(metric.compute(), n_timesteps)
                if name in time_preserving
                else None
            )
            if values is None:
                # Metrics that vectorize over time (for example the default
                # energy score) need to be evaluated on one lead at a time.
                values = self._fallback_time_values(self.metric_fns[name], preds, trues)
            for lead_time, value in enumerate(values.detach().cpu().tolist()):
                per_lead[lead_time][name] = float(value)

        return per_lead

    def _add_per_timestep_results(
        self,
        *,
        trajectory_idx: int,
        preds: torch.Tensor,
        trues: torch.Tensor,
    ) -> None:
        if int(trues.shape[0]) != 1:
            msg = "Per-timestep metrics require one source sample per trajectory."
            raise RuntimeError(msg)

        per_lead_values = self._compute_per_timestep_values(preds, trues)
        for lead_time, metric_values in enumerate(per_lead_values):
            key = (trajectory_idx, lead_time)
            if key in self._per_timestep_results:
                msg = f"Duplicate per-timestep metric result for {key}."
                raise RuntimeError(msg)
            self._per_timestep_results[key] = _PerTimestepResult(
                metric_values=metric_values,
                vrmse_count=int(trues.shape[-1]),
                coverage_count=math.prod(trues.shape[2:]),
            )

    def _add(
        self,
        totals: _MetricTotals,
        *,
        preds: torch.Tensor,
        trues: torch.Tensor,
        window_indices: list[int],
    ) -> None:
        for metric in totals.metrics.values():
            metric.update(preds, trues)

        n_samples = int(trues.shape[0])
        n_timesteps = int(trues.shape[1])
        totals.n_samples += n_samples
        totals.n_timesteps += n_samples * n_timesteps
        totals.vrmse_count += n_samples * n_timesteps * int(trues.shape[-1])
        totals.coverage_count += n_samples * n_timesteps * math.prod(trues.shape[2:])
        totals.window_indices.update(window_indices)

    def update(self, preds: TensorBTSCM, trues: TensorBTSC) -> None:
        """Consume one inference batch without retaining prediction tensors."""
        batch_size = int(trues.shape[0])
        batch_end = self._next_sample + batch_size
        if batch_end > len(self.sample_trajectory_indices):
            msg = (
                "Inference produced more samples than the trajectory metadata: "
                f"need {batch_end}, have {len(self.sample_trajectory_indices)}."
            )
            raise RuntimeError(msg)

        grouped_local_indices: dict[int, list[int]] = {}
        grouped_window_indices: dict[int, list[int]] = {}
        for local_idx in range(batch_size):
            sample_idx = self._next_sample + local_idx
            trajectory_idx = self.sample_trajectory_indices[sample_idx]
            grouped_local_indices.setdefault(trajectory_idx, []).append(local_idx)
            grouped_window_indices.setdefault(trajectory_idx, []).append(
                self.sample_window_indices[sample_idx]
            )

        n_timesteps = int(trues.shape[1])
        for trajectory_idx, local_indices in grouped_local_indices.items():
            index = torch.tensor(local_indices, device=trues.device, dtype=torch.long)
            trajectory_preds = preds.index_select(0, index)
            trajectory_trues = trues.index_select(0, index)
            window_indices = grouped_window_indices[trajectory_idx]
            for window in self.windows:
                start, end = (0, n_timesteps) if window is None else window
                end = min(end, n_timesteps)
                if start >= end:
                    continue
                key = (trajectory_idx, window)
                totals = self._window_totals.get(key)
                if totals is None:
                    totals = self._new_totals(trues.device)
                    self._window_totals[key] = totals
                self._add(
                    totals,
                    preds=trajectory_preds[:, start:end],
                    trues=trajectory_trues[:, start:end],
                    window_indices=window_indices,
                )

            if self.include_per_timestep:
                self._add_per_timestep_results(
                    trajectory_idx=trajectory_idx,
                    preds=trajectory_preds,
                    trues=trajectory_trues,
                )

        self._next_sample = batch_end

    def validate_complete(self) -> None:
        """Require every source sample to have contributed exactly once."""
        expected = len(self.sample_trajectory_indices)
        if self._next_sample != expected:
            msg = (
                "Trajectory metric aggregation consumed "
                f"{self._next_sample} of {expected} source samples."
            )
            raise RuntimeError(msg)

    def _row(
        self,
        *,
        trajectory_idx: int,
        task: TrajectoryTask,
        totals: _MetricTotals,
    ) -> dict[str, CsvValue]:
        row = self._base_row(
            trajectory_idx=trajectory_idx,
            task=task,
            n_windows=len(totals.window_indices),
            n_samples=totals.n_samples,
            n_timesteps=totals.n_timesteps,
            vrmse_count=totals.vrmse_count,
            coverage_count=totals.coverage_count,
        )

        for name, metric in totals.metrics.items():
            if name == "coverage" and isinstance(metric, MultiCoverage):
                detailed = metric.compute_detailed()
                empirical_coverage = [
                    detailed[f"coverage_{level}"] for level in metric.coverage_levels
                ]
                coverage_mae = sum(
                    abs(observed - nominal)
                    for observed, nominal in zip(
                        empirical_coverage, metric.coverage_levels, strict=True
                    )
                ) / len(metric.coverage_levels)
                row["coverage"] = coverage_mae
                row["coverage_mae"] = coverage_mae
                for level, observed in zip(
                    metric.coverage_levels, empirical_coverage, strict=True
                ):
                    row[_coverage_column(level)] = observed
                continue

            value = metric.compute()
            row[name] = float(
                value.item() if value.numel() == 1 else value.mean().item()
            )
        return row

    def _base_row(
        self,
        *,
        trajectory_idx: int,
        task: TrajectoryTask,
        n_windows: int,
        n_samples: int,
        n_timesteps: int,
        vrmse_count: int,
        coverage_count: int,
    ) -> dict[str, CsvValue]:
        row = dict(self.row_context)
        row.update(self.trajectory_metadata.get(trajectory_idx, {}))
        row.update(
            {
                "task": task.value,
                "trajectory_idx": trajectory_idx,
                "trajectory_id": f"test_{trajectory_idx:04d}",
                "n_windows": n_windows,
                "n_samples": n_samples,
                "n_timesteps": n_timesteps,
                "n_vrmse_components": vrmse_count,
                "n_coverage_components": coverage_count,
            }
        )
        return row

    def window_rows(self, *, task: TrajectoryTask) -> list[dict[str, CsvValue]]:
        """Return one row per trajectory and configured metric window."""
        if task not in (
            TrajectoryTask.SINGLE_STEP,
            TrajectoryTask.ROLLOUT_WINDOW,
        ):
            msg = f"Window rows do not support task {task}."
            raise ValueError(msg)

        rows = []
        trajectory_indices = sorted(
            {trajectory_idx for trajectory_idx, _ in self._window_totals}
        )
        for trajectory_idx in trajectory_indices:
            for window in self.windows:
                totals = self._window_totals.get((trajectory_idx, window))
                if totals is None:
                    continue
                row = self._row(
                    trajectory_idx=trajectory_idx,
                    task=task,
                    totals=totals,
                )
                row["window"] = _window_label(window)
                rows.append(row)
        return rows

    def per_timestep_rows(self) -> list[dict[str, CsvValue]]:
        """Return one row per trajectory and rollout lead time."""
        rows = []
        for trajectory_idx, lead_time in sorted(self._per_timestep_results):
            result = self._per_timestep_results[(trajectory_idx, lead_time)]
            row = self._base_row(
                trajectory_idx=trajectory_idx,
                task=TrajectoryTask.ROLLOUT_LEAD,
                n_windows=1,
                n_samples=1,
                n_timesteps=1,
                vrmse_count=result.vrmse_count,
                coverage_count=result.coverage_count,
            )
            row.update(result.metric_values)
            row["lead_time"] = lead_time
            row["window"] = _window_label((lead_time, lead_time + 1))
            rows.append(row)
        return rows
