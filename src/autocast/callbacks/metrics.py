import logging
import math
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torchmetrics import Metric, MetricCollection

log = logging.getLogger(__name__)


class ValidationMetricPlotCallback(Callback):
    """Plot validation metric histories and custom metric diagnostics.

    Lightning/W&B already logs scalar validation metrics, but a single compact
    history figure makes it easier to inspect calibration drift across a run.
    Metrics with a custom ``plot`` method, such as ``MultiCoverage``, are also
    plotted at validation end.
    """

    def __init__(
        self,
        *,
        metric_prefixes: Sequence[str] = ("val_",),
        dirname: str = "validation_metrics",
        filename: str = "validation_metrics.png",
        max_cols: int = 3,
        save_local: bool = True,
        log_to_logger: bool = True,
        plot_metric_objects: bool = True,
        metric_plot_dirname: str = "metric_plots",
    ) -> None:
        if not metric_prefixes:
            msg = "metric_prefixes must contain at least one prefix."
            raise ValueError(msg)
        if max_cols < 1:
            msg = f"max_cols must be >= 1, got {max_cols}."
            raise ValueError(msg)

        self.metric_prefixes = tuple(metric_prefixes)
        self.dirname = dirname
        self.filename = filename
        self.max_cols = max_cols
        self.save_local = save_local
        self.log_to_logger = log_to_logger
        self.plot_metric_objects = plot_metric_objects
        self.metric_plot_dirname = metric_plot_dirname
        self._history: dict[str, list[tuple[int, float]]] = {}

    def state_dict(self) -> dict[str, Any]:
        return {"history": self._history}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        history = state_dict.get("history", {})
        if isinstance(history, dict):
            self._history = {
                str(name): [(int(step), float(value)) for step, value in values]
                for name, values in history.items()
            }

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return

        self._ensure_headless_backend()
        step = int(getattr(trainer, "global_step", 0))
        scalars = self._collect_scalar_metrics(trainer.callback_metrics)
        if scalars:
            for name, value in scalars.items():
                self._history.setdefault(name, []).append((step, value))
            self._plot_scalar_history(trainer, step)

        if self.plot_metric_objects:
            self._plot_metric_objects(trainer, pl_module, step)

    def _collect_scalar_metrics(self, metrics: Mapping[str, Any]) -> dict[str, float]:
        scalars: dict[str, float] = {}
        for name, value in metrics.items():
            if not name.startswith(self.metric_prefixes):
                continue
            scalar = self._to_float(value)
            if scalar is None or not math.isfinite(scalar):
                continue
            scalars[name] = scalar
        return dict(sorted(scalars.items()))

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _ensure_headless_backend() -> None:
        if plt.get_backend().lower() != "agg":
            plt.switch_backend("Agg")

    def _plot_scalar_history(self, trainer: L.Trainer, step: int) -> None:
        if not self._history:
            return

        metric_names = sorted(self._history)
        n_metrics = len(metric_names)
        n_cols = min(self.max_cols, n_metrics)
        n_rows = math.ceil(n_metrics / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.0 * n_cols, 3.0 * n_rows),
            squeeze=False,
        )

        for ax, name in zip(axes.ravel(), metric_names, strict=False):
            values = self._history[name]
            xs = [point_step for point_step, _ in values]
            ys = [point_value for _, point_value in values]
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3)
            ax.set_title(name)
            ax.set_xlabel("global_step")
            ax.grid(True, linestyle=":", alpha=0.5)

        for ax in axes.ravel()[n_metrics:]:
            ax.axis("off")

        fig.tight_layout()
        save_path = self._save_figure(trainer, fig, self.filename)
        self._log_figure(trainer, "validation/metrics", fig, step)
        plt.close(fig)
        if save_path is not None:
            log.debug("Saved validation metric history plot to %s", save_path)

    def _plot_metric_objects(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        step: int,
    ) -> None:
        val_metrics = getattr(pl_module, "val_metrics", None)
        if val_metrics is None:
            return

        for name, metric in self._iter_metrics(val_metrics):
            if not self._has_custom_plot(metric):
                continue
            safe_name = self._safe_filename(name)
            save_path = None
            if self.save_local:
                save_path = self._local_plot_dir(trainer) / self.metric_plot_dirname
                save_path = save_path / f"{safe_name}.png"
            try:
                fig = self._render_metric_plot(metric, name)
            except (NotImplementedError, RuntimeError, ValueError) as exc:
                log.debug("Skipping validation plot for %s: %s", name, exc)
                continue

            if fig is None:
                continue
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, bbox_inches="tight")
            self._log_figure(trainer, f"validation/{name}_plot", fig, step)
            plt.close(fig)

    @staticmethod
    def _render_metric_plot(metric: Metric, name: str) -> Figure | None:
        plot_kwargs_options = (
            {"title": name, "save_csv": False},
            {"title": name},
            {},
        )
        for kwargs in plot_kwargs_options:
            try:
                return metric.plot(**kwargs)
            except TypeError:
                continue
        return None

    @staticmethod
    def _iter_metrics(metrics: MetricCollection | Metric) -> list[tuple[str, Metric]]:
        if isinstance(metrics, MetricCollection):
            return [
                (str(name), metric)
                for name, metric in metrics.items()
                if isinstance(metric, Metric)
            ]
        if isinstance(metrics, Metric):
            name = getattr(metrics, "name", type(metrics).__name__.lower())
            return [(str(name), metrics)]
        return []

    @staticmethod
    def _has_custom_plot(metric: Metric) -> bool:
        plot_method = getattr(type(metric), "plot", None)
        base_plot_method = getattr(Metric, "plot", None)
        return plot_method is not None and plot_method is not base_plot_method

    def _save_figure(
        self, trainer: L.Trainer, fig: Figure, filename: str
    ) -> Path | None:
        if not self.save_local:
            return None
        save_path = self._local_plot_dir(trainer) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        return save_path

    def _local_plot_dir(self, trainer: L.Trainer) -> Path:
        root = getattr(trainer, "default_root_dir", None) or "."
        return Path(root) / self.dirname

    def _log_figure(self, trainer: L.Trainer, key: str, fig: Figure, step: int) -> None:
        if not self.log_to_logger:
            return
        for logger in self._iter_loggers(trainer):
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                continue
            if hasattr(experiment, "add_figure"):
                experiment.add_figure(key, fig, global_step=step)
                continue
            if hasattr(experiment, "log"):
                payload: dict[str, Any] = {}
                if self._is_wandb_experiment(experiment):
                    wandb_image = self._try_make_wandb_image(fig)
                    if wandb_image is not None:
                        payload[key] = wandb_image
                if not payload:
                    payload[key] = fig
                experiment.log(payload, step=step)

    @staticmethod
    def _is_wandb_experiment(experiment: Any) -> bool:
        module = getattr(type(experiment), "__module__", "") or ""
        if module.startswith("wandb."):
            return True
        return hasattr(experiment, "project") and hasattr(experiment, "entity")

    @staticmethod
    def _try_make_wandb_image(fig: Figure) -> Any | None:
        try:
            wandb = import_module("wandb")
        except Exception:
            return None
        image_cls = getattr(wandb, "Image", None)
        if image_cls is None:
            return None
        try:
            return image_cls(fig)
        except Exception:
            return None

    @staticmethod
    def _iter_loggers(trainer: L.Trainer) -> list[Any]:
        loggers = getattr(trainer, "loggers", None)
        if loggers is not None:
            return [logger for logger in loggers if logger]

        logger = getattr(trainer, "logger", None)
        if logger in (None, False):
            return []
        if isinstance(logger, list | tuple):
            return [single_logger for single_logger in logger if single_logger]
        return [logger]

    @staticmethod
    def _safe_filename(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
