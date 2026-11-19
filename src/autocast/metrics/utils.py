"""Mixin for handling metrics in Lightning modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, cast

import lightning as L
from torch import Tensor, nn

try:
    from lightning.pytorch.loggers import WandbLogger
except Exception:  # pragma: no cover - optional dependency / version-specific
    WandbLogger = None  # type: ignore[assignment]
from torchmetrics import Metric, MetricCollection

from autocast.metrics import (
    ALL_DETERMINISTIC_METRICS,
    ALL_ENSEMBLE_METRICS,
    ALL_GAUSSIAN_METRICS,
    VRMSE,
)
from autocast.metrics.deterministic import MAE, MSE, RMSE
from autocast.types.types import TensorBTSC, TensorBTSCM


class MetricStage(StrEnum):
    """The stage a logged metric column belongs to; its value is the prefix."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MetricUnits(StrEnum):
    """The space a metric column is computed in, stated as its name's suffix.

    ``NORMALISED`` is the model's working space, the one its loss sees.
    ``PHYSICAL`` is that space mapped back through the model's own
    denormalisation. With no normalisation configured the two coincide.
    """

    PHYSICAL = "physical"
    NORMALISED = "normalised"

    @property
    def other(self) -> MetricUnits:
        if self is MetricUnits.PHYSICAL:
            return MetricUnits.NORMALISED
        return MetricUnits.PHYSICAL


#: ``(prediction, truth)`` as a metric collection's ``update`` takes them.
MetricInputs = tuple[Tensor, Tensor]


class MetricsMixin:
    """Mixin for building and managing metrics in Lightning modules.

    ⛔ **A bare column name does not say which space it was computed in**, and
    the stages differ: ``train_rmse`` is NORMALISED while ``val_rmse`` and
    ``test_rmse`` are PHYSICAL, in the same logged row as a normalised
    ``val_loss``. Comparing the two cost the whole Lorenz-96 "under-fitting"
    story (D169; RESUME trap 12). So every stage ALSO logs each metric under
    both ``<stage>_<metric>_physical`` and ``<stage>_<metric>_normalised``. The
    bare columns are kept exactly as they were, because redefining them would
    change the meaning of every run already on disk and of every checkpoint
    monitor that reads them (``val_multiwinkler``).
    """

    #: The units each stage's BARE columns have always been computed in.
    #: A subclass whose steps feed different units overrides it.
    BARE_METRIC_UNITS: ClassVar[Mapping[MetricStage, MetricUnits]] = MappingProxyType(
        {
            MetricStage.TRAIN: MetricUnits.NORMALISED,
            MetricStage.VAL: MetricUnits.PHYSICAL,
            MetricStage.TEST: MetricUnits.PHYSICAL,
        }
    )

    def _build_unit_twins(self) -> None:
        """Build, per stage, a copy of its metrics for the OTHER units.

        Call after ``train_metrics``/``val_metrics``/``test_metrics`` exist.
        Only the other units are computed a second time: the columns in the
        bare columns' own units are the same metric objects logged under a
        second name. Metric state is non-persistent, so the checkpoint layout
        does not change.
        """
        twins: dict[str, MetricCollection] = {}
        for stage, units in self.BARE_METRIC_UNITS.items():
            bare: MetricCollection | None = getattr(self, f"{stage}_metrics")
            if bare is not None:
                twins[self._twin_key(stage)] = bare.clone(postfix=f"_{units.other}")
        self.metrics_in_other_units = nn.ModuleDict(twins)

    def _twin_key(self, stage: MetricStage) -> str:
        # Named by what the twin logs, e.g. ``train_physical``. A bare stage name
        # cannot be a key: ``train`` collides with ``nn.Module.train``.
        return f"{stage}_{self.BARE_METRIC_UNITS[stage].other}"

    def _update_and_log_stage_metrics(
        self,
        stage: MetricStage,
        normalised: MetricInputs,
        physical: MetricInputs | None,
        batch_size: int,
    ) -> None:
        """Update and log one stage's metrics under all three names.

        Args:
            stage: Which stage's collections to use.
            normalised: ``(prediction, truth)`` in the model's working space.
            physical: The same pair denormalised, or ``None`` when there is
                no normalisation to undo (both names then share one metric).
            batch_size: Batch size.
        """
        bare: MetricCollection | None = getattr(self, f"{stage}_metrics")
        if bare is None:
            return
        model: L.LightningModule = self  # type: ignore[assignment]
        units = self.BARE_METRIC_UNITS[stage]
        inputs = {
            MetricUnits.NORMALISED: normalised,
            MetricUnits.PHYSICAL: normalised if physical is None else physical,
        }
        self._update_and_log_metrics(model, bare, *inputs[units], batch_size)
        # Lightning's own `log_dict` iterates a collection this way; with compute
        # groups the states are shared by reference and must not be copied.
        aliases: dict[str, Metric] = {}
        for name, metric in bare.items(keep_base=False, copy_state=False):
            aliases[f"{name}_{units}"] = metric
            if physical is None:
                aliases[f"{name}_{units.other}"] = metric
        if physical is not None:
            twin = cast(
                MetricCollection, self.metrics_in_other_units[self._twin_key(stage)]
            )
            self._update_and_log_metrics(model, twin, *inputs[units.other], batch_size)
        model.log_dict(
            aliases,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    @staticmethod
    def _build_metrics(
        metrics: Sequence[Metric] | None,
        prefix: str,
    ) -> MetricCollection | None:
        # If no metrics provided, default to a single MSE
        metrics_list = [VRMSE(), RMSE(), MSE(), MAE()] if metrics is None else metrics

        metric_dict: dict[str, Metric | MetricCollection] = {}

        allowed_types = (
            ALL_DETERMINISTIC_METRICS + ALL_ENSEMBLE_METRICS + ALL_GAUSSIAN_METRICS
        )
        for metric in metrics_list:
            if not isinstance(metric, allowed_types):
                allowed = ", ".join(cls.__name__ for cls in allowed_types)
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

        Args:
            model: The Lightning module to log metrics to.
            metrics: The metric collection to update and log. If None, does nothing.
            y_pred: Model predictions.
            y_true: Ground truth targets.
            batch_size: Batch size.
        """
        if metrics is not None:
            metrics.update(y_pred, y_true)
            model.log_dict(
                metrics,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    # TODO: consider re-adding plot method for directly logging to wandb later since
    # this currently doesn't work as it does not appear in the dashboard
    #
    # @staticmethod
    # def log_plots(
    #     model: L.LightningModule, metrics: MetricCollection | None, prefix: str = "val_"  # noqa: E501
    # ) -> None:
    #     """Log custom plots from metrics to WandB.

    #     This should be called at the end of an epoch (e.g. on_validation_epoch_end).
    #     """
    #     if metrics is None:
    #         return

    #     # Check if we are logging to WandB
    #     if WandbLogger is None:
    #         return

    #     if not isinstance(model.logger, WandbLogger):
    #         return

    #     for name, metric in metrics.items():
    #         if hasattr(metric, "plot") and callable(metric.plot):
    #             try:
    #                 plot = metric.plot()
    #             except NotImplementedError:
    #                 continue
    #             if plot is not None:
    #                 print(f"log_plots: Logging plot for {name}, type={type(plot)}")
    #                 model.logger.experiment.log({f"{prefix}{name}_plot": plot})
