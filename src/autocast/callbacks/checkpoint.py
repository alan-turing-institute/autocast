import logging
import math
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)


class ProgressModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint with progress-based cadence and delayed monitored saves.

    This adds two repo-specific conveniences on top of Lightning's standard
    callback:

    - ``every_n_train_steps_fraction`` resolves ``every_n_train_steps`` from the
      estimated number of optimizer steps in the run.
    - ``start_after_fraction`` delays monitored top-k checkpointing until a
      chosen fraction of training has completed.
    - ``stop_after_fraction`` stops monitored top-k checkpointing once a chosen
      fraction of training has completed.

    ``save_last`` continues to behave exactly as in ``ModelCheckpoint``.
    """

    def __init__(
        self,
        *,
        start_after_fraction: float = 0.0,
        stop_after_fraction: float | None = None,
        every_n_train_steps_fraction: float | None = None,
        monitor_optional: bool = False,
        **kwargs: Any,
    ) -> None:
        if every_n_train_steps_fraction is not None:
            # Lightning defaults to every_n_epochs=1 when no explicit trigger is
            # set. Fractional step snapshots should not silently add epoch saves.
            kwargs.setdefault("every_n_epochs", 0)
        super().__init__(**kwargs)

        if not 0.0 <= start_after_fraction <= 1.0:
            raise ValueError(
                f"start_after_fraction must be in [0, 1], got {start_after_fraction}."
            )
        if stop_after_fraction is not None and not 0.0 <= stop_after_fraction <= 1.0:
            raise ValueError(
                f"stop_after_fraction must be in [0, 1], got {stop_after_fraction}."
            )
        if (
            stop_after_fraction is not None
            and stop_after_fraction > 0.0
            and stop_after_fraction < start_after_fraction
        ):
            raise ValueError(
                "stop_after_fraction must be >= start_after_fraction when set, "
                f"got start_after_fraction={start_after_fraction}, "
                f"stop_after_fraction={stop_after_fraction}."
            )
        if every_n_train_steps_fraction is not None and not (
            0.0 < every_n_train_steps_fraction <= 1.0
        ):
            raise ValueError(
                "every_n_train_steps_fraction must be in (0, 1], "
                f"got {every_n_train_steps_fraction}."
            )
        if every_n_train_steps_fraction is not None:
            if self._every_n_train_steps >= 1:
                msg = (
                    "Specify either every_n_train_steps or "
                    "every_n_train_steps_fraction, not both."
                )
                raise ValueError(msg)
            if self._every_n_epochs >= 1:
                msg = (
                    "Specify either every_n_epochs or "
                    "every_n_train_steps_fraction, not both."
                )
                raise ValueError(msg)
            if self._train_time_interval is not None:
                msg = (
                    "Specify either train_time_interval or "
                    "every_n_train_steps_fraction, not both."
                )
                raise ValueError(msg)

        self.start_after_fraction = start_after_fraction
        self.stop_after_fraction = stop_after_fraction
        self.every_n_train_steps_fraction = every_n_train_steps_fraction
        self.monitor_optional = monitor_optional
        self._resolved_fractional_steps = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)
        self._maybe_resolve_fractional_train_steps(trainer)

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            every_n_train_steps=self._every_n_train_steps,
            every_n_epochs=self._every_n_epochs,
            train_time_interval=self._train_time_interval,
            start_after_fraction=self.start_after_fraction,
            stop_after_fraction=self.stop_after_fraction,
            every_n_train_steps_fraction=self.every_n_train_steps_fraction,
            monitor_optional=self.monitor_optional,
        )

    def _maybe_resolve_fractional_train_steps(self, trainer: L.Trainer) -> None:
        if self.every_n_train_steps_fraction is None or self._resolved_fractional_steps:
            return

        total_steps = self._resolve_total_training_steps(trainer)
        every_n_steps = max(
            1, math.ceil(total_steps * self.every_n_train_steps_fraction)
        )
        self._every_n_train_steps = every_n_steps
        self._resolved_fractional_steps = True

        log.info(
            "Resolved %s every_n_train_steps=%d from %.1f%% of %d estimated steps.",
            type(self).__name__,
            every_n_steps,
            100.0 * self.every_n_train_steps_fraction,
            total_steps,
        )

    @staticmethod
    def _resolve_total_training_steps(trainer: L.Trainer) -> int:
        estimated_steps = getattr(trainer, "estimated_stepping_batches", None)
        if estimated_steps is not None:
            try:
                estimated_steps_int = int(estimated_steps)
            except (TypeError, ValueError):
                estimated_steps_int = 0
            if estimated_steps_int > 0:
                return estimated_steps_int

        max_steps = getattr(trainer, "max_steps", None)
        if max_steps is not None and int(max_steps) > 0:
            return int(max_steps)

        max_epochs = getattr(trainer, "max_epochs", None)
        num_training_batches = getattr(trainer, "num_training_batches", None)
        if max_epochs is not None and int(max_epochs) > 0:
            if num_training_batches is None:
                n_batches = 0
            else:
                try:
                    n_batches = int(num_training_batches)
                except (TypeError, ValueError):
                    n_batches = 0
            if n_batches > 0:
                accumulate_grad_batches = max(
                    1, int(getattr(trainer, "accumulate_grad_batches", 1))
                )
                steps_per_epoch = math.ceil(n_batches / accumulate_grad_batches)
                if steps_per_epoch > 0:
                    return int(max_epochs) * steps_per_epoch

        raise ValueError(
            "Unable to infer total training steps for "
            f"{type(trainer).__name__}. Set trainer.max_epochs/max_steps or use "
            "an explicit every_n_train_steps value instead."
        )

    def _training_progress_fraction(self, trainer: L.Trainer) -> float:
        try:
            total_steps = self._resolve_total_training_steps(trainer)
        except ValueError:
            max_epochs = getattr(trainer, "max_epochs", None)
            if max_epochs is None or int(max_epochs) <= 0:
                return 0.0
            current_epoch = max(0, int(getattr(trainer, "current_epoch", 0)))
            return min(1.0, float(current_epoch + 1) / float(max_epochs))

        global_step = max(0, int(getattr(trainer, "global_step", 0)))
        return min(1.0, float(global_step) / float(total_steps))

    def _monitor_candidates(self, trainer: L.Trainer) -> dict[str, Any]:
        monitor_candidates: dict[str, Any] = dict(super()._monitor_candidates(trainer))
        progress_fraction = self._training_progress_fraction(trainer)
        progress_pct = round(100.0 * progress_fraction)
        monitor_candidates["progress_pct"] = torch.tensor(progress_pct)
        monitor_candidates["progress_token"] = self._format_progress_token(
            progress_fraction
        )
        return monitor_candidates

    @staticmethod
    def _format_progress_token(progress_fraction: float) -> str:
        progress_hundredths = round(100.0 * progress_fraction)
        progress_hundredths = max(0, min(100, progress_hundredths))
        return f"{progress_hundredths // 100}p{progress_hundredths % 100:02d}"

    def _monitor_ready(self, trainer: L.Trainer) -> bool:
        if self.monitor is None:
            return True

        progress = self._training_progress_fraction(trainer)
        if self.start_after_fraction > 0.0 and progress < self.start_after_fraction:
            return False
        return not (
            self.stop_after_fraction is not None and progress > self.stop_after_fraction
        )

    def _save_topk_checkpoint(
        self, trainer: L.Trainer, monitor_candidates: dict[str, Any]
    ) -> None:
        if (
            self.monitor is not None
            and self.monitor not in monitor_candidates
            and self.monitor_optional
        ):
            return

        if self.monitor is not None and not self._monitor_ready(trainer):
            return

        super()._save_topk_checkpoint(trainer, monitor_candidates)
