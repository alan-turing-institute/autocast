"""Graceful stopping on a sustained deterioration in validation loss."""

import logging
import math
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn

log = logging.getLogger(__name__)


class SustainedValidationRise(Callback):
    """Stop after consecutive validation losses exceed the best by a margin.

    Unlike patience-based early stopping, a flat validation curve does not
    trigger this callback. A check counts only when its loss is greater than
    ``best + max(absolute_tolerance, relative_tolerance * abs(best))``.
    Returning inside that band resets the count. The absolute tolerance makes
    the criterion well-defined near zero; ``abs(best)`` also supports negative
    scores. The monitored metric must be a scalar synchronized across ranks.

    ``patience`` counts distinct validation passes, not epochs or updates.
    Finite losses before ``min_steps`` update the best but cannot accumulate a
    deterioration streak. Nonfinite losses request an immediate graceful stop
    even before that threshold. Sanity checks and standalone validation are
    ignored. Checkpoint callbacks remain responsible for saving best/latest
    weights and optimizer state.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        relative_tolerance: float = 0.01,
        absolute_tolerance: float = 1e-8,
        patience: int = 10,
        min_steps: int = 10000,
    ) -> None:
        super().__init__()
        if not isinstance(monitor, str) or not monitor:
            msg = "monitor must be a nonempty metric name."
            raise ValueError(msg)
        for name, value in (
            ("relative_tolerance", relative_tolerance),
            ("absolute_tolerance", absolute_tolerance),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name, value, minimum in (
            ("patience", patience, 1),
            ("min_steps", min_steps, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")

        self.monitor = monitor
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.patience = patience
        self.min_steps = min_steps
        self.best_score: float | None = None
        self.wait_count = 0
        self.last_checked_step = -1
        self.stopped_step: int | None = None
        self.stop_reason: str | None = None

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            patience=self.patience,
            min_steps=self.min_steps,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "wait_count": self.wait_count,
            "last_checked_step": self.last_checked_step,
            "stopped_step": self.stopped_step,
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_score = state_dict["best_score"]
        self.wait_count = state_dict["wait_count"]
        self.last_checked_step = state_dict["last_checked_step"]
        self.stopped_step = state_dict["stopped_step"]
        self.stop_reason = state_dict["stop_reason"]

    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        if (
            trainer.sanity_checking
            or trainer.state.fn != TrainerFn.FITTING
            or trainer.global_step <= self.last_checked_step
        ):
            return
        if self.monitor not in trainer.callback_metrics:
            raise RuntimeError(
                f"SustainedValidationRise requires validation metric {self.monitor!r}."
            )
        metric = torch.as_tensor(trainer.callback_metrics[self.monitor]).detach()
        if metric.numel() != 1 or metric.is_complex():
            raise ValueError(
                f"Validation metric {self.monitor!r} must be a real scalar."
            )
        current = float(metric.item())
        self.last_checked_step = trainer.global_step
        reason = self._check_score(current, trainer.global_step)
        # Match Lightning's collective early-stop convention; all ranks must
        # leave fit together so the final checkpoint's barrier can complete.
        should_stop = trainer.strategy.reduce_boolean_decision(
            reason is not None, all=False
        )
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_step = trainer.global_step
            self.stop_reason = reason or "Another rank requested validation stopping."
            if trainer.is_global_zero:
                log.info(
                    "Stopping at update %d: %s", self.stopped_step, self.stop_reason
                )

    def _check_score(self, current: float, step: int) -> str | None:
        if not math.isfinite(current):
            return f"Monitored {self.monitor} is nonfinite ({current})."
        if self.best_score is None or current < self.best_score:
            self.best_score = current
            self.wait_count = 0
            return None
        if step < self.min_steps:
            self.wait_count = 0
            return None
        tolerance = max(
            self.absolute_tolerance,
            self.relative_tolerance * abs(self.best_score),
        )
        if current <= self.best_score + tolerance:
            self.wait_count = 0
            return None
        self.wait_count += 1
        if self.wait_count < self.patience:
            return None
        return (
            f"{self.monitor} exceeded best {self.best_score:.8g} by more than "
            f"{tolerance:.8g} for {self.wait_count} consecutive validations "
            f"(current {current:.8g})."
        )
