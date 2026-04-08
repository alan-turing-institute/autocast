"""Dynamic loss weight scheduling via a Lightning Callback.

Provides schedule classes and a callback that adjusts
``ambient_loss_weight`` / ``latent_loss_weight`` on an
:class:`~autocast.models.encoder_processor_decoder.EncoderProcessorDecoder`
during training.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, cast

import lightning as L

if TYPE_CHECKING:
    from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule callables
# ---------------------------------------------------------------------------


class ConstantSchedule:
    """Returns a fixed value regardless of progress."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, progress: float) -> float:  # noqa: ARG002
        return self.value


class LinearRampSchedule:
    """Linearly interpolates between *start_value* and *end_value*.

    The ramp occupies ``[start_progress, end_progress]``; values outside that
    range are clamped to the nearest endpoint.
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        start_progress: float = 0.0,
        end_progress: float = 1.0,
    ) -> None:
        self.start_value = start_value
        self.end_value = end_value
        self.start_progress = start_progress
        self.end_progress = end_progress

    def __call__(self, progress: float) -> float:
        if progress <= self.start_progress:
            return self.start_value
        if progress >= self.end_progress:
            return self.end_value
        t = (progress - self.start_progress) / (self.end_progress - self.start_progress)
        return self.start_value + t * (self.end_value - self.start_value)


class CosineSchedule:
    """Cosine interpolation between *start_value* and *end_value*.

    Uses a half-cosine curve over ``[start_progress, end_progress]``.
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        start_progress: float = 0.0,
        end_progress: float = 1.0,
    ) -> None:
        self.start_value = start_value
        self.end_value = end_value
        self.start_progress = start_progress
        self.end_progress = end_progress

    def __call__(self, progress: float) -> float:
        if progress <= self.start_progress:
            return self.start_value
        if progress >= self.end_progress:
            return self.end_value
        t = (progress - self.start_progress) / (self.end_progress - self.start_progress)
        cos_t = 0.5 * (1.0 - math.cos(math.pi * t))
        return self.start_value + cos_t * (self.end_value - self.start_value)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

Schedule = ConstantSchedule | LinearRampSchedule | CosineSchedule


class LossWeightScheduleCallback(L.Callback):
    """Dynamically adjust loss weights on an EncoderProcessorDecoder during training.

    Parameters
    ----------
    ambient_schedule:
        Callable ``(progress: float) -> float`` that returns the ambient loss
        weight at the given training progress.  *None* leaves the weight
        unchanged.
    latent_schedule:
        Same as *ambient_schedule* but for the latent loss weight.
    schedule_by:
        ``"step"`` uses ``global_step / estimated_stepping_batches`` as
        progress; ``"epoch"`` uses ``current_epoch / max_epochs``.
    """

    def __init__(
        self,
        ambient_schedule: Schedule | None = None,
        latent_schedule: Schedule | None = None,
        schedule_by: str = "step",
    ) -> None:
        self.ambient_schedule = ambient_schedule
        self.latent_schedule = latent_schedule
        self.schedule_by = schedule_by
        self._encoder_decoder_unfrozen = False

    # -- progress helper ----------------------------------------------------

    def _progress(self, trainer: L.Trainer) -> float:
        if self.schedule_by == "step":
            total = trainer.estimated_stepping_batches
            if total is None or total <= 0:
                msg = (
                    "Cannot compute step-based progress: "
                    "trainer.estimated_stepping_batches is unavailable. "
                    "Set max_epochs or max_steps, or use schedule_by='epoch'."
                )
                raise RuntimeError(msg)
            return min(trainer.global_step / total, 1.0)

        if trainer.max_epochs is None or trainer.max_epochs <= 0:
            msg = (
                "Cannot compute epoch-based progress: "
                "trainer.max_epochs is unavailable. "
                "Set max_epochs or use schedule_by='step'."
            )
            raise RuntimeError(msg)
        return min(trainer.current_epoch / trainer.max_epochs, 1.0)

    # -- main hook ----------------------------------------------------------

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        if not hasattr(pl_module, "ambient_loss_weight"):
            return

        model = cast("EncoderProcessorDecoder", pl_module)
        progress = self._progress(trainer)

        if self.ambient_schedule is not None:
            new_ambient = self.ambient_schedule(progress)
            model.ambient_loss_weight = new_ambient

            # Unfreeze encoder/decoder when ambient weight becomes positive,
            # unless the user explicitly froze it.
            if (
                new_ambient > 0
                and not self._encoder_decoder_unfrozen
                and not model.freeze_encoder_decoder
            ):
                for p in model.encoder_decoder.parameters():
                    p.requires_grad = True
                self._encoder_decoder_unfrozen = True
                log.info(
                    "LossWeightScheduleCallback: unfroze encoder_decoder "
                    "(ambient_loss_weight became positive at progress=%.3f)",
                    progress,
                )

        if self.latent_schedule is not None:
            model.latent_loss_weight = self.latent_schedule(progress)

        model.log(
            "ambient_loss_weight",
            model.ambient_loss_weight,
            prog_bar=False,
        )
        model.log(
            "latent_loss_weight",
            model.latent_loss_weight,
            prog_bar=False,
        )

    # -- checkpoint ---------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {"_encoder_decoder_unfrozen": self._encoder_decoder_unfrozen}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._encoder_decoder_unfrozen = state_dict.get(
            "_encoder_decoder_unfrozen", False
        )
