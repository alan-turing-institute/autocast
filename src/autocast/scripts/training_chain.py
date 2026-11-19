"""Allocation timer and checkpoint handoff for Slurm training chains."""

import hashlib
import json
import logging
import os
from pathlib import Path
from time import time
from typing import Self, cast

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, Timer

log = logging.getLogger(__name__)


class TrainingChain(Callback):
    """Stop at an epoch boundary and hand off one full-state checkpoint."""

    def __init__(self, work_dir: Path, segment: int, deadline: int, run_id: str):
        self.directory = work_dir / ".chain"
        self.segment = segment
        self.run_id = run_id
        self.deadline = deadline
        self.checkpoint = self.directory / f"segment-{segment:04d}.ckpt"
        self._last_epoch_end = 0.0
        self._longest_epoch = 0.0
        self.expired = False
        self._handoff = False
        self._config_hash = ""

    @classmethod
    def from_environment(cls, work_dir: Path) -> Self | None:
        """Read allocation settings exported by the generated batch script."""
        value = os.environ.get("AUTOCAST_SLURM_CHAIN")
        if value is None:
            return None
        segment, deadline, run_id = value.split(":")
        return cls(work_dir, int(segment), int(deadline), run_id)

    def configure_logging(self, logging_config: object) -> None:
        """Give enabled W&B logging one run identity across allocations."""
        if not isinstance(logging_config, dict):
            return
        wandb = logging_config.get("wandb", logging_config)
        if isinstance(wandb, dict) and wandb.get("enabled"):
            wandb["id"] = wandb.get("id") or self.run_id
            if wandb.get("mode") != "offline":
                wandb["resume"] = "allow" if self.segment == 1 else "must"

    def prepare(
        self,
        trainer: L.Trainer,
        config: object,
        resume_checkpoint: str | os.PathLike | None,
        *,
        weights_only: bool,
    ) -> None:
        """Configure the deadline callback and require the preceding full checkpoint."""
        if not isinstance(config, dict):
            msg = "Expected a mapping for the resolved training configuration."
            raise TypeError(msg)
        if str(config["optimizer"].get("scheduler_interval")).lower() == "time":
            msg = "Use an epoch- or step-based scheduler for a training chain."
            raise ValueError(msg)

        previous = self.directory / f"segment-{self.segment - 1:04d}.ckpt"
        expected = None if self.segment == 1 else previous
        actual = Path(resume_checkpoint) if resume_checkpoint is not None else None
        if actual != expected or weights_only:
            msg = f"Expected full-state resume from {expected!r} for this allocation."
            raise ValueError(msg)
        if expected and (previous.is_symlink() or not previous.is_file()):
            msg = (
                f"Missing concrete checkpoint from the preceding allocation: {previous}"
            )
            raise ValueError(msg)

        identity = config.copy()
        identity.pop("resume_from_checkpoint", None)
        self._config_hash = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()
        ).hexdigest()

        callbacks = cast(list[Callback], getattr(trainer, "callbacks", []))
        callbacks[:] = [cb for cb in callbacks if not isinstance(cb, Timer)]
        callbacks.append(self)
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint) and callback.dirpath is None:
                # The automatic directory can vary with the logger's version.
                callback.dirpath = str(self.directory.parent / "checkpoints")

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        self._last_epoch_end = time()
        if trainer.strategy.broadcast(
            self._last_epoch_end + self._longest_epoch >= self.deadline
        ):
            msg = "Allocation has too little time to start a training epoch."
            raise RuntimeError(msg)
        # Lightning's default of zero is also the index of the first epoch.
        for callback in getattr(trainer, "callbacks", []):
            if isinstance(callback, EarlyStopping):
                callback.stopped_epoch = -1

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        del pl_module
        now = time()
        self._longest_epoch = max(self._longest_epoch, now - self._last_epoch_end)
        self._last_epoch_end = now
        time_to_stop = trainer.strategy.broadcast(
            now + self._longest_epoch >= self.deadline
        )
        self.expired |= time_to_stop and not trainer.should_stop
        trainer.should_stop |= time_to_stop

    def on_save_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict
    ) -> None:
        del trainer, pl_module
        if self._handoff:
            checkpoint["autocast_slurm_handoff"] = [
                self.run_id,
                self.segment,
                self._config_hash,
                self._longest_epoch,
            ]

    def on_load_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict
    ) -> None:
        del trainer, pl_module
        handoff = checkpoint.get("autocast_slurm_handoff")
        if (
            not isinstance(handoff, list)
            or len(handoff) != 4
            or handoff[:3] != [self.run_id, self.segment - 1, self._config_hash]
        ):
            msg = "Checkpoint lineage or training configuration changed."
            raise ValueError(msg)
        self._longest_epoch = handoff[3]

    def save_endpoint(self, trainer: L.Trainer) -> bool:
        """Save the post-fit checkpoint and return whether training should resume."""
        max_epochs = trainer.max_epochs
        continue_training = (
            self.expired
            and (
                max_epochs is None
                or max_epochs == -1
                or trainer.current_epoch < max_epochs
            )
            and (trainer.max_steps == -1 or trainer.global_step < trainer.max_steps)
            and not any(
                isinstance(callback, EarlyStopping) and callback.stopped_epoch >= 0
                for callback in getattr(trainer, "callbacks", [])
            )
        )
        if (
            continue_training
            and not trainer.fit_loop.epoch_loop.batch_progress.is_last_batch
        ):
            msg = "The continuation requires a completed training epoch."
            raise RuntimeError(msg)

        if self.checkpoint.exists() or self.checkpoint.is_symlink():
            raise FileExistsError(self.checkpoint)
        self._handoff = continue_training
        trainer.save_checkpoint(self.checkpoint, weights_only=False)
        if trainer.is_global_zero:
            log.info(
                "Saved %s at epoch %d, step %d (%s).",
                self.checkpoint,
                trainer.current_epoch,
                trainer.global_step,
                "continue chain" if continue_training else "training finished",
            )
        return continue_training
