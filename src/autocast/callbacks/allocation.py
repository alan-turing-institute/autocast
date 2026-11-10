"""Clean allocation boundaries and explicit full-state resume verification."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback, Timer

from autocast.callbacks.ema import EMACallback

log = logging.getLogger(__name__)


class AllocationDeadline(Callback):
    """Stop after a complete epoch with time reserved for the final checkpoint.

    The deadline belongs to the allocation and is deliberately not checkpointed.
    Lightning's separate Timer retains the experiment's cumulative time budget.
    """

    def __init__(self, reserve_seconds: int = 900):
        if reserve_seconds <= 0:
            msg = "reserve_seconds must be positive"
            raise ValueError(msg)
        self.reserve_seconds = reserve_seconds
        self.deadline: float | None = None

    def on_fit_start(self, trainer, pl_module) -> None:
        del trainer, pl_module
        value = os.environ.get("AUTOCAST_ALLOCATION_END_TIME") or os.environ.get(
            "SLURM_JOB_END_TIME"
        )
        if value is None:
            msg = "Set AUTOCAST_ALLOCATION_END_TIME or SLURM_JOB_END_TIME."
            raise ValueError(msg)
        self.deadline = float(value)
        if self.deadline - self.reserve_seconds <= time.time():
            msg = "Insufficient allocation time remains to start training."
            raise RuntimeError(msg)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        del pl_module
        assert self.deadline is not None
        stop = time.time() >= self.deadline - self.reserve_seconds
        stop = trainer.strategy.broadcast(stop)
        if stop:
            log.info("Allocation deadline approaching; stopping at epoch boundary.")
            trainer.should_stop = True


def _same_state(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


class VerifyResumeState(Callback):
    """Check restored model, optimizer, scheduler and timer before any update."""

    def __init__(self, checkpoint: str):
        self.checkpoint = Path(checkpoint)

    def on_train_start(self, trainer, pl_module) -> None:
        error = None
        try:
            source = torch.load(
                self.checkpoint, map_location="cpu", mmap=True, weights_only=False
            )
            assert trainer.global_step == source["global_step"]
            assert _same_state(pl_module.state_dict(), source["state_dict"])
            assert _same_state(
                [optimizer.state_dict() for optimizer in trainer.optimizers],
                source["optimizer_states"],
            )
            assert _same_state(
                [item.scheduler.state_dict() for item in trainer.lr_scheduler_configs],
                source["lr_schedulers"],
            )
            for callback in getattr(trainer, "callbacks", []):
                if isinstance(callback, EMACallback):
                    assert _same_state(
                        callback.state_dict(), source["callbacks"][callback.state_key]
                    )
                if isinstance(callback, Timer):
                    elapsed = source["callbacks"][callback.state_key]["time_elapsed"][
                        "train"
                    ]
                    assert elapsed <= callback.time_elapsed() < elapsed + 120
        except Exception as exc:  # Every DDP rank must observe a failed check.
            error = f"rank {trainer.global_rank}: {type(exc).__name__}: {exc}"
        errors = [error]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            errors = [None] * trainer.world_size
            torch.distributed.all_gather_object(errors, error)
        if any(errors):
            msg = f"Full-state resume verification failed: {errors}"
            raise RuntimeError(msg)
        log.info("Verified full-state resume from %s", self.checkpoint)
