"""Prepare the AE diversity continuation without changing its update-wise LR."""

# Callback hook signatures include arguments this study does not need.
# ruff: noqa: ARG002

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from omegaconf import OmegaConf

SOURCE_SHA256 = "df27e12bb9dc46f44fa5d1b594d3f90b534ac6d0a0d2487eb1b6bad800f0824b"
SOURCE_STEP = 29116
STOP_STEP = 77308


def sha256(path: Path) -> str:
    """Hash checkpoint/config inputs and derived outputs."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_lr(
    step: int,
    *,
    original_steps_per_epoch: int = 1004,
    cosine_epochs: int = 512,
    base_lr: float = 1e-5,
) -> float:
    """Return LR for the next update after ``step`` completed optimizer updates."""
    if step < 0 or original_steps_per_epoch <= 0 or cosine_epochs <= 0:
        msg = "Step must be nonnegative and schedule lengths must be positive"
        raise ValueError(msg)
    if not math.isfinite(base_lr) or base_lr <= 0:
        msg = "Base LR must be finite and positive"
        raise ValueError(msg)
    epoch = step // original_steps_per_epoch
    return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / cosine_epochs))


class HistoricalAELearningRate(Callback):
    """Replay historical epoch-cosine LRs on the original global-update clock."""

    def __init__(
        self,
        original_steps_per_epoch: int = 1004,
        cosine_epochs: int = 512,
        base_lr: float = 1e-5,
        expected_resume_step: int = SOURCE_STEP,
        stop_step: int = STOP_STEP,
        expected_world_size: int = 4,
        expected_batch_size: int = 16,
    ) -> None:
        """Pin the schedule independently of the new training-loader length."""
        super().__init__()
        self.schedule = {
            "original_steps_per_epoch": original_steps_per_epoch,
            "cosine_epochs": cosine_epochs,
            "base_lr": base_lr,
        }
        historical_lr(expected_resume_step, **self.schedule)
        if stop_step <= expected_resume_step:
            msg = "Stop step must exceed the original resume step"
            raise ValueError(msg)
        self.expected_resume_step = expected_resume_step
        self.stop_step = stop_step
        self.expected_world_size = expected_world_size
        self.expected_batch_size = expected_batch_size
        self.last_completed_step: int | None = None

    def state_dict(self) -> dict[str, Any]:
        """Keep the resume guard valid for subsequent continuation checkpoints."""
        return {
            "last_completed_step": self.last_completed_step,
            "schedule": self.schedule,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Reject silently changed schedules when resuming this continuation."""
        if state_dict["schedule"] != self.schedule:
            msg = "Continuation checkpoint has a different historical LR schedule"
            raise ValueError(msg)
        self.last_completed_step = state_dict["last_completed_step"]

    def _set_lr(self, trainer: Trainer) -> None:
        rate = historical_lr(trainer.global_step, **self.schedule)
        for group in trainer.optimizers[0].param_groups:
            # PSGD's cached base_lr/prev_lr are its own internal state. Only the
            # public LR is controlled, as it was by the original LambdaLR.
            group["lr"] = rate

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Check full-state restoration after Lightning restores loop counters."""
        expected = self.last_completed_step
        if expected is None:
            expected = self.expected_resume_step
        if trainer.global_step != expected:
            msg = f"Expected restored global step {expected}, got {trainer.global_step}"
            raise ValueError(msg)
        if len(trainer.optimizers) != 1 or trainer.lr_scheduler_configs:
            msg = "Historical LR callback requires one optimizer and scheduler=null"
            raise ValueError(msg)
        if trainer.accumulate_grad_batches != 1:
            msg = "This continuation preserves the original one-update-per-batch task"
            raise ValueError(msg)
        if trainer.check_val_every_n_epoch not in (None, 1):
            msg = "Validation must be enabled in every epoch for the exact endpoint"
            raise ValueError(msg)
        if trainer.max_steps != self.stop_step:
            msg = "Trainer max_steps must match the continuation stop step"
            raise ValueError(msg)
        if trainer.world_size != self.expected_world_size:
            msg = "World size differs from the pinned continuation"
            raise ValueError(msg)
        if (
            getattr(trainer.train_dataloader, "batch_size", None)
            != self.expected_batch_size
        ):
            msg = "Per-device batch size differs from the pinned continuation"
            raise ValueError(msg)
        self._set_lr(trainer)

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Set the exact LR used by the upcoming optimizer update."""
        self._set_lr(trainer)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Store the next-update LR and request endpoint validation before exit."""
        self._set_lr(trainer)
        self.last_completed_step = trainer.global_step
        if trainer.global_step >= self.stop_step:
            # This forces validation at the endpoint even when the changed
            # loader means that max_steps falls partway through an epoch.
            trainer.should_stop = True


def _zero_current(progress: dict[str, Any]) -> None:
    for key in progress["current"]:
        progress["current"][key] = 0
    if "is_last_batch" in progress:
        progress["is_last_batch"] = False


def normalize_checkpoint(  # noqa: PLR0912, PLR0915
    checkpoint: dict[str, Any],
    *,
    original_steps_per_epoch: int = 1004,
    cosine_epochs: int = 512,
    base_lr: float = 1e-5,
    expected_step: int = SOURCE_STEP,
) -> dict[str, Any]:
    """Finish saved epoch bookkeeping; preserve all model/optimizer/EMA tensors."""
    if (
        checkpoint.get("global_step") != expected_step
        or expected_step % original_steps_per_epoch
    ):
        msg = "Expected the selected checkpoint at a completed historical epoch"
        raise ValueError(msg)
    optimizers, schedulers = (
        checkpoint.get("optimizer_states", []),
        checkpoint.get("lr_schedulers", []),
    )
    if len(optimizers) != 1 or len(schedulers) != 1 or not optimizers[0]["state"]:
        msg = "A full-state checkpoint with one optimizer and scheduler is required"
        raise ValueError(msg)
    epochs = expected_step // original_steps_per_epoch
    scheduler = schedulers[0]
    if scheduler["last_epoch"] != epochs or scheduler["_step_count"] != epochs + 1:
        msg = "Saved scheduler does not match the original epoch/update clock"
        raise ValueError(msg)
    rate = historical_lr(
        expected_step,
        original_steps_per_epoch=original_steps_per_epoch,
        cosine_epochs=cosine_epochs,
        base_lr=base_lr,
    )
    groups = optimizers[0]["param_groups"]
    if len(scheduler["base_lrs"]) != len(groups):
        msg = "Optimizer and scheduler groups differ"
        raise ValueError(msg)
    if any(
        not math.isclose(g["lr"], rate, rel_tol=1e-12)
        or not math.isclose(g["initial_lr"], base_lr, rel_tol=1e-12)
        for g in groups
    ):
        msg = "Source LR differs from the pinned historical schedule"
        raise ValueError(msg)
    if any(
        not math.isclose(value, base_lr, rel_tol=1e-12)
        for value in scheduler["base_lrs"]
    ):
        msg = "Source scheduler base learning rates differ"
        raise ValueError(msg)
    if len(scheduler["_last_lr"]) != len(groups) or any(
        not math.isclose(value, rate, rel_tol=1e-12) for value in scheduler["_last_lr"]
    ):
        msg = "Source scheduler current learning rates differ"
        raise ValueError(msg)
    callbacks = checkpoint.get("callbacks", {})
    ema_keys = [key for key in callbacks if "EMACallback" in key]
    if len(ema_keys) != 1 or not callbacks[ema_keys[0]].get("ema_model_state"):
        msg = "Expected the saved EMA callback state, including its update count"
        raise ValueError(msg)
    loop = checkpoint["loops"]["fit_loop"]
    epoch_progress = loop["epoch_progress"]
    if any(
        p["processed"] != epochs
        or p["ready"] != epochs
        or p["started"] != epochs
        or p["completed"] not in (epochs - 1, epochs)
        for p in epoch_progress.values()
    ):
        msg = "Source loop is not at the expected completed-epoch save boundary"
        raise ValueError(msg)
    optimizer_progress = loop["epoch_loop.automatic_optimization.optim_progress"][
        "optimizer"
    ]
    if optimizer_progress["step"]["total"]["completed"] != expected_step:
        msg = "Lightning optimizer progress disagrees with global_step"
        raise ValueError(msg)
    audit = {
        "original_epoch": checkpoint["epoch"],
        "original_fit_loop": copy.deepcopy(loop),
        "original_scheduler": copy.deepcopy(scheduler),
        "global_step": expected_step,
        "completed_epochs": epochs,
        "optimizer_lr_unchanged": rate,
    }
    # Best checkpoints are saved inside an epoch-end hook: processed=29 but
    # completed=28. Mark it completed, then begin the new loader at batch zero.
    for progress in epoch_progress.values():
        for key in progress:
            progress[key] = epochs
    for key in ("epoch_loop.batch_progress", "epoch_loop.scheduler_progress"):
        _zero_current(loop[key])
    for progress in optimizer_progress.values():
        _zero_current(progress)
    for key, progress in loop.items():
        if key.startswith("epoch_loop.val_loop.") and "progress" in key:
            _zero_current(progress)
            for counter in progress["total"]:
                progress["total"][counter] = 0
    loop["state_dict"].pop("combined_loader", None)
    loop["epoch_loop.state_dict"]["_batches_that_stepped"] = expected_step
    checkpoint["epoch"] = epochs
    checkpoint["lr_schedulers"] = []
    removed = [key for key in callbacks if key not in ema_keys]
    for key in removed:
        del callbacks[key]
    audit["removed_callback_histories"] = removed
    return audit


def prepare_checkpoint(
    source: Path,
    target: Path,
    source_config: Path,
    *,
    expected_sha256: str = SOURCE_SHA256,
) -> dict[str, Any]:
    """Derive a fresh resume file and provenance record from the pinned original."""
    # Check the supplied directory entries before resolve follows symlinks.
    # A dangling destination link is still an existing, user-owned entry.
    if os.path.lexists(target) or os.path.lexists(
        target.with_suffix(".provenance.json")
    ):
        msg = "Refusing to overwrite a checkpoint or its provenance manifest"
        raise FileExistsError(msg)
    source, target, source_config = (
        source.resolve(),
        target.resolve(),
        source_config.resolve(),
    )
    manifest = target.with_suffix(".provenance.json")
    if source == target or os.path.lexists(target) or os.path.lexists(manifest):
        msg = "Refusing to overwrite a checkpoint or its provenance manifest"
        raise FileExistsError(msg)
    source_hash = sha256(source)
    if source_hash != expected_sha256:
        msg = "Source checkpoint SHA256 does not match the selected input"
        raise ValueError(msg)
    optimizer = OmegaConf.to_container(
        OmegaConf.load(source_config).optimizer, resolve=True
    )
    if not isinstance(optimizer, dict) or any(
        optimizer.get(k, default) != expected
        for k, default, expected in (
            ("optimizer", None, "psgd"),
            ("scheduler", None, "cosine"),
            ("scheduler_interval", "epoch", "epoch"),
            ("cosine_epochs", None, 512),
            ("learning_rate", None, 1e-5),
            ("warmup", 0, 0),
            ("min_lr_ratio", 0.0, 0.0),
        )
    ):
        msg = "Source config must describe the original PSGD epoch-cosine AE"
        raise ValueError(msg)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    audit = normalize_checkpoint(checkpoint)
    with target.open("xb") as stream:
        torch.save(checkpoint, stream)
    if sha256(source) != source_hash:
        msg = "Source checkpoint changed during preparation"
        raise RuntimeError(msg)
    record = {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "source_config": str(source_config),
        "source_config_sha256": sha256(source_config),
        "derived_checkpoint": str(target),
        "derived_checkpoint_sha256": sha256(target),
        "audit": audit,
        "weights_optimizer_and_ema_preserved": True,
        "schedule": (
            "Historical epoch cosine indexed by floor(global_step/1004); no LR change"
        ),
        "loader": "New epoch starts at batch zero; cumulative optimizer steps retained",
    }
    with manifest.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def main() -> None:
    """Prepare a checkpoint only; this command never launches training."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    for name in ("source", "target", "source-config"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_checkpoint(args.source, args.target, args.source_config), indent=2
        )
    )


if __name__ == "__main__":
    main()
