"""Regression checks for full-state training continuation."""

import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader, TensorDataset

from autocast.callbacks.checkpoint import ProgressModelCheckpoint
from autocast.callbacks.ema import EMACallback
from autocast.scripts.training import TrainingTimerCallback


def test_timer_resume_preserves_elapsed_time_and_epoch_history(monkeypatch):
    clock = [1000.0]
    trainer = cast(L.Trainer, None)
    model = cast(L.LightningModule, None)
    monkeypatch.setattr("autocast.scripts.training.perf_counter", lambda: clock[0])
    callback = TrainingTimerCallback()
    callback.load_state_dict(
        {
            "training_runtime_total_s": None,
            "training_runtime_elapsed_s": 60.0,
            "epoch_times_s": [20.0, 30.0],
        }
    )
    assert callback.state_dict()["training_runtime_elapsed_s"] == 60.0
    callback.on_train_start(trainer, model)
    callback.on_train_epoch_start(trainer, model)
    clock[0] += 5.0
    callback.on_train_end(trainer, model)
    saved = callback.state_dict()
    assert saved["training_runtime_total_s"] == 65.0
    assert saved["training_runtime_elapsed_s"] == 65.0
    assert saved["epoch_times_s"] == [20.0, 30.0, 5.0]

    # A second restart must neither lose nor double-count either segment.
    resumed = TrainingTimerCallback()
    resumed.load_state_dict(saved)
    clock[0] += 10000.0
    resumed.on_train_start(trainer, model)
    resumed.on_train_epoch_start(trainer, model)
    clock[0] += 7.0
    resumed.on_train_end(trainer, model)
    assert resumed.training_runtime_total_s == 72.0
    assert resumed.state_dict()["epoch_times_s"] == [20.0, 30.0, 5.0, 7.0]


def test_fractional_snapshot_matches_legacy_state_before_restore(tmp_path):
    trainer = cast(L.Trainer, SimpleNamespace(estimated_stepping_batches=101))
    original = ProgressModelCheckpoint(
        dirpath=tmp_path, every_n_train_steps_fraction=0.05, save_top_k=-1
    )
    original._maybe_resolve_fractional_train_steps(trainer)
    original.best_model_path = str(tmp_path / "existing.ckpt")
    checkpoint = {"callbacks": {original.state_key: original.state_dict()}}
    restored = ProgressModelCheckpoint(
        dirpath=tmp_path, every_n_train_steps_fraction=0.05, save_top_k=-1
    )
    restored.on_load_checkpoint(trainer, cast(L.LightningModule, None), checkpoint)
    assert restored.state_key in checkpoint["callbacks"]
    restored.load_state_dict(checkpoint["callbacks"][restored.state_key])
    assert restored._every_n_train_steps == 6
    assert restored.state_dict() == original.state_dict()


class _DeterministicModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return ((self.weight * batch[0] - 0.25) ** 2).mean()

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        # The earliest checkpoint must remain best across the interruption.
        self.log("val_loss", float(self.current_epoch + 1), batch_size=1)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW([self.weight], lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: (1 + math.cos(math.pi * epoch / 4)) / 2
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class _StopAfterTwoEpochs(Callback):
    def on_train_epoch_end(self, trainer, pl_module):  # noqa: ARG002
        if trainer.current_epoch == 1:
            trainer.should_stop = True


def _trainer(root: Path, *, interrupted: bool = False):
    timer = TrainingTimerCallback()
    snapshot = ProgressModelCheckpoint(
        dirpath=root,
        every_n_train_steps_fraction=0.25,
        filename="snapshot-{step}",
        save_top_k=-1,
        save_last=True,
    )
    best = ModelCheckpoint(
        dirpath=root,
        monitor="val_loss",
        filename="best-{epoch}",
        save_top_k=1,
    )
    ema = EMACallback(decay=0.9)
    callbacks: list[Callback] = [timer, snapshot, best, ema]
    if interrupted:
        callbacks.append(_StopAfterTwoEpochs())
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=4,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=root,
    )
    return trainer, timer, snapshot, best, ema


def test_interrupted_fit_matches_uninterrupted_state(tmp_path):
    loader = DataLoader(TensorDataset(torch.arange(1.0, 6.0)), batch_size=1)
    baseline, _, _, _, baseline_ema = _trainer(tmp_path / "baseline")
    baseline_model = _DeterministicModel()
    baseline.fit(baseline_model, loader, loader)

    first, first_timer, _, first_best, _ = _trainer(
        tmp_path / "resumed", interrupted=True
    )
    first.fit(_DeterministicModel(), loader, loader)
    saved_best = first_best.best_model_path
    saved_history = list(first_timer.state_dict()["epoch_times_s"])
    checkpoint = tmp_path / "resume.ckpt"
    first.save_checkpoint(checkpoint)

    resumed, timer, snapshot, best, ema = _trainer(tmp_path / "resumed")
    resumed_model = _DeterministicModel()
    resumed.fit(resumed_model, loader, loader, ckpt_path=checkpoint)
    assert resumed.global_step == baseline.global_step == 20
    assert torch.equal(resumed_model.weight, baseline_model.weight)
    assert resumed.optimizers[0].state_dict() == baseline.optimizers[0].state_dict()
    assert (
        resumed.lr_scheduler_configs[0].scheduler.state_dict()
        == baseline.lr_scheduler_configs[0].scheduler.state_dict()
    )
    assert best.best_model_path == saved_best
    assert best.best_model_score == 1.0
    assert Path(saved_best).exists()
    assert snapshot._every_n_train_steps == 5
    assert timer.state_dict()["epoch_times_s"][:2] == saved_history
    assert len(timer.state_dict()["epoch_times_s"]) == 4
    assert timer.training_runtime_total_s is not None
    assert first_timer.training_runtime_total_s is not None
    assert timer.training_runtime_total_s > first_timer.training_runtime_total_s
    assert ema.ema_model is not None
    assert baseline_ema.ema_model is not None
    for key, value in baseline_ema.ema_model.state_dict().items():
        assert torch.equal(ema.ema_model.state_dict()[key], value)
