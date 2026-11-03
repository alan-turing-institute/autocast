from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader, TensorDataset

from autocast.callbacks.ema import EMA_CHECKPOINT_KEY, EMACallback
from autocast.callbacks.sustained_validation import SustainedValidationRise


def _trainer() -> Any:
    strategy = SimpleNamespace(
        reduce_boolean_decision=Mock(side_effect=lambda decision, **_: decision)
    )
    return SimpleNamespace(
        global_step=0,
        state=SimpleNamespace(fn=TrainerFn.FITTING),
        sanity_checking=False,
        callback_metrics={},
        strategy=strategy,
        should_stop=False,
        is_global_zero=True,
    )


def _check(
    callback: SustainedValidationRise,
    trainer: Any,
    step: int,
    value: float,
) -> None:
    trainer.global_step = step
    trainer.callback_metrics[callback.monitor] = torch.tensor(
        value, dtype=torch.float64
    )
    callback.on_validation_end(trainer, cast(Any, None))


@pytest.mark.parametrize(
    "values",
    [[1.0] * 20, [1.0, 0.99, 0.98, 0.97], [1.0, 1.005, 1.003, 1.0] * 5],
)
def test_does_not_stop_improving_flat_or_in_band_noise(values: list[float]) -> None:
    callback = SustainedValidationRise(patience=2, min_steps=0)
    trainer = _trainer()
    for step, value in enumerate(values):
        _check(callback, trainer, step, value)
    assert not trainer.should_stop
    assert callback.wait_count == 0


def test_requires_consecutive_rises_and_resets_on_recovery() -> None:
    callback = SustainedValidationRise(patience=3, min_steps=0)
    trainer = _trainer()
    for step, value in enumerate([1.0, 1.02, 1.03, 1.005, 1.02, 1.025]):
        _check(callback, trainer, step, value)
        assert not trainer.should_stop
    assert callback.wait_count == 2
    _check(callback, trainer, 6, 1.03)
    assert trainer.should_stop
    assert callback.stopped_step == 6
    assert "3 consecutive" in cast(str, callback.stop_reason)
    trainer.strategy.reduce_boolean_decision.assert_called_with(True, all=False)


def test_warmup_tracks_best_without_accumulating_a_streak() -> None:
    callback = SustainedValidationRise(patience=2, min_steps=10)
    trainer = _trainer()
    for step, value in [(1, 1.0), (2, 0.9), (5, 1.1), (9, 1.1)]:
        _check(callback, trainer, step, value)
    assert callback.best_score == 0.9
    assert callback.wait_count == 0
    _check(callback, trainer, 10, 1.1)
    assert not trainer.should_stop
    _check(callback, trainer, 11, 1.1)
    assert trainer.should_stop


def test_margin_supports_negative_scores_and_near_zero() -> None:
    negative = SustainedValidationRise(patience=1, min_steps=0)
    trainer = _trainer()
    _check(negative, trainer, 0, -2.0)
    _check(negative, trainer, 1, -1.99)
    assert not trainer.should_stop
    _check(negative, trainer, 2, -1.97)
    assert trainer.should_stop

    zero = SustainedValidationRise(patience=1, min_steps=0, absolute_tolerance=1e-6)
    trainer = _trainer()
    _check(zero, trainer, 0, 0.0)
    _check(zero, trainer, 1, 1e-7)
    assert not trainer.should_stop
    _check(zero, trainer, 2, 2e-6)
    assert trainer.should_stop


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_stops_before_warmup_but_preserves_finite_best(value: float) -> None:
    callback = SustainedValidationRise(min_steps=10000)
    trainer = _trainer()
    _check(callback, trainer, 1, 1.0)
    _check(callback, trainer, 2, value)
    assert trainer.should_stop
    assert callback.best_score == 1.0
    assert "nonfinite" in cast(str, callback.stop_reason)


def test_ignores_sanity_standalone_and_duplicate_validation() -> None:
    callback = SustainedValidationRise(patience=2, min_steps=0)
    trainer = _trainer()
    trainer.sanity_checking = True
    _check(callback, trainer, 0, float("nan"))
    assert callback.best_score is None
    trainer.sanity_checking = False
    trainer.state.fn = TrainerFn.VALIDATING
    _check(callback, trainer, 0, float("nan"))
    assert callback.last_checked_step == -1
    trainer.state.fn = TrainerFn.FITTING
    _check(callback, trainer, 0, 1.0)
    _check(callback, trainer, 1, 1.1)
    _check(callback, trainer, 1, 1.1)
    assert callback.wait_count == 1
    assert not trainer.should_stop
    _check(callback, trainer, 2, 1.1)
    assert trainer.should_stop


def test_resume_retains_streak_and_does_not_repeat_previous_check() -> None:
    callback = SustainedValidationRise(patience=2, min_steps=0)
    trainer = _trainer()
    _check(callback, trainer, 1, 1.0)
    _check(callback, trainer, 2, 1.1)
    restored = SustainedValidationRise(patience=2, min_steps=0)
    restored.load_state_dict(callback.state_dict())
    assert restored.state_key == callback.state_key
    assert restored.state_dict() == callback.state_dict()
    _check(restored, trainer, 2, 1.1)
    assert not trainer.should_stop
    _check(restored, trainer, 3, 1.1)
    assert trainer.should_stop


def test_other_rank_stop_is_honoured_and_existing_stop_is_not_cleared() -> None:
    callback = SustainedValidationRise(min_steps=0)
    trainer = _trainer()
    trainer.strategy.reduce_boolean_decision = Mock(return_value=True)
    _check(callback, trainer, 1, 1.0)
    assert trainer.should_stop
    assert "Another rank" in cast(str, callback.stop_reason)
    trainer.strategy.reduce_boolean_decision = Mock(return_value=False)
    _check(callback, trainer, 2, 0.9)
    assert trainer.should_stop


def test_missing_or_nonscalar_metric_fails_clearly() -> None:
    callback = SustainedValidationRise()
    trainer = _trainer()
    with pytest.raises(RuntimeError, match="requires validation metric"):
        callback.on_validation_end(trainer, cast(Any, None))
    for value in [torch.ones(2), torch.tensor(1 + 2j)]:
        trainer.callback_metrics["val_loss"] = value
        with pytest.raises(ValueError, match="real scalar"):
            callback.on_validation_end(trainer, cast(Any, None))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"monitor": ""},
        {"patience": 0},
        {"patience": 1.5},
        {"patience": True},
        {"min_steps": -1},
        {"min_steps": True},
        {"relative_tolerance": -0.1},
        {"relative_tolerance": float("nan")},
        {"relative_tolerance": True},
        {"absolute_tolerance": -0.1},
        {"absolute_tolerance": float("inf")},
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be"):
        SustainedValidationRise(**kwargs)


class _ValidationSequenceModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        return self.weight.square()

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # noqa: ARG002
        score = [1.0, 1.1, 1.2][self.current_epoch]
        self.log("val_loss", score, batch_size=1)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def test_lightning_stop_saves_best_latest_and_full_resume_state(tmp_path: Path) -> None:
    stopper = SustainedValidationRise(patience=2, min_steps=0)
    latest = ModelCheckpoint(
        dirpath=tmp_path,
        filename="snapshot-{epoch}",
        every_n_epochs=64,
        save_on_train_epoch_end=False,
        save_top_k=-1,
        save_last=True,
    )
    best = ModelCheckpoint(
        dirpath=tmp_path,
        filename="best-{epoch}",
        monitor="val_loss",
        save_on_train_epoch_end=False,
        save_top_k=1,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=20,
        callbacks=[latest, best, stopper, EMACallback(decay=0.9)],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
    )
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    model = _ValidationSequenceModel()
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 6
    last_checkpoint = torch.load(latest.last_model_path, weights_only=False)
    best_checkpoint = torch.load(best.best_model_path, weights_only=False)
    assert best_checkpoint["global_step"] == 2
    assert last_checkpoint["global_step"] == 6
    assert last_checkpoint["callbacks"][stopper.state_key] == stopper.state_dict()
    assert last_checkpoint["callbacks"][stopper.state_key]["wait_count"] == 2
    assert last_checkpoint["optimizer_states"][0]["state"][0]["step"].item() == 6
    assert last_checkpoint["lr_schedulers"][0]["last_epoch"] == 3
    assert EMA_CHECKPOINT_KEY in last_checkpoint
    assert not torch.equal(
        last_checkpoint["state_dict"]["weight"],
        best_checkpoint["state_dict"]["weight"],
    )
    torch.testing.assert_close(last_checkpoint["state_dict"]["weight"], model.weight)
