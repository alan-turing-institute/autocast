from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, Timer
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
        {"window_size": 0},
        {"window_size": True},
        {"window_size": 1.5},
        {"check_every": 0},
        {"check_every": True},
        {"noise_tolerance": -1},
        {"noise_tolerance": float("nan")},
        {"noise_tolerance": True},
        {"reset_on_resume": "true"},
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be"):
        SustainedValidationRise(**kwargs)


class _ValidationSequenceModel(LightningModule):
    def __init__(self, scores: list[float] | None = None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.validation_scores = scores if scores is not None else [1.0, 1.1, 1.2]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        return self.weight.square()

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # noqa: ARG002
        score = self.validation_scores[self.current_epoch]
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


def test_averages_full_windows_and_counts_only_scheduled_checks() -> None:
    callback = SustainedValidationRise(
        window_size=3, check_every=3, patience=2, min_steps=0
    )
    trainer = _trainer()
    for step, value in enumerate([1.0] * 3 + [1.2] * 3 + [1.3] * 3):
        _check(callback, trainer, step, value)
        assert trainer.should_stop == (step == 8)
    assert callback.best_score == 1.0
    assert callback.wait_count == 2
    assert callback.validation_count == 9
    assert "mean of 3" in cast(str, callback.stop_reason)


def test_noise_allowance_ignores_a_lucky_low_draw_but_stops_real_rise() -> None:
    callback = SustainedValidationRise(
        window_size=4,
        check_every=4,
        noise_tolerance=1.0,
        relative_tolerance=0.05,
        patience=1,
        min_steps=0,
    )
    trainer = _trainer()
    for step, value in enumerate([1.0, 1.0, 1.0, 0.1] + [1.0] * 4):
        _check(callback, trainer, step, value)
        assert not trainer.should_stop
    assert callback.best_score == pytest.approx(0.775)
    for step in range(8, 12):
        _check(callback, trainer, step, 2.0)
    assert trainer.should_stop


def test_partial_window_and_check_cadence_survive_resume() -> None:
    settings: dict[str, Any] = {
        "window_size": 4,
        "check_every": 2,
        "patience": 2,
        "min_steps": 0,
    }
    original = SustainedValidationRise(**settings)
    trainer = _trainer()
    for step in range(3):
        _check(original, trainer, step, 1.0)
    resumed = SustainedValidationRise(**settings)
    resumed.load_state_dict(original.state_dict())
    assert list(resumed.window_values) == [1.0] * 3
    other_trainer = _trainer()
    for step, value in enumerate([1.0] + [1.2] * 8, start=3):
        _check(original, trainer, step, value)
        _check(resumed, other_trainer, step, value)
        assert original.state_dict() == resumed.state_dict()
        assert trainer.should_stop == other_trainer.should_stop
    assert trainer.should_stop


def test_nonfinite_stops_before_an_averaging_window_is_full() -> None:
    callback = SustainedValidationRise(window_size=100, check_every=25)
    trainer = _trainer()
    _check(callback, trainer, 1, 1.0)
    _check(callback, trainer, 2, float("nan"))
    assert trainer.should_stop
    assert callback.best_score is None


def test_opt_in_resume_reset_only_clears_stopping_history() -> None:
    callback = SustainedValidationRise(
        window_size=2, check_every=2, patience=1, min_steps=0
    )
    trainer = _trainer()
    for step, value in enumerate([1.0, 1.0, 1.2, 1.2]):
        _check(callback, trainer, step, value)
    assert trainer.should_stop
    resumed = SustainedValidationRise(
        window_size=2,
        check_every=2,
        patience=1,
        min_steps=0,
        reset_on_resume=True,
    )
    assert resumed.state_key == callback.state_key
    resumed.load_state_dict(callback.state_dict())
    assert resumed.best_score is None
    assert resumed.best_spread == 0
    assert not resumed.window_values
    assert resumed.wait_count == resumed.validation_count == 0
    assert resumed.last_checked_step == -1
    assert resumed.stopped_step is resumed.stop_reason is None
    assert callback.wait_count == 1


def test_existing_swe_state_key_and_checkpoint_remain_compatible() -> None:
    callback = SustainedValidationRise(patience=2, min_steps=0)
    expected_key = callback._generate_state_key(
        monitor="val_loss",
        relative_tolerance=0.01,
        absolute_tolerance=1e-8,
        patience=2,
        min_steps=0,
    )
    assert callback.state_key == expected_key
    callback.load_state_dict(
        {
            "best_score": 1.0,
            "wait_count": 1,
            "last_checked_step": 1,
            "stopped_step": None,
            "stop_reason": None,
        }
    )
    trainer = _trainer()
    _check(callback, trainer, 2, 1.2)
    assert trainer.should_stop
    assert callback.state_key != SustainedValidationRise(window_size=100).state_key


def test_windowed_lightning_stop_preserves_best_and_terminal_state(
    tmp_path: Path,
) -> None:
    stopper = SustainedValidationRise(
        window_size=2, check_every=2, patience=2, min_steps=0
    )
    best = ModelCheckpoint(
        dirpath=tmp_path,
        filename="best-{epoch}",
        monitor="val_loss",
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=20,
        callbacks=[best, stopper, EMACallback(decay=0.9)],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
    )
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    model = _ValidationSequenceModel([1.0, 1.0, 1.1, 1.1, 1.2, 1.2])
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    terminal = tmp_path / "processor.ckpt"
    trainer.save_checkpoint(terminal)
    state = torch.load(terminal, weights_only=False)
    selected = torch.load(best.best_model_path, weights_only=False)
    assert trainer.global_step == state["global_step"] == 12
    assert selected["global_step"] == 2
    assert state["callbacks"][stopper.state_key] == stopper.state_dict()
    assert state["callbacks"][stopper.state_key]["validation_count"] == 6
    assert state["optimizer_states"][0]["state"][0]["step"].item() == 12
    assert state["lr_schedulers"][0]["last_epoch"] == 6
    assert EMA_CHECKPOINT_KEY in state
    torch.testing.assert_close(state["state_dict"]["weight"], model.weight)


def test_full_resume_resets_only_stopper_and_continues_cosine(tmp_path: Path) -> None:
    class CosineSequenceModel(_ValidationSequenceModel):
        def configure_optimizers(self) -> Any:
            optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    settings: dict[str, Any] = {
        "window_size": 2,
        "check_every": 2,
        "patience": 2,
        "min_steps": 0,
    }
    scores = [1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3]
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)

    def make_best() -> ModelCheckpoint:
        return ModelCheckpoint(
            dirpath=tmp_path,
            filename="best-{epoch}",
            monitor="val_loss",
            save_on_train_epoch_end=True,
            save_top_k=1,
        )

    callbacks: list[Callback] = [
        make_best(),
        SustainedValidationRise(**settings),
        EMACallback(decay=0.9),
        Timer(duration={"hours": 1}),
    ]
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=8,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        CosineSequenceModel(scores),
        train_dataloaders=loader,
        val_dataloaders=loader,
    )
    assert trainer.global_step == 12
    terminal = tmp_path / "processor.ckpt"
    trainer.save_checkpoint(terminal)
    saved = torch.load(terminal, weights_only=False)

    stopper = SustainedValidationRise(**settings, reset_on_resume=True)
    ema = EMACallback(decay=0.9)
    timer = Timer(duration={"hours": 1})
    best = make_best()
    observed = {}

    class ResumeProbe(Callback):
        def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
            assert trainer.global_step == 12
            assert trainer.current_epoch == 6
            assert stopper.best_score is None
            assert stopper.wait_count == 0
            assert stopper.validation_count == 0
            assert not stopper.window_values
            optimizer = trainer.optimizers[0]
            expected = saved["optimizer_states"][0]
            assert optimizer.param_groups[0]["lr"] == expected["param_groups"][0]["lr"]
            assert optimizer.param_groups[0]["lr"] < 0.01
            for key, value in expected["state"][0].items():
                torch.testing.assert_close(
                    optimizer.state[next(pl_module.parameters())][key], value
                )
            scheduler = trainer.lr_scheduler_configs[0].scheduler
            assert scheduler.state_dict() == saved["lr_schedulers"][0]
            torch.testing.assert_close(pl_module.weight, saved["state_dict"]["weight"])
            restored_ema = ema.state_dict()["ema_model_state"]
            for key, value in saved["callbacks"][ema.state_key][
                "ema_model_state"
            ].items():
                torch.testing.assert_close(restored_ema[key], value)
            assert (
                timer._offset
                == saved["callbacks"][timer.state_key]["time_elapsed"]["train"]
            )
            assert timer._offset > 0
            assert (
                best.best_model_path
                == saved["callbacks"][best.state_key]["best_model_path"]
            )
            observed["restored"] = True

    resumed = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=8,
        callbacks=[best, stopper, ema, timer, ResumeProbe()],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    resumed.fit(
        CosineSequenceModel(scores),
        train_dataloaders=loader,
        val_dataloaders=loader,
        ckpt_path=terminal,
    )
    assert observed["restored"]
    assert resumed.global_step == 16
    assert not resumed.should_stop
    assert resumed.lr_scheduler_configs[0].scheduler.last_epoch == 8
    assert stopper.validation_count == 2
    assert stopper.best_score == pytest.approx(1.3)
    assert ema.ema_model is not None
    assert ema.ema_model.n_averaged.item() == 16
    assert best.best_model_score is not None
    assert best.best_model_score.item() == 1.0
