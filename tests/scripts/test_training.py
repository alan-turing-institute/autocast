"""Tests that exercise real configs end-to-end."""

from datetime import timedelta
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import cast

import lightning as L
import pytest
import torch
from conftest import get_optimizer_config
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict
from torchmetrics import Metric, MetricCollection

from autocast.callbacks.checkpoint import ProgressModelCheckpoint
from autocast.callbacks.metrics import ValidationMetricPlotCallback
from autocast.encoders.base import EncoderWithCond
from autocast.scripts.setup import (
    _infer_latent_spatial_resolution,
    setup_autoencoder_model,
    setup_epd_model,
    setup_processor_model,
)
from autocast.scripts.training import (
    ResetResumeTimerCallback,
    TrainingTimerCallback,
    _attach_reset_timer_callback,
    _validate_resume_settings,
)
from autocast.types import Batch, EncodedBatch


@pytest.fixture
def config_dir(REPO_ROOT: Path) -> str:
    return str(REPO_ROOT / "src" / "autocast" / "configs")


def _load_config(
    config_dir: str, config_name: str, overrides: list[str] | None = None
) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides or [])


def _wrap_model_config(model_cfg: DictConfig) -> DictConfig:
    if "model" in model_cfg:
        return model_cfg
    return OmegaConf.create({"model": model_cfg})


def _stats_from_batch(batch: Batch) -> dict:
    return {
        "channel_count": batch.input_fields.shape[-1],
        "n_steps_input": batch.input_fields.shape[1],
        "n_steps_output": batch.output_fields.shape[1],
        "input_shape": batch.input_fields.shape,
        "output_shape": batch.output_fields.shape,
        "example_batch": batch,
    }


def _stats_from_encoded_batch(batch: EncodedBatch) -> dict:
    return {
        "channel_count": batch.encoded_inputs.shape[-1],
        "n_steps_input": batch.encoded_inputs.shape[1],
        "n_steps_output": batch.encoded_output_fields.shape[1],
        "input_shape": batch.encoded_inputs.shape,
        "output_shape": batch.encoded_output_fields.shape,
        "example_batch": batch,
    }


def test_autoencoder_config_trainer_fit_smoke(
    config_dir: str, toy_batch: Batch, dummy_loader, dummy_datamodule
):
    model_cfg = _load_config(config_dir, "model/autoencoder")
    cfg = _wrap_model_config(model_cfg)
    with open_dict(cfg):
        cfg.optimizer = get_optimizer_config()
        cfg.datamodule = {
            "n_steps_input": toy_batch.input_fields.shape[1],
            "n_steps_output": toy_batch.output_fields.shape[1],
        }
    stats = _stats_from_batch(toy_batch)
    model = setup_autoencoder_model(cfg, stats, dummy_datamodule)

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=dummy_loader, val_dataloaders=dummy_loader)


def test_processor_config_training_step_smoke(config_dir: str, dummy_datamodule):
    processor_cfg = _load_config(config_dir, "processor/flow_matching").processor
    with open_dict(processor_cfg):
        processor_cfg.backbone.include_global_cond = False
        processor_cfg.backbone.global_cond_channels = 0

    encoded_inputs = torch.randn(2, 2, 4, 4, 1)
    encoded_outputs = torch.randn(2, 2, 4, 4, 1)
    cfg = OmegaConf.create(
        {
            "model": {
                "processor": processor_cfg,
                "loss_func": {"_target_": "torch.nn.MSELoss"},
            },
            "optimizer": get_optimizer_config(learning_rate=1e-3),
            "datamodule": {
                "stride": 1,
                "n_steps_input": encoded_inputs.shape[1],
                "n_steps_output": encoded_outputs.shape[1],
            },
        }
    )
    batch = EncodedBatch(
        encoded_inputs=encoded_inputs,
        encoded_output_fields=encoded_outputs,
        global_cond=None,
        encoded_info={},
    )
    stats = _stats_from_encoded_batch(batch)
    model = setup_processor_model(cfg, stats, dummy_datamodule)

    loss = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0


def test_processor_metric_overrides_are_forwarded(config_dir: str, dummy_datamodule):
    processor_cfg = _load_config(config_dir, "processor/flow_matching").processor
    with open_dict(processor_cfg):
        processor_cfg.backbone.include_global_cond = False
        processor_cfg.backbone.global_cond_channels = 0

    encoded_inputs = torch.randn(2, 2, 4, 4, 1)
    encoded_outputs = torch.randn(2, 2, 4, 4, 1)
    cfg = OmegaConf.create(
        {
            "model": {
                "processor": processor_cfg,
                "loss_func": {"_target_": "torch.nn.MSELoss"},
                "val_metrics": [],
                "test_metrics": [],
            },
            "optimizer": get_optimizer_config(learning_rate=1e-3),
            "datamodule": {
                "stride": 1,
                "n_steps_input": encoded_inputs.shape[1],
                "n_steps_output": encoded_outputs.shape[1],
            },
        }
    )
    batch = EncodedBatch(
        encoded_inputs=encoded_inputs,
        encoded_output_fields=encoded_outputs,
        global_cond=None,
        encoded_info={},
    )
    stats = _stats_from_encoded_batch(batch)
    model = setup_processor_model(cfg, stats, dummy_datamodule)

    assert model.val_metrics is None
    assert model.test_metrics is None


# --- TrainingTimerCallback ---


def test_training_timer_callback_state_dict_empty_before_training():
    cb = TrainingTimerCallback()
    sd = cb.state_dict()
    assert sd["training_runtime_total_s"] is None
    assert sd["training_runtime_elapsed_s"] is None
    assert sd["epoch_times_s"] == []


def test_training_timer_callback_state_dict_and_round_trip():
    cb = TrainingTimerCallback()
    cb._train_start = 0.0
    cb._epoch_times_s = [1.0, 2.0, 3.0]
    cb.training_runtime_total_s = 10.0

    sd = cb.state_dict()
    assert sd["training_runtime_total_s"] == 10.0
    assert isinstance(sd["training_runtime_elapsed_s"], float)
    assert sd["training_runtime_elapsed_s"] >= 0.0
    assert sd["epoch_times_s"] == [1.0, 2.0, 3.0]
    assert sd["mean_epoch_s"] == pytest.approx(2.0)
    assert sd["min_epoch_s"] == pytest.approx(1.0)
    assert sd["max_epoch_s"] == pytest.approx(3.0)

    cb2 = TrainingTimerCallback()
    cb2.load_state_dict(sd)
    assert cb2.training_runtime_total_s == 10.0
    assert cb2._epoch_times_s == [1.0, 2.0, 3.0]


def test_training_timer_callback_state_dict_reports_elapsed_before_train_end():
    cb = TrainingTimerCallback()
    cb._train_start = perf_counter() - 0.05
    cb.training_runtime_total_s = None

    sd = cb.state_dict()
    assert sd["training_runtime_total_s"] is None
    assert isinstance(sd["training_runtime_elapsed_s"], float)
    assert sd["training_runtime_elapsed_s"] > 0.0


def test_validate_resume_settings_raises_for_weights_only_without_checkpoint():
    with pytest.raises(ValueError, match="resume_weights_only=true requires"):
        _validate_resume_settings(
            resume_checkpoint=None,
            resume_weights_only=True,
        )


def test_validate_resume_settings_allows_weights_only_with_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "model.ckpt"
    _validate_resume_settings(
        resume_checkpoint=ckpt,
        resume_weights_only=True,
    )


def test_validate_resume_settings_raises_for_reset_timer_without_checkpoint():
    with pytest.raises(ValueError, match="reset_resume_time_budget=true requires"):
        _validate_resume_settings(
            resume_checkpoint=None,
            resume_weights_only=False,
            reset_resume_time_budget=True,
        )


def test_validate_resume_settings_raises_for_reset_timer_with_weights_only(
    tmp_path: Path,
):
    ckpt = tmp_path / "model.ckpt"
    with pytest.raises(
        ValueError,
        match="reset_resume_time_budget=true is only meaningful for full-state resume",
    ):
        _validate_resume_settings(
            resume_checkpoint=ckpt,
            resume_weights_only=True,
            reset_resume_time_budget=True,
        )


def test_validate_resume_settings_allows_reset_timer_with_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "model.ckpt"
    _validate_resume_settings(
        resume_checkpoint=ckpt,
        resume_weights_only=False,
        reset_resume_time_budget=True,
    )


def test_attach_reset_timer_callback_inserts_before_timer():
    trainer = L.Trainer(
        max_time="00:00:00:10",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    _attach_reset_timer_callback(trainer, enabled=True)
    callbacks = list(getattr(trainer, "callbacks", []))
    reset_idx = next(
        idx
        for idx, cb in enumerate(callbacks)
        if isinstance(cb, ResetResumeTimerCallback)
    )
    timer_idx = next(idx for idx, cb in enumerate(callbacks) if isinstance(cb, Timer))
    assert reset_idx < timer_idx


def _trainer_stub(**overrides) -> SimpleNamespace:
    defaults = {
        "estimated_stepping_batches": 100,
        "max_steps": -1,
        "max_epochs": 10,
        "num_training_batches": 10,
        "accumulate_grad_batches": 1,
        "global_step": 0,
        "current_epoch": 0,
        "callback_metrics": {},
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_progress_model_checkpoint_resolves_fractional_train_steps():
    callback = ProgressModelCheckpoint(
        every_n_train_steps_fraction=0.05,
        save_top_k=-1,
        save_last=True,
        filename="snapshot-{step:08d}",
    )

    trainer = _trainer_stub(estimated_stepping_batches=101)
    callback.on_fit_start(cast(L.Trainer, trainer), cast(L.LightningModule, object()))

    assert callback._every_n_train_steps == 6


def test_progress_model_checkpoint_disables_default_epoch_trigger():
    callback = ProgressModelCheckpoint(
        every_n_train_steps_fraction=0.05,
        save_top_k=-1,
        save_last=True,
        filename="snapshot-{step:08d}",
    )

    assert callback._every_n_epochs == 0


@pytest.mark.parametrize(
    "trigger_kwargs",
    [
        {"every_n_train_steps": 10},
        {"every_n_epochs": 1},
        {"train_time_interval": timedelta(minutes=30)},
    ],
)
def test_progress_model_checkpoint_rejects_fraction_with_other_triggers(
    trigger_kwargs: dict,
):
    with pytest.raises(ValueError, match="every_n_train_steps_fraction"):
        ProgressModelCheckpoint(
            every_n_train_steps_fraction=0.05,
            save_top_k=-1,
            filename="snapshot-{step:08d}",
            **trigger_kwargs,
        )


@pytest.mark.parametrize(
    ("progress_fraction", "expected"),
    [
        (0.0, "0p00"),
        (0.05, "0p05"),
        (0.25, "0p25"),
        (1.0, "1p00"),
    ],
)
def test_progress_model_checkpoint_formats_progress_token(
    progress_fraction: float, expected: str
):
    assert ProgressModelCheckpoint._format_progress_token(progress_fraction) == expected


def test_progress_model_checkpoint_adds_progress_filename_fields():
    callback = ProgressModelCheckpoint(
        every_n_train_steps_fraction=0.05,
        save_top_k=-1,
        filename="snapshot-{progress_token}-{progress_pct:03d}-{step:08d}",
        auto_insert_metric_name=False,
    )
    trainer = _trainer_stub(estimated_stepping_batches=100, global_step=25)

    monitor_candidates = callback._monitor_candidates(cast(L.Trainer, trainer))
    filename = callback.format_checkpoint_name(monitor_candidates)

    assert monitor_candidates["progress_token"] == "0p25"
    assert monitor_candidates["progress_pct"].item() == 25
    assert filename == "snapshot-0p25-025-00000025.ckpt"


def test_progress_model_checkpoint_delays_monitored_topk(monkeypatch):
    callback = ProgressModelCheckpoint(
        monitor="val_multicoverage",
        mode="min",
        save_top_k=1,
        start_after_fraction=0.5,
        filename="best-{epoch:04d}",
    )

    calls: list[dict[str, torch.Tensor]] = []

    def fake_super_save(self, trainer, monitor_candidates):
        del self, trainer
        calls.append(monitor_candidates)

    monkeypatch.setattr(ModelCheckpoint, "_save_topk_checkpoint", fake_super_save)

    trainer = _trainer_stub(global_step=40)
    monitor_candidates = {"val_multicoverage": torch.tensor(0.2)}
    callback._save_topk_checkpoint(cast(L.Trainer, trainer), monitor_candidates)
    assert calls == []

    trainer.global_step = 50
    callback._save_topk_checkpoint(cast(L.Trainer, trainer), monitor_candidates)
    assert len(calls) == 1


def test_progress_model_checkpoint_skips_optional_missing_monitor(monkeypatch):
    callback = ProgressModelCheckpoint(
        monitor="val_multicoverage",
        save_top_k=1,
        monitor_optional=True,
        filename="best-{epoch:04d}",
    )

    called = False

    def fake_super_save(self, trainer, monitor_candidates):
        del self, trainer, monitor_candidates
        nonlocal called
        called = True

    monkeypatch.setattr(ModelCheckpoint, "_save_topk_checkpoint", fake_super_save)

    trainer = _trainer_stub(global_step=80)
    callback._save_topk_checkpoint(
        cast(L.Trainer, trainer), {"val_loss": torch.tensor(0.1)}
    )

    assert not called


def test_validation_metric_plot_callback_saves_val_metric_history(tmp_path: Path):
    callback = ValidationMetricPlotCallback(
        save_local=True,
        log_to_logger=False,
        plot_metric_objects=False,
    )
    trainer = SimpleNamespace(
        callback_metrics={
            "val_loss": torch.tensor(1.5),
            "val_vector": torch.ones(2),
            "train_loss": torch.tensor(2.0),
        },
        default_root_dir=tmp_path,
        global_step=12,
        is_global_zero=True,
        sanity_checking=False,
        loggers=[],
    )
    pl_module = SimpleNamespace(val_metrics=None)

    callback.on_validation_end(
        cast(L.Trainer, trainer), cast(L.LightningModule, pl_module)
    )

    assert callback._history == {"val_loss": [(12, 1.5)]}
    assert (tmp_path / "validation_metrics" / "validation_metrics.png").exists()


def test_validation_metric_plot_callback_saves_custom_metric_plot(tmp_path: Path):
    class PlotMetric(Metric):
        def update(self) -> None:
            pass

        def compute(self) -> torch.Tensor:
            return torch.tensor(0.0)

        def plot(self, save_path=None, title=None):
            fig, ax = plt.subplots()
            ax.set_title(title or "")
            ax.plot([0, 1], [0, 1])
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path)
            return fig

    callback = ValidationMetricPlotCallback(
        save_local=True,
        log_to_logger=False,
        plot_metric_objects=True,
    )
    trainer = SimpleNamespace(
        callback_metrics={},
        default_root_dir=tmp_path,
        global_step=12,
        is_global_zero=True,
        sanity_checking=False,
        loggers=[],
    )
    pl_module = SimpleNamespace(
        val_metrics=MetricCollection({"diagnostic": PlotMetric()}).clone(prefix="val_")
    )

    callback.on_validation_end(
        cast(L.Trainer, trainer), cast(L.LightningModule, pl_module)
    )

    assert (
        tmp_path / "validation_metrics" / "metric_plots" / "val_diagnostic.png"
    ).exists()


def test_validation_metric_plot_callback_wandb_log_omits_step(tmp_path: Path):
    callback = ValidationMetricPlotCallback(
        save_local=False,
        log_to_logger=True,
        plot_metric_objects=False,
    )

    class FakeWandbRun:
        project = "p"
        entity = "e"

        def __init__(self):
            self.calls: list[dict] = []

        def log(self, payload, step=None):
            self.calls.append({"payload": payload, "step": step})

    run = FakeWandbRun()
    logger = SimpleNamespace(experiment=run)
    trainer = SimpleNamespace(
        callback_metrics={"val_loss": torch.tensor(1.5)},
        default_root_dir=tmp_path,
        global_step=1189,
        is_global_zero=True,
        sanity_checking=False,
        loggers=[logger],
    )
    pl_module = SimpleNamespace(val_metrics=None)

    callback.on_validation_end(
        cast(L.Trainer, trainer), cast(L.LightningModule, pl_module)
    )

    assert run.calls, "expected at least one wandb.log call"
    for call in run.calls:
        assert call["step"] is None, (
            "wandb.log must not receive an explicit step= argument; "
            "passing trainer.global_step forces wandb's internal _step forward "
            "and breaks the default X-axis for all other metrics."
        )


def test_default_trainer_config_tracks_coverage_winkler_and_plots(config_dir: str):
    trainer_cfg = OmegaConf.load(Path(config_dir) / "trainer" / "default.yaml")
    callbacks = list(trainer_cfg.callbacks)
    monitors = [callback.get("monitor") for callback in callbacks]

    assert "val_multicoverage" in monitors
    assert "val_multiwinkler" in monitors
    assert any(
        callback.get("_target_")
        == "autocast.callbacks.metrics.ValidationMetricPlotCallback"
        for callback in callbacks
    )


def test_epd_config_forward_smoke(config_dir: str, toy_batch: Batch, dummy_datamodule):
    model_cfg = _load_config(
        config_dir,
        "model/encoder_processor_decoder",
        overrides=[
            "encoder@model.encoder=dc",
            "decoder@model.decoder=dc",
            "processor@model.processor=flow_matching",
        ],
    )
    cfg = _wrap_model_config(model_cfg)
    with open_dict(cfg):
        cfg.optimizer = get_optimizer_config()
        cfg.datamodule = {
            "stride": 1,
            "n_steps_input": toy_batch.input_fields.shape[1],
            "n_steps_output": toy_batch.output_fields.shape[1],
        }
        cfg.model.processor.backbone.include_global_cond = False
        cfg.model.processor.backbone.global_cond_channels = 0
    stats = _stats_from_batch(toy_batch)
    model = setup_epd_model(cfg, stats, dummy_datamodule)

    output = model(toy_batch)
    assert output.shape == toy_batch.output_fields.shape


def test_epd_metric_overrides_are_forwarded(
    config_dir: str, toy_batch: Batch, dummy_datamodule
):
    model_cfg = _load_config(
        config_dir,
        "model/encoder_processor_decoder",
        overrides=[
            "encoder@model.encoder=dc",
            "decoder@model.decoder=dc",
            "processor@model.processor=flow_matching",
        ],
    )
    cfg = _wrap_model_config(model_cfg)
    with open_dict(cfg):
        cfg.optimizer = get_optimizer_config()
        cfg.datamodule = {
            "stride": 1,
            "n_steps_input": toy_batch.input_fields.shape[1],
            "n_steps_output": toy_batch.output_fields.shape[1],
        }
        cfg.model.processor.backbone.include_global_cond = False
        cfg.model.processor.backbone.global_cond_channels = 0
        cfg.model.val_metrics = []
        cfg.model.test_metrics = []
    stats = _stats_from_batch(toy_batch)
    model = setup_epd_model(cfg, stats, dummy_datamodule)

    assert model.val_metrics is None
    assert model.test_metrics is None


def test_infer_latent_spatial_resolution_channels_last_with_time():
    class DummyEncoder:
        channel_axis = -1
        outputs_time_channel_concat = False

    encoded = torch.randn(2, 3, 16, 16, 8)  # B,T,W,H,C
    spatial = _infer_latent_spatial_resolution(
        encoded, cast(EncoderWithCond, DummyEncoder())
    )
    assert spatial == (16, 16)


def test_infer_latent_spatial_resolution_channels_first_time_concat():
    class DummyEncoder:
        channel_axis = 1
        outputs_time_channel_concat = True

    encoded = torch.randn(2, 24, 16, 16)  # B,C*T,W,H
    spatial = _infer_latent_spatial_resolution(
        encoded, cast(EncoderWithCond, DummyEncoder())
    )
    assert spatial == (16, 16)
