"""Tests that exercise real configs end-to-end."""

from datetime import timedelta
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import lightning as L
import pytest
import torch
from conftest import get_optimizer_config
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Metric, MetricCollection

from autocast.callbacks.checkpoint import ProgressModelCheckpoint
from autocast.callbacks.ema import EMACallback
from autocast.callbacks.metrics import ValidationMetricPlotCallback
from autocast.callbacks.residual_statistics import ResidualStatisticsCallback
from autocast.data.datamodule import SpatioTemporalDataModule
from autocast.encoders.base import EncoderWithCond
from autocast.encoders.identity import IdentityEncoder
from autocast.nn.noise.source import TemporalOUSource
from autocast.nn.vit import SpatialViTBackbone
from autocast.processors.azula_vit import AzulaViTProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.scripts import training
from autocast.scripts.setup import (
    _infer_latent_spatial_resolution,
    setup_autoencoder_model,
    setup_epd_model,
    setup_processor_model,
)
from autocast.scripts.training import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    ResetResumeTimerCallback,
    TrainingTimerCallback,
    _attach_reset_timer_callback,
    _early_stopping_callbacks,
    _ensure_residual_statistics_callback,
    _validate_resume_settings,
    run_training,
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


class _EncodedDataModule(L.LightningDataModule):
    def __init__(self, batch: EncodedBatch) -> None:
        super().__init__()
        self.train_dataset = [batch]

    def train_dataloader(self):
        return DataLoader(
            cast(Dataset, self.train_dataset),
            batch_size=None,
            num_workers=0,
        )

    def val_dataloader(self):
        return self.train_dataloader()


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


def test_autoencoder_trainer_fit_bf16_mixed_smoke(
    config_dir: str, toy_batch: Batch, dummy_loader, dummy_datamodule
):
    """Verify bf16-mixed + gradient_clip_val=1.0 don't produce NaN/Inf.

    bf16-mixed is a supported but non-default option (not recommended — earlier
    experiments showed fp32-vs-bf16 degradation). This smoke fit with both
    enabled catches regressions where mixed precision introduces NaN/Inf in our
    model paths or where clipping interferes with the optimizer step.
    """
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
        precision="bf16-mixed",
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dataloaders=dummy_loader, val_dataloaders=dummy_loader)

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"param {name} has NaN/Inf after bf16-mixed fit"


def test_autoencoder_trainer_fit_time_cosine_smoke(
    config_dir: str, toy_batch: Batch, dummy_loader, dummy_datamodule
):
    """scheduler_interval='time' wires up and runs under a real max_time Trainer.

    Confirms the wall-clock cosine path finds Lightning's Timer in
    trainer.callbacks and that the time->step interval mapping in
    configure_optimizers is accepted by Lightning's scheduler config.
    """
    model_cfg = _load_config(config_dir, "model/autoencoder")
    cfg = _wrap_model_config(model_cfg)
    with open_dict(cfg):
        cfg.optimizer = get_optimizer_config(
            scheduler="cosine", scheduler_interval="time"
        )
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
        limit_train_batches=2,
        limit_val_batches=1,
        max_time="00:01:00:00",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=dummy_loader, val_dataloaders=dummy_loader)

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"param {name} not finite after time-cosine fit"


def test_default_trainer_config_omits_clip_and_precision(config_dir: str):
    """Pin trainer/default.yaml to the reproducibility-preserving defaults.

    gradient_clip_val stays null (no clipping) and precision is left unset
    (Lightning's 32-true) so existing runs reproduce unchanged; the tuned values
    live in the opt-in ``updated_defaults`` configs. Fails loudly if a refactor
    silently re-introduces a behaviour-changing default.
    """
    trainer_cfg = OmegaConf.load(Path(config_dir) / "trainer" / "default.yaml")
    assert isinstance(trainer_cfg, DictConfig)
    assert trainer_cfg.get("gradient_clip_val") is None
    assert trainer_cfg.get("precision") is None


def test_updated_defaults_configs_opt_in(config_dir: str):
    """The opt-in recommended configs select cleanly and carry the tuned values.

    Per review: live defaults stay untouched and the tuned values ship as
    selectable configs (``optimizer=updated_defaults/adamw``,
    ``trainer=updated_defaults``) rather than changing the defaults in place.
    """
    cfg = _load_config(
        config_dir,
        "autoencoder",
        overrides=[
            "optimizer=updated_defaults/adamw",
            "trainer=updated_defaults",
        ],
    )
    assert cfg.optimizer.weight_decay == 0.01
    assert cfg.optimizer.warmup == 0.05
    assert cfg.trainer.gradient_clip_val == 1.0
    # bf16 deliberately not recommended (see review); precision stays unset.
    assert cfg.trainer.get("precision") is None
    # Inheritance from trainer/default.yaml preserved the callbacks.
    assert len(cfg.trainer.callbacks) > 0


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


def test_residual_flow_config_fits_and_restores_statistics(  # noqa: PLR0915
    config_dir: str,
    tmp_path: Path,
    monkeypatch,
):
    model_cfg = _load_config(
        config_dir,
        "model/processor",
        overrides=[
            "processor@model.processor=residual_flow_matching_vit",
            "flow_source@model.processor.source=temporal_ou",
        ],
    )
    with open_dict(model_cfg):
        processor_cfg = model_cfg.model.processor
        processor_cfg.flow_ode_steps = 1
        processor_cfg.backbone.include_global_cond = False
        processor_cfg.backbone.global_cond_channels = 0
        processor_cfg.backbone.mod_features = 8
        processor_cfg.backbone.hid_channels = 16
        processor_cfg.backbone.hid_blocks = 1
        processor_cfg.backbone.attention_heads = 2
        model_cfg.optimizer = get_optimizer_config(learning_rate=1e-3)
        model_cfg.datamodule = {
            "stride": 1,
            "n_steps_input": 1,
            "n_steps_output": 3,
        }

    inputs = torch.randn(2, 1, 4, 4, 1)
    residual = torch.randn(2, 3, 4, 4, 1)
    targets = inputs[:, -1:].expand(-1, 3, -1, -1, -1) + residual
    batch = EncodedBatch(
        encoded_inputs=inputs,
        encoded_output_fields=targets,
        global_cond=None,
        encoded_info={},
    )
    datamodule = _EncodedDataModule(batch)
    setup_datamodule = cast(SpatioTemporalDataModule, datamodule)
    model = setup_processor_model(
        model_cfg,
        _stats_from_encoded_batch(batch),
        setup_datamodule,
    )
    ema = EMACallback()
    callbacks: list = [ema, Callback()]
    callbacks = _ensure_residual_statistics_callback(callbacks, model)

    assert isinstance(model.processor, ResidualFlowMatchingProcessor)
    assert isinstance(model.processor.source, TemporalOUSource)
    assert isinstance(callbacks[0], ResidualStatisticsCallback)
    assert len(callbacks) == 3

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=datamodule)

    residual_normalizer = model.processor.residual_normalizer
    assert residual_normalizer is not None
    expected_scale, expected_mean = torch.std_mean(
        residual.movedim(1, 0).reshape(3, -1, 1),
        dim=1,
        correction=0,
    )
    assert residual_normalizer.fitted
    assert torch.allclose(residual_normalizer.mean, expected_mean)
    assert torch.allclose(residual_normalizer.scale, expected_scale)
    assert ema.ema_model is not None
    ema_state = ema.ema_model.module.state_dict()
    assert ema_state["processor.residual_normalizer._fitted"].item()
    assert torch.equal(
        ema_state["processor.residual_normalizer.mean"], residual_normalizer.mean
    )
    assert torch.equal(
        ema_state["processor.residual_normalizer.scale"], residual_normalizer.scale
    )

    checkpoint = tmp_path / "residual.ckpt"
    trainer.save_checkpoint(checkpoint)
    # Recreate the pre-rename artifact format, including EMA callback resume state.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    def legacy_names(state):
        return {
            key.replace("residual_normalizer.", "standardizer."): value
            for key, value in state.items()
        }

    for key in ("state_dict", "ema_state_dict"):
        payload[key] = legacy_names(payload[key])
    for state in payload["callbacks"].values():
        if state.get("ema_model_state") is not None:
            state["ema_model_state"] = legacy_names(state["ema_model_state"])
    torch.save(payload, checkpoint)
    with open_dict(model_cfg.model.processor):
        legacy_config = model_cfg.model.processor.pop("residual_normalizer")
        legacy_config._target_ = (
            "autocast.processors.residual_normalization.ResidualStandardizer"
        )
        model_cfg.model.processor.standardizer = legacy_config

    # Real Lightning resume restores statistics before on_fit_start callbacks.
    resumed = setup_processor_model(
        model_cfg, _stats_from_encoded_batch(batch), setup_datamodule
    )

    def unexpected_statistics_fit(*args, **kwargs):
        pytest.fail("Checkpoint resume must skip residual statistics fitting")

    monkeypatch.setattr(
        ResidualStatisticsCallback, "_fit_statistics", unexpected_statistics_fit
    )
    resumed_ema = EMACallback()
    resumed_trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=_ensure_residual_statistics_callback([resumed_ema], resumed),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    resumed_trainer.fit(resumed, datamodule=datamodule, ckpt_path=checkpoint)
    assert resumed_trainer.global_step == 2
    assert resumed_ema.ema_model is not None
    resumed_state = resumed_ema.ema_model.module.state_dict()
    assert torch.equal(
        resumed_state["processor.residual_normalizer.mean"], residual_normalizer.mean
    )
    assert torch.equal(
        resumed_state["processor.residual_normalizer.scale"], residual_normalizer.scale
    )


def test_masked_window_flow_matching_config_smoke(config_dir: str, dummy_datamodule):
    processor_cfg = _load_config(
        config_dir, "processor/flow_matching_masked_window_vit"
    ).processor
    with open_dict(processor_cfg):
        processor_cfg.flow_ode_steps = 1
        processor_cfg.backbone.include_global_cond = False
        processor_cfg.backbone.global_cond_channels = 0
        processor_cfg.backbone.hid_channels = 32
        processor_cfg.backbone.hid_blocks = 1
        processor_cfg.backbone.mod_features = 16

    encoded_inputs = torch.randn(2, 1, 4, 4, 1)
    encoded_outputs = torch.randn(2, 4, 4, 4, 1)
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

    processor = model.processor
    assert processor.flow_matching_model.n_steps_input == 5  # type: ignore  # noqa: PGH003
    assert processor.flow_matching_model.n_steps_output == 5  # type: ignore  # noqa: PGH003
    assert processor.flow_matching_model.cond_channels == 1  # type: ignore  # noqa: PGH003

    output = model.map(encoded_inputs, None)
    assert output.shape == encoded_outputs.shape

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


def _trainer_stub(**overrides) -> Mock:
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
    return Mock(spec=L.Trainer, **defaults)


def test_progress_model_checkpoint_resolves_fractional_train_steps():
    callback = ProgressModelCheckpoint(
        every_n_train_steps_fraction=0.05,
        save_top_k=-1,
        save_last=True,
        filename="snapshot-{step:08d}",
    )

    trainer = _trainer_stub(estimated_stepping_batches=101)
    callback.on_fit_start(cast(L.Trainer, trainer), Mock(spec=L.LightningModule))

    assert callback._every_n_train_steps == 6


def test_progress_model_checkpoint_disables_default_epoch_trigger():
    callback = ProgressModelCheckpoint(
        every_n_train_steps_fraction=0.05,
        save_top_k=-1,
        save_last=True,
        filename="snapshot-{step:08d}",
    )

    assert callback._every_n_epochs == 0


def test_progress_model_checkpoint_state_key_includes_progress_window():
    callbacks = [
        ProgressModelCheckpoint(
            monitor="val_multicoverage",
            monitor_optional=True,
            stop_after_fraction=0.25,
            mode="min",
            save_top_k=1,
            filename="best-pre-{epoch:04d}",
        ),
        ProgressModelCheckpoint(
            monitor="val_multicoverage",
            monitor_optional=True,
            start_after_fraction=0.25,
            mode="min",
            save_top_k=1,
            filename="best-from0p25-{epoch:04d}",
        ),
        ProgressModelCheckpoint(
            monitor="val_multicoverage",
            monitor_optional=True,
            start_after_fraction=0.5,
            mode="min",
            save_top_k=1,
            filename="best-from0p50-{epoch:04d}",
        ),
    ]

    state_keys = [callback.state_key for callback in callbacks]
    assert len(state_keys) == len(set(state_keys))


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


def _timer_stub(elapsed_s: float, remaining_s: float | None) -> Timer:
    """A real ``Timer`` with controlled elapsed/remaining for progress tests."""
    duration = (
        None if remaining_s is None else timedelta(seconds=elapsed_s + remaining_s)
    )
    timer = Timer(duration=duration)
    timer.time_elapsed = lambda *_: float(elapsed_s)  # type: ignore[method-assign]
    timer.time_remaining = lambda *_: remaining_s  # type: ignore[method-assign]
    return timer


def _windowed_callback() -> ProgressModelCheckpoint:
    return ProgressModelCheckpoint(
        monitor="val_multicoverage",
        monitor_optional=True,
        start_after_fraction=0.25,
        mode="min",
        save_top_k=1,
        filename="best-from0p25-{epoch:04d}",
    )


def test_progress_fraction_uses_time_budget_over_placeholder_max_epochs():
    """Time-limited run: the step fraction stays ~0 because ``max_epochs`` is a
    large placeholder, but the wall-clock budget is half spent, so windowed
    checkpoints must see 0.5 and open the ``start_after_fraction=0.25`` window.

    This is the regression guard for the checkpoint-progress bug: before the
    fix the ``from0p25``/``0p50``/``0p75`` checkpoints never fired on
    ``max_time``-bounded runs.
    """
    callback = _windowed_callback()
    timer = _timer_stub(elapsed_s=1800.0, remaining_s=1800.0)  # 50% of budget
    trainer = _trainer_stub(
        estimated_stepping_batches=10_000_000,  # placeholder max_epochs => ~0 steps
        global_step=300,
        callbacks=[timer],
    )

    assert callback._training_progress_fraction(
        cast(L.Trainer, trainer)
    ) == pytest.approx(0.5)
    assert callback._monitor_ready(cast(L.Trainer, trainer)) is True


def test_progress_fraction_falls_back_to_steps_without_timer():
    callback = _windowed_callback()
    trainer = _trainer_stub(estimated_stepping_batches=100, global_step=50)

    assert callback._training_progress_fraction(
        cast(L.Trainer, trainer)
    ) == pytest.approx(0.5)


def test_progress_fraction_takes_max_of_time_and_steps():
    """Step-bounded run with a large, non-binding ``max_time`` must use the
    (larger) step fraction, not the tiny wall-clock fraction."""
    callback = _windowed_callback()
    timer = _timer_stub(elapsed_s=60.0, remaining_s=43_140.0)  # ~0.14% of 12h
    trainer = _trainer_stub(
        estimated_stepping_batches=100, global_step=50, callbacks=[timer]
    )

    assert callback._training_progress_fraction(
        cast(L.Trainer, trainer)
    ) == pytest.approx(0.5)


def test_progress_fraction_ignores_durationless_timer():
    callback = _windowed_callback()
    timer = _timer_stub(elapsed_s=1800.0, remaining_s=None)  # no duration set
    trainer = _trainer_stub(
        estimated_stepping_batches=100, global_step=50, callbacks=[timer]
    )

    assert callback._training_progress_fraction(
        cast(L.Trainer, trainer)
    ) == pytest.approx(0.5)


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
    trainer = Mock(
        spec=L.Trainer,
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
    pl_module = Mock(spec=L.LightningModule, val_metrics=None)

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
    trainer = Mock(
        spec=L.Trainer,
        callback_metrics={},
        default_root_dir=tmp_path,
        global_step=12,
        is_global_zero=True,
        sanity_checking=False,
        loggers=[],
    )
    pl_module = Mock(
        spec=L.LightningModule,
        val_metrics=MetricCollection({"diagnostic": PlotMetric()}).clone(prefix="val_"),
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
    trainer = Mock(
        spec=L.Trainer,
        callback_metrics={"val_loss": torch.tensor(1.5)},
        default_root_dir=tmp_path,
        global_step=1189,
        is_global_zero=True,
        sanity_checking=False,
        loggers=[logger],
    )
    pl_module = Mock(spec=L.LightningModule, val_metrics=None)

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


def test_default_trainer_config_tracks_coverage_winkler_without_plots(
    config_dir: str,
):
    trainer_cfg = OmegaConf.load(Path(config_dir) / "trainer" / "default.yaml")
    callbacks = list(trainer_cfg.callbacks)
    monitors = [callback.get("monitor") for callback in callbacks]

    assert "val_multicoverage" in monitors
    assert "val_multiwinkler" in monitors
    assert not any(
        callback.get("_target_")
        == "autocast.callbacks.metrics.ValidationMetricPlotCallback"
        for callback in callbacks
    )


def test_default_trainer_config_instantiates_callbacks(config_dir: str):
    trainer_cfg = OmegaConf.load(Path(config_dir) / "trainer" / "default.yaml")

    trainer = instantiate(trainer_cfg)
    progress_state_keys = [
        callback.state_key
        for callback in trainer.callbacks
        if isinstance(callback, ProgressModelCheckpoint)
    ]

    assert len(progress_state_keys) == len(set(progress_state_keys))


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


def test_epd_rejects_ambient_only_loss_for_latent_training(
    config_dir: str,
    toy_batch: Batch,
    dummy_datamodule,
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
        cfg.model.loss_func = {
            "_target_": "autocast.losses.MCDropoutMSEL2Loss",
            "l2_coefficient": 1e-5,
        }

    stats = _stats_from_batch(toy_batch)
    with pytest.raises(ValueError, match="requires ambient predictions"):
        setup_epd_model(cfg, stats, dummy_datamodule)


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
    class DummyEncoder(IdentityEncoder):
        channel_axis = -1
        outputs_time_channel_concat = False

    encoded = torch.randn(2, 3, 16, 16, 8)  # B,T,W,H,C
    spatial = _infer_latent_spatial_resolution(
        encoded, cast(EncoderWithCond, DummyEncoder(in_channels=1))
    )
    assert spatial == (16, 16)


def test_infer_latent_spatial_resolution_channels_first_time_concat():
    class DummyEncoder(IdentityEncoder):
        channel_axis = 1
        outputs_time_channel_concat = True

    encoded = torch.randn(2, 24, 16, 16)  # B,C*T,W,H
    spatial = _infer_latent_spatial_resolution(
        encoded, cast(EncoderWithCond, DummyEncoder(in_channels=1))
    )
    assert spatial == (16, 16)


@pytest.mark.parametrize("flow_matching", [False, True])
def test_opt_in_compile_backbone_preserves_full_resume(
    config_dir: str, tmp_path: Path, monkeypatch, flow_matching: bool
):
    cfg = _load_config(config_dir, "encoder_processor_decoder")
    assert cfg.get("compile_backbone") is False
    compile_calls = []
    original_compile = torch.nn.Module.compile

    def compile_eager(module, **kwargs):
        compile_calls.append(module)
        assert kwargs["backend"] == "inductor"
        original_compile(module, backend="eager")

    monkeypatch.setattr(torch.nn.Module, "compile", compile_eager)

    class ToyModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            if flow_matching:
                backbone = SpatialViTBackbone(
                    in_channels=1,
                    out_channels=1,
                    cond_channels=1,
                    n_steps_input=1,
                    n_steps_output=1,
                    global_cond_channels=None,
                    include_global_cond=False,
                    hid_channels=16,
                    hid_blocks=1,
                    attention_heads=2,
                    mod_features=8,
                    patch_size=2,
                )
                self.processor = FlowMatchingProcessor(
                    backbone=backbone,
                    flow_ode_steps=1,
                    n_steps_output=1,
                    n_channels_out=1,
                )
            else:
                self.processor = AzulaViTProcessor(
                    in_channels=1,
                    out_channels=1,
                    spatial_resolution=(4, 4),
                    hidden_dim=16,
                    n_layers=1,
                    num_heads=2,
                    patch_size=2,
                    temporal_method="none",
                    n_noise_channels=8,
                )

        def training_step(self, batch, batch_idx):
            del batch_idx
            (fields,) = batch
            if isinstance(self.processor, FlowMatchingProcessor):
                pred = self.processor.flow_field(
                    fields, fields.new_full((fields.shape[0],), 0.5), fields
                )
            else:
                pred = self.processor(fields, fields.new_zeros(fields.shape[0], 8))
            return (pred - fields).square().mean()

        def configure_optimizers(self) -> OptimizerLRScheduler:
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=4
                    ),
                    "interval": "step",
                },
            }

    class ToyData(L.LightningDataModule):
        def train_dataloader(self):
            fields = torch.linspace(-1, 1, steps=64).reshape(4, 1, 4, 4, 1)
            return DataLoader(TensorDataset(fields), batch_size=2)

    def fit(name, max_steps, compile_backbone=False, parent=None):
        config = OmegaConf.create(
            {
                "compile_backbone": compile_backbone,
                "resume_from_checkpoint": str(parent) if parent is not None else None,
                "logging": {"wandb": {"enabled": False}},
                "output": {"save_config": False},
                "trainer": {
                    "_target_": "lightning.pytorch.Trainer",
                    "accelerator": "cpu",
                    "devices": 1,
                    "max_steps": max_steps,
                    "callbacks": [{"_target_": "autocast.callbacks.ema.EMACallback"}],
                    "enable_checkpointing": False,
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "num_sanity_val_steps": 0,
                },
            }
        )
        path = tmp_path / name / "final.ckpt"
        model = ToyModel()
        run_training(
            config,
            model,
            ToyData(),
            path.parent,
            skip_test=True,
            output_checkpoint_path=path,
        )
        return path, torch.load(path, weights_only=False, map_location="cpu")

    parent, _ = fit("parent", 2)
    _, eager = fit("eager", 4, parent=parent)
    assert compile_calls == []
    _, compiled = fit("compiled", 4, compile_backbone=True, parent=parent)
    assert len(compile_calls) == 1
    assert isinstance(compile_calls[0], SpatialViTBackbone)
    assert eager["global_step"] == compiled["global_step"] == 4
    assert eager["lr_schedulers"] == compiled["lr_schedulers"]
    for key in ("state_dict", "ema_state_dict"):
        assert eager[key].keys() == compiled[key].keys()
        for name, weights in eager[key].items():
            torch.testing.assert_close(compiled[key][name], weights)


def test_final_checkpoint_survives_standalone_validation(tmp_path: Path):
    """Post-fit validation must not replace the endpoint with an older alias."""
    final_path = tmp_path / "final.ckpt"

    class Toy(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def training_step(self, batch, batch_idx):
            del batch, batch_idx
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            del batch, batch_idx
            # Keep the best checkpoint at epoch zero, before the endpoint.
            self.log("val_loss", self.weight * 0 + self.current_epoch + 1, batch_size=1)

        def on_train_epoch_start(self):
            if self.current_epoch > 0:
                assert final_path.is_symlink()

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=0.01)

    class Data(L.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(TensorDataset(torch.arange(4.0)), batch_size=1)

        def val_dataloader(self):
            return DataLoader(TensorDataset(torch.arange(2.0)), batch_size=1)

    cfg = OmegaConf.create(
        {
            "logging": {"wandb": {"enabled": False}},
            "output": {"save_config": False},
            "trainer": {
                "_target_": "lightning.pytorch.Trainer",
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
                "max_epochs": 3,
                "max_steps": -1,
                "enable_model_summary": False,
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
                "callbacks": [
                    {
                        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "dirpath": str(tmp_path / "checkpoints"),
                        "monitor": "val_loss",
                        "mode": "min",
                        "save_top_k": 1,
                        "save_last": False,
                        "save_on_train_epoch_end": False,
                    }
                ],
            },
        }
    )
    model, data = Toy(), Data()
    run_training(
        cfg, model, data, tmp_path, skip_test=True, output_checkpoint_path=final_path
    )
    assert model.trainer.global_step == 12
    assert not final_path.is_symlink()
    assert torch.load(final_path, weights_only=False)["global_step"] == 12
    saved_endpoint = final_path.read_bytes()

    model.trainer.validate(model, datamodule=data, ckpt_path=None, verbose=False)

    assert not final_path.is_symlink()
    assert final_path.read_bytes() == saved_endpoint


@pytest.mark.parametrize(
    "name", ["autoencoder", "encoder_processor_decoder", "processor"]
)
def test_early_stopping_is_off_by_default_and_patience_100_when_on(
    config_dir: str, name: str
):
    """Every top-level config leaves the stop off, at patience 100 once enabled.

    Off because a flow-matching ``val_loss`` is a fresh draw at every validation
    and patience 100 ends those runs far from their best (aiphys#246). The
    values live in the config so each run's resolved config records them, which
    the grid scorers read back; an absent block is off as well.
    """
    cfg = _load_config(config_dir, name)
    assert cfg.early_stopping.enabled is False
    assert _early_stopping_callbacks(cfg) == []
    assert cfg.early_stopping.patience == DEFAULT_EARLY_STOPPING_PATIENCE == 100
    assert cfg.early_stopping.min_delta == 0.0
    with open_dict(cfg):
        cfg.early_stopping.enabled = True
    (stop,) = _early_stopping_callbacks(cfg)
    assert cast(EarlyStopping, stop).patience == 100
    with open_dict(cfg):
        del cfg["early_stopping"]
    assert _early_stopping_callbacks(cfg) == []


@pytest.mark.parametrize(("every_n", "checks"), [(1, 100), (5, 20), (3, 34)])
def test_early_stopping_patience_is_in_epochs_not_validation_checks(
    config_dir: str, every_n: int, checks: int
):
    """Validating every k-th epoch must not multiply the patience by k."""
    cfg = _load_config(config_dir, "encoder_processor_decoder")
    with open_dict(cfg):
        cfg.trainer.check_val_every_n_epoch = every_n
        cfg.early_stopping.enabled = True
    (stop,) = _early_stopping_callbacks(cfg)
    assert cast(EarlyStopping, stop).patience == checks


def test_early_stopping_can_be_declined_and_never_replaces_callbacks(config_dir: str):
    """A run can switch the stop off, and it is APPENDED, never merged into the list.

    The trainer config's ``callbacks`` list carries every checkpoint callback
    plus the non-finite-loss guard, and Hydra list overrides *replace* rather
    than append -- so adding early stopping via a child config would silently
    drop all of them. It is therefore appended in ``run_training`` instead, and
    this test pins both halves of that contract.
    """
    cfg = _load_config(config_dir, "autoencoder")

    # Off only when a run declines it.
    with open_dict(cfg):
        cfg.early_stopping.enabled = False
    assert _early_stopping_callbacks(cfg) == []

    targets = [str(c.get("_target_", "")) for c in cfg.trainer.callbacks]
    assert any(t.endswith("ModelCheckpoint") for t in targets)
    n_callbacks_before = len(cfg.trainer.callbacks)

    patience = 7
    with open_dict(cfg):
        cfg.early_stopping = {
            "enabled": True,
            "patience": patience,
            "monitor": "val_loss",
        }

    stops = _early_stopping_callbacks(cfg)
    assert len(stops) == 1
    stop = cast(EarlyStopping, stops[0])
    assert stop.monitor == "val_loss"
    assert stop.patience == patience
    # The stop is on a validation plateau only, so its reason is unambiguous.
    assert stop.check_finite is False

    # Enabling it must not have disturbed the configured callback list.
    assert len(cfg.trainer.callbacks) == n_callbacks_before
    targets_after = [str(c.get("_target_", "")) for c in cfg.trainer.callbacks]
    assert targets_after == targets


@pytest.mark.parametrize("enabled", [True, False])
def test_train_autoencoder_honours_early_stopping(
    config_dir: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
):
    """The autoencoder path carries the stop.

    Until 2026-10-16 ``train_autoencoder`` did not append it, so
    ``early_stopping.enabled: true`` was ignored: the 32x32 shallow-water
    autoencoder trained ~1000 epochs past its validation minimum (epoch 34).
    Data, model, fitting and saving are stubbed; the trainer is the one the
    composed config builds.
    """
    cfg = _load_config(
        config_dir,
        "autoencoder",
        ["trainer.accelerator=cpu", f"early_stopping.enabled={enabled}"],
    )
    seen: list[list[Any]] = []
    monkeypatch.setattr(
        training, "apply_float32_matmul_precision", lambda *_a, **_k: None
    )
    monkeypatch.setattr(training, "create_wandb_logger", lambda *_, **__: (None, None))
    monkeypatch.setattr(training, "setup_datamodule", lambda c: (None, c, {}))
    monkeypatch.setattr(
        training, "setup_autoencoder_model", lambda *_, **__: torch.nn.Linear(1, 1)
    )
    monkeypatch.setattr(training, "_save_reconstructions", lambda *_: None)
    monkeypatch.setattr(
        L.Trainer, "fit", lambda self, **_: seen.append(list(self.callbacks))
    )
    monkeypatch.setattr(L.Trainer, "save_checkpoint", lambda *_, **__: None)

    training.train_autoencoder(cfg, tmp_path)

    (callbacks,) = seen
    stops = [c for c in callbacks if isinstance(c, EarlyStopping)]
    if enabled:
        (stop,) = stops
        assert stop.monitor == "val_loss"
        assert stop.patience == DEFAULT_EARLY_STOPPING_PATIENCE
    else:
        assert stops == []
