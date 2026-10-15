"""Epoch-based SWE64 config, scheduler and tiny CPU checkpoint checks."""

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from autocast.callbacks.ema import EMACallback
from autocast.models.optimizer_mixin import OptimizerMixin

ROOT = Path(__file__).resolve().parents[2]
PRESET = "afcrps_global_residual_one_output_512ep"


def _config(name=PRESET, *, distributed=False, epochs=512):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
    ]
    if name == PRESET:
        overrides.append(f"trainer.max_epochs={epochs}")
    if distributed:
        overrides += [
            "+distributed=ddp_4gpu_slurm",
            "trainer.max_time=00:23:30:00",
            "++hydra.launcher.nodes=1",
            "++hydra.launcher.cpus_per_task=72",
            "hydra.launcher.timeout_min=1440",
            "logging.wandb.enabled=true",
        ]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=overrides,
            return_hydra_config=distributed,
        )


@pytest.mark.parametrize("distributed", [False, True])
def test_epoch_control_preserves_data_model_and_optimizer_settings(distributed):
    reference = _config("afcrps_global_residual_one_output_2h30")
    cfg = _config(distributed=distributed)
    for key in ("model", "datamodule"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(
            reference[key], resolve=True
        )
    expected = OmegaConf.to_container(reference.optimizer, resolve=True)
    assert isinstance(expected, dict)
    expected.update(
        name=cfg.optimizer.name,
        scheduler="cosine",
        scheduler_interval="epoch",
        cosine_epochs=512,
    )
    assert OmegaConf.to_container(cfg.optimizer, resolve=True) == expected
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.min_lr_ratio == cfg.optimizer.warmup == 0
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 512
    assert cfg.trainer.max_steps == -1
    assert cfg.trainer.max_time == "00:23:30:00"
    assert cfg.trainer.limit_train_batches == cfg.trainer.limit_val_batches == 1.0
    assert cfg.trainer.check_val_every_n_epoch == cfg.trainer.val_check_interval == 1
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.seed == reference.seed == 42
    assert cfg.float32_matmul_precision == reference.float32_matmul_precision
    assert cfg.trainer.gradient_clip_val == reference.trainer.gradient_clip_val
    assert cfg.datamodule.batch_size == 32
    assert cfg.model.processor.noise_mode == "global"
    assert cfg.model.processor.n_noise_input_channels == 1024
    assert cfg.model.n_members == 8
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert not cfg.model.encoder.with_constants
    assert not cfg.model.processor.include_global_cond
    assert cfg.datamodule.start_frame == 0
    assert "shallow_water2d_stochastic_64_b43bd32" in cfg.datamodule.data_path
    assert OmegaConf.select(cfg, "resume_from_checkpoint") is None
    assert cfg.autoencoder_checkpoint is None
    if distributed:
        assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
        assert cfg.trainer.num_nodes == cfg.hydra.launcher.nodes == 1
        assert cfg.hydra.launcher.tasks_per_node == 4
        assert cfg.hydra.launcher.cpus_per_task == 72
        assert cfg.hydra.launcher.timeout_min == 1440
        assert cfg.logging.wandb.enabled
    else:
        assert not cfg.logging.wandb.enabled


def test_512_full_epochs_are_256000_updates_at_the_intended_global_batch():
    cfg = _config(distributed=True)
    manifest = json.loads(
        (ROOT / "local_hydra/local_experiment/swe64s/dataset.json").read_text()
    )
    windows = (
        manifest["frames"]
        - cfg.datamodule.n_steps_input
        - cfg.datamodule.n_steps_output
    ) // cfg.datamodule.stride + 1
    assert manifest["splits"]["train"] == 200
    assert windows == 320
    data = TensorDataset(torch.zeros(200 * windows))
    sampler = DistributedSampler(data, num_replicas=4, rank=0, shuffle=True)
    loader = DataLoader(data, batch_size=cfg.datamodule.batch_size, sampler=sampler)
    assert len(loader) == 500
    assert len(loader) * cfg.trainer.max_epochs == 256000


class _EpochFit(OptimizerMixin, LightningModule):
    def __init__(self, optimizer_config):
        super().__init__()
        self.optimizer_config = optimizer_config
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.rates = []
        self.validation_epochs = []

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return (self.weight - batch[0]).square().mean()

    def on_before_optimizer_step(self, optimizer):
        self.rates.append((self.current_epoch, optimizer.param_groups[0]["lr"]))

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self.validation_epochs.append(self.current_epoch)
        # Deliberately select a non-final best checkpoint in the tiny smoke fit.
        self.log("val_loss", float(abs(self.current_epoch - 1)))


def test_full_cosine_uses_epochs_and_reaches_zero_after_512_epochs():
    cfg = _config()
    model = _EpochFit(cfg.optimizer)
    configured = model.configure_optimizers()
    assert isinstance(configured, dict)
    configured = cast("dict[str, Any]", configured)
    schedule = configured["lr_scheduler"]
    assert schedule["interval"] == "epoch"
    assert schedule["frequency"] == 1
    scheduler = schedule["scheduler"]
    optimizer = configured["optimizer"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert isinstance(optimizer, torch.optim.Optimizer)
    rates = []
    for epoch in range(512):
        rate = optimizer.param_groups[0]["lr"]
        assert rate == pytest.approx(2e-4 * (1 + math.cos(math.pi * epoch / 512)) / 2)
        rates.append(rate)
        optimizer.step()
        scheduler.step()
    assert rates[0] == 2e-4
    assert rates[256] == pytest.approx(1e-4)
    assert rates[-1] > 0
    assert optimizer.param_groups[0]["lr"] == 0


def test_epoch_override_updates_the_cosine_horizon():
    cfg = _config(epochs=64)
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 64


def test_epoch_policy_keeps_best_last_and_periodic_full_checkpoints(tmp_path):
    cfg = _config(epochs=4)
    callbacks = [instantiate(spec) for spec in cfg.trainer.callbacks]
    checkpoints = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
    assert len(checkpoints) == 2
    snapshot, best = checkpoints
    assert snapshot._every_n_epochs == 64
    assert snapshot.save_last
    assert snapshot.save_top_k == -1
    assert best.monitor == "val_loss"
    assert best.mode == "min"
    assert best.save_top_k == 1
    assert best._every_n_epochs == 1
    assert not best.save_last
    assert all(not cb.save_weights_only for cb in checkpoints)
    ema = next(cb for cb in callbacks if isinstance(cb, EMACallback))
    assert ema.decay == 0.999
    snapshot._every_n_epochs = 2  # Shorten cadence only for this four-epoch smoke.
    model = _EpochFit(cfg.optimizer)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        callbacks=[snapshot, best, ema],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    data = TensorDataset(torch.ones(4))
    trainer.fit(
        model,
        train_dataloaders=DataLoader(data, batch_size=2),
        val_dataloaders=DataLoader(data, batch_size=4),
    )
    assert trainer.global_step == 8
    assert model.validation_epochs == [0, 1, 2, 3]
    for epoch, rate in model.rates:
        assert rate == pytest.approx(2e-4 * (1 + math.cos(math.pi * epoch / 4)) / 2)
    assert trainer.optimizers[0].param_groups[0]["lr"] == 0
    assert len(list((tmp_path / "checkpoints").glob("snapshot-*.ckpt"))) == 2
    best_state = torch.load(
        best.best_model_path, weights_only=False, map_location="cpu"
    )
    last_state = torch.load(
        snapshot.last_model_path, weights_only=False, map_location="cpu"
    )
    assert best_state["epoch"] == 1
    assert best_state["global_step"] == 4
    assert last_state["epoch"] == 3
    assert last_state["global_step"] == 8
    assert last_state["optimizer_states"][0]["state"]
    assert last_state["lr_schedulers"][0]["last_epoch"] == 4
    assert "ema_state_dict" in last_state
    restored = _EpochFit(cfg.optimizer)
    restored.load_state_dict(last_state["state_dict"], strict=True)
    torch.testing.assert_close(restored.weight, model.weight, rtol=0, atol=0)
