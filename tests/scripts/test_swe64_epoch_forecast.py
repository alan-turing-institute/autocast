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
from autocast.scripts.workflow.slurm import (
    _build_sbatch_command,
    _resolve_launcher_submission_context,
)

ROOT = Path(__file__).resolve().parents[2]
PRESET = "afcrps_global_residual_one_output_512ep"
SPATIAL_PRESET = "afcrps_spatial_residual_one_output_256ep"
BOUNDED_GLOBAL_PRESET = "afcrps_global_residual_one_output_512ep_8h"
BOUNDED_SPATIAL_PRESET = "afcrps_spatial_residual_one_output_512ep_8h"
MATCHED_GLOBAL_PRESET = "afcrps_global256_residual_one_output_12block_512ep_8h"
MATCHED_SPATIAL_PRESET = "afcrps_spatial1d_residual_one_output_12block_512ep_8h"
MATCHED_PRESETS = {MATCHED_GLOBAL_PRESET, MATCHED_SPATIAL_PRESET}
SPATIAL_PRESETS = {SPATIAL_PRESET, BOUNDED_SPATIAL_PRESET, MATCHED_SPATIAL_PRESET}
EPOCH_PRESETS = {
    PRESET: 512,
    "afcrps_global_residual_one_output_256ep": 256,
    SPATIAL_PRESET: 256,
    BOUNDED_GLOBAL_PRESET: 512,
    BOUNDED_SPATIAL_PRESET: 512,
    MATCHED_GLOBAL_PRESET: 512,
    MATCHED_SPATIAL_PRESET: 512,
}
BOUNDED_LIMITS = {
    BOUNDED_GLOBAL_PRESET: ("00:07:45:00", 480),
    BOUNDED_SPATIAL_PRESET: ("00:07:45:00", 480),
    MATCHED_GLOBAL_PRESET: ("00:07:45:00", 480),
    MATCHED_SPATIAL_PRESET: ("00:07:45:00", 480),
}


def _config(name=PRESET, *, distributed=False, epochs=None):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
    ]
    if epochs is not None:
        overrides.append(f"trainer.max_epochs={epochs}")
    if distributed:
        time_cap, allocation_minutes = BOUNDED_LIMITS.get(name, ("00:23:30:00", 1440))
        overrides += [
            "+distributed=ddp_4gpu_slurm",
            f"trainer.max_time={time_cap}",
            "++hydra.launcher.nodes=1",
            "++hydra.launcher.cpus_per_task=72",
            f"hydra.launcher.timeout_min={allocation_minutes}",
            "logging.wandb.enabled=true",
            "logging.wandb.mode=online",
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
@pytest.mark.parametrize(("name", "epochs"), EPOCH_PRESETS.items())
def test_epoch_control_preserves_data_model_and_optimizer_settings(
    distributed, name, epochs
):
    reference = _config("afcrps_global_residual_one_output_2h30")
    if name in MATCHED_PRESETS:
        reference.model.processor.update(
            hidden_dim=224,
            n_layers=12,
            num_heads=8,
            n_noise_channels=256,
            n_noise_input_channels=256,
        )
    if name in SPATIAL_PRESETS:
        reference.model.processor.noise_mode = "spatial"
        reference.model.processor.n_noise_input_channels = (
            1 if name in MATCHED_PRESETS else 4
        )
    cfg = _config(name, distributed=distributed)
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
        cosine_epochs=epochs,
    )
    assert OmegaConf.to_container(cfg.optimizer, resolve=True) == expected
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.min_lr_ratio == cfg.optimizer.warmup == 0
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == epochs
    assert cfg.trainer.max_steps == -1
    time_cap, allocation_minutes = BOUNDED_LIMITS.get(name, ("00:23:30:00", 1440))
    assert cfg.trainer.max_time == time_cap
    assert cfg.trainer.limit_train_batches == cfg.trainer.limit_val_batches == 1.0
    assert cfg.trainer.check_val_every_n_epoch == cfg.trainer.val_check_interval == 1
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.seed == reference.seed == 42
    assert cfg.float32_matmul_precision == reference.float32_matmul_precision
    assert cfg.trainer.gradient_clip_val == reference.trainer.gradient_clip_val
    assert cfg.datamodule.batch_size == 32
    assert cfg.model.processor.noise_mode == reference.model.processor.noise_mode
    assert (
        cfg.model.processor.n_noise_input_channels
        == reference.model.processor.n_noise_input_channels
    )
    assert (
        cfg.model.processor.n_noise_channels
        == reference.model.processor.n_noise_channels
    )
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
        assert cfg.hydra.launcher.timeout_min == allocation_minutes
        assert cfg.logging.wandb.enabled
        assert cfg.logging.wandb.mode == "online"
    else:
        assert not cfg.logging.wandb.enabled


def test_shorter_preset_changes_only_experiment_name_and_epoch_horizon():
    reference = _config()
    shorter = _config("afcrps_global_residual_one_output_256ep")
    expected = OmegaConf.to_container(reference, resolve=False)
    assert isinstance(expected, dict)
    expected["experiment_name"] = shorter.experiment_name
    expected["trainer"]["max_epochs"] = 256
    assert OmegaConf.to_container(shorter, resolve=False) == expected


@pytest.mark.parametrize(
    ("global_name", "spatial_name"),
    [
        ("afcrps_global_residual_one_output_256ep", SPATIAL_PRESET),
        (BOUNDED_GLOBAL_PRESET, BOUNDED_SPATIAL_PRESET),
        (MATCHED_GLOBAL_PRESET, MATCHED_SPATIAL_PRESET),
    ],
)
def test_spatial_preset_changes_only_experiment_name_and_noise_layout(
    global_name, spatial_name
):
    reference = _config(global_name)
    spatial = _config(spatial_name)
    expected = OmegaConf.to_container(reference, resolve=False)
    assert isinstance(expected, dict)
    expected["experiment_name"] = spatial.experiment_name
    noise_values = 1 if spatial_name == MATCHED_SPATIAL_PRESET else 4
    expected["model"]["processor"].update(
        noise_mode="spatial", n_noise_input_channels=noise_values
    )
    assert OmegaConf.to_container(spatial, resolve=False) == expected
    assert spatial.datamodule.n_steps_input == spatial.datamodule.n_steps_output == 1
    assert spatial.datamodule.stride == 1
    tokens = (64 // spatial.model.processor.patch_size) ** 2
    assert tokens == 256
    assert (
        tokens * spatial.model.processor.n_noise_input_channels
        == reference.model.processor.n_noise_input_channels
        == (256 if spatial_name == MATCHED_SPATIAL_PRESET else 1024)
    )
    with torch.device("meta"):
        processor = instantiate(
            spatial.model.processor,
            in_channels=3,
            out_channels=3,
            spatial_resolution=[64, 64],
        )
    expected_count = (
        10_232_016 if spatial_name == MATCHED_SPATIAL_PRESET else 10_662_384
    )
    assert sum(p.numel() for p in processor.parameters()) == expected_count


@pytest.mark.parametrize(
    ("name", "reference_name", "noise_input", "expected_count"),
    [
        (MATCHED_GLOBAL_PRESET, BOUNDED_GLOBAL_PRESET, 256, 10_231_504),
        (MATCHED_SPATIAL_PRESET, BOUNDED_SPATIAL_PRESET, 1, 10_232_016),
    ],
)
def test_matched_noise_presets_change_only_capacity_and_noise_dimensions(
    name, reference_name, noise_input, expected_count
):
    reference = _config(reference_name)
    cfg = _config(name)
    expected = OmegaConf.to_container(reference, resolve=False)
    assert isinstance(expected, dict)
    expected["experiment_name"] = cfg.experiment_name
    expected["model"]["processor"].update(
        hidden_dim=224,
        n_layers=12,
        num_heads=8,
        n_noise_channels=256,
        n_noise_input_channels=noise_input,
    )
    assert OmegaConf.to_container(cfg, resolve=False) == expected
    assert cfg.model.processor.patch_size == cfg.model.processor.ffn_factor == 4
    assert cfg.model.processor.qk_norm
    assert cfg.model.processor.rpb
    assert not cfg.model.processor.rope
    assert cfg.model.processor.dropout == 0
    with torch.device("meta"):
        processor = instantiate(
            cfg.model.processor,
            in_channels=3,
            out_channels=3,
            spatial_resolution=[64, 64],
        )
    assert sum(p.numel() for p in processor.parameters()) == expected_count
    if name == MATCHED_GLOBAL_PRESET:
        assert processor.modulation_proj is None
    else:
        projection = processor.modulation_proj
        assert isinstance(projection, torch.nn.Linear)
        assert projection.in_features == 1
        assert projection.out_features == 256
        assert sum(p.numel() for p in projection.parameters()) == 512


@pytest.mark.parametrize(
    ("name", "reference_name"),
    [
        (BOUNDED_GLOBAL_PRESET, "afcrps_global_residual_one_output_256ep"),
        (BOUNDED_SPATIAL_PRESET, SPATIAL_PRESET),
    ],
)
def test_bounded_512_profiles_preserve_their_256_epoch_scientific_settings(
    name, reference_name
):
    reference = _config(reference_name)
    longer = _config(name)
    expected = OmegaConf.to_container(reference, resolve=False)
    assert isinstance(expected, dict)
    expected["experiment_name"] = longer.experiment_name
    expected["trainer"].update(max_epochs=512, max_time=BOUNDED_LIMITS[name][0])
    assert OmegaConf.to_container(longer, resolve=False) == expected
    assert OmegaConf.select(longer, "resume_from_checkpoint") is None
    assert longer.autoencoder_checkpoint is None


@pytest.mark.parametrize("name", BOUNDED_LIMITS)
def test_bounded_512_launcher_defaults_and_submission_budget(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    cap, minutes = BOUNDED_LIMITS[name]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"local_experiment=swe64s/{name}",
                f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
            ],
            return_hydra_config=True,
        )
    assert cfg.trainer.max_time == cap
    assert cfg.hydra.launcher.timeout_min == minutes
    assert minutes - int(cap.split(":")[1]) * 60 - int(cap.split(":")[2]) == 15
    overrides = [
        f"local_experiment=swe64s/{name}",
        "+distributed=ddp_4gpu_slurm",
        f"trainer.max_time={cap}",
        "++hydra.launcher.nodes=1",
        "++hydra.launcher.cpus_per_task=72",
        "logging.wandb.enabled=true",
        "logging.wandb.mode=online",
    ]
    launcher, module_overrides = _resolve_launcher_submission_context(overrides)
    assert launcher["timeout_min"] == minutes
    assert launcher["nodes"] == 1
    assert launcher["gpus_per_node"] == launcher["tasks_per_node"] == 4
    assert launcher["cpus_per_task"] == 72
    assert f"trainer.max_time={cap}" in module_overrides
    preview = _build_sbatch_command(
        job_name=name,
        log_dir=Path("/preflight"),
        launcher_cfg=launcher,
        batch_script_path=Path("/preflight/preview.sh"),
    )
    assert f"--time={minutes // 60:02d}:00:00" in preview
    assert "--nodes=1" in preview
    assert "--gpus-per-node=4" in preview
    assert "--ntasks-per-node=4" in preview


@pytest.mark.parametrize(("name", "epochs"), EPOCH_PRESETS.items())
def test_full_epochs_match_the_intended_update_count(name, epochs):
    cfg = _config(name, distributed=True)
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
    assert len(loader) * cfg.trainer.max_epochs == {512: 256000, 256: 128000}[epochs]


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


@pytest.mark.parametrize(("name", "epochs"), EPOCH_PRESETS.items())
def test_full_cosine_uses_epochs_and_reaches_zero_at_the_endpoint(name, epochs):
    cfg = _config(name)
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
    for epoch in range(epochs):
        rate = optimizer.param_groups[0]["lr"]
        assert rate == pytest.approx(
            2e-4 * (1 + math.cos(math.pi * epoch / epochs)) / 2
        )
        rates.append(rate)
        optimizer.step()
        scheduler.step()
    assert rates[0] == 2e-4
    assert rates[epochs // 2] == pytest.approx(1e-4)
    assert rates[-1] > 0
    assert optimizer.param_groups[0]["lr"] == 0


@pytest.mark.parametrize("name", EPOCH_PRESETS)
def test_epoch_override_updates_the_cosine_horizon(name):
    cfg = _config(name, epochs=64)
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 64


@pytest.mark.parametrize("name", EPOCH_PRESETS)
def test_epoch_policy_keeps_best_last_and_periodic_full_checkpoints(tmp_path, name):
    cfg = _config(name, epochs=4)
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
