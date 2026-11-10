"""Forecast windows and sampled epochs for the larger SWE64 training pool."""

import json
import runpy
from itertools import islice
from pathlib import Path
from typing import Literal

import pytest
import torch
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationToMissingValueError
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    TensorDataset,
)

from autocast.data.dataset import SpatioTemporalDataset
from autocast.scripts.workflow.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]
PRESET = "crps_fft_spatial1d_1000traj_256ep"
BASELINE = "crps_fft_spatial1d_residual_one_output_12block_256ep_8h"
HELPER = runpy.run_path(str(ROOT / "scripts/swe64s/prepare_followups.py"))


def config(name=PRESET, *, data: str | Path | None = "/preflight/pool", extra=()):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
        "+distributed=ddp_4gpu_slurm",
        *extra,
    ]
    if data is not None:
        data_key = (
            "forecast_diversity_data_path" if name == PRESET else "datamodule.data_path"
        )
        overrides.append(f"{data_key}={json.dumps(str(data))}")
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(config_name="encoder_processor_decoder", overrides=overrides)


def test_larger_pool_keeps_forecast_model_and_optimization_protocol():
    cfg = config()
    baseline = config(BASELINE, extra=("trainer=swe64s_noise_ablations",))
    for key in ("model", "optimizer"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(
            baseline[key], resolve=True
        )
    expected_data = OmegaConf.to_container(baseline.datamodule, resolve=True)
    assert isinstance(expected_data, dict)
    expected_data.update(autoencoder_mode=False, full_trajectory_mode=False)
    assert OmegaConf.to_container(cfg.datamodule, resolve=True) == expected_data
    baseline.trainer.limit_train_batches = 500
    assert cfg.trainer == baseline.trainer
    assert cfg.seed == baseline.seed == 42
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert not cfg.model.train_in_latent_space
    assert cfg.model.loss_func.weights == [1.0, 0.1]
    assert cfg.model.processor.noise_mode == "spatial"
    assert cfg.model.processor.modulation_mode == "adaln"
    assert cfg.model.processor.noise_source is None
    assert cfg.model.processor.n_noise_input_channels == 1
    assert cfg.model.processor.n_noise_channels == 256
    assert cfg.model.processor.hidden_dim == 224
    assert cfg.model.processor.n_layers == 12
    assert cfg.model.processor.num_heads == 8
    assert cfg.model.processor.patch_size == 4
    assert not cfg.model.processor.include_global_cond
    assert cfg.model.n_members == 8
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert not cfg.datamodule.autoencoder_mode
    assert not cfg.datamodule.full_trajectory_mode
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 256
    assert cfg.trainer.max_steps == -1
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.scheduler_interval == "epoch"
    assert cfg.optimizer.min_lr_ratio == cfg.optimizer.warmup == 0
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.devices == 4
    assert cfg.trainer.limit_train_batches * cfg.trainer.max_epochs == 128000
    assert cfg.trainer.limit_train_batches * 4 * cfg.datamodule.batch_size == 64000
    assert 64000 / (1000 * (321 - 1)) == 0.2


def test_fresh_start_requires_explicit_pool_and_retains_safeguards():
    with pytest.raises(
        InterpolationToMissingValueError, match="forecast_diversity_data_path"
    ):
        OmegaConf.to_container(
            config(data=None).datamodule, resolve=True, throw_on_missing=True
        )
    cfg = config()
    for key in (
        "resume_from_checkpoint",
        "resume_weights_only",
        "reset_resume_time_budget",
        "autoencoder_checkpoint",
        "output.resume_from_checkpoint",
    ):
        assert not OmegaConf.select(cfg, key)
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "online"
    assert cfg.logging.wandb.id is cfg.logging.wandb.resume is None
    assert cfg.trainer.limit_val_batches == 1.0
    assert cfg.trainer.val_check_interval == 1.0
    assert cfg.trainer.check_val_every_n_epoch == 1
    stop = next(
        c
        for c in cfg.trainer.callbacks
        if c._target_.endswith("SustainedValidationRise")
    )
    assert (stop.monitor, stop.relative_tolerance, stop.patience, stop.min_steps) == (
        "val_loss",
        0.01,
        10,
        10000,
    )
    checkpoint = next(
        c
        for c in cfg.trainer.callbacks
        if c.get("monitor") == "val_loss" and c._target_.endswith("ModelCheckpoint")
    )
    assert checkpoint.save_last
    assert checkpoint.save_top_k == 1
    assert not checkpoint.get("save_weights_only", False)


def test_opt_in_preview_uses_pool_provenance_and_unchanged_held_out_data(tmp_path):
    plan = HELPER["build_plan"](
        datasets_root=tmp_path / "unmounted data",
        output_root=tmp_path / "new outputs",
        run_names=["spatial_crps_fft_1000traj"],
    )
    (job,) = plan["jobs"]
    pool = plan["forecast_diversity_pool"]
    assert pool["shape"] == [1000, 321, 64, 64, 3]
    assert pool["splits"] == {"train": 1000, "valid": 20, "test": 20}
    baseline = json.loads(
        (ROOT / "local_hydra/local_experiment/swe64s/dataset.json").read_text()
    )
    for split in ("valid", "test"):
        assert (
            pool["sha256"][f"{split}/data.pt"] == baseline["sha256"][f"{split}/data.pt"]
        )
    assert pool["sha256"]["stats.yml"] == plan["latent_cache"]["normalization_sha256"]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        cfg = compose(
            config_name=job["config_name"],
            overrides=job["overrides"],
            return_hydra_config=True,
        )
    parsed = build_parser().parse_args(job["preview_command"][5:])
    assert parsed.dry_run
    assert parsed.mode == "slurm"
    assert parsed.command == "epd"
    assert parsed.overrides == job["overrides"]
    assert cfg.datamodule.data_path == pool["path"]
    assert cfg.datamodule.normalization_path == pool["path"] + "/stats.yml"
    assert cfg.provenance.dataset_id == pool["directory"]
    assert cfg.provenance.training_pool_sha256 == pool["sha256"]["train/data.pt"]
    assert cfg.provenance.reference_dataset_uuid == plan["raw_dataset"]["uuid"]
    assert cfg.provenance.normalization_sha256 == pool["sha256"]["stats.yml"]
    assert "dataset_uuid" not in cfg.provenance
    assert "raw_dataset_uuid" not in cfg.provenance
    assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
    assert cfg.hydra.launcher.nodes == cfg.trainer.num_nodes == 1
    assert cfg.hydra.launcher.timeout_min == job["allocation_minutes"] == 480
    assert cfg.trainer.max_time == job["trainer_max_time"] == "00:07:45:00"
    assert cfg.trainer.limit_train_batches == job["sampled_batches_per_epoch"] == 500
    assert (
        cfg.trainer.max_epochs * cfg.trainer.limit_train_batches
        == job["expected_updates"]
    )
    assert cfg.logging.wandb.mode == "online"
    assert cfg.logging.wandb.enabled
    assert not plan["jobs_submitted"]
    assert not plan["target_data_verified"]
    assert not (tmp_path / "new outputs").exists()
    assert not (tmp_path / "unmounted data").exists()
    default = HELPER["build_plan"](
        datasets_root=tmp_path, output_root=tmp_path / "other"
    )
    assert [j["name"] for j in default["jobs"]] == [
        "global_crps_fft",
        "latent_residual_fm",
    ]


def test_full_cadence_forecast_windows_keep_adjacent_targets_and_share_storage():
    cfg = config()
    time = rearrange(torch.arange(3 * 321, dtype=torch.float32), "(n t) -> n t", n=3)
    fields = repeat(time, "n t -> n t h w c", h=2, w=2, c=3).contiguous()
    dataset = SpatioTemporalDataset(
        data_path=None,
        data={"data": fields, "constant_scalars": None, "constant_fields": None},
        n_steps_input=cfg.datamodule.n_steps_input,
        n_steps_output=cfg.datamodule.n_steps_output,
        stride=cfg.datamodule.stride,
        autoencoder_mode=cfg.datamodule.autoencoder_mode,
    )
    assert len(dataset) == 3 * 320
    for index in (0, 319, 320, 959):
        trajectory, frame = divmod(index, 320)
        sample = dataset[index]
        torch.testing.assert_close(
            sample.input_fields, fields[trajectory, frame : frame + 1]
        )
        torch.testing.assert_close(
            sample.output_fields, fields[trajectory, frame + 1 : frame + 2]
        )
        assert dataset.sample_trajectory_indices[index] == trajectory
        assert dataset.sample_window_indices[index] == frame
        for view in (dataset.all_input_fields[index], dataset.all_output_fields[index]):
            assert (
                view.untyped_storage().data_ptr() == fields.untyped_storage().data_ptr()
            )


def test_data_view_uses_original_normalization_and_full_held_out_loaders(tmp_path):
    original, view = tmp_path / "original", tmp_path / "view"
    original.mkdir()
    view.mkdir()
    (view / "train").mkdir()
    fields = torch.arange(6 * 3, dtype=torch.float32)
    fields = repeat(
        rearrange(fields, "(t c) -> 1 t c", c=3), "n t c -> n t h w c", h=2, w=2
    )
    payload = {"data": fields, "constant_scalars": None, "constant_fields": None}
    torch.save(payload, view / "train/data.pt")
    for split in ("valid", "test"):
        (original / split).mkdir()
        torch.save(payload, original / split / "data.pt")
        (view / split).symlink_to(original / split, target_is_directory=True)
    stats = {
        "core_field_names": ["h", "u", "v"],
        "constant_field_names": [],
        "stats": {
            key: dict.fromkeys(("h", "u", "v"), value)
            for key, value in (
                ("mean", 2.0),
                ("std", 3.0),
                ("mean_delta", 0.1),
                ("std_delta", 0.2),
            )
        },
    }
    OmegaConf.save(OmegaConf.create(stats), original / "stats.yml")
    (view / "stats.yml").symlink_to(original / "stats.yml")
    cfg = config(data=view)
    module = instantiate(cfg.datamodule, num_workers=0, pin_memory=False)
    assert isinstance(module.train_dataloader().sampler, RandomSampler)
    for dataset in (module.train_dataset, module.val_dataset, module.test_dataset):
        assert len(dataset) == 5
        torch.testing.assert_close(dataset[0].input_fields, (fields[0, :1] - 2) / 3)
        torch.testing.assert_close(dataset[0].output_fields, (fields[0, 1:2] - 2) / 3)
    for loader in (module.val_dataloader(), module.test_dataloader()):
        assert sum(batch.input_fields.shape[0] for batch in loader) == 5
    assert OmegaConf.to_container(OmegaConf.load(view / "stats.yml")) == stats


def test_sampled_ddp_epochs_are_disjoint_reproducible_and_cover_the_pool():
    cfg = config()
    windows = TensorDataset(torch.arange(1000 * 320))
    per_rank = cfg.trainer.limit_train_batches * cfg.datamodule.batch_size
    selections = []
    for epoch in (0, 1, 0):
        ranks = []
        for rank in range(4):
            sampler = DistributedSampler(
                windows, num_replicas=4, rank=rank, shuffle=True, seed=cfg.seed
            )
            sampler.set_epoch(epoch)
            indices = list(islice(sampler, per_rank))
            assert len(indices) == len(set(indices)) == 16000
            ranks.append(set(indices))
        combined = set.union(*ranks)
        assert sum(map(len, ranks)) == len(combined) == 64000
        assert len(combined) / len(windows) == 0.2
        assert {index // 320 for index in combined} == set(range(1000))
        selections.append(combined)
    assert selections[0] != selections[1]
    assert selections[0] == selections[2]


class _EpochProbe(LightningModule):
    def __init__(self, initial_lr: float, interval: Literal["epoch", "step"]):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.initial_lr = initial_lr
        self.interval = interval
        self.seen = {}
        self.validation_steps = []
        self.rates = []

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        self.seen.setdefault(self.current_epoch, []).extend(batch[0].tolist())
        self.rates.append(self.trainer.optimizers[0].param_groups[0]["lr"])
        return self.weight.square()

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        if batch_idx == 0:
            self.validation_steps.append(self.global_step)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.SGD(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": self.interval},
        }


def test_sampled_lightning_epochs_validate_and_decay_at_the_truncated_boundary(
    tmp_path,
):
    cfg = config()
    # A single simulated DDP rank: five of 25 batches is the same 20% prefix.
    windows = TensorDataset(torch.arange(200))
    sampler = DistributedSampler(
        windows, num_replicas=4, rank=0, shuffle=True, seed=cfg.seed
    )
    model = _EpochProbe(cfg.optimizer.learning_rate, cfg.optimizer.scheduler_interval)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        max_steps=cfg.trainer.max_steps,
        limit_train_batches=5,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(
        model,
        train_dataloaders=DataLoader(windows, batch_size=2, sampler=sampler),
        val_dataloaders=DataLoader(windows, batch_size=40),
    )
    assert trainer.global_step == 15
    assert model.validation_steps == [5, 10, 15]
    assert model.seen[0] != model.seen[1] != model.seen[2]
    for epoch, seen in model.seen.items():
        sampler.set_epoch(epoch)
        assert seen == list(islice(sampler, 10))
    assert model.rates[0] == 2e-4
    assert model.rates[5] == pytest.approx(1.5e-4)
    assert model.rates[10] == pytest.approx(5e-5)
    assert trainer.optimizers[0].param_groups[0]["lr"] == 0
