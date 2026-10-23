"""Check short sensitivity branches retain the matched full training state."""

import copy
import math
from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autocast.callbacks.ema import EMACallback
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.scripts.training import run_training

ROOT = Path(__file__).resolve().parents[2]
PRESET = "crps_fft_moments_spatial1d_sensitivity_5k"
PARENT = "crps_fft_moments_spatial1d_residual_one_output_12block_256ep_8h"


def _compose(name=PRESET, overrides=()):
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"local_experiment=swe64s/{name}",
                f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
                *overrides,
            ],
            return_hydra_config=True,
        )


@pytest.mark.parametrize("weight", [0, 5e-4, 2e-3])
def test_sensitivity_preserves_model_data_and_original_schedule(weight):
    cfg = _compose(overrides=[f"model.loss_func.weights.2={weight}"])
    parent = _compose(PARENT)
    parent.model.loss_func.weights[2] = weight
    assert cfg.model == parent.model
    assert cfg.datamodule == parent.datamodule
    assert cfg.optimizer == parent.optimizer
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 256
    assert cfg.optimizer.scheduler == "cosine"
    assert cfg.optimizer.scheduler_interval == "epoch"
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.min_lr_ratio == cfg.optimizer.warmup == 0
    assert cfg.trainer.max_steps == 36000 + 5000
    assert OmegaConf.is_missing(cfg, "resume_from_checkpoint")
    assert not cfg.resume_weights_only
    assert cfg.reset_resume_time_budget
    assert cfg.trainer.num_sanity_val_steps == 0
    assert cfg.trainer.check_val_every_n_epoch == 1
    assert cfg.trainer.val_check_interval == 1.0
    assert cfg.model.loss_func.weights == [1.0, 0.1, weight]
    assert cfg.trainer.callbacks[-1].decay == 0.999
    # The unchanged four-GPU loader has 500 updates/epoch. Saving each epoch
    # gives exactly ten equally spaced validation/snapshot points per branch.
    assert 64000 // (cfg.datamodule.batch_size * 4) == 500
    assert cfg.trainer.callbacks[0].every_n_epochs == 1
    assert cfg.trainer.callbacks[0].save_top_k == -1
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "offline"
    assert not cfg.logging.wandb.log_model
    assert cfg.logging.wandb.id is cfg.logging.wandb.resume is None
    assert cfg.output.checkpoint_path is None
    assert cfg.output.skip_test


def test_ddp_override_requires_explicit_short_timer():
    cfg = _compose(
        overrides=[
            "+distributed=ddp_4gpu_slurm",
            "trainer.max_time=00:00:25:00",
        ]
    )
    assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
    assert cfg.trainer.max_time == "00:00:25:00"
    assert cfg.hydra.launcher.timeout_min == 30


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_equal(a, b)
    else:
        assert left == right


class _Probe(OptimizerMixin, L.LightningModule):
    def __init__(self, *, weight: float = 0.0):
        super().__init__()
        self.layer = nn.Linear(2, 1)
        self.weight = weight
        self.optimizer_config = {
            "optimizer": "adamw",
            "learning_rate": 2e-4,
            "scheduler": "cosine",
            "scheduler_interval": "epoch",
            "cosine_epochs": 8,
            "min_lr_ratio": 0,
            "warmup": 0,
        }
        self.started = None
        self.trace = []
        self.val_steps = []

    def on_train_start(self):
        ema = next(
            c
            for c in getattr(self.trainer, "callbacks", [])
            if isinstance(c, EMACallback)
        )
        self.started = {
            "weights": copy.deepcopy(self.state_dict()),
            "optimizer": copy.deepcopy(self.trainer.optimizers[0].state_dict()),
            "scheduler": copy.deepcopy(
                self.trainer.lr_scheduler_configs[0].scheduler.state_dict()
            ),
            "ema": copy.deepcopy(ema.state_dict()),
            "global_step": self.global_step,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        noise = torch.randn_like(y)
        self.trace.append(
            (
                self.global_step,
                self.current_epoch,
                batch_idx,
                x.clone(),
                noise.clone(),
                self.trainer.optimizers[0].param_groups[0]["lr"],
            )
        )
        prediction = self.layer(x) + noise
        error = prediction - y
        return error.square().mean() + self.weight * prediction.pow(4).mean()

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self.val_steps.append(self.global_step)
        x, y = batch
        self.log("val_loss", (self.layer(x) - y).square().mean())


class _ProbeData(L.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            TensorDataset(torch.arange(6.0).reshape(3, 2), torch.ones(3, 1)),
            batch_size=1,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(1, 2), torch.ones(1, 1)))


def test_full_resume_retains_state_clock_and_matched_random_streams(tmp_path):
    L.seed_everything(42, workers=True)
    original_dir = tmp_path / "original"
    source_saver = ModelCheckpoint(
        dirpath=original_dir,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        monitor="val_loss",
        save_top_k=-1,
        save_last=True,
    )
    source = _Probe()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        max_time="00:00:01:00",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[EMACallback(decay=0.999), source_saver],
    )
    trainer.fit(source, datamodule=_ProbeData())
    checkpoint = torch.load(source_saver.last_model_path, weights_only=False)
    before = copy.deepcopy(checkpoint)
    original_files = {p.name: p.read_bytes() for p in original_dir.iterdir()}
    assert checkpoint["global_step"] == 6
    # Prepare a new destination's metadata only. Preserve training tensors,
    # counters and cosine state rather than creating a weights-only warm start.
    checkpoint["callbacks"] = {
        key: value
        for key, value in checkpoint["callbacks"].items()
        if "EMACallback" in key or key == "Timer"
    }
    # Model the actual source's old elapsed timer exceeding this short branch
    # budget. The existing explicit timer-reset path must prevent an early stop.
    checkpoint["callbacks"]["Timer"]["time_elapsed"]["train"] = 120.0
    derived = tmp_path / "prepared.ckpt"
    torch.save(checkpoint, derived)
    for key in (
        "state_dict",
        "ema_state_dict",
        "optimizer_states",
        "lr_schedulers",
        "loops",
    ):
        _assert_equal(checkpoint[key], before[key])

    traces = []
    for index, weight in enumerate((0, 5e-4, 2e-3)):
        L.seed_everything(42, workers=True)
        model = _Probe(weight=weight)
        cfg = _compose(overrides=[f"resume_from_checkpoint={derived}"])
        del cfg.hydra
        cfg.logging.wandb.enabled = False
        cfg.trainer = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 8,
                "max_steps": 12,
                "max_time": "00:00:01:00",
                "num_sanity_val_steps": 0,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "callbacks": [
                    {
                        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "every_n_epochs": 1,
                        "save_on_train_epoch_end": False,
                        "monitor": "val_loss",
                        "save_top_k": 1,
                        "save_last": True,
                    },
                    {"_target_": "autocast.callbacks.ema.EMACallback", "decay": 0.999},
                ],
            }
        )
        destination = tmp_path / f"arm-{index}"
        destination.mkdir()
        run_training(cfg, model, _ProbeData(), destination, skip_test=True)
        assert model.started is not None
        assert model.started["global_step"] == 6
        _assert_equal(model.started["weights"], before["state_dict"])
        _assert_equal(model.started["optimizer"], before["optimizer_states"][0])
        _assert_equal(model.started["scheduler"], before["lr_schedulers"][0])
        _assert_equal(model.started["ema"], before["callbacks"]["EMACallback"])
        assert [row[0] for row in model.trace] == list(range(6, 12))
        assert [row[2] for row in model.trace] == [0, 1, 2, 0, 1, 2]
        assert [row[1] for row in model.trace] == [2, 2, 2, 3, 3, 3]
        assert model.val_steps == [9, 12]
        for row in model.trace:
            expected_lr = 2e-4 * 0.5 * (1 + math.cos(math.pi * row[1] / 8))
            assert row[5] == pytest.approx(expected_lr)
        final = torch.load(destination / cfg.output.checkpoint_name, weights_only=False)
        assert final["global_step"] == 12
        assert (
            int(final["callbacks"]["EMACallback"]["ema_model_state"]["n_averaged"])
            == 12
        )
        traces.append(model.trace)
    _assert_equal(traces[0], traces[1])
    _assert_equal(traces[1], traces[2])
    assert original_files == {p.name: p.read_bytes() for p in original_dir.iterdir()}
