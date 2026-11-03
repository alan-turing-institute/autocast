"""Check the one-step controls before adding matched rollout-training support."""

import math
from pathlib import Path
from typing import Any, cast

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.scripts.training import run_training

ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    "crps_fft_global256_finetune_32k": (
        "crps_fft_global256_residual_one_output_12block_256ep_8h"
    ),
    "crps_fft_spatial1d_finetune_32k": (
        "crps_fft_spatial1d_residual_one_output_12block_256ep_8h"
    ),
}


def _compose(name, overrides=()):
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


@pytest.mark.parametrize("name", PRESETS)
def test_control_preserves_checkpoint_architecture_task_and_loss(name):
    cfg, source = _compose(name), _compose(PRESETS[name])
    assert cfg.model == source.model
    assert cfg.datamodule == source.datamodule
    assert cfg.model.loss_func.weights == [1.0, 0.1]
    assert cfg.model.n_members == 8
    assert cfg.datamodule.batch_size == 32
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert not cfg.model.processor.include_global_cond
    for key in ("optimizer", "betas", "weight_decay"):
        assert cfg.optimizer[key] == source.optimizer[key]


@pytest.mark.parametrize("name", PRESETS)
def test_control_requires_weights_and_has_new_step_budget(name):
    cfg = _compose(
        name, ["+distributed=ddp_4gpu_slurm", "trainer.max_time=00:07:45:00"]
    )
    assert OmegaConf.is_missing(cfg, "resume_from_checkpoint")
    assert cfg.resume_weights_only
    assert not cfg.reset_resume_time_budget
    assert cfg.trainer.max_steps == cfg.optimizer.cosine_steps == 32000
    assert cfg.trainer.max_epochs > cfg.trainer.max_steps
    assert cfg.optimizer.cosine_epochs is None
    assert cfg.optimizer.scheduler_interval == "step"
    assert cfg.optimizer.scheduler == "cosine"
    assert cfg.optimizer.warmup == 100
    assert cfg.optimizer.learning_rate == 2e-5
    assert cfg.optimizer.min_lr_ratio == 0
    assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
    assert cfg.trainer.max_time == "00:07:45:00"
    assert cfg.hydra.launcher.timeout_min == 480
    assert cfg.trainer.check_val_every_n_epoch is None
    assert cfg.trainer.val_check_interval == 500
    snapshot, best = cfg.trainer.callbacks[:2]
    assert snapshot.every_n_train_steps == 8000
    assert snapshot.save_top_k == -1
    assert snapshot.save_last
    assert best.monitor == "val_loss"
    assert best.save_top_k == 1
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "offline"
    assert not cfg.logging.wandb.log_model
    assert cfg.logging.wandb.id is cfg.logging.wandb.resume is None
    assert cfg.output.skip_test


class _ControlProbe(OptimizerMixin, L.LightningModule):
    normalizer_probe: torch.Tensor

    def __init__(self, optimizer_config, expected=None):
        super().__init__()
        self.layer = nn.Linear(2, 1)
        self.register_buffer("normalizer_probe", torch.tensor(1.0))
        self.optimizer_config = optimizer_config
        self.expected = expected
        self.start_checked = False
        self.seen_lrs = []

    def on_fit_start(self):
        assert self.global_step == self.current_epoch == 0
        assert not self.trainer.optimizers[0].state
        scheduler = self.trainer.lr_scheduler_configs[0]
        assert scheduler.interval == "step"
        assert scheduler.scheduler.last_epoch == 0
        if self.expected is not None:
            for name, value in self.state_dict().items():
                torch.testing.assert_close(value, self.expected[name])
        self.start_checked = True

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        self.seen_lrs.append(self.trainer.optimizers[0].param_groups[0]["lr"])
        x, y = batch
        return nn.functional.mse_loss(self.layer(x), y)


class _ProbeData(L.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            TensorDataset(torch.ones(4, 2), torch.ones(4, 1)), batch_size=2
        )


@pytest.mark.parametrize("name", PRESETS)
def test_control_schedule_uses_existing_warmup_and_reaches_zero(name):
    model = _ControlProbe(_compose(name).optimizer)
    configured = model.configure_optimizers()
    assert isinstance(configured, dict)
    config_dict = cast("dict[str, Any]", configured)
    optimizer = config_dict["optimizer"]
    scheduler = config_dict["lr_scheduler"]["scheduler"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert scheduler.get_last_lr()[0] == pytest.approx(2e-5 / 101)
    selected = {}
    # Exercise the actual scheduler at the production horizon without training.
    for step in range(1, 32001):
        optimizer.step()
        scheduler.step()
        if step in {50, 100, 16000, 32000}:
            selected[step] = scheduler.get_last_lr()[0]
    assert selected[50] < selected[100]
    assert selected[100] == pytest.approx(
        2e-5 * (1 + math.cos(math.pi * 100 / 32000)) / 2
    )
    assert selected[16000] == pytest.approx(1e-5)
    assert selected[32000] == 0


@pytest.mark.parametrize("name", PRESETS)
def test_weights_only_control_resets_optimizer_and_scheduler(tmp_path, name):
    cfg = _compose(name, [f"resume_from_checkpoint={tmp_path / 'source.ckpt'}"])
    del cfg.hydra
    # Resolve the production horizon before shortening only the CPU toy fit.
    cfg.optimizer = OmegaConf.create(
        OmegaConf.to_container(cfg.optimizer, resolve=True)
    )
    source = _ControlProbe(cfg.optimizer)
    source.normalizer_probe.fill_(3)
    old_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    source.layer(torch.ones(2, 2)).square().mean().backward()
    old_optimizer.step()
    expected = {key: value.clone() for key, value in source.state_dict().items()}
    torch.save(
        {
            "state_dict": source.state_dict(),
            "optimizer_states": [old_optimizer.state_dict()],
            "lr_schedulers": [{"last_epoch": 36000}],
            "global_step": 36000,
            "epoch": 71,
        },
        cfg.resume_from_checkpoint,
    )
    cfg.trainer = OmegaConf.create(
        {
            "_target_": "lightning.pytorch.Trainer",
            "accelerator": "cpu",
            "devices": 1,
            "max_steps": 2,
            "max_epochs": 1,
            "max_time": "00:00:01:00",
            "limit_val_batches": 0,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "callbacks": [],
        }
    )
    cfg.logging.wandb.enabled = False
    target = _ControlProbe(cfg.optimizer, expected)
    destination = tmp_path / "fit"
    destination.mkdir()
    run_training(cfg, target, _ProbeData(), destination, skip_test=True)
    assert target.start_checked
    assert target.global_step == 2
    assert target.seen_lrs[0] == pytest.approx(2e-5 / 101)
    assert target.seen_lrs[1] > target.seen_lrs[0]
    assert target.trainer.lr_scheduler_configs[0].scheduler.last_epoch == 2
    assert all(
        int(state["step"]) == 2 for state in target.trainer.optimizers[0].state.values()
    )
    assert all(torch.isfinite(parameter).all() for parameter in target.parameters())
