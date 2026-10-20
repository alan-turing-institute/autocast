"""Pin the warm-start control and prove weights-only means a fresh optimizer."""

from pathlib import Path

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
PRESET = "crps_fft_anomaly_spatial1d_finetune_5k"
LONGER_PRESET = "crps_fft_anomaly_spatial1d_finetune_15k"
PARENT = "crps_fft_anomaly_spatial1d_residual_one_output_12block_256ep_8h"


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


@pytest.mark.parametrize("preset", [PRESET, LONGER_PRESET])
def test_finetune_preserves_model_task_and_calibrated_auxiliary(preset):
    baseline, cfg = _compose(PARENT), _compose(preset)
    assert cfg.model == baseline.model
    assert cfg.datamodule == baseline.datamodule
    assert cfg.model.loss_func.weights == [1.0, 0.1, 0.12492216134130599]
    assert cfg.model.processor.noise_mode == "spatial"
    assert cfg.model.processor.n_noise_input_channels == 1
    assert cfg.model.n_members == 8
    assert cfg.datamodule.batch_size == 32
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert cfg.model.loss_func.losses[2].power_scale == 1
    for key in ("optimizer", "betas", "weight_decay"):
        assert cfg.optimizer[key] == baseline.optimizer[key]


@pytest.mark.parametrize(("preset", "lr"), [(PRESET, 2e-5), (LONGER_PRESET, 5e-5)])
def test_checkpoint_is_required_and_optimizer_schedule_is_fresh_constant(preset, lr):
    cfg = _compose(preset)
    assert OmegaConf.is_missing(cfg, "resume_from_checkpoint")
    assert cfg.resume_weights_only
    assert not cfg.reset_resume_time_budget
    assert cfg.optimizer.learning_rate == lr
    assert cfg.optimizer.scheduler is None
    assert cfg.optimizer.cosine_epochs is cfg.optimizer.cosine_steps is None
    assert cfg.optimizer.warmup == 0
    assert cfg.logging.wandb.mode == "offline"
    assert cfg.logging.wandb.enabled
    assert not cfg.logging.wandb.log_model
    assert cfg.logging.wandb.id is cfg.logging.wandb.resume is None
    assert cfg.output.skip_test


@pytest.mark.parametrize(
    ("preset", "updates", "minutes", "cap"),
    [
        (PRESET, 5000, 30, "00:00:25:00"),
        (LONGER_PRESET, 15000, 60, "00:00:55:00"),
    ],
)
def test_ddp_budget_has_frequent_validation_and_checkpoints(
    preset, updates, minutes, cap
):
    cfg = _compose(
        preset, overrides=["+distributed=ddp_4gpu_slurm", f"trainer.max_time={cap}"]
    )
    assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
    assert cfg.trainer.max_steps == updates
    assert cfg.trainer.max_epochs > updates
    assert cfg.trainer.max_time == cap
    assert cfg.hydra.launcher.timeout_min == minutes
    assert cfg.trainer.check_val_every_n_epoch is None
    assert cfg.trainer.val_check_interval == 500
    snapshot, best = cfg.trainer.callbacks[:2]
    assert snapshot.every_n_train_steps == 500
    assert snapshot.save_last
    assert snapshot.save_top_k == -1
    assert best.monitor == "val_loss"
    assert best.save_top_k == 1


class _WarmStartProbe(OptimizerMixin, L.LightningModule):
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
        assert self.global_step == 0
        assert self.current_epoch == 0
        assert not self.trainer.optimizers[0].state
        assert not self.trainer.lr_scheduler_configs
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


@pytest.mark.parametrize(("preset", "lr"), [(PRESET, 2e-5), (LONGER_PRESET, 5e-5)])
def test_weights_only_entrypoint_does_not_restore_training_state(tmp_path, preset, lr):
    cfg = _compose(
        preset, overrides=[f"resume_from_checkpoint={tmp_path / 'source.ckpt'}"]
    )
    # Hydra runtime fields belong to the launcher, not the training payload.
    del cfg.hydra
    source = _WarmStartProbe(cfg.optimizer)
    source.normalizer_probe.fill_(3)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    source.layer(torch.ones(2, 2)).square().mean().backward()
    source_optimizer.step()
    assert source_optimizer.state
    expected = {name: value.clone() for name, value in source.state_dict().items()}
    torch.save(
        {
            "state_dict": source.state_dict(),
            "optimizer_states": [source_optimizer.state_dict()],
            "lr_schedulers": [{"last_epoch": 36000}],
            "global_step": 36000,
            "epoch": 71,
        },
        cfg.resume_from_checkpoint,
    )
    # Exercise the real training entrypoint on a two-update CPU toy. Only the
    # test trainer/data are tiny; the production preset remains unchanged.
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
    target = _WarmStartProbe(cfg.optimizer, expected)
    work_dir = tmp_path / "fit"
    work_dir.mkdir()
    run_training(cfg, target, _ProbeData(), work_dir, skip_test=True)
    assert target.start_checked
    assert target.global_step == 2
    assert target.seen_lrs == [lr, lr]
    assert all(
        int(state["step"]) == 2 for state in target.trainer.optimizers[0].state.values()
    )
    for parameter in target.parameters():
        assert torch.isfinite(parameter).all()
