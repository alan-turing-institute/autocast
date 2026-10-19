"""Matched-update residual FM settings and synthetic cached-latent checks."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DistributedSampler, TensorDataset

from autocast.callbacks.residual_statistics import ResidualStatisticsCallback
from autocast.models.processor import ProcessorModel
from autocast.nn.noise.source import IIDGaussianSource, ZeroSource
from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.processors.residual_normalization import ResidualNormalizer
from autocast.processors.residual_reference import LastFrameReference
from autocast.scripts.setup import (
    setup_datamodule,
    setup_epd_model,
    setup_processor_model,
)
from autocast.scripts.training import _ensure_residual_statistics_callback, run_training
from autocast.types import Batch

ROOT = Path(__file__).resolve().parents[2]
PRESET = "flow_matching_residual_12block_one_output_128k"
AMBIENT_PRESET = "flow_matching_residual_ambient_12block_one_output_128k"
CRPS_PRESET = "afcrps_global256_residual_one_output_12block_256ep_8h"


def _config(name=PRESET, *, distributed=False):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
    ]
    if distributed:
        overrides += [
            "+distributed=ddp_4gpu_slurm",
            "trainer.max_time=00:23:30:00",
            "++hydra.launcher.nodes=1",
            "++hydra.launcher.cpus_per_task=72",
            "hydra.launcher.timeout_min=1440",
            "logging.wandb.enabled=true",
            "logging.wandb.mode=online",
        ]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="processor" if name == PRESET else "encoder_processor_decoder",
            overrides=overrides,
            return_hydra_config=distributed,
        )


@pytest.mark.parametrize("distributed", [False, True])
def test_matched_residual_fm_settings(distributed):
    cfg = _config(distributed=distributed)
    assert OmegaConf.is_missing(cfg.datamodule, "data_path")
    assert cfg.datamodule.batch_size == 256
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert cfg.trainer.max_steps == cfg.optimizer.cosine_steps == 128000
    assert cfg.trainer.max_epochs == 1000000
    assert cfg.trainer.max_time == "00:23:30:00"
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.val_check_interval == 500
    assert cfg.trainer.check_val_every_n_epoch is None
    assert cfg.trainer.limit_train_batches == cfg.trainer.limit_val_batches == 1.0
    assert cfg.trainer.gradient_clip_val is None
    assert cfg.optimizer.scheduler == "cosine"
    assert cfg.optimizer.scheduler_interval == "step"
    assert cfg.optimizer.cosine_epochs is None
    assert cfg.optimizer.learning_rate == 1e-4
    assert cfg.optimizer.warmup == cfg.optimizer.min_lr_ratio == 0
    assert cfg.optimizer.betas == [0.9, 0.99]
    assert cfg.optimizer.weight_decay == 0
    assert cfg.model.n_members == 1
    assert cfg.model.loss_func is None
    assert cfg.model.train_metrics == cfg.model.val_metrics == []
    assert not OmegaConf.select(cfg.model, "residual_prediction", default=False)
    assert cfg.model.processor.flow_ode_steps == 50
    assert cfg.model.processor.integrator == "euler"
    assert cfg.model.processor.source._target_.endswith("IIDGaussianSource")
    assert cfg.model.processor.reference._target_.endswith("LastFrameReference")
    assert cfg.model.processor.residual_normalizer.granularity == "lead_channel"
    assert "standardizer" not in cfg.model.processor
    assert not cfg.model.processor.backbone.include_global_cond
    assert cfg.model.processor.backbone.include_time_embedding
    assert OmegaConf.select(cfg, "resume_from_checkpoint") is None
    assert cfg.output.skip_test
    if distributed:
        assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
        assert cfg.trainer.num_nodes == cfg.hydra.launcher.nodes == 1
        assert cfg.hydra.launcher.tasks_per_node == 4
        assert cfg.hydra.launcher.cpus_per_task == 72
        assert cfg.hydra.launcher.timeout_min == 1440
        assert cfg.logging.wandb.enabled
        assert cfg.logging.wandb.mode == "online"
    else:
        assert not cfg.logging.wandb.enabled


def test_residual_fm_backbone_settings():
    backbone = _config().model.processor.backbone
    assert backbone.hid_channels == 224
    assert backbone.hid_blocks == 12
    assert backbone.attention_heads == 8
    assert backbone.mod_features == 256
    assert backbone.ffn_factor == 4
    assert backbone.patch_size == 1
    assert backbone.dropout == 0.0
    assert backbone.qk_norm
    assert backbone.rpb
    assert not backbone.rope
    assert backbone.temporal_method == "none"


def test_backbone_batch_matches_crps_and_update_budget_is_not_epoch_limited():
    fm, crps = _config(), _config(CRPS_PRESET)
    windows = TensorDataset(torch.empty(200 * (321 - 1)))
    per_rank = len(DistributedSampler(windows, num_replicas=4, rank=0))
    crps_steps_per_epoch = (per_rank + crps.datamodule.batch_size - 1) // 32
    fm_steps_per_epoch = (per_rank + fm.datamodule.batch_size - 1) // 256
    assert fm.datamodule.batch_size == crps.datamodule.batch_size * crps.model.n_members
    assert crps_steps_per_epoch == 500
    assert fm_steps_per_epoch == 63
    assert fm.trainer.max_steps == crps_steps_per_epoch * crps.trainer.max_epochs
    assert fm.trainer.max_epochs * fm_steps_per_epoch > fm.trainer.max_steps


@pytest.mark.parametrize("distributed", [False, True])
def test_ambient_residual_fm_matches_latent_budget_and_backbone(distributed):
    cfg = _config(AMBIENT_PRESET, distributed=distributed)
    latent = _config(distributed=distributed)
    assert OmegaConf.to_container(
        cfg.optimizer, resolve=True
    ) == OmegaConf.to_container(latent.optimizer, resolve=True)
    assert OmegaConf.to_container(cfg.trainer, resolve=True) == OmegaConf.to_container(
        latent.trainer, resolve=True
    )
    expected_processor = OmegaConf.to_container(latent.model.processor, resolve=True)
    assert isinstance(expected_processor, dict)
    expected_processor["backbone"]["patch_size"] = 4
    assert (
        OmegaConf.to_container(cfg.model.processor, resolve=True) == expected_processor
    )
    assert cfg.trainer.max_steps == cfg.optimizer.cosine_steps == 128000
    assert cfg.optimizer.scheduler_interval == "step"
    assert cfg.datamodule.batch_size == 256
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert cfg.datamodule.start_frame == 0
    assert "shallow_water2d_stochastic_64_b43bd32" in cfg.datamodule.data_path
    assert cfg.datamodule.normalization_path == f"{cfg.datamodule.data_path}/stats.yml"
    assert cfg.datamodule.use_normalization
    assert cfg.autoencoder_checkpoint is None
    assert cfg.model.encoder._target_.endswith("IdentityEncoder")
    assert cfg.model.decoder._target_.endswith("IdentityDecoder")
    assert cfg.model.train_in_latent_space
    assert not cfg.model.residual_prediction
    assert cfg.model.freeze_encoder_decoder
    assert cfg.model.n_members == 1
    assert cfg.model.loss_func is None
    assert cfg.model.train_metrics == cfg.model.val_metrics == []
    assert (64 // cfg.model.processor.backbone.patch_size) ** 2 == 16**2
    if distributed:
        assert cfg.trainer.devices == 4
        assert cfg.logging.wandb.enabled
        assert cfg.logging.wandb.mode == "online"


def test_ambient_residual_fm_gradients_and_zero_correction_persistence():
    cfg = _config(AMBIENT_PRESET)
    batch = Batch(
        input_fields=torch.randn(2, 1, 64, 64, 3),
        output_fields=torch.randn(2, 1, 64, 64, 3),
        constant_scalars=torch.full((2, 1), 0.1),
        constant_fields=None,
    )
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": batch,
    }
    datamodule = SimpleNamespace(train_dataset=SimpleNamespace(norm=None))
    model = setup_epd_model(cfg, stats, cast(Any, datamodule))
    processor = cast(ResidualFlowMatchingProcessor, model.processor)
    encoded = model.encoder_decoder.encoder.encode_batch(batch)
    residual = processor.target_residual(encoded)
    torch.testing.assert_close(residual, batch.output_fields - batch.input_fields)
    assert not model.residual_prediction
    assert not any(p.requires_grad for p in model.encoder_decoder.parameters())
    normalizer = processor.residual_normalizer
    assert normalizer is not None
    values = rearrange(residual, "b t ... c -> t (b ...) c")
    scale, mean = torch.std_mean(values, dim=1, correction=0)
    normalizer.set_statistics(mean=mean, scale=scale)
    loss, prediction = model.loss(batch)
    assert prediction is None
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in processor.parameters()
        if p.requires_grad
    )
    blocks = processor.flow_matching_model.get_submodule("vit.blocks")
    assert sum(p.numel() for p in blocks.parameters()) == 10109376
    scheduler_config = cast(dict[str, Any], model.configure_optimizers())[
        "lr_scheduler"
    ]
    assert scheduler_config["interval"] == "step"
    multiplier = scheduler_config["scheduler"].lr_lambdas[0]
    assert multiplier(0) == 1
    assert multiplier(64000) == pytest.approx(0.5)
    assert multiplier(128000) == 0

    # Zero residual in identity-codec coordinates is exact persistence.
    processor.source = ZeroSource()
    processor.flow_ode_steps = 1
    with torch.no_grad():
        for parameter in processor.parameters():
            parameter.zero_()
        normalizer.set_statistics(mean=torch.zeros(1, 3), scale=torch.ones(1, 3))
        forecast = model(batch)
    torch.testing.assert_close(forecast, batch.input_fields)


@pytest.fixture
def synthetic_cache(tmp_path):
    # Hold-out increments are deliberately different; only train may fit stats.
    with torch.random.fork_rng():
        torch.manual_seed(7)
        for split, scale in [("train", 0.1), ("valid", 9.0), ("test", 19.0)]:
            directory = tmp_path / split
            directory.mkdir()
            fields = torch.randn(5, 16, 16, 8).cumsum(dim=0) * scale
            torch.save(
                {"encoded_fields": fields, "global_cond": torch.tensor([0.1])},
                directory / "traj_000000.pt",
            )
    return tmp_path


def _setup_cached_model(cache_path, *, small=False):
    cfg = _config()
    cfg.datamodule.data_path = str(cache_path)
    if small:
        cfg.model.processor.backbone.hid_channels = 16
        cfg.model.processor.backbone.hid_blocks = 1
        cfg.model.processor.backbone.attention_heads = 2
    datamodule, cfg, stats = setup_datamodule(cfg)
    return cfg, datamodule, setup_processor_model(cfg, stats, datamodule)


def _residual_normalizer(model: ProcessorModel) -> ResidualNormalizer:
    assert isinstance(model.processor, ResidualFlowMatchingProcessor)
    residual_normalizer = model.processor.residual_normalizer
    assert residual_normalizer is not None
    return residual_normalizer


def test_full_capacity_matches_crps_blocks_and_has_finite_gradients(synthetic_cache):
    _, datamodule, model = _setup_cached_model(synthetic_cache)
    processor = model.processor
    assert isinstance(processor, ResidualFlowMatchingProcessor)
    assert isinstance(processor.reference, LastFrameReference)
    assert isinstance(processor.source, IIDGaussianSource)
    assert type(model) is ProcessorModel
    assert model.loss_func is model.train_metrics is model.val_metrics is None
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 10346920

    crps_cfg = _config(CRPS_PRESET)
    with torch.device("meta"):
        crps = instantiate(
            crps_cfg.model.processor,
            in_channels=3,
            out_channels=3,
            spatial_resolution=[64, 64],
        )
    fm_blocks = processor.flow_matching_model.get_submodule("vit.blocks")
    crps_blocks = crps.model.vit.get_submodule("blocks")
    assert {name: p.shape for name, p in fm_blocks.named_parameters()} == {
        name: p.shape for name, p in crps_blocks.named_parameters()
    }
    assert sum(p.numel() for p in fm_blocks.parameters()) == 10109376
    assert sum(p.numel() for p in crps.parameters() if p.requires_grad) == 10231504

    batch = next(iter(datamodule.train_dataloader()))
    residual = processor.target_residual(batch)
    values = rearrange(residual, "b t ... c -> t (b ...) c")
    scale, mean = torch.std_mean(values, dim=1, correction=0)
    _residual_normalizer(model).set_statistics(mean=mean, scale=scale)
    loss = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    assert residual.shape == batch.encoded_output_fields.shape == (4, 1, 16, 16, 8)


def test_step_cosine_endpoints_and_checkpoint_policy(synthetic_cache):
    cfg, _, model = _setup_cached_model(synthetic_cache, small=True)
    optimizers = cast(dict[str, Any], model.configure_optimizers())
    scheduler_config = optimizers["lr_scheduler"]
    scheduler = scheduler_config["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert scheduler_config["interval"] == "step"
    assert scheduler_config["frequency"] == 1
    multiplier = scheduler.lr_lambdas[0]
    assert multiplier(0) == 1
    assert multiplier(64000) == pytest.approx(0.5)
    assert multiplier(128000) == 0
    snapshot, best, *_ = [instantiate(callback) for callback in cfg.trainer.callbacks]
    assert isinstance(snapshot, ModelCheckpoint)
    assert snapshot._every_n_train_steps == 8000
    assert snapshot.save_last
    assert snapshot.save_top_k == -1
    assert best.monitor == "val_loss"
    assert best.save_top_k == 1
    assert best._every_n_epochs == 1
    assert best._save_on_train_epoch_end is False
    callbacks = _ensure_residual_statistics_callback([], model)
    assert isinstance(callbacks[0], ResidualStatisticsCallback)


def test_cached_fit_sanity_validation_and_full_checkpoint_resume(
    synthetic_cache, tmp_path, monkeypatch
):
    cfg, datamodule, model = _setup_cached_model(synthetic_cache, small=True)
    cfg.trainer.max_steps = 2
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.val_check_interval = 1
    cfg.optimizer.cosine_steps = 128000  # Stop the smoke early, not its LR schedule.
    with open_dict(cfg.trainer):
        cfg.trainer.enable_progress_bar = False
        cfg.trainer.enable_model_summary = False
    work_dir = tmp_path / "fit"
    work_dir.mkdir()
    run_training(cfg, model, datamodule, work_dir, skip_test=True)

    checkpoints = list(work_dir.rglob("best-val-*.ckpt"))
    assert len(checkpoints) == 1
    checkpoint = work_dir / "processor.ckpt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 2
    assert payload["optimizer_states"]
    assert payload["lr_schedulers"]
    residual_normalizer = _residual_normalizer(model)
    assert residual_normalizer.fitted
    train_fields = torch.load(
        synthetic_cache / "train/traj_000000.pt", weights_only=False
    )["encoded_fields"]
    increments = train_fields[1:] - train_fields[:-1]
    scale, mean = torch.std_mean(
        rearrange(increments, "t ... c -> 1 (t ...) c"), dim=1, correction=0
    )
    torch.testing.assert_close(residual_normalizer.mean, mean)
    torch.testing.assert_close(residual_normalizer.scale, scale)

    resumed_cfg, _, restored = _setup_cached_model(synthetic_cache, small=True)
    resumed_cfg.trainer = cfg.trainer.copy()
    resumed_cfg.trainer.max_steps = 3
    with open_dict(resumed_cfg):
        resumed_cfg.resume_from_checkpoint = str(checkpoint)

    # Restored buffers must skip the statistics pass, preserving full-state resume.
    def fail_if_refitted(*args, **kwargs):
        pytest.fail("Residual statistics were refitted after checkpoint restoration")

    monkeypatch.setattr(ResidualStatisticsCallback, "_fit_statistics", fail_if_refitted)
    run_training(resumed_cfg, restored, datamodule, work_dir, skip_test=True)
    resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert resumed["global_step"] == 3
    assert resumed["lr_schedulers"][0]["last_epoch"] == 3
    assert "ema_state_dict" in resumed
    torch.testing.assert_close(_residual_normalizer(restored).mean, mean)
    torch.testing.assert_close(_residual_normalizer(restored).scale, scale)
