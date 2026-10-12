"""Composition and full-state LR checks; no real data or GPU allocation."""

import copy
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.types import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPERS = REPO_ROOT / "scripts/swe64s"


def _config(name, *, four_gpu=False):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{REPO_ROOT / 'local_hydra'}]",
    ]
    is_ae = name == "autoencoder_refine"
    if is_ae:
        overrides.append("resume_from_checkpoint=/preflight/derived.ckpt")
    if four_gpu:
        overrides += [
            "+distributed=ddp_4gpu_slurm",
            f"trainer.max_time={'00:00:30:00' if is_ae else '00:02:00:00'}",
        ]
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="autoencoder" if is_ae else "encoder_processor_decoder",
            overrides=overrides,
        )


def test_short_afcrps_pair_changes_capacity_only():
    large, small = (_config(f"afcrps_{size}_2h") for size in ("large", "small"))
    small_model = OmegaConf.to_container(small.model)
    assert isinstance(small_model, dict)
    small_model["processor"].update(hidden_dim=568, n_layers=12)
    assert small_model == OmegaConf.to_container(large.model)
    assert OmegaConf.to_container(small.optimizer) == OmegaConf.to_container(
        large.optimizer
    )
    assert OmegaConf.to_container(small.datamodule) == OmegaConf.to_container(
        large.datamodule
    )
    assert OmegaConf.to_container(small.trainer) == OmegaConf.to_container(
        large.trainer
    )
    for cfg, expected_count in ((large, 80849688), (small, 10694272)):
        with torch.device("meta"):
            processor = instantiate(
                cfg.model.processor,
                in_channels=3,
                out_channels=12,
                spatial_resolution=[64, 64],
            )
        assert sum(p.numel() for p in processor.parameters()) == expected_count
        assert cfg.optimizer.scheduler_interval == "time"
        assert cfg.optimizer.learning_rate == 2e-4
        assert cfg.trainer.max_steps == -1
        assert cfg.trainer.max_time == "00:02:00:00"
        assert cfg.trainer.limit_val_batches == 1.0
        assert any(
            cb._target_.endswith("GradNormCallback") for cb in cfg.trainer.callbacks
        )


@pytest.mark.parametrize(
    "name",
    [
        "afcrps_large_2h",
        "afcrps_small_2h",
        "afcrps_spatial_small_2h",
        "autoencoder_refine",
    ],
)
def test_short_distributed_launch_preserves_explicit_cap(name):
    cfg = _config(name, four_gpu=True)
    assert cfg.trainer.devices == 4
    assert cfg.trainer.strategy == "ddp"
    assert cfg.trainer.max_time == (
        "00:00:30:00" if name == "autoencoder_refine" else "00:02:00:00"
    )


def test_spatial_small_profile_changes_noise_and_scheduler_only():
    baseline = _config("afcrps_small_2h")
    spatial = _config("afcrps_spatial_small_2h")
    # Compare interpolation expressions, since log names include experiment_name.
    spatial_config = OmegaConf.to_container(spatial, resolve=False)
    baseline_config = OmegaConf.to_container(baseline, resolve=False)
    assert isinstance(spatial_config, dict)
    assert isinstance(baseline_config, dict)
    spatial_config["experiment_name"] = baseline_config["experiment_name"]
    spatial_config["model"]["processor"].update(
        noise_mode="global", n_noise_input_channels=1024
    )
    spatial_config["optimizer"]["scheduler"] = baseline_config["optimizer"]["scheduler"]
    assert spatial_config == baseline_config
    assert spatial.model.processor.noise_mode == "spatial"
    assert spatial.model.processor.n_noise_input_channels == 4
    assert spatial.model.processor.n_noise_channels == 1024
    assert spatial.optimizer.learning_rate == 2e-4
    assert spatial.optimizer.scheduler is None
    assert spatial.trainer.max_time == "00:02:00:00"
    assert OmegaConf.select(spatial, "resume_from_checkpoint") is None
    with torch.device("meta"):
        processor = instantiate(
            spatial.model.processor,
            in_channels=3,
            out_channels=12,
            spatial_resolution=[64, 64],
        )
    assert sum(p.numel() for p in processor.parameters()) == 10_699_392


def test_spatial_small_optimizer_keeps_constant_lr_without_scheduler():
    cfg = _config("afcrps_spatial_small_2h", four_gpu=True)
    model = OptimizerMixin()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    model.optimizer_config = cfg.optimizer
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    for _ in range(2):
        optimizer.zero_grad()
        model.get_parameter("weight").square().sum().backward()
        optimizer.step()
        assert all(group["lr"] == 2e-4 for group in optimizer.param_groups)
    assert any(cb.get("save_last", False) for cb in cfg.trainer.callbacks)


def test_spatial_small_four_frame_ensemble_training_and_rollout():
    cfg = _config("afcrps_spatial_small_2h")
    # Keep the real grid, patching, modulation and member count; shrink capacity.
    processor = instantiate(
        cfg.model.processor,
        in_channels=3,
        out_channels=12,
        spatial_resolution=[64, 64],
        hidden_dim=16,
        n_layers=1,
    )
    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=EncoderDecoder(
            encoder=instantiate(cfg.model.encoder, in_channels=3, n_steps_input=1),
            decoder=instantiate(cfg.model.decoder, output_channels=3, time_steps=4),
        ),
        processor=processor,
        n_members=cfg.model.n_members,
        loss_func=instantiate(cfg.model.loss_func),
        train_in_latent_space=cfg.model.train_in_latent_space,
    )
    batch = Batch(
        input_fields=torch.randn(1, 1, 64, 64, 3),
        output_fields=torch.randn(1, 4, 64, 64, 3),
        constant_scalars=torch.tensor([[0.1]]),
        constant_fields=None,
    )
    noise_draws = []
    assert processor.modulation_proj is not None
    handle = processor.modulation_proj.register_forward_pre_hook(
        lambda _module, args: noise_draws.append(args[0].detach().clone())
    )
    try:
        loss, prediction = model.loss(batch)
    finally:
        handle.remove()
    assert prediction is not None
    assert prediction.shape == (1, 4, 64, 64, 3, 8)
    assert noise_draws[0].shape == (8, 256, 4)
    assert not torch.equal(noise_draws[0][0], noise_draws[0][1])
    assert not torch.equal(noise_draws[0][:, 0], noise_draws[0][:, 1])
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in processor.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    batch.output_fields = torch.randn(1, 8, 64, 64, 3)
    model.eval()
    with torch.no_grad():
        for ratio in (0.0, 1.0):
            prediction, truth = model.rollout(
                batch,
                stride=4,
                max_rollout_steps=2,
                n_members=2,
                teacher_forcing_ratio=ratio,
                free_running_only=ratio == 0.0,
            )
            assert prediction.shape == (1, 8, 64, 64, 3, 2)
            assert torch.isfinite(prediction).all()
            torch.testing.assert_close(truth, batch.output_fields)


def test_ae_refinement_keeps_architecture_and_constant_lr():
    cfg = _config("autoencoder_refine")
    baseline = runpy.run_path(str(HELPERS / "check_inputs.py"))["check_configs"]()[
        "autoencoder"
    ]
    assert OmegaConf.to_container(cfg.model) == OmegaConf.to_container(baseline.model)
    assert cfg.reset_resume_time_budget
    assert cfg.optimizer.learning_rate == 3e-6
    assert cfg.optimizer.min_lr_ratio == 1.0
    assert cfg.optimizer.scheduler_interval == "epoch"
    assert cfg.optimizer.cosine_epochs == cfg.trainer.max_epochs == 512
    assert cfg.trainer.val_check_interval == 250
    assert cfg.trainer.check_val_every_n_epoch is None
    assert cfg.trainer.limit_val_batches == 1.0


def _checkpoint():
    return {
        "epoch": 28,
        "global_step": 29116,
        "state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer_states": [
            {
                "state": {0: {"moment": torch.ones(2), "step": 29116}},
                "param_groups": [
                    {
                        "params": [0],
                        "lr": 1e-5,
                        "initial_lr": 1e-5,
                        "betas": (0.9, 0.99),
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                    }
                ],
            }
        ],
        "lr_schedulers": [
            {
                "base_lrs": [1e-5],
                "_last_lr": [1e-5],
                "last_epoch": 28,
                "lr_lambdas": [None],
            }
        ],
        "hyper_parameters": {
            "optimizer_config": {"scheduler": "cosine", "learning_rate": 1e-5}
        },
        "callbacks": {
            "ModelCheckpoint{'path': 'old'}": {"best_model_path": "old.ckpt"},
            "ValidationMetricPlotCallback": {},
            "TrainingTimerCallback": {},
            "EMACallback": {"weight": torch.tensor([3.0])},
            "Timer": {"time_elapsed": {"train": 1000.0}},
        },
    }


@pytest.mark.parametrize("saved_optimizer_config", [True, False])
def test_lr_retarget_preserves_state_and_restores_constant_schedule(
    tmp_path, saved_optimizer_config
):
    helpers = runpy.run_path(str(HELPERS / "prepare_ae_refinement.py"))
    source, target = tmp_path / "source.ckpt", tmp_path / "derived.ckpt"
    original = _checkpoint()
    source_config = None
    if not saved_optimizer_config:
        # Real AE checkpoints do not save optimizer_config as a hyperparameter.
        original.pop("hyper_parameters")
        source_config = tmp_path / "resolved_autoencoder_config.yaml"
        OmegaConf.save(
            OmegaConf.create(
                {
                    "optimizer": {
                        "scheduler": "cosine",
                        "learning_rate": 1e-5,
                        "scheduler_interval": "epoch",
                        "cosine_epochs": 512,
                    }
                }
            ),
            source_config,
        )
    torch.save(original, source)
    source_hash = helpers["sha256"](source)
    kwargs = {"source_config": source_config} if source_config is not None else {}
    record = helpers["prepare_checkpoint"](
        source, target, expected_sha256=source_hash, **kwargs
    )
    derived = torch.load(target, weights_only=False)
    assert helpers["sha256"](source) == source_hash
    assert derived["global_step"] == original["global_step"]
    assert derived["epoch"] == original["epoch"]
    assert torch.equal(
        derived["state_dict"]["weight"], original["state_dict"]["weight"]
    )
    assert torch.equal(
        derived["optimizer_states"][0]["state"][0]["moment"], torch.ones(2)
    )
    assert derived["optimizer_states"][0]["state"][0]["step"] == 29116
    assert set(derived["callbacks"]) == {"EMACallback", "Timer"}
    assert record["learning_rate"] == 3e-6
    assert target.with_suffix(".provenance.json").is_file()
    if source_config is not None:
        assert "hyper_parameters" not in derived
        assert record["source_config"] == str(source_config.resolve())
        assert record["source_config_sha256"] == helpers["sha256"](source_config)

    # Exercise the actual scheduler restore rather than only its serialized LR.
    model = OptimizerMixin()
    cast(Any, model).trainer = SimpleNamespace(max_epochs=512)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=3e-6)
    optimizer_config = OmegaConf.to_container(_config("autoencoder_refine").optimizer)
    assert isinstance(optimizer_config, dict)
    scheduler = model._create_scheduler(
        optimizer, cast(dict[str, Any], optimizer_config)
    )
    optimizer.load_state_dict(derived["optimizer_states"][0])
    scheduler.load_state_dict(derived["lr_schedulers"][0])
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)
    with pytest.raises(FileExistsError):
        helpers["prepare_checkpoint"](source, target)
    with pytest.raises(ValueError, match="SHA256"):
        helpers["prepare_checkpoint"](
            source, tmp_path / "wrong.ckpt", expected_sha256="wrong"
        )


def test_lr_retarget_requires_explicit_config_when_metadata_is_missing(tmp_path):
    helper = runpy.run_path(str(HELPERS / "prepare_ae_refinement.py"))
    checkpoint = _checkpoint()
    checkpoint.pop("hyper_parameters")
    source, target = tmp_path / "source.ckpt", tmp_path / "derived.ckpt"
    torch.save(checkpoint, source)
    with pytest.raises(ValueError, match="source-config"):
        helper["prepare_checkpoint"](source, target)
    assert not target.exists()


def test_lr_retarget_rejects_config_from_a_different_base_lr(tmp_path):
    helper = runpy.run_path(str(HELPERS / "prepare_ae_refinement.py"))
    checkpoint = _checkpoint()
    checkpoint.pop("hyper_parameters")
    source, target = tmp_path / "source.ckpt", tmp_path / "derived.ckpt"
    torch.save(checkpoint, source)
    config = tmp_path / "wrong.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "optimizer": {
                    "scheduler": "cosine",
                    "learning_rate": 5e-5,
                    "scheduler_interval": "epoch",
                }
            }
        ),
        config,
    )
    with pytest.raises(ValueError, match="base learning rate"):
        helper["prepare_checkpoint"](source, target, source_config=config)
    assert not target.exists()


@pytest.mark.parametrize("learning_rate", [0, -1, float("nan"), float("inf")])
def test_lr_retarget_rejects_invalid_lr(learning_rate):
    helper = runpy.run_path(str(HELPERS / "prepare_ae_refinement.py"))[
        "retarget_checkpoint"
    ]
    with pytest.raises(ValueError, match="finite and positive"):
        helper(copy.deepcopy(_checkpoint()), learning_rate)
