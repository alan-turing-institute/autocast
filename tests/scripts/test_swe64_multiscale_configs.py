"""Both multiscale injection paths retain the SWE64 residual training protocol."""

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from the_well.data.normalization import ZScoreNormalization

from autocast.losses import AlphaFairCRPSLoss, WeightedSumLoss
from autocast.nn.noise import MultiscaleGaussianSource
from autocast.nn.noise.noise_injector import ConcatenatedNoiseInjector
from autocast.processors.azula_vit import AzulaViTProcessor
from autocast.scripts.setup import _build_loss_func, setup_epd_model
from autocast.scripts.workflow.slurm import (
    _build_sbatch_command,
    _resolve_launcher_submission_context,
)
from autocast.types import Batch

ROOT = Path(__file__).resolve().parents[2]
SUFFIX = "_residual_one_output_12block_256ep_8h"
PRESETS = {
    f"{loss}_multiscale_{mode}4d{SUFFIX}": (loss, mode)
    for loss in ("afcrps", "crps_fft")
    for mode in ("spatial", "concat")
}


def _config(name, *, distributed=False):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
    ]
    if distributed:
        overrides += [
            "+distributed=ddp_4gpu_slurm",
            "trainer.max_time=00:07:45:00",
            "++hydra.launcher.nodes=1",
            "++hydra.launcher.cpus_per_task=72",
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


def _noise_source(model: Any) -> MultiscaleGaussianSource:
    if model.input_noise_injector is not None:
        assert isinstance(model.input_noise_injector, ConcatenatedNoiseInjector)
        source = model.input_noise_injector.noise_source
        assert model.encoder_decoder.encoder.latent_channels == 7
    else:
        assert isinstance(model.processor, AzulaViTProcessor)
        source = model.processor.noise_source
        projection = model.processor.modulation_proj
        assert isinstance(projection, torch.nn.Linear)
        assert projection.in_features == 4
        assert projection.out_features == 256
    assert isinstance(source, MultiscaleGaussianSource)
    return source


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("distributed", [False, True])
def test_multiscale_protocol_source_scales_and_optional_wandb(name, distributed):
    cfg = _config(name, distributed=distributed)
    reference = _config(f"afcrps_global256{SUFFIX}", distributed=distributed)
    for key in ("datamodule", "trainer", "optimizer"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(
            reference[key], resolve=True
        )
    assert cfg.seed == 42
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert cfg.datamodule.batch_size == 32
    assert cfg.model.n_members == 8
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert not cfg.model.train_in_latent_space
    assert not cfg.model.encoder.with_constants
    assert not cfg.model.processor.include_global_cond
    assert cfg.autoencoder_checkpoint is None
    assert cfg.model.processor.hidden_dim == 224
    assert cfg.model.processor.n_layers == 12
    assert cfg.model.processor.num_heads == 8
    assert cfg.model.processor.patch_size == 4
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.scheduler == "cosine"
    assert cfg.optimizer.scheduler_interval == "epoch"
    assert cfg.optimizer.min_lr_ratio == cfg.optimizer.warmup == 0
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 256
    assert cfg.trainer.max_time == "00:07:45:00"
    assert cfg.logging.wandb.enabled == distributed
    if distributed:
        assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
        assert cfg.hydra.launcher.tasks_per_node == 4
        assert cfg.hydra.launcher.timeout_min == 480
        assert cfg.logging.wandb.mode == "online"


@pytest.mark.parametrize("name", PRESETS)
def test_consumer_specific_noise_grid_and_filter_scales(name):
    cfg = _config(name)
    _, mode = PRESETS[name]
    if mode == "spatial":
        assert cfg.model.processor.noise_mode == "spatial"
        assert cfg.model.processor.n_noise_input_channels == 4
        assert cfg.model.processor.n_noise_channels == 256
        assert cfg.model.get("input_noise_injector") is None
        source_cfg = cfg.model.processor.noise_source
        expected_cutoffs, grid = (1, 2, 4, 6), 16
    else:
        assert cfg.model.processor.noise_mode == "global"
        assert cfg.model.processor.n_noise_input_channels is None
        assert cfg.model.processor.n_noise_channels is None
        assert cfg.model.processor.noise_source is None
        assert cfg.model.input_noise_injector.n_channels == 4
        assert cfg.model.input_noise_injector.std == 1
        source_cfg = cfg.model.input_noise_injector.noise_source
        expected_cutoffs, grid = (2, 4, 8, 16), 64
    source = instantiate(source_cfg)
    assert isinstance(source, MultiscaleGaussianSource)
    assert source.half_power_wavenumbers == expected_cutoffs
    assert source.domain_lengths == (1, 1)
    sample = source.sample_like(torch.empty(2, 1, grid, grid, 4))
    assert sample.shape == (2, 1, grid, grid, 4)
    assert torch.isfinite(sample).all()


@pytest.mark.parametrize("name", PRESETS)
def test_loss_selection_changes_neither_source_nor_injection(name):
    cfg = _config(name)
    loss_name, mode = PRESETS[name]
    loss = _build_loss_func(cfg.model)
    if loss_name == "afcrps":
        assert isinstance(loss, AlphaFairCRPSLoss)
        assert loss.alpha == 0.95
    else:
        baseline = _config(f"afcrps_multiscale_{mode}4d{SUFFIX}")
        assert cfg.model.processor == baseline.model.processor
        assert cfg.model.get("input_noise_injector") == baseline.model.get(
            "input_noise_injector"
        )
        assert isinstance(loss, WeightedSumLoss)
        assert loss.weights == (1.0, 0.1)
        assert loss.requires_ambient_predictions


@pytest.mark.parametrize("name", PRESETS)
def test_both_paths_score_clean_reconstructed_states_and_roll_out(name):
    cfg = _config(name)
    cfg.model.processor.hidden_dim = 16
    cfg.model.processor.n_layers = 1
    cfg.model.processor.num_heads = 2
    norm = ZScoreNormalization(
        stats={
            "mean": {"h": 1.0, "u": 0.2, "v": -0.3},
            "std": {"h": 0.02, "u": 0.1, "v": 0.2},
            "mean_delta": {"h": 0.001, "u": 0.02, "v": -0.04},
            "std_delta": {"h": 0.003, "u": 0.04, "v": 0.06},
        },
        core_field_names=["h", "u", "v"],
        core_constant_field_names=[],
    )
    batch = Batch(
        input_fields=torch.randn(1, 1, 64, 64, 3),
        output_fields=torch.randn(1, 1, 64, 64, 3),
        constant_scalars=torch.tensor([[0.1]]),
        constant_fields=None,
    )
    original_input, original_target = (
        batch.input_fields.clone(),
        batch.output_fields.clone(),
    )
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": batch,
    }
    datamodule = cast(Any, SimpleNamespace(train_dataset=SimpleNamespace(norm=norm)))
    model = setup_epd_model(cfg, stats, datamodule)
    source = _noise_source(model)
    decoded, references = [], []
    decode = model.encoder_decoder.decoder.decode
    sample_like = source.sample_like

    def capture_decode(mapped):
        values = decode(mapped)
        decoded.append(values)
        return values

    def capture_sample(reference):
        references.append(reference.shape)
        return sample_like(reference)

    with (
        patch.object(
            model.encoder_decoder.decoder, "decode", side_effect=capture_decode
        ),
        patch.object(source, "sample_like", side_effect=capture_sample),
    ):
        loss, prediction = model.loss(batch)
    assert prediction is not None
    assert prediction.shape == (1, 1, 64, 64, 3, 8)
    grid = 16 if PRESETS[name][1] == "spatial" else 64
    assert references == [torch.Size((8, 1, grid, grid, 4))]
    correction = (
        decoded[0] * torch.tensor([0.003, 0.04, 0.06])
        + torch.tensor([0.001, 0.02, -0.04])
    ) / torch.tensor([0.02, 0.1, 0.2])
    expected_state = original_input[..., None] + rearrange(
        correction, "(b m) t ... c -> b t ... c m", b=1, m=8
    )
    torch.testing.assert_close(prediction, expected_state)
    criterion = _build_loss_func(cfg.model)
    assert criterion is not None
    torch.testing.assert_close(loss, criterion(prediction, original_target))
    torch.testing.assert_close(batch.input_fields, original_input)
    torch.testing.assert_close(batch.output_fields, original_target)
    assert not torch.equal(prediction[..., 0], prediction[..., 1])
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.processor.parameters()
        if p.requires_grad
    )
    batch.output_fields = torch.randn(1, 2, 64, 64, 3)
    model.eval()
    with torch.no_grad():
        for ratio in (0.0, 1.0):
            forecast, truth = model.rollout(
                batch,
                stride=1,
                max_rollout_steps=2,
                n_members=2,
                teacher_forcing_ratio=ratio,
                free_running_only=ratio == 0,
            )
            assert forecast.shape == (1, 2, 64, 64, 3, 2)
            assert torch.isfinite(forecast).all()
            torch.testing.assert_close(
                truth, norm.denormalize_flattened(batch.output_fields, "variable")
            )


def test_multiscale_adaln_checkpoint_roundtrip_and_explicit_noise():
    kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "spatial_resolution": [32, 32],
        "hidden_dim": 16,
        "num_heads": 2,
        "n_layers": 1,
        "patch_size": 2,
        "n_noise_input_channels": 4,
        "n_noise_channels": 8,
        "noise_mode": "spatial",
        "temporal_method": "none",
    }
    original = AzulaViTProcessor(
        **kwargs, noise_source=MultiscaleGaussianSource([1, 2, 4, 6])
    )
    stream = io.BytesIO()
    torch.save(original.state_dict(), stream)
    stream.seek(0)
    restored = AzulaViTProcessor(
        **kwargs, noise_source=MultiscaleGaussianSource([1, 2, 4, 6])
    )
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    iid = AzulaViTProcessor(**kwargs)
    iid.load_state_dict(original.state_dict(), strict=True)
    assert original.state_dict().keys() == iid.state_dict().keys()
    fields = torch.randn(2, 3, 32, 32)
    torch.manual_seed(11)
    expected = original.map(fields)
    torch.manual_seed(11)
    torch.testing.assert_close(restored.map(fields), expected, rtol=0, atol=0)
    explicit = torch.randn(2, 256, 4, requires_grad=True)
    torch.testing.assert_close(original(fields, explicit), iid(fields, explicit))
    original(fields, explicit).square().mean().backward()
    assert explicit.grad is not None
    assert torch.isfinite(explicit.grad).all()


@pytest.mark.parametrize("name", PRESETS)
def test_explicit_four_gpu_allocation_preview_has_no_submission(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    launcher, _ = _resolve_launcher_submission_context(
        [
            f"local_experiment=swe64s/{name}",
            "+distributed=ddp_4gpu_slurm",
            "++hydra.launcher.nodes=1",
            "++hydra.launcher.cpus_per_task=72",
        ]
    )
    preview = _build_sbatch_command(
        job_name=name,
        log_dir=Path("/preflight"),
        launcher_cfg=launcher,
        batch_script_path=Path("/preflight/preview.sh"),
    )
    assert "--time=08:00:00" in preview
    assert "--nodes=1" in preview
    assert "--gpus-per-node=4" in preview
    assert "--ntasks-per-node=4" in preview
