"""Verify the matched SWE64 full-grid FNO and concatenation controls."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule
from omegaconf import OmegaConf
from torch import nn

from autocast.processors.periodic_hybrid_fno import _GlobalFourierConv
from autocast.scripts.setup import setup_epd_model
from autocast.types import Batch

ROOT = Path(__file__).resolve().parents[2]
LOSS_ARMS = (
    "campaign_nested_hybrid_fno_helmholtz",
    "campaign_nested_hybrid_fno_fourier_pairs",
    "campaign_nested_hybrid_fno_pairs_singletons",
)


def _compose(preset):
    with initialize_config_dir(
        config_dir=str(ROOT / "src/autocast/configs"), version_base=None
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"local_experiment=swe64s/{preset}",
                f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
                "swe64_pool_root=/example/pool",
                "swe64_train_trajectories=1000",
            ],
        )


def test_controls_match_campaign_and_parameter_budget():
    prefix = "campaign_nested_crps_fft_"
    reference = _compose(prefix + "fno")
    fft, iid, vit = (
        _compose(prefix + name) for name in ("hybrid_fno", "hybrid_fno_iid", "concat4d")
    )
    for cfg in (fft, iid, vit):
        for key in ("datamodule", "trainer", "optimizer"):
            assert OmegaConf.to_container(cfg[key], resolve=True) == (
                OmegaConf.to_container(reference[key], resolve=True)
            )
        assert cfg.seed == 42
        assert cfg.logging.wandb.enabled
        assert not cfg.logging.wandb.log_model
        assert cfg.logging.wandb.mode == "online"
        assert cfg.model.loss_func == reference.model.loss_func
        assert cfg.model.residual_prediction == reference.model.residual_prediction
        assert cfg.model.n_members == 8
        assert cfg.trainer.max_steps == cfg.optimizer.cosine_steps == 128000
        assert cfg.resume_from_checkpoint is None
        assert cfg.model.input_noise_injector.n_channels == 4
        assert cfg.model.input_noise_injector.std == 1.0
    assert fft.model.processor == iid.model.processor
    assert fft.model.input_noise_injector == vit.model.input_noise_injector
    assert iid.model.input_noise_injector.get("noise_source") is None
    assert vit.model.processor.n_noise_channels is None
    assert fft.model.input_noise_injector.noise_source.half_power_wavenumbers == [
        2,
        4,
        8,
        16,
    ]

    def build(cfg):
        return instantiate(
            cfg.model.processor,
            in_channels=7,
            out_channels=3,
            spatial_resolution=(64, 64),
        )

    hybrid = build(fft)
    assert len(hybrid.blocks) == 10
    assert sum(isinstance(b.mixer, _GlobalFourierConv) for b in hybrid.blocks) == 2
    assert sum(isinstance(b.mixer, nn.Conv2d) for b in hybrid.blocks) == 8
    assert hybrid.blocks[0].mixer.weight.shape[:2] == (43, 22)
    assert sum(p.numel() for p in hybrid.parameters()) == pytest.approx(
        sum(p.numel() for p in build(vit).parameters()), rel=0.01
    )
    assert all(not p.is_complex() for p in hybrid.parameters())


def test_periodic_swin_matches_correlated_concat_and_trains():
    cfg = _compose("campaign_nested_crps_fft_swin")
    reference = _compose("campaign_nested_crps_fft_concat4d")
    for key in ("datamodule", "trainer", "optimizer", "seed"):
        assert cfg[key] == reference[key]
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "online"
    assert not cfg.logging.wandb.log_model
    for key in (
        "loss_func",
        "input_noise_injector",
        "residual_prediction",
        "n_members",
    ):
        assert cfg.model[key] == reference.model[key]
    assert cfg.model.processor.periodic
    assert cfg.model.processor.n_noise_channels is None
    assert not cfg.compile_backbone

    field = torch.randn(1, 1, 64, 64, 3)
    batch = Batch(field, torch.randn_like(field), None, None)
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": batch,
    }
    datamodule = LightningDataModule()
    model = setup_epd_model(cfg, stats, datamodule)
    with pytest.raises(ValueError, match="field sizes must be multiples of 16"):
        model.processor.map(torch.zeros(1, 7, 63, 64), global_cond=None)
    vit = instantiate(
        reference.model.processor,
        in_channels=7,
        out_channels=3,
        spatial_resolution=(64, 64),
    )
    assert sum(p.numel() for p in model.processor.parameters()) == pytest.approx(
        sum(p.numel() for p in vit.parameters()), rel=0.02
    )
    loss, members = model.loss(batch)
    assert torch.isfinite(loss)
    assert members is not None
    assert members.shape == (*field.shape, 8)
    assert not torch.equal(members[..., 0], members[..., 1])
    loss.backward()
    for parameter in model.processor.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    reloaded = setup_epd_model(cfg, stats, datamodule)
    reloaded.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        before = model(batch)
        torch.manual_seed(42)
        after = reloaded(batch)
    torch.testing.assert_close(before, after)


def test_wind_loss_controls_match_fno_and_common_validation():
    reference = _compose("campaign_nested_crps_fft_hybrid_fno")
    for name in LOSS_ARMS:
        cfg = _compose(name)
        for key in ("datamodule", "trainer", "optimizer", "seed"):
            assert OmegaConf.select(cfg, key) == OmegaConf.select(reference, key)
        assert cfg.logging.wandb.enabled
        assert cfg.logging.wandb.mode == "online"
        assert not cfg.logging.wandb.log_model
        for key in (
            "processor",
            "input_noise_injector",
            "residual_prediction",
            "n_members",
        ):
            assert cfg.model[key] == reference.model[key]
        assert cfg.model.validation_loss_func == reference.model.loss_func
        assert cfg.model.loss_func.weights[:2] == [1.0, 0.1]
        assert cfg.model.loss_func.losses[0] == reference.model.loss_func.losses[0]
        assert cfg.resume_from_checkpoint is None

    helmholtz, pairs, mixed = (_compose(name) for name in LOSS_ARMS)
    assert helmholtz.model.loss_func.losses[2].mode == "helmholtz"
    pair_cfg = OmegaConf.to_container(pairs.fno_pair_score, resolve=True)
    mixed_cfg = OmegaConf.to_container(mixed.fno_pair_score, resolve=True)
    assert isinstance(pair_cfg, dict)
    assert mixed_cfg == {**pair_cfg, "singleton_weight": 0.5}


def test_fourier_mixer_is_periodic_and_retains_cutoff():
    layer = _GlobalFourierConv(1, 1, max_wavenumber=21)
    with torch.no_grad():
        layer.lift.weight.fill_(1)
        layer.project.weight.fill_(1)
        layer.weight.zero_()
        layer.weight[..., 0] = 1
    x = 2 * torch.pi * torch.arange(64, dtype=torch.float32) / 64
    low = torch.sin(21 * x)[None, None, None, :].expand(1, 1, 64, 64)
    high = torch.cos(22 * x)[:, None][None, None].expand_as(low)
    torch.testing.assert_close(layer(low + high), low, atol=1e-5, rtol=1e-5)
    low = low.transpose(-2, -1)
    torch.testing.assert_close(layer(low), low, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("preset", ["campaign_nested_crps_fft_hybrid_fno", *LOSS_ARMS])
def test_hybrid_runs_ensemble_residual_loss_and_reloads(preset):
    cfg = _compose(preset)
    cfg.model.processor.hidden_channels = 8
    cfg.model.processor.spectral_channels = 4
    field = torch.randn(1, 1, 48, 48, 3)
    batch = Batch(field, torch.randn_like(field), None, None)
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": batch,
    }
    model = setup_epd_model(cfg, stats, LightningDataModule())
    assert model.residual_prediction
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, members = model.loss(batch)
    assert torch.isfinite(loss)
    assert members is not None
    assert members.shape == (*field.shape, 8)
    assert not torch.equal(members[..., 0], members[..., 1])
    loss.backward()
    for param in model.processor.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
    reloaded = setup_epd_model(cfg, stats, LightningDataModule())
    reloaded.load_state_dict(model.state_dict(), strict=True)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(7)
        _, expected = model.loss(batch)
        torch.manual_seed(7)
        _, actual = reloaded.loss(batch)
        torch.testing.assert_close(actual, expected)
