"""Structured sampling leaves the spatial AdaLN interface unchanged."""

from pathlib import Path

import pytest
import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from autocast.nn.noise import FlowSource, SpatialFourierSource
from autocast.processors.azula_vit import AzulaViTProcessor

WAVES = ((1, 0), (0, 2))


def make_processor(**kwargs):
    config = {
        "in_channels": 3,
        "out_channels": 3,
        "spatial_resolution": (16, 24),
        "hidden_dim": 16,
        "num_heads": 2,
        "n_layers": 2,
        "patch_size": 2,
        "n_noise_input_channels": 4,
        "n_noise_channels": 8,
        "noise_mode": "spatial",
        "temporal_method": "none",
    }
    return AzulaViTProcessor(**(config | kwargs))


@pytest.mark.parametrize("patch", [1, 2, (2, 3)])
@pytest.mark.parametrize("with_time", [False, True])
def test_source_uses_actual_token_grid_and_preserves_backward(patch, with_time):
    source = SpatialFourierSource(WAVES)
    processor = make_processor(patch_size=patch, noise_source=source)
    iid = make_processor(patch_size=patch)
    iid.load_state_dict(processor.state_dict(), strict=True)
    assert iid.state_dict().keys() == processor.state_dict().keys()
    assert sum(p.numel() for p in iid.parameters()) == sum(
        p.numel() for p in processor.parameters()
    )
    x = torch.randn(2, 1, 16, 24, 3) if with_time else torch.randn(2, 3, 16, 24)
    ph, pw = (patch, patch) if isinstance(patch, int) else patch
    reference = x.new_empty(2, 1, 16 // ph, 24 // pw, 4)
    torch.manual_seed(7)
    expected_noise = rearrange(source.sample_like(reference), "b 1 h w d -> b (h w) d")
    torch.manual_seed(7)
    output = processor.map(x)
    assert output.shape == x.shape
    torch.testing.assert_close(output, iid(x, expected_noise), rtol=0, atol=0)
    output.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in processor.parameters()
    )
    assert not torch.equal(output, processor.map(x))


def test_source_requires_spatial_mode_and_resolved_frequencies():
    with pytest.raises(ValueError, match="requires noise_mode"):
        make_processor(noise_mode="global", noise_source=SpatialFourierSource(WAVES))
    processor = make_processor(
        patch_size=4, noise_source=SpatialFourierSource(((2, 0),))
    )
    with pytest.raises(ValueError, match="Nyquist"):
        processor.map(torch.randn(2, 3, 16, 24))


def test_explicit_noise_bypasses_source_and_preserves_gradients():
    processor = make_processor(noise_source=SpatialFourierSource(((99, 0),)))
    noise = torch.randn(2, 96, 4, requires_grad=True)
    output = processor(torch.randn(2, 3, 16, 24), noise)
    output.square().mean().backward()
    assert noise.grad is not None
    assert torch.isfinite(noise.grad).all()
    assert (noise.grad.abs().sum(-1) > 0).all()


class CaptureSource(FlowSource):
    def sample_like(self, reference):
        self.last_reference = reference
        return torch.randn_like(reference)


def test_joint_output_frames_have_one_effective_noise_slot():
    source = CaptureSource()
    processor = make_processor(noise_source=source, n_steps_input=2, n_steps_output=3)
    output = processor.map(torch.randn(2, 2, 16, 24, 3))
    assert output.shape == (2, 3, 16, 24, 3)
    assert source.last_reference.shape == (2, 1, 8, 12, 4)


class BadSource(FlowSource):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def sample_like(self, reference):
        if self.error == "shape":
            return reference[..., :1]
        if self.error == "dtype":
            return reference.double()
        return reference.to(device="meta")


@pytest.mark.parametrize("error", ["shape", "dtype", "device"])
def test_source_contract_is_checked(error):
    processor = make_processor(noise_source=BadSource(error))
    with pytest.raises(ValueError, match="Noise source"):
        processor.map(torch.randn(2, 3, 16, 24))


@pytest.mark.parametrize("mode", ["spatial", "global"])
def test_default_iid_draw_is_exactly_unchanged(mode):
    processor = make_processor(noise_mode=mode)
    x = torch.randn(2, 3, 16, 24)
    torch.manual_seed(7)
    expected = processor(x, torch.randn(processor._noise_shape(x)))
    torch.manual_seed(7)
    torch.testing.assert_close(processor.map(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("preset", ["vit_azula_small", "vit_azula_large"])
@pytest.mark.parametrize("group", ["flow_source", "source_distribution"])
def test_hydra_source_composes_for_spatial_adaln(preset, group):
    config_dir = Path(__file__).resolve().parents[2] / "src/autocast/configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"processor@model.processor={preset}",
                "model.processor.noise_mode=spatial",
                f"+{group}@model.processor.noise_source=spatial_fourier",
                "model.processor.n_noise_input_channels=4",
                "model.processor.in_channels=3",
                "model.processor.out_channels=3",
                "model.processor.spatial_resolution=[16,24]",
                "model.processor.hidden_dim=16",
                "model.processor.num_heads=2",
                "model.processor.n_layers=1",
            ],
        )
    processor = instantiate(cfg.model.processor)
    assert isinstance(processor.noise_source, SpatialFourierSource)
    assert processor.map(torch.randn(2, 3, 16, 24)).shape == (2, 3, 16, 24)


@pytest.mark.parametrize("group", ["flow_source", "source_distribution"])
def test_hydra_source_composes_for_concat(group):
    config_dir = Path(__file__).resolve().parents[2] / "src/autocast/configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                "input_noise_injector@model.input_noise_injector=concat",
                f"+{group}@model.input_noise_injector.noise_source=spatial_fourier",
            ],
        )
    injector = instantiate(cfg.model.input_noise_injector)
    assert isinstance(injector.noise_source, SpatialFourierSource)
    assert injector(torch.randn(2, 3, 8, 12, 3)).shape == (2, 3, 8, 12, 4)
