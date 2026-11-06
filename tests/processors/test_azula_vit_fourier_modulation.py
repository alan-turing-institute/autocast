"""Fourier modulation remains an opt-in extension of existing ViT processors."""

from pathlib import Path

import pytest
import torch
from azula.nn.vit import ViT
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from autocast.losses.ensemble import CRPSLoss
from autocast.nn.fourier_vit import FourierViTBlock
from autocast.processors.azula_vit import AzulaViTProcessor


def make_processor(**kwargs):
    config = {
        "in_channels": 3,
        "out_channels": 3,
        "spatial_resolution": (6, 10),
        "hidden_dim": 16,
        "num_heads": 2,
        "n_layers": 2,
        "patch_size": 2,
        "temporal_method": "none",
        "n_noise_input_channels": 4,
        "n_noise_channels": 8,
    }
    return AzulaViTProcessor(**(config | kwargs))


def test_default_and_explicit_adaln_preserve_initialization_and_rng():
    torch.manual_seed(91)
    default = make_processor()
    default_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(91)
    explicit = make_processor(modulation_mode="adaln")
    explicit_rng = torch.random.get_rng_state()

    assert isinstance(default.model.vit, ViT)
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not any(".fourier." in name for name in default.state_dict())
    for name, parameter in default.state_dict().items():
        torch.testing.assert_close(
            parameter, explicit.state_dict()[name], rtol=0, atol=0
        )
    assert torch.equal(default_rng, explicit_rng)

    x, noise = torch.randn(2, 3, 6, 10), torch.randn(2, 4)
    torch.testing.assert_close(default(x, noise), explicit(x, noise), rtol=0, atol=0)


def test_old_state_loads_strictly_without_new_constructor_options():
    original = make_processor().eval()
    # Old constructor dictionaries have neither modulation_mode nor band count.
    restored = make_processor().eval()
    result = restored.load_state_dict(original.state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    x, noise = torch.randn(2, 3, 6, 10), torch.randn(2, 4)
    torch.testing.assert_close(original(x, noise), restored(x, noise), rtol=0, atol=0)


def test_fourier_initialization_preserves_existing_seeded_parameters():
    torch.manual_seed(92)
    original = make_processor()
    torch.manual_seed(92)
    fourier = make_processor(modulation_mode="fourier")

    original_state, fourier_state = original.state_dict(), fourier.state_dict()
    extra_keys = fourier_state.keys() - original_state.keys()
    assert extra_keys
    assert all(".fourier." in name for name in extra_keys)
    assert not original_state.keys() - fourier_state.keys()
    for name, parameter in original_state.items():
        torch.testing.assert_close(parameter, fourier_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_old_state_migration_preserves_identity_outputs_and_gradients(checkpointing):
    torch.manual_seed(93)
    kwargs = {
        "checkpointing": checkpointing,
        "include_global_cond": True,
        "global_cond_channels": 2,
    }
    original = make_processor(**kwargs).double()
    fourier = make_processor(modulation_mode="fourier", **kwargs).double()
    original_state = original.state_dict()
    with pytest.raises(RuntimeError, match="Missing key"):
        fourier.load_state_dict(original_state, strict=True)
    migration = fourier.load_state_dict(original_state, strict=False)
    assert (
        set(migration.missing_keys)
        == fourier.state_dict().keys() - original_state.keys()
    )
    assert all(".fourier." in name for name in migration.missing_keys)
    assert not migration.unexpected_keys

    x = torch.randn(2, 3, 6, 10, dtype=torch.float64)
    noise = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    fourier_noise = noise.detach().clone().requires_grad_()
    cond = torch.randn(2, 2, dtype=torch.float64)
    original_output = original(x, noise, cond)
    fourier_output = fourier(x, fourier_noise, cond)
    torch.testing.assert_close(original_output, fourier_output, rtol=1e-10, atol=1e-12)

    original_output.square().mean().backward()
    fourier_output.square().mean().backward()
    fourier_parameters = dict(fourier.named_parameters())
    for name, parameter in original.named_parameters():
        assert parameter.grad is not None, name
        actual_gradient = fourier_parameters[name].grad
        assert actual_gradient is not None, name
        torch.testing.assert_close(
            parameter.grad, actual_gradient, rtol=1e-9, atol=1e-12
        )
    torch.testing.assert_close(noise.grad, fourier_noise.grad, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("modulation_mode", ["adaln", "fourier"])
def test_full_module_serialization_roundtrip(tmp_path, modulation_mode):
    processor = make_processor(modulation_mode=modulation_mode).eval()
    # Old serialized default objects do not contain these optional new fields.
    if modulation_mode == "adaln":
        for module in (processor, processor.model):
            module.__dict__.pop("modulation_mode", None)
            module.__dict__.pop("fourier_num_bands", None)
    x, noise = torch.randn(2, 3, 6, 10), torch.randn(2, 4)
    expected = processor(x, noise)
    checkpoint = tmp_path / "processor.pt"
    torch.save(processor, checkpoint)
    # This test loads only the trusted fixture just written above.
    restored = torch.load(checkpoint, weights_only=False)
    assert restored.state_dict().keys() == processor.state_dict().keys()
    torch.testing.assert_close(restored(x, noise), expected, rtol=0, atol=0)


def test_fourier_state_loads_strictly_with_learned_projection():
    processor = make_processor(modulation_mode="fourier", fourier_num_bands=3)
    with torch.no_grad():
        for block in processor.model.vit.blocks:
            assert isinstance(block, FourierViTBlock)
            block.fourier.projection.weight.normal_(std=0.05)
    restored = make_processor(modulation_mode="fourier", fourier_num_bands=3)
    result = restored.load_state_dict(processor.state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    x, noise = torch.randn(2, 3, 6, 10), torch.randn(2, 4)
    torch.testing.assert_close(restored(x, noise), processor(x, noise), rtol=0, atol=0)


def test_dynamic_rectangular_grids_with_non_square_patches():
    processor = make_processor(modulation_mode="fourier", patch_size=(2, 3))
    noise = torch.randn(2, 4)
    # Both odd and even token grids run through the same initialized processor.
    for height, width in ((6, 15), (10, 9), (8, 18)):
        x = torch.randn(2, 3, height, width)
        output = processor(x, noise)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("with_time", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_multistep_backward_uses_every_parameter_and_explicit_noise(
    with_time, checkpointing
):
    torch.manual_seed(94)
    processor = make_processor(
        modulation_mode="fourier",
        n_steps_input=2,
        n_steps_output=3,
        checkpointing=checkpointing,
    )
    x_shape = (2, 2, 6, 10, 3) if with_time else (2, 6, 6, 10)
    output_shape = (2, 3, 6, 10, 3) if with_time else (2, 9, 6, 10)
    noise = torch.randn(2, 4, requires_grad=True)
    output = processor(torch.randn(x_shape), noise)
    assert output.shape == output_shape
    output.square().mean().backward()
    for name, parameter in processor.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert noise.grad is not None
    assert torch.isfinite(noise.grad).all()
    assert (noise.grad.abs().sum(dim=-1) > 0).all()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_spectral_branch_responds_to_noise_and_learns_from_crps(checkpointing):
    torch.manual_seed(95)
    processor = make_processor(modulation_mode="fourier", checkpointing=checkpointing)
    # Hold AdaLN constant to isolate the Fourier branch's dependence on noise.
    with torch.no_grad():
        for block in processor.model.vit.blocks:
            assert isinstance(block, FourierViTBlock)
            projection = block.ada_zero[-2]
            assert isinstance(projection, torch.nn.Linear)
            projection.weight.zero_()
    x = torch.randn(2, 1, 6, 10, 3)
    noises = torch.randn(3, 2, 4, requires_grad=True)
    torch.testing.assert_close(
        processor(x, noises[0]), processor(x, noises[1]), rtol=0, atol=0
    )
    with torch.no_grad():
        for block in processor.model.vit.blocks:
            assert isinstance(block, FourierViTBlock)
            block.fourier.projection.weight.normal_(std=0.05)
    predictions = torch.stack([processor(x, noise) for noise in noises], dim=-1)
    assert not torch.allclose(predictions[..., 0], predictions[..., 1])
    loss = CRPSLoss()(predictions, torch.randn_like(x))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    for block in processor.model.vit.blocks:
        assert isinstance(block, FourierViTBlock)
        gradient = block.fourier.projection.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert noises.grad is not None
    assert torch.isfinite(noises.grad).all()
    assert (noises.grad.abs().sum(dim=-1) > 0).all()


@pytest.mark.parametrize("preset", ["vit_azula_small", "vit_azula_large"])
def test_hydra_global_fourier_opt_in(preset):
    config_dir = Path(__file__).resolve().parents[2] / "src/autocast/configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"processor@model.processor={preset}",
                "model.processor.modulation_mode=fourier",
                "model.processor.fourier_num_bands=3",
                "model.processor.n_noise_channels=8",
                "model.processor.n_noise_input_channels=4",
                "model.processor.in_channels=3",
                "model.processor.out_channels=3",
                "model.processor.spatial_resolution=[6,10]",
                "model.processor.patch_size=2",
                "model.processor.hidden_dim=16",
                "model.processor.num_heads=2",
                "model.processor.n_layers=1",
            ],
        )
    assert cfg.model.processor.noise_mode == "global"
    processor = instantiate(cfg.model.processor)
    assert processor.model.vit.blocks[0].fourier.num_bands == 3
    assert processor.map(torch.randn(2, 3, 6, 10)).shape == (2, 3, 6, 10)


@pytest.mark.parametrize("n_noise_channels", [None, 0, -1])
def test_fourier_requires_positive_global_noise(n_noise_channels):
    with pytest.raises(ValueError, match="positive n_noise_channels"):
        make_processor(
            modulation_mode="fourier",
            n_noise_channels=n_noise_channels,
            n_noise_input_channels=None,
        )


def test_fourier_rejects_spatial_noise():
    with pytest.raises(ValueError, match="global noise"):
        make_processor(modulation_mode="fourier", noise_mode="spatial")


@pytest.mark.parametrize("num_bands", [0, 1, -1, 2.5, True, "4"])
def test_fourier_rejects_invalid_band_count(num_bands):
    with pytest.raises(ValueError, match="num_bands must be an integer"):
        make_processor(modulation_mode="fourier", fourier_num_bands=num_bands)


def test_unknown_modulation_mode_is_rejected():
    with pytest.raises(ValueError, match="not a valid"):
        make_processor(modulation_mode="unknown")
