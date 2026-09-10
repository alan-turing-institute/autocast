"""Spatial AdaLN uses shared weights but independent noise at each patch token."""

from pathlib import Path

import pytest
import torch
from einops import repeat
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from autocast.decoders.channels_last import ChannelsLast
from autocast.encoders.permute_concat import PermuteConcat
from autocast.losses.ensemble import CRPSLoss
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.processors.azula_vit import AzulaViTProcessor
from autocast.types import Batch


def make_processor(**kwargs):
    config = {
        "in_channels": 3,
        "out_channels": 3,
        "spatial_resolution": (8, 8),
        "hidden_dim": 16,
        "num_heads": 2,
        "n_layers": 2,
        "patch_size": 2,
        "temporal_method": "none",
        "n_noise_input_channels": 4,
        "n_noise_channels": 8,
    }
    return AzulaViTProcessor(**(config | kwargs))


@pytest.mark.parametrize("include_global_cond", [False, True])
def test_broadcast_noise_matches_global_and_checkpoint(include_global_cond):
    kwargs = {"include_global_cond": include_global_cond, "global_cond_channels": 2}
    global_model = make_processor(**kwargs).eval()
    spatial_model = make_processor(noise_mode="spatial", **kwargs).eval()
    spatial_model.load_state_dict(global_model.state_dict(), strict=True)
    assert spatial_model.state_dict().keys() == global_model.state_dict().keys()
    assert sum(p.numel() for p in spatial_model.parameters()) == sum(
        p.numel() for p in global_model.parameters()
    )
    x = torch.randn(2, 3, 8, 8)
    z = torch.randn(2, 4)
    local_z = repeat(z, "b d -> b n d", n=16)
    cond = torch.randn(2, 2) if include_global_cond else None
    torch.testing.assert_close(
        spatial_model(x, local_z, cond), global_model(x, z, cond)
    )


@pytest.mark.parametrize(("patch_size", "n_tokens"), [(1, 64), (2, 16), ((2, 4), 8)])
@pytest.mark.parametrize("with_time", [False, True])
def test_map_samples_independent_token_noise(patch_size, n_tokens, with_time):
    processor = make_processor(noise_mode="spatial", patch_size=patch_size)
    x = torch.randn(2, 1, 8, 8, 3) if with_time else torch.randn(2, 3, 8, 8)
    captured = []
    assert processor.modulation_proj is not None
    handle = processor.modulation_proj.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        torch.manual_seed(123)
        first = processor.map(x)
        torch.manual_seed(123)
        repeated = processor.map(x)
        second = processor.map(x)
    finally:
        handle.remove()
    assert first.shape == x.shape
    assert captured[0].shape == (2, n_tokens, 4)
    assert not torch.equal(captured[0][:, 0], captured[0][:, 1])
    assert not torch.equal(captured[0][0], captured[0][1])
    assert not torch.equal(captured[0], captured[2])
    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, second)


def test_local_modulation_is_pointwise_before_attention():
    processor = make_processor(noise_mode="spatial")
    noise = torch.zeros(2, 16, 4)
    changed = noise.clone()
    changed[:, 0] = 1
    modulator = processor.model.get_submodule("vit.blocks.0.ada_zero")
    assert processor.modulation_proj is not None
    original_mod = modulator(processor.modulation_proj(noise))
    changed_mod = modulator(processor.modulation_proj(changed))
    assert original_mod.shape == (3, 2, 16, 16)
    torch.testing.assert_close(original_mod[:, :, 1:], changed_mod[:, :, 1:])
    assert not torch.equal(original_mod[:, :, 0], changed_mod[:, :, 0])


@pytest.mark.parametrize("checkpointing", [False, True])
def test_multistep_backward_uses_all_parameters(checkpointing):
    processor = make_processor(
        noise_mode="spatial",
        n_steps_input=2,
        n_steps_output=3,
        checkpointing=checkpointing,
    )
    output = processor.map(torch.randn(2, 2, 8, 8, 3))
    assert output.shape == (2, 3, 8, 8, 3)
    output.square().mean().backward()
    for name, parameter in processor.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("shape", [(2, 4), (2, 15, 4), (1, 16, 4), (2, 16, 5)])
def test_wrong_spatial_noise_shape_is_rejected(shape):
    processor = make_processor(noise_mode="spatial")
    with pytest.raises(ValueError, match="Expected x_noise with shape"):
        processor(torch.randn(2, 3, 8, 8), torch.randn(shape))


def test_spatial_requires_noise_and_divisible_grid():
    with pytest.raises(ValueError, match="positive n_noise_channels"):
        make_processor(noise_mode="spatial", n_noise_channels=0)
    with pytest.raises(ValueError, match="positive n_noise_input_channels"):
        make_processor(noise_mode="spatial", n_noise_input_channels=0)
    with pytest.raises(ValueError, match="not a valid"):
        make_processor(noise_mode="unknown")
    processor = make_processor(noise_mode="spatial")
    with pytest.raises(ValueError, match="divisible by patch_size"):
        processor.map(torch.randn(2, 3, 7, 8))
    with pytest.raises(ValueError, match="4D or 5D"):
        processor.map(torch.randn(2, 3, 8))


def test_default_global_and_disabled_noise_are_unchanged():
    default = make_processor()
    explicit = make_processor(noise_mode="global")
    explicit.load_state_dict(default.state_dict())
    x = torch.randn(2, 3, 8, 8)
    torch.manual_seed(123)
    expected = default.map(x)
    torch.manual_seed(123)
    torch.testing.assert_close(explicit.map(x), expected)
    disabled = make_processor(n_noise_channels=None, n_noise_input_channels=None)
    torch.testing.assert_close(disabled.map(x), disabled.map(x))


def test_ensemble_crps_backward():
    fields = torch.randn(2, 1, 8, 8, 3)
    batch = Batch(fields, torch.randn_like(fields), None, None)
    encoder_decoder = EncoderDecoder(
        encoder=PermuteConcat(in_channels=3, n_steps_input=1, with_constants=False),
        decoder=ChannelsLast(output_channels=3, time_steps=1),
    )
    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=encoder_decoder,
        processor=make_processor(noise_mode="spatial"),
        n_members=4,
        loss_func=CRPSLoss(),
        train_in_latent_space=False,
    )
    prediction = model(batch)
    assert prediction.shape == (2, 1, 8, 8, 3, 4)
    assert not torch.equal(prediction[..., 0], prediction[..., 1])
    loss, _ = model.loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.processor.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_noise_chunks_receive_gradients(dtype):
    processor = make_processor(noise_mode="spatial").to(dtype=dtype)
    x = torch.randn(2, 3, 8, 8, dtype=dtype)
    noise = torch.randn(2, 16, 4, dtype=dtype, requires_grad=True)
    output = processor(x, noise)
    assert output.dtype == dtype
    output.square().mean().backward()
    assert noise.grad is not None
    assert torch.isfinite(noise.grad).all()
    assert (noise.grad.abs().sum(dim=-1) > 0).all()


def test_spatial_without_input_projection():
    processor = make_processor(noise_mode="spatial", n_noise_input_channels=None)
    assert processor.modulation_proj is None
    x = torch.randn(2, 3, 8, 12)
    assert processor.map(x).shape == x.shape
    assert processor(x, torch.randn(2, 24, 8)).shape == x.shape


@pytest.mark.parametrize("patch_size", [0, -1, (2,), (2, 0), (2, 2, 2)])
def test_invalid_spatial_patch_size_is_rejected(patch_size):
    with pytest.raises(ValueError, match="two positive patch_size dimensions"):
        make_processor(noise_mode="spatial", patch_size=patch_size)


def test_explicit_spatial_forward_requires_noise_chunks():
    processor = make_processor(noise_mode="spatial")
    with pytest.raises(ValueError, match="Expected x_noise with shape"):
        processor(torch.randn(2, 3, 8, 8))


@pytest.mark.parametrize("preset", ["vit_azula_small", "vit_azula_large"])
def test_hydra_spatial_mode_override(preset):
    config_dir = Path(__file__).resolve().parents[2] / "src/autocast/configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"processor@model.processor={preset}",
                "model.processor.noise_mode=spatial",
                "model.processor.n_noise_input_channels=16",
                "model.processor.in_channels=3",
                "model.processor.out_channels=3",
                "model.processor.spatial_resolution=[8,8]",
                "model.processor.hidden_dim=16",
                "model.processor.num_heads=2",
                "model.processor.n_layers=1",
            ],
        )
    processor = instantiate(cfg.model.processor)
    assert processor.map(torch.randn(2, 3, 8, 8)).shape == (2, 3, 8, 8)
