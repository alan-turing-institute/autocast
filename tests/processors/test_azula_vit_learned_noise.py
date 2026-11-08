"""Learnable conditioning fields use the existing spatial AdaLN computation."""

import pytest
import torch
from einops import rearrange

from autocast.losses.ensemble import CRPSLoss
from autocast.processors.azula_vit import AzulaViTProcessor


def make_processor(**kwargs):
    config = {
        "in_channels": 3,
        "out_channels": 3,
        "spatial_resolution": (8, 12),
        "hidden_dim": 16,
        "num_heads": 2,
        "n_layers": 3,
        "patch_size": 2,
        "temporal_method": "none",
        "n_noise_input_channels": 1,
        "n_noise_channels": 8,
        "noise_mode": "spatial",
        "modulation_mode": "spectral_noise",
        "noise_radial_knots": 6,
    }
    return AzulaViTProcessor(**(config | kwargs))


@pytest.mark.parametrize("checkpointing", [False, True])
def test_identity_matches_spatial_baseline_outputs_parameters_rng_and_gradients(
    checkpointing,
):
    torch.manual_seed(10)
    baseline = make_processor(
        modulation_mode="adaln", checkpointing=checkpointing
    ).double()
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(10)
    learned = make_processor(checkpointing=checkpointing).double()
    assert torch.equal(baseline_rng, torch.random.get_rng_state())
    expected = baseline.state_dict()
    assert expected.keys() <= learned.state_dict().keys()
    assert all(
        name.startswith("noise_filter.")
        for name in learned.state_dict().keys() - expected.keys()
    )
    for name, parameter in expected.items():
        torch.testing.assert_close(
            learned.state_dict()[name], parameter, rtol=0, atol=0
        )
    x = torch.randn(2, 3, 8, 12).double()
    noise = torch.randn(2, 24, 1).double().requires_grad_()
    second_noise = noise.detach().clone().requires_grad_()
    first, second = baseline(x, noise), learned(x, second_noise)
    torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-12)
    first.square().mean().backward()
    second.square().mean().backward()
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(
            dict(learned.named_parameters())[name].grad,
            parameter.grad,
            rtol=1e-9,
            atol=1e-12,
        )
    torch.testing.assert_close(noise.grad, second_noise.grad, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("with_time", [False, True])
def test_eight_member_loss_reaches_noise_filter_and_all_backbone_parameters(
    checkpointing, with_time
):
    torch.manual_seed(11)
    processor = make_processor(
        checkpointing=checkpointing,
        n_steps_input=2,
        n_steps_output=3,
        include_global_cond=True,
        global_cond_channels=2,
    )
    x = torch.randn(2, 2, 8, 12, 3) if with_time else torch.randn(2, 6, 8, 12)
    cond = torch.randn(2, 2)
    noise = torch.randn(8, 2, 24, 1, requires_grad=True)
    predictions = torch.stack([processor(x, sample, cond) for sample in noise], dim=-1)
    target = torch.randn_like(predictions[..., 0])
    loss = CRPSLoss()(predictions, target)
    loss.backward()
    for name, parameter in processor.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert processor.noise_filter is not None
    for parameter in processor.noise_filter.parameters():
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
    assert noise.grad is not None
    assert torch.isfinite(noise.grad).all()


def test_filtered_noise_enters_each_block_before_projection_and_global_condition():
    processor = make_processor(
        include_global_cond=True, global_cond_channels=2
    ).double()
    assert processor.noise_filter is not None
    assert processor.modulation_proj is not None
    assert processor.model.global_cond_embedding is not None
    with torch.no_grad():
        processor.noise_filter.radial_gains[1:].zero_()
    x = torch.randn(2, 3, 8, 12).double()
    noise = torch.randn(2, 24, 1).double()
    cond = torch.randn(2, 2).double()
    observed = []
    handles = []
    for block in processor.model.vit.blocks:
        assert isinstance(block, torch.nn.Module)
        handles.append(
            block.get_submodule("ada_zero").register_forward_pre_hook(
                lambda _, args: observed.append(args[0].detach())
            )
        )
    processor(x, noise, cond)
    for handle in handles:
        handle.remove()
    embedding = processor.model.global_cond_embedding(cond)[:, None]
    filtered = rearrange(
        processor.noise_filter(rearrange(noise, "b (h w) d -> b h w d", h=4)),
        "b l h w d -> b l (h w) d",
    )
    for index, actual in enumerate(observed):
        expected = processor.modulation_proj(filtered[:, index]) + embedding
        torch.testing.assert_close(actual, expected)
    assert observed[0].std(dim=1).max() > 0
    torch.testing.assert_close(observed[1], observed[1][:, :1].expand_as(observed[1]))


def test_map_samples_once_and_explicit_forward_is_deterministic(monkeypatch):
    processor = make_processor()
    x = torch.randn(2, 3, 8, 12)
    calls = []
    randn = torch.randn

    def draw(shape, **kwargs):
        calls.append(shape)
        return randn(shape, **kwargs)

    monkeypatch.setattr(torch, "randn", draw)
    first, second = processor.map(x), processor.map(x)
    assert calls == [(2, 24, 1), (2, 24, 1)]
    assert not torch.allclose(first, second)
    explicit = randn(2, 24, 1)
    state = torch.random.get_rng_state().clone()
    torch.testing.assert_close(
        processor(x, explicit), processor(x, explicit), rtol=0, atol=0
    )
    assert torch.equal(state, torch.random.get_rng_state())


def test_strict_and_full_module_roundtrip(tmp_path):
    processor = make_processor().eval()
    assert processor.noise_filter is not None
    assert processor.noise_filter.phase_coefficients is not None
    with torch.no_grad():
        processor.noise_filter.radial_gains.normal_()
        processor.noise_filter.phase_coefficients.normal_()
    restored = make_processor().eval()
    restored.load_state_dict(processor.state_dict(), strict=True)
    x, noise = torch.randn(2, 3, 8, 12), torch.randn(2, 24, 1)
    expected = processor(x, noise)
    torch.testing.assert_close(restored(x, noise), expected, rtol=0, atol=0)
    path = tmp_path / "processor.pt"
    torch.save(processor, path)
    restored = torch.load(path, weights_only=False)
    torch.testing.assert_close(restored(x, noise), expected, rtol=0, atol=0)


def test_dynamic_rectangular_token_grids():
    processor = make_processor(patch_size=(2, 3))
    for height, width in ((6, 15), (10, 9), (8, 18)):
        x = torch.randn(2, 3, height, width)
        assert processor.map(x).shape == x.shape


def test_rejects_global_noise_layout():
    with pytest.raises(ValueError, match="noise_mode='spatial'"):
        make_processor(noise_mode="global")


def test_legacy_full_objects_need_no_new_attributes(tmp_path):
    processor = make_processor(modulation_mode="adaln").eval()
    # Simulate a trusted checkpoint serialized before these attributes existed.
    processor.__dict__.pop("noise_filter")
    processor.model.__dict__.pop("blockwise_modulation")
    x, noise = torch.randn(2, 3, 8, 12), torch.randn(2, 24, 1)
    expected = processor(x, noise)
    path = tmp_path / "legacy.pt"
    torch.save(processor, path)
    restored = torch.load(path, weights_only=False)
    torch.testing.assert_close(restored(x, noise), expected, rtol=0, atol=0)
