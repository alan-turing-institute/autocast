"""Behavioral checks for bounded, real-compatible Fourier modulation."""

import math

import pytest
import torch
from einops import rearrange

from autocast.nn.fourier_modulation import FourierModulation2d


@pytest.mark.parametrize("shape", [(2, 8, 12, 3), (2, 9, 7, 3), (1, 1, 1, 1)])
def test_identity_initialization_retains_parameter_gradients(shape):
    torch.manual_seed(12)
    layer = FourierModulation2d(5).double()
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    mod = torch.randn(shape[0], 5, dtype=torch.float64, requires_grad=True)
    output = layer(x, mod)
    torch.testing.assert_close(output, x, atol=1e-14, rtol=1e-14)
    (output * torch.randn_like(output)).sum().backward()
    assert layer.projection.weight.grad is not None
    assert layer.projection.weight.grad.norm() > 0, "Identity must retain gradients"
    assert x.grad is not None
    assert x.grad.norm() > 0


def test_radial_bands_apply_different_gains_without_truncating_high_modes():
    height, width = 16, 24
    layer = FourierModulation2d(1, num_bands=3).double()
    with torch.no_grad():
        layer.projection.weight[:3, 0] = torch.tensor(
            [-2.0, 0.3, 2.0], dtype=torch.float64
        )
    h, w = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    # DC, middle-radius diagonal, and corner Nyquist each occupy one band center.
    fields = torch.stack(
        [torch.ones_like(h), (2 * math.pi * (h / 4 + w / 4)).cos(), (-1.0) ** (h + w)],
        dim=-1,
    ).unsqueeze(0)
    output = layer(fields, torch.ones(1, 1, dtype=torch.float64))
    gains = (0.5 * torch.tensor([-2.0, 0.3, 2.0], dtype=torch.float64).tanh()).exp()
    torch.testing.assert_close(output, fields * gains, atol=1e-13, rtol=1e-13)
    assert output[..., -1].square().mean() > fields[..., -1].square().mean()


@pytest.mark.parametrize("wave", [(2, 0), (0, 3), (2, 3), (-2, 3)])
def test_directional_phase_shifts_an_isolated_sinusoid(wave):
    height, width = 15, 20
    layer = FourierModulation2d(1, num_bands=4).double()
    height_shift, width_shift = 0.35, -0.6
    with torch.no_grad():
        layer.projection.bias[4:8] = math.atanh(height_shift)
        layer.projection.bias[8:] = math.atanh(width_shift)
    h, w = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    angle = 2 * math.pi * (wave[0] * h / height + wave[1] * w / width)
    x = rearrange(angle.cos(), "h w -> 1 h w 1")
    output = layer(x, torch.ones(1, 1, dtype=torch.float64))
    phase_shift = (
        2 * math.pi * (wave[0] * height_shift / height + wave[1] * width_shift / width)
    )
    expected = rearrange((angle + phase_shift).cos(), "h w -> 1 h w 1")
    torch.testing.assert_close(output, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("shape", [(2, 8, 12, 3), (2, 9, 11, 3), (2, 8, 11, 3)])
def test_shared_filter_is_equivariant_to_spatial_rolls(shape):
    torch.manual_seed(3)
    layer = FourierModulation2d(5).double()
    torch.nn.init.normal_(layer.projection.weight)
    x = torch.randn(shape, dtype=torch.float64)
    mod = torch.randn(shape[0], 5, dtype=torch.float64)
    actual = layer(x.roll((3, -2), dims=(1, 2)), mod)
    expected = layer(x, mod).roll((3, -2), dims=(1, 2))
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("grid", [(8, 12), (9, 11), (8, 11), (9, 12), (1, 1)])
def test_phase_only_filter_preserves_real_spectrum_and_energy(grid, monkeypatch):
    torch.manual_seed(4)
    layer = FourierModulation2d(3, num_bands=4).double()
    with torch.no_grad():
        layer.projection.weight[4:] = 3 * torch.randn_like(layer.projection.weight[4:])
    x = torch.randn(2, *grid, 3, dtype=torch.float64)
    original_irfft2 = torch.fft.irfft2

    def check_real_compatible_spectrum(spectrum, *args, **kwargs):
        output = original_irfft2(spectrum, *args, **kwargs)
        recovered = torch.fft.rfft2(output, dim=(1, 2), norm="ortho")
        torch.testing.assert_close(recovered, spectrum, atol=1e-13, rtol=1e-13)
        return output

    monkeypatch.setattr(torch.fft, "irfft2", check_real_compatible_spectrum)
    output = layer(x, torch.randn(2, 3, dtype=torch.float64))
    torch.testing.assert_close(
        output.square().sum((1, 2)), x.square().sum((1, 2)), atol=1e-12, rtol=1e-13
    )


@pytest.mark.parametrize("shape", [(2, 8, 12, 3), (2, 9, 11, 3)])
def test_gain_bounds_hold_for_arbitrary_latents(shape):
    torch.manual_seed(9)
    layer = FourierModulation2d(5).double()
    torch.nn.init.normal_(layer.projection.weight, std=10)
    x = torch.randn(shape, dtype=torch.float64)
    output = layer(x, 100 * torch.randn(shape[0], 5, dtype=torch.float64))
    energy_ratio = output.square().sum((1, 2)) / x.square().sum((1, 2))
    assert (energy_ratio >= math.exp(-1) - 1e-12).all(), "All modes must retain energy"
    assert (energy_ratio <= math.exp(1) + 1e-12).all(), "Gain must remain bounded"


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_dtype_and_shape_preserved_under_cpu_autocast(dtype, autocast_enabled):
    torch.manual_seed(6)
    layer = FourierModulation2d(3).to(dtype=dtype)
    torch.nn.init.normal_(layer.projection.weight, std=0.2)
    x = torch.randn(2, 7, 10, 2, dtype=dtype)
    mod = torch.randn(2, 3, dtype=dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        output = layer(x, mod)
    assert output.shape == x.shape
    assert output.dtype == dtype
    assert output.device == x.device
    assert torch.isfinite(output).all()


def test_nonzero_projection_has_finite_gradients_to_both_inputs_and_parameters():
    torch.manual_seed(8)
    layer = FourierModulation2d(5).double()
    torch.nn.init.normal_(layer.projection.weight, std=0.1)
    x = torch.randn(2, 9, 12, 3, dtype=torch.float64, requires_grad=True)
    mod = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    (layer(x, mod) * torch.randn_like(x)).sum().backward()
    for name, tensor in [("x", x), ("mod", mod), *layer.named_parameters()]:
        assert tensor.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"Nonfinite gradient for {name}"
        assert tensor.grad.norm() > 0, f"Zero gradient for {name}"


def test_each_batch_member_uses_its_own_global_latent_and_channels_share_filter():
    layer = FourierModulation2d(1).double()
    with torch.no_grad():
        layer.projection.weight[:4] = 1
    field = torch.randn(1, 8, 12, 1, dtype=torch.float64)
    x = field.expand(2, 8, 12, 3).clone()
    mod = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    output = layer(x, mod)
    gains = (0.5 * mod.tanh()).exp()
    torch.testing.assert_close(output, x * rearrange(gains, "b 1 -> b 1 1 1"))
    assert not torch.allclose(output[0], output[1])


@pytest.mark.parametrize("mod_features", [0, -1, 1.5, True])
def test_invalid_latent_width_rejected(mod_features):
    with pytest.raises(ValueError, match="mod_features must be a positive integer"):
        FourierModulation2d(mod_features)


@pytest.mark.parametrize("num_bands", [0, 1, -1, 2.5, True])
def test_invalid_band_count_rejected(num_bands):
    with pytest.raises(ValueError, match="num_bands must be an integer"):
        FourierModulation2d(3, num_bands=num_bands)


@pytest.mark.parametrize("shape", [(2, 8, 3), (2, 0, 8, 3), (0, 8, 8, 3)])
def test_invalid_field_shape_rejected(shape):
    with pytest.raises(ValueError, match="x must have shape"):
        FourierModulation2d(3)(torch.empty(shape), torch.empty(2, 3))


@pytest.mark.parametrize("shape", [(2, 3, 1), (2, 4), (1, 3)])
def test_invalid_latent_shape_rejected(shape):
    with pytest.raises(ValueError, match="mod must have shape"):
        FourierModulation2d(3)(torch.empty(2, 8, 8, 3), torch.empty(shape))


@pytest.mark.parametrize("target", ["x", "mod"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_non_real_floating_input_rejected(target, dtype):
    x = torch.empty(2, 8, 8, 3)
    mod = torch.empty(2, 3)
    if target == "x":
        x = x.to(dtype=dtype)
    else:
        mod = mod.to(dtype=dtype)
    with pytest.raises(TypeError, match="real floating point"):
        FourierModulation2d(3)(x, mod)
