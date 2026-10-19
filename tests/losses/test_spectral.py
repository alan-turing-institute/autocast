import math

import pytest
import torch

from autocast.losses.decomposition import TransformedLoss
from autocast.losses.ensemble import CoefficientCRPSLoss
from autocast.losses.spectral import FourierDecomposition


def _explicit_dft(tensor, spatial_axes, norm):
    """Small independent DFT matrices over only the requested spatial axes."""
    coefficients = tensor.to(torch.complex128)
    for axis in spatial_axes:
        size = tensor.shape[axis]
        coordinates = torch.arange(size, dtype=torch.float64)
        phase = coordinates[:, None] * coordinates[None, :] * (-2j * math.pi / size)
        kernel = phase.exp()
        if norm == "ortho":
            kernel = kernel / math.sqrt(size)
        elif norm == "forward":
            kernel = kernel / size
        coefficients = torch.einsum(
            "kn,n...->k...", kernel, coefficients.movedim(axis, 0)
        )
        coefficients = coefficients.movedim(0, axis)
    return coefficients


@pytest.mark.parametrize("spatial_shape", [(5,), (3, 5), (2, 3, 5)])
@pytest.mark.parametrize("members", [None, 4])
@pytest.mark.parametrize("norm", ["ortho", "forward", "backward"])
@pytest.mark.parametrize("representation", ["complex", "magnitude", "power"])
def test_fourier_matches_explicit_dft_and_preserves_other_axes(
    spatial_shape, members, norm, representation
):
    generator = torch.Generator().manual_seed(4)
    shape = (2, 3, *spatial_shape, 2)
    if members is not None:
        shape = (*shape, members)
    tensor = torch.randn(shape, dtype=torch.float64, generator=generator)
    expected = _explicit_dft(tensor, range(2, 2 + len(spatial_shape)), norm)
    if representation == "magnitude":
        expected = expected.abs()
    elif representation == "power":
        expected = expected.abs().square()

    result = FourierDecomposition(representation, norm)(
        tensor, n_spatial_dims=len(spatial_shape)
    )
    assert result.keys() == {"fourier"}
    assert result["fourier"].shape == tensor.shape
    torch.testing.assert_close(result["fourier"], expected, rtol=1e-12, atol=1e-12)


def test_full_transform_retains_both_phases_of_known_real_mode():
    size, mode, phase = 9, 2, math.pi / 3
    coordinate = torch.arange(size, dtype=torch.float64)
    field = torch.cos(2 * math.pi * mode * coordinate / size + phase)
    actual = FourierDecomposition()(field.reshape(1, 1, size, 1), n_spatial_dims=1)[
        "fourier"
    ]
    expected = torch.zeros(size, dtype=torch.complex128)
    expected[mode] = (
        math.sqrt(size)
        / 2
        * torch.tensor(
            complex(math.cos(phase), math.sin(phase)), dtype=torch.complex128
        )
    )
    expected[-mode] = expected[mode].conj()
    torch.testing.assert_close(actual.flatten(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (torch.float16, torch.complex64),
        (torch.bfloat16, torch.complex64),
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
        (torch.complex64, torch.complex64),
        (torch.complex128, torch.complex128),
    ],
)
def test_fourier_dtype_device_and_gradients(dtype, expected_dtype):
    generator = torch.Generator().manual_seed(7)
    tensor = torch.randn(
        1, 2, 3, 5, 2, dtype=dtype, generator=generator, requires_grad=True
    )
    result = FourierDecomposition()(tensor, n_spatial_dims=2)["fourier"]
    expected = _explicit_dft(tensor.detach(), (2, 3), "ortho")
    assert result.dtype == expected_dtype
    assert result.device == tensor.device
    torch.testing.assert_close(
        result.to(torch.complex128), expected, rtol=1e-5, atol=1e-6
    )
    result.abs().sum().backward()
    assert tensor.grad is not None
    assert tensor.grad.dtype == tensor.dtype
    assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transform_composes_with_coefficient_loss_and_backpropagates(dtype):
    generator = torch.Generator().manual_seed(10)
    preds = torch.randn(
        2, 3, 3, 5, 2, 4, dtype=dtype, generator=generator, requires_grad=True
    )
    targets = torch.randn(2, 3, 3, 5, 2, dtype=dtype, generator=generator)
    score = CoefficientCRPSLoss(0.95, reduction="none")
    loss = TransformedLoss(FourierDecomposition(), score=score)
    result = loss(preds, targets)
    assert result.shape == targets.shape
    assert result.dtype == dtype
    result.mean().backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()
    assert torch.count_nonzero(preds.grad) > 0


def test_fourier_coefficient_objective_passes_double_gradcheck():
    generator = torch.Generator().manual_seed(14)
    preds = torch.randn(
        1, 1, 3, 1, 2, dtype=torch.float64, generator=generator, requires_grad=True
    )
    targets = torch.randn(
        1, 1, 3, 1, dtype=torch.float64, generator=generator, requires_grad=True
    )
    loss = TransformedLoss(FourierDecomposition(), score=CoefficientCRPSLoss(0.95))
    assert torch.autograd.gradcheck(loss, (preds, targets))


def test_cpu_autocast_with_bfloat16_inputs_uses_promoted_fft():
    preds = torch.arange(60.0).reshape(1, 1, 3, 5, 2, 2).bfloat16().requires_grad_()
    targets = torch.zeros(1, 1, 3, 5, 2, dtype=torch.bfloat16)
    loss = TransformedLoss(FourierDecomposition(), score=CoefficientCRPSLoss(0.95))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = loss(preds, targets)
    assert result.dtype == torch.float32
    result.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_autocast_keeps_fft_and_gradients_on_device():
    preds = torch.randn(
        1, 2, 3, 5, 2, 3, device="cuda", dtype=torch.float16, requires_grad=True
    )
    targets = torch.randn(1, 2, 3, 5, 2, device="cuda", dtype=torch.float16)
    loss = TransformedLoss(FourierDecomposition(), score=CoefficientCRPSLoss(0.95))
    with torch.autocast("cuda", dtype=torch.float16):
        result = loss(preds, targets)
    assert result.dtype == torch.float32
    assert result.device == preds.device
    result.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"representation": "phase"}, "Unknown Fourier representation"),
        ({"norm": "none"}, "Unknown Fourier normalization"),
    ],
)
def test_invalid_fourier_options_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FourierDecomposition(**kwargs)


@pytest.mark.parametrize("n_spatial_dims", [0, -1, 3])
def test_invalid_spatial_rank_is_rejected(n_spatial_dims):
    with pytest.raises(ValueError, match="Invalid n_spatial_dims"):
        FourierDecomposition()(torch.ones(1, 1, 3, 2), n_spatial_dims=n_spatial_dims)
