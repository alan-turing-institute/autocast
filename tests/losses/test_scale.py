import math

import pytest
import torch
from einops import rearrange

from autocast import losses
from autocast.losses import (
    CRPSLoss,
    FourierLowPass,
    GaussianSmoothing,
    ScaleDecomposition,
    SmoothingOperator,
    TransformedLoss,
)
from autocast.types import Tensor


class _Scale(SmoothingOperator):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, tensor: Tensor, *, n_spatial_dims: int) -> Tensor:
        del n_spatial_dims
        return self.factor * tensor


class _WrongShape(SmoothingOperator):
    def forward(self, tensor: Tensor, *, n_spatial_dims: int) -> Tensor:
        del n_spatial_dims
        return tensor[..., :1]


def test_scale_decomposition_telescopes_to_original_field():
    tensor = torch.randn((2, 1, 8, 1))
    decomposition = ScaleDecomposition(
        [("large", _Scale(0.25)), ("medium", _Scale(0.5))],
        residual_name="small",
    )

    components = decomposition(tensor, n_spatial_dims=1)

    assert components.keys() == {"large", "medium", "small"}
    assert torch.equal(sum(components.values(), torch.zeros_like(tensor)), tensor)
    assert torch.equal(components["large"], 0.25 * tensor)
    assert torch.equal(components["medium"], 0.25 * tensor)
    assert torch.equal(components["small"], 0.5 * tensor)


@pytest.mark.parametrize("boundary", ["reflect", "periodic"])
def test_gaussian_smoothing_preserves_constant_fields(boundary):
    tensor = torch.ones((1, 1, 16, 12, 2))

    smoothed = GaussianSmoothing(1, boundary=boundary)(tensor, n_spatial_dims=2)

    assert torch.allclose(smoothed, tensor)


def test_fourier_low_pass_selects_known_modes():
    n_points = 16
    coordinate = torch.arange(n_points)
    low = torch.cos(2 * math.pi * coordinate / n_points)
    high = torch.cos(2 * math.pi * 6 * coordinate / n_points)
    tensor = (low + high).reshape(1, 1, n_points, 1)

    smoothed = FourierLowPass(cutoff=0.1)(tensor, n_spatial_dims=1)

    assert torch.allclose(smoothed.flatten(), low, atol=1e-5)


@pytest.mark.parametrize(
    "operator",
    [
        GaussianSmoothing(2, boundary="periodic"),
        FourierLowPass(0.1),
    ],
)
def test_scale_decomposition_composes_with_ensemble_crps(
    operator: SmoothingOperator,
):
    preds = torch.randn((2, 1, 16, 1, 3), requires_grad=True)
    targets = torch.randn((2, 1, 16, 1))
    decomposition = ScaleDecomposition(
        [("large", operator)],
        residual_name="small",
    )

    value = TransformedLoss(decomposition, CRPSLoss())(preds, targets)
    value.backward()

    assert torch.isfinite(value)
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


def test_scale_decomposition_validates_configuration_and_shape():
    with pytest.raises(ValueError, match="at least one"):
        ScaleDecomposition([])
    with pytest.raises(ValueError, match="unique"):
        ScaleDecomposition([("same", _Scale(0.5))], residual_name="same")

    decomposition = ScaleDecomposition([("large", _WrongShape())])
    with pytest.raises(ValueError, match="preserve field shape"):
        decomposition(torch.randn((1, 1, 8, 2)), n_spatial_dims=1)


def test_scale_decomposition_accepts_pairs_as_lists():
    # Configuration supplies each pair as a list rather than a tuple, so both
    # must build the same decomposition.
    from_tuples = ScaleDecomposition([("large", _Scale(0.25)), ("medium", _Scale(0.5))])
    from_lists = ScaleDecomposition([["large", _Scale(0.25)], ["medium", _Scale(0.5)]])

    assert from_lists.names == from_tuples.names
    tensor = torch.randn(2, 3, 4, 5, 2)
    torch.testing.assert_close(
        from_lists(tensor, n_spatial_dims=2)["large"],
        from_tuples(tensor, n_spatial_dims=2)["large"],
    )


def test_scale_decomposition_rejects_a_malformed_pair():
    with pytest.raises(ValueError, match="pair a name with a smoothing operator"):
        ScaleDecomposition([["large", _Scale(0.25), "extra"]])


@pytest.mark.parametrize(
    "scale", [["large", "not an operator"], [_Scale(0.25), _Scale(0.5)]]
)
def test_scale_decomposition_rejects_a_mistyped_pair(scale):
    with pytest.raises(ValueError, match="pair a name with a SmoothingOperator"):
        ScaleDecomposition([scale])


def test_scale_operators_are_exported():
    for name in (
        "SmoothingOperator",
        "GaussianSmoothing",
        "FourierLowPass",
        "ScaleDecomposition",
    ):
        assert hasattr(losses, name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_pass_promotes_narrow_inputs_with_gradients(dtype):
    tensor = torch.randn(2, 2, 7, 9, 2, 3).to(dtype).requires_grad_()
    operator = FourierLowPass(0.25)
    actual = operator(tensor, n_spatial_dims=2)
    expected = operator(tensor.detach().float(), n_spatial_dims=2)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize(("size", "mode"), [(10, 3), (30, 3)])
def test_low_pass_keeps_cutoff_modes_in_both_precisions(size, mode):
    coordinate = torch.arange(size, dtype=torch.float64)
    signal = torch.cos(2 * math.pi * mode * coordinate / size).reshape(1, 1, size, 1)
    operator = FourierLowPass(mode / size)
    for dtype in (torch.float32, torch.float64):
        tensor = signal.to(dtype)
        torch.testing.assert_close(operator(tensor, n_spatial_dims=1), tensor)


@pytest.mark.parametrize("parameter", ["sigma", "truncate"])
@pytest.mark.parametrize("value", [0, -1, math.inf, -math.inf, math.nan])
def test_gaussian_rejects_nonpositive_or_nonfinite_parameters(parameter, value):
    arguments = {"sigma": 1, "truncate": 4, parameter: value}
    with pytest.raises(ValueError, match=f"{parameter} must be finite and positive"):
        GaussianSmoothing(**arguments)


@pytest.mark.parametrize("cutoff", [0, -0.1, 0.51, math.inf, math.nan])
def test_low_pass_rejects_invalid_cutoffs(cutoff):
    with pytest.raises(ValueError, match="cutoff must be in"):
        FourierLowPass(cutoff)


def test_gaussian_boundary_contract():
    with pytest.raises(ValueError, match="Unknown scale boundary"):
        GaussianSmoothing(1, boundary="unknown")  # pyright: ignore[reportArgumentType]
    tensor = torch.ones(1, 1, 3, 1)
    with pytest.raises(ValueError, match="longer than the kernel radius"):
        GaussianSmoothing(1)(tensor, n_spatial_dims=1)
    torch.testing.assert_close(
        GaussianSmoothing(1, boundary="periodic")(tensor, n_spatial_dims=1), tensor
    )
    zero_padded = GaussianSmoothing(1, boundary="zero")(tensor, n_spatial_dims=1)
    assert (zero_padded < tensor).all()
    assert zero_padded[0, 0, 0, 0] < zero_padded[0, 0, 1, 0]


@pytest.mark.parametrize("shape", [(5,), (5, 7), (5, 7, 3)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_periodic_gaussian_matches_explicit_axiswise_convolution(shape, dtype):
    tensor = torch.randn(2, 2, *shape, 2, 3, dtype=dtype)
    # Radius 4 exceeds the last spatial axis in the 3D case.
    operator = GaussianSmoothing(1, boundary="periodic")
    expected = tensor
    for axis in range(2, 2 + len(shape)):
        expected = sum(
            (
                weight * torch.roll(expected, shifts=shift, dims=axis)
                for shift, weight in zip(
                    range(-operator.radius, operator.radius + 1),
                    operator.kernel,
                    strict=True,
                )
            ),
            torch.zeros_like(tensor),
        )
    actual = operator(tensor, n_spatial_dims=len(shape))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(7,), (7, 9), (7, 5, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex128])
def test_low_pass_preserves_layout_and_only_transforms_spatial_axes(shape, dtype):
    tensor = torch.randn(2, 2, *shape, 2, 3, dtype=dtype)
    operator = FourierLowPass(0.25)
    actual = operator(tensor, n_spatial_dims=len(shape))
    # Independently filter each batch/time/channel/member slice.
    slices = rearrange(tensor, "b t ... c m -> (b t c m) ...")
    expected_slices = []
    for field in slices:
        expanded = rearrange(field, "... -> 1 1 ... 1")
        filtered = operator(expanded, n_spatial_dims=len(shape))
        expected_slices.append(rearrange(filtered, "1 1 ... 1 -> ..."))
    expected = rearrange(
        torch.stack(expected_slices),
        "(b t c m) ... -> b t ... c m",
        b=2,
        t=2,
        c=2,
        m=3,
    )
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)


def test_low_pass_rejects_diagonal_corners_at_axis_nyquist():
    axis = torch.arange(8, dtype=torch.float64)
    corner = torch.cos(math.pi * (axis[:, None] + axis[None, :]))
    tensor = rearrange(corner, "x y -> 1 1 x y 1")
    actual = FourierLowPass(0.5)(tensor, n_spatial_dims=2)
    torch.testing.assert_close(actual, torch.zeros_like(tensor))


@pytest.mark.parametrize(
    "operator", [GaussianSmoothing(0.5, boundary="periodic"), FourierLowPass(0.3)]
)
def test_smoothing_operators_pass_gradcheck(operator):
    tensor = torch.randn(1, 1, 3, 4, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: operator(value, n_spatial_dims=2), (tensor,), fast_mode=True
    )


@pytest.mark.parametrize("ensemble", [False, True])
def test_real_scale_bands_reconstruct_and_roundtrip_buffers(ensemble):
    shape = (2, 2, 7, 9, 2, 3) if ensemble else (2, 2, 7, 9, 2)
    tensor = torch.randn(shape, dtype=torch.float64)
    decomposition = ScaleDecomposition(
        [
            ("large", GaussianSmoothing(2, boundary="periodic")),
            ("medium", GaussianSmoothing(1, boundary="periodic")),
        ]
    ).double()
    components = decomposition(tensor, n_spatial_dims=2)
    torch.testing.assert_close(sum(components.values()), tensor)
    restored = ScaleDecomposition(
        [
            ("large", GaussianSmoothing(2, boundary="periodic")),
            ("medium", GaussianSmoothing(1, boundary="periodic")),
        ]
    ).double()
    restored.load_state_dict(decomposition.state_dict(), strict=True)
    actual = restored(tensor, n_spatial_dims=2)
    for name in components:
        torch.testing.assert_close(actual[name], components[name])


def test_low_pass_under_cpu_autocast_has_finite_gradients():
    tensor = torch.randn(2, 1, 7, 9, 2, 3, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = FourierLowPass(0.25)(tensor, n_spatial_dims=2)
        loss = result.square().mean()
    loss.backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "operator", [GaussianSmoothing(1, boundary="periodic"), FourierLowPass(0.3)]
)
def test_scale_operator_moves_to_cuda_and_backpropagates(operator):
    tensor = torch.randn(1, 1, 7, 9, 2, 3, device="cuda", requires_grad=True)
    expected = operator(tensor.detach().cpu(), n_spatial_dims=2)
    result = operator.to("cuda")(tensor, n_spatial_dims=2)
    torch.testing.assert_close(result.cpu(), expected)
    result.square().mean().backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
