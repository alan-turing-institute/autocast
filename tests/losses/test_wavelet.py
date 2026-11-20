import math

import pytest
import torch
from torch import nn

from autocast.losses import (
    CoefficientLpLoss,
    ComponentBranch,
    ComponentObjective,
    CRPSLoss,
    DTCWTDecomposition,
    FieldDecomposition,
    FourierDecomposition,
    MagnitudeTransform,
    PowerTransform,
    RMSAmplitudeTransform,
    TransformedLoss,
    WaveletDecomposition,
    WeightedSumLoss,
)
from autocast.types import Tensor


def test_coefficient_lp_loss_supports_complex_ensembles():
    preds = torch.tensor([1 + 2j, 3 + 4j]).reshape(1, 1, 1, 1, 2)
    targets = torch.tensor([1 + 1j]).reshape(1, 1, 1, 1)

    assert torch.isclose(CoefficientLpLoss(p=2)(preds, targets), torch.tensor(7.0))

    with pytest.raises(ValueError, match="Unknown reduction"):
        CoefficientLpLoss(reduction="median")


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_coefficient_lp_loss_keeps_coincident_gradients_finite(dtype, p):
    preds = torch.zeros(1, 1, 2, 1, 3, dtype=dtype, requires_grad=True)
    targets = torch.zeros(1, 1, 2, 1, dtype=dtype)
    value = CoefficientLpLoss(p=p)(preds, targets)
    assert value.item() == 0
    value.backward()
    assert preds.grad is not None
    torch.testing.assert_close(preds.grad, torch.zeros_like(preds))


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_coefficient_lp_matches_nonzero_values_and_gradients(
    complex_values, p, reduction
):
    values = [1 + 2j, 3 + 4j] if complex_values else [2.0, 4.0]
    dtype = torch.complex128 if complex_values else torch.float64
    preds = torch.tensor(values, dtype=dtype).reshape(1, 1, 1, 1, 2).requires_grad_()
    targets = torch.full((1, 1, 1, 1), 1 + 1j if complex_values else 1.0, dtype=dtype)
    difference = preds.detach() - targets.unsqueeze(-1)
    radius = difference.abs()
    expected_values = radius**p
    if reduction != "none":
        expected_values = getattr(expected_values, reduction)()
    value = CoefficientLpLoss(p=p, reduction=reduction)(preds, targets)
    torch.testing.assert_close(value, expected_values)
    value.sum().backward()
    expected_gradient = p * radius ** (p - 2) * difference
    if reduction == "mean":
        expected_gradient = expected_gradient / preds.numel()
    assert preds.grad is not None
    torch.testing.assert_close(preds.grad, expected_gradient)


def test_haar_wavelet_matches_known_multilevel_coefficients():
    tensor = torch.tensor([1.0, 3.0, 5.0, 7.0]).reshape(1, 1, 4, 1)
    decomposition = WaveletDecomposition("haar", levels=2)

    components = decomposition(tensor, n_spatial_dims=1)

    assert components.keys() == {
        "level_1/H",
        "level_2/H",
        "level_2/approximation",
    }
    assert torch.allclose(
        components["level_1/H"].flatten(),
        torch.tensor([-math.sqrt(2.0), -math.sqrt(2.0)]),
    )
    assert torch.allclose(components["level_2/H"].flatten(), torch.tensor([-4.0]))
    assert torch.allclose(
        components["level_2/approximation"].flatten(), torch.tensor([8.0])
    )


@pytest.mark.parametrize(
    ("boundary", "last_high"),
    [
        ("periodic", math.sqrt(2.0)),
        ("reflect", 1 / math.sqrt(2.0)),
    ],
)
def test_haar_wavelet_boundary_modes(boundary, last_high):
    tensor = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1)
    decomposition = WaveletDecomposition("haar", boundary=boundary)

    high = decomposition(tensor, n_spatial_dims=1)["level_1/H"].flatten()

    assert torch.allclose(high, torch.tensor([-1 / math.sqrt(2.0), last_high]))


def test_haar_wavelet_nd_transform_preserves_energy():
    tensor = torch.randn((2, 1, 8, 4, 3))
    decomposition = WaveletDecomposition("haar", levels=2)

    components = decomposition(tensor, n_spatial_dims=2)
    transformed_energy = torch.stack(
        [component.square().sum() for component in components.values()]
    ).sum()

    assert torch.allclose(transformed_energy, tensor.square().sum(), atol=1e-5)


def test_wavelet_decomposition_supports_named_bases():
    tensor = torch.randn((1, 1, 4, 4, 1))
    haar = WaveletDecomposition("haar")(tensor, n_spatial_dims=2)
    db2 = WaveletDecomposition("db2")(tensor, n_spatial_dims=2)

    assert haar.keys() == db2.keys()
    assert haar["level_1/HH"].shape != db2["level_1/HH"].shape


def test_wavelet_decomposition_supports_three_spatial_axes():
    tensor = torch.randn((1, 1, 4, 4, 4, 1))

    components = WaveletDecomposition()(tensor, n_spatial_dims=3)

    assert components.keys() == {
        "level_1/LLH",
        "level_1/LHL",
        "level_1/LHH",
        "level_1/HLL",
        "level_1/HLH",
        "level_1/HHL",
        "level_1/HHH",
        "level_1/approximation",
    }


@pytest.mark.parametrize(
    ("biort", "qshift"),
    [("near_sym_a", "qshift_a"), ("near_sym_b", "qshift_b")],
)
def test_dtcwt_decomposition_matches_backend_coefficients(biort, qshift):
    tensor = torch.randn((1, 1, 16, 12, 2))
    decomposition = DTCWTDecomposition(levels=2, biort=biort, qshift=qshift)

    components = decomposition(tensor, n_spatial_dims=2)
    channels_first = tensor.permute(0, 1, 4, 2, 3).flatten(0, 1)
    expected_lowpass, expected_highpasses = decomposition.transform(channels_first)
    expected_direction = torch.view_as_complex(expected_highpasses[0].contiguous())[
        :, :, 0
    ]
    expected_keys = {
        f"level_{level}/direction_{direction}"
        for level in (1, 2)
        for direction in range(6)
    }
    expected_keys.add("level_2/lowpass")

    assert components.keys() == expected_keys
    assert torch.equal(
        components["level_1/direction_0"],
        expected_direction.permute(0, 2, 3, 1).unsqueeze(1),
    )
    assert torch.equal(
        components["level_2/lowpass"],
        expected_lowpass.permute(0, 2, 3, 1).unsqueeze(1),
    )


def test_dtcwt_composes_with_magnitude_crps_for_ensembles():
    preds = torch.randn((2, 1, 8, 8, 2, 3), requires_grad=True)
    targets = torch.randn((2, 1, 8, 8, 2))
    objective = ComponentObjective(
        [
            ComponentBranch(
                CRPSLoss(),
                transforms=[MagnitudeTransform()],
                components=["level_1/direction_0"],
            )
        ]
    )

    value = TransformedLoss(DTCWTDecomposition(include_lowpass=False), objective)(
        preds, targets
    )
    value.backward()

    assert torch.isfinite(value)
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


def test_coefficient_transforms_support_complex_values():
    # Transforms consume components, which carry the field layout.
    tensor = torch.tensor([3 + 4j]).reshape(1, 1, 1, 1)

    magnitude = MagnitudeTransform()(tensor, n_spatial_dims=1)
    power = PowerTransform()(tensor, n_spatial_dims=1)

    assert torch.equal(magnitude, torch.tensor([5.0]).reshape(1, 1, 1, 1))
    assert torch.equal(power, torch.tensor([25.0]).reshape(1, 1, 1, 1))


def test_dtcwt_requires_a_real_2d_field():
    decomposition = DTCWTDecomposition()

    with pytest.raises(ValueError, match="exactly two"):
        decomposition(torch.randn((1, 1, 8, 1)), n_spatial_dims=1)
    with pytest.raises(ValueError, match="real-valued"):
        decomposition(
            torch.randn((1, 1, 8, 8, 1), dtype=torch.complex64),
            n_spatial_dims=2,
        )


@pytest.mark.parametrize("decomposition", [DTCWTDecomposition, WaveletDecomposition])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_wavelet_backends_keep_mixed_precision_gradients_finite(decomposition, dtype):
    field = torch.randn(1, 1, 8, 8, 1, 3).to(dtype).requires_grad_()
    transform = decomposition()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        components = transform(field, n_spatial_dims=2)
        loss = torch.stack(
            [component.abs().square().sum() for component in components.values()]
        ).sum()
    assert all(value.real.dtype == torch.float32 for value in components.values())
    loss.backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


@pytest.mark.parametrize("decomposition", [DTCWTDecomposition, WaveletDecomposition])
def test_wavelet_double_precision_and_registered_filters(decomposition):
    field = torch.randn(1, 2, 8, 8, 2, 3, dtype=torch.float64, requires_grad=True)
    transform = decomposition().double()
    components = transform(field, n_spatial_dims=2)
    assert all(value.real.dtype == torch.float64 for value in components.values())
    restored = decomposition().double()
    restored.load_state_dict(transform.state_dict())
    expected = restored(field, n_spatial_dims=2)
    for name, value in components.items():
        torch.testing.assert_close(value, expected[name])
    torch.stack(
        [value.abs().square().sum() for value in components.values()]
    ).sum().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_coefficient_lp_loss_validates_a_finite_positive_power(value):
    with pytest.raises(ValueError, match="finite and positive"):
        CoefficientLpLoss(p=value)


@pytest.mark.parametrize(
    ("decomposition", "score"),
    [
        (FourierDecomposition("magnitude"), CRPSLoss()),
        (WaveletDecomposition("db2"), CRPSLoss()),
        (WaveletDecomposition(), CoefficientLpLoss(p=1)),
    ],
)
def test_decompositions_compose_with_scores(decomposition, score):
    preds = torch.randn((2, 2, 4, 4, 1, 3), requires_grad=True)
    targets = torch.randn((2, 2, 4, 4, 1))

    value = TransformedLoss(decomposition, score)(preds, targets)
    value.backward()

    assert torch.isfinite(value)
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


class _TwoBandDecomposition(FieldDecomposition):
    def forward(self, tensor: Tensor, *, n_spatial_dims: int) -> dict[str, Tensor]:
        del n_spatial_dims
        return {"low": tensor, "high": 2 * tensor}


def test_transformed_loss_normalizes_partial_component_weights():
    preds = torch.ones((1, 1, 2, 1))
    targets = torch.zeros_like(preds)
    loss = TransformedLoss(
        _TwoBandDecomposition(),
        CoefficientLpLoss(p=1),
        component_weights={"high": 0.5},
    )

    assert torch.isclose(loss(preds, targets), torch.tensor(4 / 3))
    with pytest.raises(ValueError, match="Unknown component"):
        TransformedLoss(
            _TwoBandDecomposition(),
            CoefficientLpLoss(),
            component_weights={"missing": 1},
        )(preds, targets)
    with pytest.raises(ValueError, match="At least one component weight"):
        TransformedLoss(
            _TwoBandDecomposition(),
            CoefficientLpLoss(),
            component_weights={"low": 0, "high": 0},
        )(preds, targets)


def test_weighted_sum_loss_composes_losses():
    preds = torch.ones((1, 1, 1, 1))
    targets = torch.zeros_like(preds)
    loss = WeightedSumLoss([nn.L1Loss(), nn.MSELoss()], weights=[0.25, 2])

    assert torch.isclose(loss(preds, targets), torch.tensor(2.25))


def _complex_field(shape: tuple[int, ...] = (2, 3, 8, 6, 4)) -> torch.Tensor:
    return torch.randn(*shape) + 1j * torch.randn(*shape)


def test_rms_amplitude_matches_the_manual_reduction():
    torch.manual_seed(53)
    tensor = _complex_field()

    reduced = RMSAmplitudeTransform()(tensor, n_spatial_dims=2)

    torch.testing.assert_close(
        reduced, tensor.abs().square().mean(dim=(2, 3), keepdim=True).sqrt()
    )


def test_rms_amplitude_keeps_the_field_layout():
    # Spatial axes are kept with extent one, so the result is still a
    # component and every retained axis survives.
    tensor = _complex_field((2, 3, 8, 6, 4))

    reduced = RMSAmplitudeTransform()(tensor, n_spatial_dims=2)

    assert reduced.shape == (2, 3, 1, 1, 4)


def test_rms_amplitude_discards_position_but_not_strength():
    # Permuting a component within its spatial axes leaves the reduction
    # unchanged, while scaling it does not.
    torch.manual_seed(59)
    tensor = _complex_field()
    transform = RMSAmplitudeTransform()

    reduced = transform(tensor, n_spatial_dims=2)
    shuffled = transform(
        torch.roll(tensor, shifts=(3, 2), dims=(2, 3)), n_spatial_dims=2
    )
    scaled = transform(2 * tensor, n_spatial_dims=2)

    torch.testing.assert_close(reduced, shuffled)
    torch.testing.assert_close(scaled, 2 * reduced)


def test_rms_amplitude_keeps_gradients_finite_at_zero():
    tensor = torch.zeros(1, 1, 3, 2, requires_grad=True)

    RMSAmplitudeTransform()(tensor, n_spatial_dims=1).sum().backward()

    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
