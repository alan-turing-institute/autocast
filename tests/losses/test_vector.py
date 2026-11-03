import math

import pytest
import torch

from autocast.losses import HelmholtzDecomposition, VelocityScoreLoss


def _vector_score_reference(preds, target, alpha):
    """Independent real-vector implementation, with channels before members."""
    members = preds.shape[-1]
    skill = torch.linalg.vector_norm(preds - target.unsqueeze(-1), dim=-2).mean(-1)
    pairs = sum(
        torch.linalg.vector_norm(preds[..., i] - preds[..., j], dim=-1)
        for i in range(members)
        for j in range(i + 1, members)
    )
    coefficient = (1 - alpha) / members**2 + alpha / (members * (members - 1))
    return (skill - coefficient * pairs).mean()


@pytest.mark.parametrize("alpha", [0.0, 0.95, 1.0])
def test_joint_matches_real_bivariate_score(alpha):
    generator = torch.Generator().manual_seed(91)
    preds = torch.randn(2, 3, 5, 7, 2, 8, dtype=torch.float64, generator=generator)
    targets = torch.randn(2, 3, 5, 7, 2, dtype=torch.float64, generator=generator)
    result = VelocityScoreLoss(alpha=alpha)(preds, targets)
    torch.testing.assert_close(
        result, _vector_score_reference(preds, targets, alpha), atol=1e-12, rtol=1e-12
    )


def test_channel_selection_units_and_unused_height_gradient():
    generator = torch.Generator().manual_seed(101)
    preds = torch.randn(2, 1, 8, 9, 3, 4, dtype=torch.float64, generator=generator)
    targets = torch.randn(2, 1, 8, 9, 3, dtype=torch.float64, generator=generator)
    preds.requires_grad_()
    physical_preds = torch.stack((preds[..., 1, :] * 2, preds[..., 2, :] * 3), dim=4)
    physical_targets = torch.stack((targets[..., 1] * 2, targets[..., 2] * 3), dim=4)
    loss = VelocityScoreLoss(channel_stds=(2, 3), channels=(1, 2), velocity_scale=4)
    actual = loss(preds, targets)
    expected = _vector_score_reference(physical_preds, physical_targets, 0.95) / 4
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    grad = torch.autograd.grad(actual, preds)[0]
    assert torch.count_nonzero(grad[..., 0, :]) == 0
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("mode", ["joint", "helmholtz"])
def test_fixed_channel_means_cancel_after_scaling_and_projection(mode):
    generator = torch.Generator().manual_seed(102)
    preds = torch.randn(2, 1, 8, 9, 2, 4, dtype=torch.float64, generator=generator)
    targets = torch.randn(2, 1, 8, 9, 2, dtype=torch.float64, generator=generator)
    offset = torch.tensor((12.0, -4.0)).reshape(1, 1, 1, 1, 2)
    loss = VelocityScoreLoss(channel_stds=(2, 3), mode=mode)
    torch.testing.assert_close(
        loss(preds, targets), loss(preds + offset.unsqueeze(-1), targets + offset)
    )


def test_joint_rotation_invariance_and_locations_retained():
    generator = torch.Generator().manual_seed(103)
    preds = torch.randn(1, 2, 8, 9, 2, 4, dtype=torch.float64, generator=generator)
    targets = torch.randn(1, 2, 8, 9, 2, dtype=torch.float64, generator=generator)
    rotated_preds = torch.stack((-preds[..., 1, :], preds[..., 0, :]), dim=4)
    rotated_targets = torch.stack((-targets[..., 1], targets[..., 0]), dim=4)
    score = VelocityScoreLoss(reduction="none")
    actual = score(preds, targets)
    assert actual.shape == (1, 2, 8, 9, 1)
    torch.testing.assert_close(actual, score(rotated_preds, rotated_targets))
    # Distinct physical locations are not combined into one vector norm.
    assert not torch.allclose(actual, actual.mean().expand_as(actual))


@pytest.mark.parametrize("shape", [(7, 9), (8, 10), (7, 10), (1, 4)])
@pytest.mark.parametrize("members", [None, 3])
def test_helmholtz_real_reconstruction_orthogonality_and_idempotence(shape, members):
    generator = torch.Generator().manual_seed(104)
    layout = (2, 1, *shape, 2) + (() if members is None else (members,))
    tensor = torch.randn(*layout, dtype=torch.float64, generator=generator)
    split = HelmholtzDecomposition(domain_lengths=(2.0, 3.0))
    parts = split(tensor, n_spatial_dims=2)
    sol, div = parts["solenoidal"], parts["divergent"]
    assert not sol.is_complex()
    assert not div.is_complex()
    torch.testing.assert_close(sol + div, tensor, atol=1e-12, rtol=1e-12)
    assert (sol * div).sum().abs().item() < 1e-11
    sol_parts = split(sol, n_spatial_dims=2)
    div_parts = split(div, n_spatial_dims=2)
    torch.testing.assert_close(sol_parts["solenoidal"], sol, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(div_parts["divergent"], div, atol=1e-12, rtol=1e-12)
    assert sol_parts["divergent"].abs().max().item() < 1e-12
    assert div_parts["solenoidal"].abs().max().item() < 1e-12


def test_pure_rotational_and_divergent_modes_respect_physical_periods():
    n_x, n_y = 17, 19
    x = torch.arange(n_x, dtype=torch.float64)[:, None] / n_x
    y = torch.arange(n_y, dtype=torch.float64)[None, :] / n_y
    wave = torch.sin(2 * math.pi * (2 * x + 3 * y))
    # Periods (2,3) make the physical wavevector (2/2,3/3) = (1,1).
    divergent = torch.stack((wave, wave), dim=-1)[None, None]
    solenoidal = torch.stack((-wave, wave), dim=-1)[None, None]
    parts = HelmholtzDecomposition((2, 3))(divergent + 2 * solenoidal, n_spatial_dims=2)
    torch.testing.assert_close(parts["divergent"], divergent, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        parts["solenoidal"], 2 * solenoidal, atol=1e-12, rtol=1e-12
    )


def test_velocity_scales_are_restored_before_helmholtz_projection():
    x = torch.arange(9, dtype=torch.float64)[:, None] / 9
    y = torch.arange(11, dtype=torch.float64)[None, :] / 11
    wave = torch.sin(2 * math.pi * (x + y))
    # Physical velocity is (wave,wave), purely divergent. Its normalized
    # channels differ, so projecting before restoring their scales is wrong.
    normalized = torch.stack((wave / 2, wave / 3), dim=-1)[None, None]
    preds = normalized.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3)
    target = torch.zeros_like(normalized)
    actual = VelocityScoreLoss(channel_stds=(2, 3), velocity_scale=1, mode="helmholtz")(
        preds, target
    )
    expected = math.sqrt(2) * wave.abs().mean() / 2
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", ["dc", "x_nyquist", "y_nyquist", "xy_nyquist"])
def test_derivative_null_modes_are_kept_in_solenoidal_part(mode):
    x = torch.arange(8)[:, None]
    y = torch.arange(10)[None, :]
    phase = torch.zeros(8, 10, dtype=torch.int64)
    if mode in ("x_nyquist", "xy_nyquist"):
        phase = phase + x
    if mode in ("y_nyquist", "xy_nyquist"):
        phase = phase + y
    wave = (-1.0) ** phase
    tensor = torch.stack((wave, -2 * wave), dim=-1)[None, None].double()
    parts = HelmholtzDecomposition()(tensor, n_spatial_dims=2)
    torch.testing.assert_close(parts["solenoidal"], tensor, atol=1e-12, rtol=1e-12)
    assert parts["divergent"].abs().max().item() < 1e-12


def test_mixed_nyquist_mode_is_projected_with_real_derivative_convention():
    x = torch.arange(8, dtype=torch.float64)[:, None]
    y = torch.arange(10, dtype=torch.float64)[None, :] / 10
    wave = (-1.0) ** x * torch.cos(2 * math.pi * y)
    tensor = torch.stack((wave, 3 * wave), dim=-1)[None, None]
    parts = HelmholtzDecomposition()(tensor, n_spatial_dims=2)
    expected_div = torch.stack((torch.zeros_like(wave), 3 * wave), dim=-1)[None, None]
    torch.testing.assert_close(parts["divergent"], expected_div, atol=1e-12, rtol=1e-12)


def test_decomposed_score_is_mean_of_two_local_scores_and_keeps_dc():
    generator = torch.Generator().manual_seed(105)
    preds = torch.randn(1, 2, 8, 9, 2, 4, dtype=torch.float64, generator=generator)
    targets = torch.randn(1, 2, 8, 9, 2, dtype=torch.float64, generator=generator)
    split = HelmholtzDecomposition()
    pp, tp = split(preds, n_spatial_dims=2), split(targets, n_spatial_dims=2)
    expected = sum(_vector_score_reference(pp[k], tp[k], 0.95) for k in pp) / 2
    actual = VelocityScoreLoss(mode="helmholtz")(preds, targets)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    constant_preds = torch.ones_like(preds)
    constant_targets = torch.zeros_like(targets)
    joint = VelocityScoreLoss()(constant_preds, constant_targets)
    decomposed = VelocityScoreLoss(mode="helmholtz")(constant_preds, constant_targets)
    torch.testing.assert_close(decomposed, joint / 2)


@pytest.mark.parametrize("mode", ["joint", "helmholtz"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zero_loss_and_finite_gradients_at_coincidences(mode, dtype):
    preds = torch.ones(1, 1, 6, 8, 2, 3, dtype=dtype, requires_grad=True)
    targets = torch.ones(1, 1, 6, 8, 2, dtype=dtype)
    result = VelocityScoreLoss(mode=mode)(preds, targets)
    assert result.dtype == torch.float32
    assert result.item() == 0
    gradient = torch.autograd.grad(result, preds)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("mode", ["joint", "helmholtz"])
def test_gradcheck(mode):
    generator = torch.Generator().manual_seed(106)
    preds = torch.randn(1, 1, 3, 3, 2, 2, dtype=torch.float64, generator=generator)
    targets = torch.randn(1, 1, 3, 3, 2, dtype=torch.float64, generator=generator)
    preds.requires_grad_()
    assert torch.autograd.gradcheck(VelocityScoreLoss(mode=mode), (preds, targets))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channels": (1, 1)},
        {"channels": (-1, 2)},
        {"channels": (False, 1)},
        {"channel_stds": (1, 0)},
        {"channel_stds": (1, math.nan)},
        {"velocity_scale": 0},
        {"mode": "invalid"},
        {"domain_lengths": (1, -2)},
        {"domain_lengths": (1,)},
        {"alpha": 2},
    ],
)
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(
        ValueError,
        match=r"channels|channel_stds|velocity_scale|mode|domain_lengths|alpha",
    ):
        VelocityScoreLoss(**kwargs)


def test_invalid_layout_dtype_and_channel_selection_fail():
    loss = VelocityScoreLoss(channels=(1, 2))
    with pytest.raises(ValueError, match="channel index"):
        loss(torch.zeros(1, 1, 4, 4, 2, 3), torch.zeros(1, 1, 4, 4, 2))
    with pytest.raises(ValueError, match="requires"):
        loss(torch.zeros(1, 1, 4, 3, 3), torch.zeros(1, 1, 4, 3))
    with pytest.raises(TypeError, match="real"):
        loss(
            torch.zeros(1, 1, 4, 4, 3, 3, dtype=torch.complex64),
            torch.zeros(1, 1, 4, 4, 3, dtype=torch.complex64),
        )
