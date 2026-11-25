"""Regression tests for matched cross-mode SWE64 spectral objectives."""

import copy
import math
from unittest.mock import patch

import pytest
import torch

from autocast.losses.mode_groups import (
    SampledModeEnergyScore,
    _mode_grid,
    _sample_triads,
)


def test_triad_geometry_and_reproducibility():
    groups = _sample_triads(64, 64, 1024, torch.Generator().manual_seed(713))
    points = torch.stack((groups // 64, groups % 64), dim=-1)
    points = torch.where(points > 31, points - 64, points)
    k, q, r = points.unbind(dim=1)
    torch.testing.assert_close(k + q, r)
    assert torch.all(points.abs() <= 21)
    assert torch.all((k != q).any(-1))
    assert torch.all(
        (points[..., 0] > 0) | ((points[..., 0] == 0) & (points[..., 1] > 0))
    )
    valid, outside = _mode_grid(64, 64)
    assert 2 * len(valid) + len(outside) == 4096
    assert 0 in outside


def test_triad_sees_three_way_phase_relation_with_identical_pair_laws():
    # At r=k+q, flip the sign of a*b. All two-mode marginals are identical.
    groups = _sample_triads(8, 8, 1, torch.Generator().manual_seed(42))
    k, q, r = groups[0].tolist()
    signs = torch.tensor([-1.0, 1.0])
    a, b = torch.cartesian_prod(signs, signs).T.repeat(1, 2)
    intact = torch.zeros(64, 1, 8, dtype=torch.complex128)
    for index, values in zip((k, q, r), (a, b, a * b), strict=True):
        opposite = (-(index // 8) % 8) * 8 + (-(index % 8) % 8)
        intact[index, 0] = values
        intact[opposite, 0] = values
    broken = intact.clone()
    opposite_r = (-(r // 8) % 8) * 8 + (-(r % 8) % 8)
    broken[[r, opposite_r]] *= -1
    intact = intact.reshape(1, 1, 8, 8, 1, 8)
    broken = broken.reshape_as(intact)

    def mean_score(objective, forecast):
        return torch.stack(
            [objective(forecast, target) for target in intact.unbind(-1)]
        ).mean()

    pairs = SampledModeEnergyScore("pairs", 1, seed=42, alpha=0.95).eval()
    triads = SampledModeEnergyScore("triads", 1, seed=42, alpha=0.95).eval()
    torch.testing.assert_close(mean_score(pairs, intact), mean_score(pairs, broken))
    assert mean_score(triads, intact) < mean_score(triads, broken)


@pytest.mark.parametrize(
    ("group", "norm"),
    [
        ("singletons", "rms"),
        ("pairs", None),
        ("pairs", "raw"),
        ("pairs", "rms"),
        ("triads", None),
        ("triads", "raw"),
        ("triads", "rms"),
    ],
)
def test_grouped_score_matches_manual_distances_and_resumes(group, norm):
    # A full complex FFT of a real grid includes both orientations of each mode.
    generator = torch.Generator().manual_seed(15)
    preds = torch.fft.fftn(
        torch.randn(1, 1, 8, 10, 3, 4, generator=generator, dtype=torch.float64),
        dim=(2, 3),
        norm="ortho",
    ).requires_grad_()
    truth = torch.fft.fftn(
        torch.randn(1, 1, 8, 10, 3, generator=generator, dtype=torch.float64),
        dim=(2, 3),
        norm="ortho",
    )
    scales = [0.7, 1.1, 1.1]
    options = {} if norm is None else {"group_norm": norm}
    loss = SampledModeEnergyScore(
        group,
        triads_per_draw=9,
        seed=42,
        alpha=0.95,
        channel_multipliers=scales,
        **options,
    )
    sampled = _sample_triads(8, 10, 9, torch.Generator().manual_seed(42))
    _, outside = _mode_grid(8, 10)
    scale = torch.tensor(scales, dtype=torch.float64)
    flat_p = preds.detach().reshape(80, 3, 4) * scale[:, None]
    flat_y = truth.reshape(80, 3) * scale

    def reference(indices):
        members = flat_p[indices].reshape(-1, 4).T
        target = flat_y[indices].flatten()
        distances = torch.linalg.vector_norm(members - target, dim=1)
        spread = torch.linalg.vector_norm(members[:, None] - members[None, :], dim=-1)
        return distances.mean() - 0.5 * (0.05 + 0.95 * 4 / 3) * spread.mean()

    if group == "singletons":
        # Preserve every sampled occurrence, including repeated vertices.
        assert sampled.unique().numel() < sampled.numel()
        tuples = sampled.reshape(-1, 1)
    elif group == "pairs":
        tuples = sampled[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)
    else:
        tuples = sampled
    grouped = torch.stack([reference(indices) for indices in tuples]).mean()
    if norm == "rms":
        grouped /= math.sqrt(tuples.shape[1])
    expected = (
        grouped * (80 - len(outside)) / 80
        + torch.stack([reference(i) for i in outside]).sum() / 80
    )
    # Lightning's validation sanity check may populate the shared index cache.
    _mode_grid.cache_clear()
    with torch.inference_mode():
        loss.eval()(preds, truth)
    actual = loss.train()(preds, truth)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()

    state = copy.deepcopy(loss.state_dict())
    loss.eval()
    torch.testing.assert_close(loss(preds, truth), actual)
    loss.train()
    next_score = loss(preds, truth)
    assert not torch.isclose(next_score, actual)
    loss.load_state_dict(state)
    torch.testing.assert_close(loss(preds, truth), next_score)


def test_pair_singleton_mix_uses_one_shared_draw():
    options = {"triads_per_draw": 9, "seed": 42, "alpha": 0.95, "group_norm": "rms"}
    pairs = SampledModeEnergyScore("pairs", **options)
    singletons = SampledModeEnergyScore("singletons", **options)
    mixture = SampledModeEnergyScore("pairs", singleton_weight=0.5, **options)
    generator = torch.Generator().manual_seed(15)
    prediction = torch.randn(
        1, 1, 8, 10, 3, 4, dtype=torch.complex64, generator=generator
    )
    truth = torch.randn(1, 1, 8, 10, 3, dtype=torch.complex64, generator=generator)
    for _ in range(2):
        expected = 0.5 * (pairs(prediction, truth) + singletons(prediction, truth))
        torch.testing.assert_close(mixture(prediction, truth), expected)
        torch.testing.assert_close(mixture.get_extra_state(), pairs.get_extra_state())


def test_sampled_arms_and_rank_local_streams_draw_the_same_triads():
    scores = [
        SampledModeEnergyScore(group, triads_per_draw=64, seed=153164)
        for group in ("singletons", "pairs", "triads")
        for _rank in range(2)
    ]
    truth = torch.zeros(1, 1, 8, 10, 3, dtype=torch.complex64)
    preds = truth.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 2)

    def draw(nx, ny, count, generator):
        triads = _sample_triads(nx, ny, count, generator)
        draws.append(triads)
        return triads

    def step():
        draws.clear()
        for score in scores:
            score(preds, truth)
        assert len(draws) == len(scores)
        assert draws[0].shape == (64, 3)
        assert all(torch.equal(triads, draws[0]) for triads in draws)
        return draws[0]

    draws = []
    with patch("autocast.losses.mode_groups._sample_triads", side_effect=draw):
        assert not torch.equal(step(), step())
