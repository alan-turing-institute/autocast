import itertools

import pytest
import torch

from autocast.processors.drifting import (
    _compute_drift_field,
    _energy_score,
    _pairwise_distances,
)


def test_pairwise_distances_shape():
    x = torch.randn(2, 4, 8)
    y = torch.randn(2, 6, 8)
    out = _pairwise_distances(x, y)
    assert out.shape == (2, 4, 6)


def test_pairwise_distances_zero_distance_finite_backward():
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = x.detach().clone()
    d = _pairwise_distances(x, y)
    d.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_pairwise_distances_matches_cdist_for_separated_points():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8) + 10.0
    y = torch.randn(2, 6, 8)
    ours = _pairwise_distances(x, y)
    ref = torch.cdist(x, y)
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


_SHAPE_PARAMS = list(
    itertools.product(
        [1, 2],  # batch_size
        [2, 4, 8],  # n_gen (>=2; n_gen=1 produces trivial zero drift)
        [1, 3],  # n_pos
        [4, 16],  # feature_dim
    )
)


@pytest.mark.parametrize(("b", "n_gen", "n_pos", "d"), _SHAPE_PARAMS)
def test_compute_drift_field_shapes_and_info_keys(b, n_gen, n_pos, d):
    gen = torch.randn(b, n_gen, d)
    fixed_pos = torch.randn(b, n_pos, d)
    goal, scaled, info = _compute_drift_field(gen, fixed_pos)
    assert goal.shape == (b, n_gen, d)
    assert scaled.shape == (b, n_gen, d)
    assert "scale" in info
    for tau in (0.02, 0.05, 0.2):
        assert f"loss_{tau}" in info


def test_compute_drift_field_output_dtype_fp32():
    gen = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    fixed_pos = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    goal, scaled, _ = _compute_drift_field(gen, fixed_pos)
    assert goal.dtype == torch.float32
    assert scaled.dtype == torch.float32


def test_compute_drift_field_finite_with_duplicate_positive():
    torch.manual_seed(0)
    b, n_gen, d = 2, 4, 8
    gen = torch.randn(b, n_gen, d)
    fixed_pos = gen[:, :1, :].clone()
    goal, scaled, _ = _compute_drift_field(gen, fixed_pos)
    assert torch.isfinite(goal).all()
    assert torch.isfinite(scaled).all()


def test_compute_drift_field_gradient_flows_to_gen():
    torch.manual_seed(0)
    b, n_gen, d = 2, 4, 8
    gen = torch.randn(b, n_gen, d, requires_grad=True)
    fixed_pos = torch.randn(b, 1, d)
    _, scaled, _ = _compute_drift_field(gen, fixed_pos)
    scaled.sum().backward()
    assert gen.grad is not None
    assert torch.isfinite(gen.grad).all()


def test_compute_drift_field_scale_invariance():
    torch.manual_seed(0)
    b, n_gen, d = 2, 6, 4
    gen = torch.randn(b, n_gen, d)
    fixed_pos = torch.randn(b, 1, d)

    goal_1, scaled_1, _ = _compute_drift_field(gen, fixed_pos, tau_list=(0.05,))

    lam = 7.0
    goal_2, scaled_2, _ = _compute_drift_field(
        gen * lam, fixed_pos * lam, tau_list=(0.05,)
    )

    torch.testing.assert_close(scaled_1, scaled_2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(goal_1, goal_2, atol=1e-4, rtol=1e-4)


def test_compute_drift_field_pulls_toward_unique_positive():
    torch.manual_seed(0)
    b, n_gen, d = 1, 8, 2
    gen = torch.randn(b, n_gen, d) * 0.1
    fixed_pos = torch.tensor([[[2.0, 0.0]]])

    goal, scaled, _ = _compute_drift_field(gen, fixed_pos, tau_list=(0.5,))
    movement = (goal - scaled).squeeze(0)
    # Direction (fixed_pos - gen) is in original space; scaled space is a
    # positive-scalar transform of it so direction is preserved.
    direction_to_pos = (fixed_pos - gen).squeeze(0)
    per_sample_alignment = (movement * direction_to_pos).sum(dim=-1)
    assert (per_sample_alignment > 0).all()


def test_compute_drift_field_single_vs_multi_scale_keys():
    b, n_gen, d = 2, 4, 8
    gen = torch.randn(b, n_gen, d)
    fixed_pos = torch.randn(b, 1, d)

    _, scaled_single, info_single = _compute_drift_field(
        gen, fixed_pos, tau_list=(0.05,)
    )
    _, scaled_multi, info_multi = _compute_drift_field(
        gen, fixed_pos, tau_list=(0.02, 0.05, 0.2)
    )
    torch.testing.assert_close(scaled_single, scaled_multi)
    torch.testing.assert_close(info_single["scale"], info_multi["scale"])
    assert set(info_single) == {"scale", "loss_0.05"}
    assert set(info_multi) == {"scale", "loss_0.02", "loss_0.05", "loss_0.2"}


def test_compute_drift_field_with_extra_negatives():
    b, n_gen, n_pos, n_neg, d = 2, 4, 1, 3, 8
    gen = torch.randn(b, n_gen, d)
    fixed_pos = torch.randn(b, n_pos, d)
    fixed_neg = torch.randn(b, n_neg, d)
    goal, scaled, _ = _compute_drift_field(gen, fixed_pos, fixed_neg)
    assert goal.shape == (b, n_gen, d)
    assert scaled.shape == (b, n_gen, d)


def test_pairwise_distances_zero_distance_returns_sqrt_eps():
    eps = 1e-6
    x = torch.zeros(1, 3, 4)
    d = _pairwise_distances(x, x, eps=eps)
    diag = torch.diagonal(d.squeeze(0))
    expected = torch.full_like(diag, eps**0.5)
    torch.testing.assert_close(diag, expected, atol=0.0, rtol=1e-5)


def test_compute_drift_field_fixed_negatives_change_goal():
    torch.manual_seed(0)
    b, n_gen, n_neg, d = 1, 4, 3, 8
    gen = torch.randn(b, n_gen, d)
    fixed_pos = torch.randn(b, 1, d)
    fixed_neg = torch.randn(b, n_neg, d) * 3.0

    goal_no_neg, _, _ = _compute_drift_field(gen, fixed_pos, tau_list=(0.3,))
    goal_with_neg, _, _ = _compute_drift_field(
        gen, fixed_pos, fixed_neg, tau_list=(0.3,)
    )
    assert torch.isfinite(goal_no_neg).all()
    assert torch.isfinite(goal_with_neg).all()
    assert (goal_no_neg - goal_with_neg).abs().max() > 1e-4


def test_compute_drift_field_nonuniform_weights_change_goal():
    torch.manual_seed(0)
    b, n_gen, d = 1, 4, 2
    gen = torch.randn(b, n_gen, d)
    fixed_pos = torch.randn(b, 1, d)

    goal_default, _, _ = _compute_drift_field(gen, fixed_pos, tau_list=(0.05,))
    weight_gen = torch.tensor([[2.0, 1.0, 1.0, 1.0]])
    goal_weighted, _, _ = _compute_drift_field(
        gen, fixed_pos, weight_gen=weight_gen, tau_list=(0.05,)
    )
    assert torch.isfinite(goal_default).all()
    assert torch.isfinite(goal_weighted).all()
    assert (goal_default - goal_weighted).abs().max() > 1e-4


def test_compute_drift_field_uniform_weight_scaling_preserves_output():
    torch.manual_seed(0)
    b, n_gen, d = 2, 4, 8
    gen = torch.randn(b, n_gen, d)
    fixed_pos = torch.randn(b, 1, d)

    goal_a, scaled_a, info_a = _compute_drift_field(gen, fixed_pos, tau_list=(0.05,))
    goal_b, scaled_b, info_b = _compute_drift_field(
        gen,
        fixed_pos,
        weight_gen=torch.full((b, n_gen), 3.0),
        weight_pos=torch.full((b, 1), 3.0),
        tau_list=(0.05,),
    )
    torch.testing.assert_close(scaled_a, scaled_b)
    torch.testing.assert_close(info_a["scale"], info_b["scale"])
    torch.testing.assert_close(goal_a, goal_b)


# ---------------------------------------------------------------------------
# _energy_score tests
# ---------------------------------------------------------------------------


def _es_naive_reference(gen_b, pos_b):
    """Brute-force naive (1/(2K^2)) joint-field energy score, batch-meaned."""
    k = gen_b.shape[1]
    d_truth = torch.cdist(gen_b, pos_b).squeeze(-1)  # (B, K)
    term1 = d_truth.mean(dim=1)  # (B,)
    d_pair = torch.cdist(gen_b, gen_b)  # (B, K, K)
    term2 = d_pair.sum(dim=(1, 2)) / (2 * k * k)  # (B,) naive incl. zero diagonal
    return (term1 - term2).mean()


def test_energy_score_naive_matches_bruteforce():
    torch.manual_seed(0)
    gen_b = torch.randn(3, 5, 7)
    pos_b = torch.randn(3, 1, 7)
    got = _energy_score(gen_b, pos_b, a_f=0.0)
    torch.testing.assert_close(
        got, _es_naive_reference(gen_b, pos_b), atol=1e-4, rtol=1e-4
    )


def test_energy_score_fair_coefficient_at_af_one():
    """a_f=1 must use the fair 1/(2K(K-1)) spread coefficient."""
    torch.manual_seed(1)
    gen_b = torch.randn(2, 4, 6)
    pos_b = torch.randn(2, 1, 6)
    k = gen_b.shape[1]
    d_truth = torch.cdist(gen_b, pos_b).squeeze(-1).mean(dim=1)
    d_pair = torch.cdist(gen_b, gen_b)
    eye = torch.eye(k)
    pair_sum = (d_pair * (1 - eye)).sum(dim=(1, 2))
    fair = (d_truth - pair_sum / (2 * k * (k - 1))).mean()
    torch.testing.assert_close(
        _energy_score(gen_b, pos_b, a_f=1.0), fair, atol=1e-4, rtol=1e-4
    )


def test_energy_score_zero_spread_when_members_identical():
    gen_b = torch.ones(2, 3, 5)
    pos_b = torch.zeros(2, 1, 5)
    # All members equal => spread term is EXACTLY 0: the shared
    # ``_energy_score_terms`` kernel gives exactly-zero self-distances, so no
    # soft-floor bias leaks in. ES == mean distance to truth == sqrt(5).
    got = _energy_score(gen_b, pos_b, a_f=0.95)
    torch.testing.assert_close(got, torch.tensor(5.0).sqrt())


def test_energy_score_sigma_rescales():
    torch.manual_seed(2)
    gen_b = torch.randn(2, 4, 6)
    pos_b = torch.randn(2, 1, 6)
    base = _energy_score(gen_b, pos_b, a_f=0.95)
    scaled = _energy_score(2.0 * gen_b, 2.0 * pos_b, a_f=0.95, sigma=2.0)
    torch.testing.assert_close(base, scaled, atol=1e-5, rtol=1e-5)


def test_energy_score_is_scalar_and_differentiable():
    gen_b = torch.randn(2, 4, 6, requires_grad=True)
    pos_b = torch.randn(2, 1, 6)
    loss = _energy_score(gen_b, pos_b)
    assert loss.shape == ()
    loss.backward()
    assert gen_b.grad is not None
    assert torch.isfinite(gen_b.grad).all()


def test_energy_score_minimised_at_true_predictive_spread():
    """The energy score is a proper scoring rule: averaged over a dispersed
    predictive ``y ~ N(0, sigma^2 I)``, a cohort whose spread equals ``sigma``
    scores strictly better (lower) than both a collapsed cohort (all members at
    the mean) and a 2x over-dispersed one. This is the property the whole
    research direction rests on — the loss rewards *calibrated* spread, not
    maximal spread, so it pulls an under-disperser up and an over-disperser
    down toward the truth. (A single truth is correctly minimised by collapse;
    the spread reward only emerges in expectation over the predictive.)
    """
    torch.manual_seed(0)
    b, k, d, sigma = 2048, 16, 3, 1.0
    truths = sigma * torch.randn(b, 1, d)
    collapsed = torch.zeros(b, k, d)
    calibrated = sigma * torch.randn(b, k, d)
    over_dispersed = 2.0 * sigma * torch.randn(b, k, d)

    es_collapsed = _energy_score(collapsed, truths, a_f=1.0)
    es_calibrated = _energy_score(calibrated, truths, a_f=1.0)
    es_over = _energy_score(over_dispersed, truths, a_f=1.0)

    assert es_calibrated < es_collapsed
    assert es_calibrated < es_over


def test_scrps_uses_unbiased_pairwise_denominator_not_spread_term():
    """SCRPS denominator ``b`` must be the unbiased mean pairwise distance
    ``pair_sum / (K (K - 1))`` — NOT the energy-score spread term, which carries
    an extra ``1/2 (1 - eps)`` factor. A wrong constant in the denominator moves
    the calibrated minimiser and breaks propriety, so pin the value here.
    """
    torch.manual_seed(0)
    gen_b = torch.randn(3, 8, 5)
    pos_b = torch.randn(3, 1, 5)
    k = gen_b.shape[1]
    eps = 1e-6
    a = torch.cdist(gen_b, pos_b).squeeze(-1).mean(dim=1)  # (B,)
    d_pair = torch.cdist(gen_b, gen_b)
    pair_sum = (d_pair * (1 - torch.eye(k))).sum(dim=(1, 2))

    b = pair_sum / (k * (k - 1))  # unbiased mean pairwise distance
    expected = (a / (b + eps) + 0.5 * (b + eps).log()).mean()
    torch.testing.assert_close(
        _energy_score(gen_b, pos_b, scaled=True), expected, atol=1e-4, rtol=1e-4
    )

    # The spread-term denominator (the fair 1/(2K(K-1)) coefficient => b / 2)
    # gives a materially different value; confirm the helper does NOT use it.
    b_wrong = pair_sum / (2 * k * (k - 1))
    wrong = (a / (b_wrong + eps) + 0.5 * (b_wrong + eps).log()).mean()
    assert not torch.isclose(_energy_score(gen_b, pos_b, scaled=True), wrong, atol=1e-3)


def test_scrps_minimised_at_true_predictive_spread():
    """SCRPS is a proper scoring rule: over a dispersed predictive
    ``y ~ N(0, sigma^2)``, the loss ``-SCRPS`` is minimised by a cohort whose
    spread equals ``sigma``, scoring strictly below an under- and an
    over-dispersed cohort. K is large to keep the finite-K ratio bias small.
    """
    torch.manual_seed(0)
    b, k, d, sigma = 4096, 64, 1, 1.0
    truths = sigma * torch.randn(b, 1, d)
    under = 0.5 * sigma * torch.randn(b, k, d)
    calibrated = sigma * torch.randn(b, k, d)
    over = 2.0 * sigma * torch.randn(b, k, d)

    s_under = _energy_score(under, truths, scaled=True)
    s_cal = _energy_score(calibrated, truths, scaled=True)
    s_over = _energy_score(over, truths, scaled=True)

    assert s_cal < s_under
    assert s_cal < s_over


def test_scrps_scale_invariant_ranking():
    """Scaling ``(X, y) -> (c X, c y)`` shifts ``-SCRPS`` by a constant
    (``1/2 log c``) that cancels across forecasters, so the gap between two
    forecasters is invariant — the property that makes lambda scale-free.
    """
    torch.manual_seed(0)
    b, k, d = 1024, 32, 1
    truth = torch.randn(b, 1, d)
    good = torch.randn(b, k, d)
    bad = 2.0 * torch.randn(b, k, d)

    gap = _energy_score(good, truth, scaled=True) - _energy_score(
        bad, truth, scaled=True
    )
    c = 3.7
    gap_scaled = _energy_score(c * good, c * truth, scaled=True) - _energy_score(
        c * bad, c * truth, scaled=True
    )
    torch.testing.assert_close(gap, gap_scaled, atol=1e-3, rtol=1e-3)


def test_scrps_is_scalar_and_differentiable():
    gen_b = torch.randn(2, 8, 6, requires_grad=True)
    pos_b = torch.randn(2, 1, 6)
    loss = _energy_score(gen_b, pos_b, scaled=True)
    assert loss.shape == ()
    loss.backward()
    assert gen_b.grad is not None
    assert torch.isfinite(gen_b.grad).all()


def test_scrps_eps_floor_keeps_loss_and_grad_finite_near_collapse():
    """As the cohort collapses (``b -> 0``) the ``a / b`` barrier and ``log b``
    diverge; the eps floor (with the pairwise-distance soft-floor) keeps both the
    loss and its gradient finite. The collapse gradient is large but finite --
    training relies on gradient clipping for conditioning, and the barrier itself
    pushes away from collapse, so a healthy random init never starts here.
    """
    gen_b = (torch.ones(2, 8, 5) + 1e-9 * torch.randn(2, 8, 5)).requires_grad_(True)
    pos_b = torch.zeros(2, 1, 5)
    loss = _energy_score(gen_b, pos_b, scaled=True, eps=1e-6)
    assert torch.isfinite(loss)
    loss.backward()
    assert gen_b.grad is not None
    assert torch.isfinite(gen_b.grad).all()
