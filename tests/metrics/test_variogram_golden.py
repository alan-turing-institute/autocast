"""Every variogram entry point must reproduce the golden fixture.

The fixture was generated ONCE from the pre-optimisation implementation at
661a8963 and is never regenerated. float64 tolerance 1e-10 absolute on values
of order 1e1..1e4: reordering noise is ~1e-13, a real defect (a dropped factor
of two, a wrong triangle, a missing weight orientation) is ~1e0.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autocast.metrics.ensemble import (
    SpatialVariogramScore,
    SpatiotemporalVariogramScore,
    TemporalVariogramScore,
    VariogramScore,
    _variogram_score,
)
from autocast.metrics.rank_histogram import _weighted_variogram

FIXTURE = Path(__file__).parent / "fixtures" / "variogram_golden.pt"
ATOL = 1e-10
P_VALUES = (0.5, 1.0, 2.0, 0.7)
VECTOR_DIMS = ("spatial", "temporal", "spatial_temporal", "spatial_temporal_channels")


@pytest.fixture(scope="module")
def golden() -> dict[str, torch.Tensor]:
    return torch.load(FIXTURE)


@pytest.mark.parametrize("vd", VECTOR_DIMS)
@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize(
    "max_pair_elements", [2**30, 97]
)  # one block, and forced chunking
def test_metric_path_matches_golden(golden, vd, p, max_pair_elements):
    got = VariogramScore(
        p=p, vector_dims=vd, max_pair_elements=max_pair_elements
    )._score(golden["y_pred"], golden["y_true"])
    torch.testing.assert_close(got, golden[f"metric/{vd}/p={p}"], atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("vd", VECTOR_DIMS)
@pytest.mark.parametrize("p", P_VALUES)
def test_loss_path_matches_golden(golden, vd, p):
    got = _variogram_score(golden["y_pred"], golden["y_true"], p, None, vd)
    torch.testing.assert_close(got, golden[f"loss/{vd}/p={p}"], atol=ATOL, rtol=0.0)


def test_asymmetric_weights_match_golden(golden):
    got = VariogramScore(
        p=0.5, weights=golden["weights_asym"], vector_dims="spatial"
    )._score(golden["y_pred"], golden["y_true"])
    torch.testing.assert_close(
        got, golden["metric/spatial/p=0.5/weighted"], atol=ATOL, rtol=0.0
    )


def test_named_subclasses_match_golden(golden):
    for cls, key in (
        (SpatialVariogramScore, "subclass/spatial"),
        (TemporalVariogramScore, "subclass/temporal"),
        (SpatiotemporalVariogramScore, "subclass/spatial_temporal"),
    ):
        got = cls()._score(golden["y_pred"], golden["y_true"])
        torch.testing.assert_close(got, golden[key], atol=ATOL, rtol=0.0)


def test_rank_histogram_pair_sum_matches_golden(golden):
    got = _weighted_variogram(
        golden["rank_hist/fields"], golden["rank_hist/weights"], 0.5, 1 << 20
    )
    torch.testing.assert_close(got, golden["rank_hist/p=0.5"], atol=ATOL, rtol=0.0)


def test_loss_path_gradient_is_finite_and_nonzero(golden):
    """The loss path must stay differentiable, including through exact zeros
    on the pair diagonal (the reason _safe_abs_pow exists)."""
    y_pred = golden["y_pred"].clone().requires_grad_(True)
    score = _variogram_score(y_pred, golden["y_true"], 0.5, None, "spatial").sum()
    (grad,) = torch.autograd.grad(score, y_pred)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_loss_path_gradient_matches_golden(golden):
    """The rewritten kernel must reproduce the pre-optimisation gradient, not
    merely a finite one -- see loss/spatial/p=0.5/grad in the fixture."""
    y_pred = golden["y_pred"].clone().requires_grad_(True)
    score = _variogram_score(y_pred, golden["y_true"], 0.5, None, "spatial").sum()
    (grad,) = torch.autograd.grad(score, y_pred)
    torch.testing.assert_close(
        grad, golden["loss/spatial/p=0.5/grad"], atol=1e-9, rtol=0.0
    )


def test_loss_path_gradcheck():
    """Finite-difference check on a tiny shape, independent of the fixture."""
    torch.manual_seed(0)
    b, t, h, w, c, m = 1, 2, 3, 3, 1, 3
    y_true = torch.randn(b, t, h, w, c, dtype=torch.float64)
    y_pred = (
        y_true[..., None] + 0.3 * torch.randn(b, t, h, w, c, m, dtype=torch.float64)
    ).requires_grad_(True)

    def fn(yp: torch.Tensor) -> torch.Tensor:
        return _variogram_score(yp, y_true, 0.5, None, "spatial").sum()

    assert torch.autograd.gradcheck(fn, (y_pred,), atol=1e-6)
