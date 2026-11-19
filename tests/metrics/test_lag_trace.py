"""Tests for the lag-1 trace residual gauge."""

import math

import torch

from autocast.metrics.lag_trace import lag_trace_rho


def test_rho_near_zero_on_white_noise():
    torch.manual_seed(0)
    f, n, n_traj = 400, 6, 8
    e = torch.randn(f, n)
    traj_id = torch.arange(f) // (f // n_traj)
    rho = lag_trace_rho(e, traj_id)
    assert isinstance(rho, float)
    assert abs(rho) < 0.2


def test_rho_large_on_persistent_bias():
    """A constant per-trajectory bias persists across steps -> large rho."""
    torch.manual_seed(0)
    f, n, n_traj = 400, 6, 8
    traj_id = torch.arange(f) // (f // n_traj)
    bias = torch.randn(n_traj, n)[traj_id]  # same vector for every frame of a traj
    noise = 0.01 * torch.randn(f, n)
    rho = lag_trace_rho(bias + noise, traj_id)
    # Near-unit lag-1 correlation -> tiny denominator (S0 - S1) -> large rho.
    assert rho > 5.0


def test_returns_nan_when_denominator_degenerate():
    """Identical consecutive frames make S0 == S1 -> nan, not a divide error."""
    e = torch.ones(4, 3)
    traj_id = torch.zeros(4, dtype=torch.long)
    rho = lag_trace_rho(e, traj_id)
    assert math.isnan(rho)
