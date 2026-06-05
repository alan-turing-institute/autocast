"""Tests for time-varying-scalar conditioning in EncoderWithCond.encode_cond."""

import pytest
import torch
from einops import rearrange

from autocast.encoders.permute_concat import PermuteConcat
from autocast.types import Batch


def _make_batch(batch_size=2, t_in=3, w=4, h=4, c=1, c_tvs=2, tvs_steps=None):
    tvs_steps = tvs_steps if tvs_steps is not None else t_in
    return Batch(
        input_fields=torch.randn(batch_size, t_in, w, h, c),
        output_fields=torch.randn(batch_size, t_in, w, h, c),
        constant_scalars=None,
        constant_fields=None,
        time_varying_scalars=torch.randn(batch_size, tvs_steps, c_tvs),
    )


def test_encode_cond_default_uses_last_input_step():
    t_in, c_tvs = 3, 2
    batch = _make_batch(t_in=t_in, c_tvs=c_tvs)
    encoder = PermuteConcat(in_channels=1, n_steps_input=t_in)  # n_tvs_steps=1
    cond = encoder.encode_cond(batch)
    assert cond is not None
    assert cond.shape == (batch.input_fields.shape[0], c_tvs)
    # Default n_tvs_steps=1 -> last input step (index n_steps_input - 1).
    assert torch.equal(cond, batch.time_varying_scalars[:, t_in - 1, :])


def test_encode_cond_rearranges_multiple_steps():
    t_in, c_tvs, n_tvs_steps = 3, 2, 2
    batch = _make_batch(t_in=t_in, c_tvs=c_tvs)
    encoder = PermuteConcat(in_channels=1, n_steps_input=t_in, n_tvs_steps=n_tvs_steps)
    cond = encoder.encode_cond(batch)
    assert cond is not None
    assert cond.shape == (batch.input_fields.shape[0], n_tvs_steps * c_tvs)
    window = batch.time_varying_scalars[:, t_in - n_tvs_steps : t_in, :]
    assert torch.equal(cond, rearrange(window, "b t c -> b (t c)"))


def test_encode_cond_concatenates_after_constant_scalars():
    t_in, c_tvs, n_const = 3, 2, 4
    batch = _make_batch(t_in=t_in, c_tvs=c_tvs)
    batch.constant_scalars = torch.randn(batch.input_fields.shape[0], n_const)
    encoder = PermuteConcat(in_channels=1, n_steps_input=t_in)
    cond = encoder.encode_cond(batch)
    assert cond is not None
    assert cond.shape == (batch.input_fields.shape[0], n_const + c_tvs)
    assert torch.equal(cond[:, :n_const], batch.constant_scalars)
    assert torch.equal(cond[:, n_const:], batch.time_varying_scalars[:, t_in - 1, :])


def test_encode_cond_rejects_window_longer_than_input():
    t_in = 2
    batch = _make_batch(t_in=t_in)
    encoder = PermuteConcat(in_channels=1, n_steps_input=t_in, n_tvs_steps=t_in + 1)
    with pytest.raises(ValueError, match="cannot exceed"):
        encoder.encode_cond(batch)


def test_encode_cond_raises_when_tvs_exhausted():
    t_in = 3
    # Fewer TVS steps than the input window (e.g. rollout consumed them).
    batch = _make_batch(t_in=t_in, tvs_steps=t_in - 1)
    encoder = PermuteConcat(in_channels=1, n_steps_input=t_in)
    with pytest.raises(RuntimeError, match="exhausted"):
        encoder.encode_cond(batch)
