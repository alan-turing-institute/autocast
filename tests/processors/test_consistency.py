from typing import cast

import pytest
import torch
from torch import nn

from autocast.processors.base import Processor
from autocast.processors.consistency import ConsistencyDistilledProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.types import EncodedBatch

# Archetypes: scalar (ou/dw), vector (mvou), 1D (l96), 2D (gs).
SHAPES = [((1, 1), 1), ((1, 1), 3), ((40, 1), 1), ((8, 8), 2)]


class _CondField(nn.Module):
    """Stub flow field f(z,t) = cond, so a 1-step Euler step adds the conditioning.

    The step depends on the conditioning x_t, so the teacher carries real
    structure for the student to be consistent with.
    """

    include_time_embedding = False

    def forward(self, z, t=None, cond=None, global_cond=None):  # noqa: ARG002
        return cond


def _teacher(channels):
    return FlowMatchingProcessor(
        backbone=_CondField(),
        n_steps_output=1,
        n_channels_out=channels,
        flow_ode_steps=1,
    )


def _student(spatial, channels, hidden=(64, 64)):
    return ConsistencyDistilledProcessor(
        n_channels_in=channels,
        n_channels_out=channels,
        spatial_shape=spatial,
        hidden_features=hidden,
        n_consistency_steps=8,
    )


def _batch(b, spatial, channels):
    shape = (b, 1, *spatial, channels)
    return EncodedBatch(
        encoded_inputs=torch.randn(*shape),
        encoded_output_fields=torch.randn(*shape),
        global_cond=None,
        encoded_info={},
    )


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_map_shape_and_finite_without_teacher(spatial, channels):
    proc = _student(spatial, channels)
    x = torch.randn(3, 1, *spatial, channels)
    y = proc.map(x, None)
    assert y.shape == (3, 1, *spatial, channels)
    assert torch.isfinite(y).all()


def test_map_is_stochastic():
    """Two draws from the same input differ (the student is a generative sampler)."""
    torch.manual_seed(0)
    proc = _student((40, 1), 1)
    x = torch.randn(5, 1, 40, 1)
    a = proc.map(x, None)
    b = proc.map(x, None)
    assert not torch.allclose(a, b)


def test_boundary_condition_identity_at_data_end():
    """f(z, t=1 | x) = z exactly (the data-end identity), for any z."""
    proc = _student((40, 1), 1)
    z = torch.randn(4, 1, 40, 1)
    ctx = proc._context(torch.randn(4, 1, 40, 1), None)
    t1 = torch.ones(4)
    out = proc._consistency_map(z, t1, ctx)
    assert torch.allclose(out, z, atol=1e-6)


class _RecordingField(nn.Module):
    """Flow field that records every time t it is queried at (returns cond)."""

    include_time_embedding = False

    def __init__(self):
        super().__init__()
        self.seen_t: list[float] = []

    def forward(self, z, t: torch.Tensor, cond=None, global_cond=None):  # noqa: ARG002
        self.seen_t.extend(t.reshape(-1).tolist())
        return cond


def test_teacher_stepped_from_noisier_time_toward_data():
    """The teacher must be queried at the noisier grid time t_n = n/N (n in
    {0..N-1}), stepping toward the data end — never at the data boundary t=1, which
    would only occur if the step originated from the closer-to-data t_{n+1}. This
    pins the teacher-step time-direction (a t-independent stub teacher cannot)."""
    torch.manual_seed(0)
    n_steps = 5
    field = _RecordingField()
    teacher = FlowMatchingProcessor(
        backbone=field,
        n_steps_output=1,
        n_channels_out=1,
        flow_ode_steps=1,
    )
    proc = _student((4, 1), 1)
    proc.n_consistency_steps = n_steps
    proc.set_teacher(teacher)
    proc.loss(_batch(64, (4, 1), 1))
    assert field.seen_t, "teacher was never queried"
    # t_n grid is {0, 1/N, ..., (N-1)/N}; max is (N-1)/N < 1. A swap to t_{n+1}
    # would put 1.0 in the set.
    assert max(field.seen_t) <= (n_steps - 1) / n_steps + 1e-6, (
        f"teacher queried at/above the data boundary: max t = {max(field.seen_t)}"
    )
    assert min(field.seen_t) == 0.0, "the noisiest slice t=0 was never sampled"


def test_loss_requires_teacher():
    proc = _student((4, 1), 1)
    with pytest.raises(RuntimeError, match="requires a teacher"):
        proc.loss(_batch(2, (4, 1), 1))


def test_set_teacher_rejects_non_flow_matching():
    """The teacher must be flow-matching-capable (expose flow_field); anything
    else is rejected up front rather than failing cryptically during training."""
    proc = _student((4, 1), 1)
    with pytest.raises(TypeError, match="flow-matching-capable"):
        proc.set_teacher(cast(Processor, object()))


def test_teacher_excluded_from_checkpoint():
    """The injected teacher must not leak into the student state_dict (scoring
    loads the student alone via a strict state_dict load)."""
    proc = _student((8, 8), 2)
    keys_before = set(proc.state_dict())
    proc.set_teacher(_teacher(2))
    assert set(proc.state_dict()) == keys_before


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_consistency_loss_backward_only_student(spatial, channels):
    proc = _student(spatial, channels)
    teacher = _teacher(channels)
    proc.set_teacher(teacher)
    loss = proc.loss(_batch(4, spatial, channels))
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in proc.net.parameters())
    # Teacher is frozen and not part of the student graph.
    assert all(not p.requires_grad for p in teacher.parameters())
