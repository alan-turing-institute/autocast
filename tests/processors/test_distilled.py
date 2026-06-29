from typing import cast

import pytest
import torch
from torch import nn

from autocast.processors.base import Processor
from autocast.processors.distilled import DistilledProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.types import EncodedBatch

# Archetypes: scalar (ou/dw), vector (mvou), 1D (l96), 2D (gs).
SHAPES = [((1, 1), 1), ((1, 1), 3), ((40, 1), 1), ((8, 8), 2)]


class _CondField(nn.Module):
    """Stub flow field f(z,t) = cond, so a 1-step Euler endpoint is z + x_t.

    The endpoint depends on the noise z (the teacher has real spread) and on the
    conditioning x_t, so a successful student must reproduce both.
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
    return DistilledProcessor(
        n_channels_in=channels,
        n_channels_out=channels,
        spatial_shape=spatial,
        hidden_features=hidden,
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


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_distill_loss_backward_only_student(spatial, channels):
    proc = _student(spatial, channels)
    teacher = _teacher(channels)
    proc.set_teacher(teacher)
    loss = proc.loss(_batch(4, spatial, channels))
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in proc.student.parameters())
    # Teacher is frozen and not part of the student graph.
    assert all(p.grad is None for p in teacher.parameters())


def test_teacher_excluded_from_checkpoint():
    """The injected teacher must not leak into the student state_dict (scoring
    loads the student alone via a strict state_dict load)."""
    proc = _student((8, 8), 2)
    keys_before = set(proc.state_dict())
    proc.set_teacher(_teacher(2))
    assert set(proc.state_dict()) == keys_before


def test_distillation_matches_teacher_and_keeps_spread():
    """Training drives the student toward the teacher's noise->sample map: the
    distillation MSE drops and the student's member spread stays non-trivial
    (sample-matching, not a collapsed mean)."""
    torch.manual_seed(0)
    spatial, channels = (4, 1), 1
    proc = _student(spatial, channels, hidden=(128, 128))
    proc.set_teacher(_teacher(channels))
    batch = _batch(64, spatial, channels)
    opt = torch.optim.Adam(proc.student.parameters(), lr=3e-3)
    losses = []
    for _ in range(400):
        opt.zero_grad()
        loss = proc.loss(batch)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.25 * losses[0]  # student learned the teacher map
    # Member spread over a fixed input must be > 0 (UQ preserved).
    x = torch.randn(8, 1, *spatial, channels)
    members = torch.stack([proc.map(x, None) for _ in range(32)], dim=0)
    assert members.std(dim=0).mean() > 0.1
