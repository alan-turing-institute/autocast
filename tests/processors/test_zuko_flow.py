import pytest
import torch

from autocast.types import EncodedBatch

pytest.importorskip("zuko")

from autocast.processors.zuko_flow import ZukoFlowProcessor

# Cover a coupling flow (one-forward sampling) and an autoregressive flow.
FLOW_TYPES = ["realnvp", "nsf"]


def _batch(b, spatial, channels):
    shape = (b, 1, *spatial, channels)
    return EncodedBatch(
        encoded_inputs=torch.randn(*shape),
        encoded_output_fields=torch.randn(*shape),
        global_cond=None,
        encoded_info={},
    )


def _processor(flow_type, spatial=(4, 1), channels=1):
    return ZukoFlowProcessor(
        n_channels_in=channels,
        n_channels_out=channels,
        spatial_shape=spatial,
        flow_type=flow_type,
        transforms=2,
        hidden_features=(32, 32),
    )


@pytest.mark.parametrize("flow_type", FLOW_TYPES)
def test_loss_finite_and_backward(flow_type):
    proc = _processor(flow_type)
    loss = proc.loss(_batch(8, (4, 1), 1))
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in proc.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("flow_type", FLOW_TYPES)
def test_map_shape_and_finite(flow_type):
    proc = _processor(flow_type)
    x = torch.randn(3, 1, 4, 1, 1)
    y = proc.map(x, None)
    assert y.shape == (3, 1, 4, 1, 1)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("flow_type", FLOW_TYPES)
def test_two_train_steps_decrease(flow_type):
    """Overfitting a fixed batch reduces the exact NLL (the log score is real)."""
    torch.manual_seed(0)
    proc = _processor(flow_type)
    batch = _batch(32, (4, 1), 1)
    opt = torch.optim.Adam(proc.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = proc.loss(batch)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]


def test_invalid_flow_type_raises():
    with pytest.raises(ValueError, match="flow_type must be one of"):
        ZukoFlowProcessor(flow_type="not_a_flow")


def test_loss_rejects_mismatched_field_shape():
    """A target whose dims don't match the configured n_steps_output /
    n_channels_out raises a clear error instead of a cryptic reshape failure."""
    proc = _processor("realnvp", spatial=(8, 8), channels=1)
    bad = EncodedBatch(
        encoded_inputs=torch.randn(4, 1, 8, 8, 1),
        encoded_output_fields=torch.randn(4, 2, 8, 8, 1),  # wrong n_steps_output
        global_cond=None,
        encoded_info={},
    )
    with pytest.raises(ValueError, match="does not match configured"):
        proc.loss(bad)
