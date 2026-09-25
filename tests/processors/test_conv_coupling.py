import pytest
import torch

from autocast.processors.conv_coupling import ConvCouplingFlowProcessor
from autocast.types import EncodedBatch

# (spatial_shape, n_channels) covering a 2D grid and a 1D field as [L, 1].
SHAPES = [((8, 8), 2), ((40, 1), 1)]


def _batch(b, spatial, channels):
    h, w = spatial
    return EncodedBatch(
        encoded_inputs=torch.randn(b, 1, h, w, channels),
        encoded_output_fields=torch.randn(b, 1, h, w, channels),
        global_cond=None,
        encoded_info={},
    )


def _processor(spatial, channels, transforms=6):
    return ConvCouplingFlowProcessor(
        n_steps_input=1,
        n_steps_output=1,
        n_channels_in=channels,
        n_channels_out=channels,
        spatial_shape=spatial,
        transforms=transforms,
        hidden_channels=16,
        n_blocks=2,
    )


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_loss_finite_and_backward(spatial, channels):
    proc = _processor(spatial, channels)
    loss = proc.loss(_batch(4, spatial, channels))
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in proc.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_map_shape_and_finite(spatial, channels):
    proc = _processor(spatial, channels)
    h, w = spatial
    x = torch.randn(3, 1, h, w, channels)
    y = proc.map(x, None)
    assert y.shape == (3, 1, h, w, channels)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_bijective_roundtrip(spatial, channels):
    """encode (data->latent) then decode (latent->data) must recover the field
    exactly — the property that makes the change-of-variables likelihood exact."""
    proc = _processor(spatial, channels).eval()
    # Random (non-identity) parameters so the test is not trivially satisfied
    # by the zero-init identity start.
    with torch.no_grad():
        for p in proc.parameters():
            p.add_(0.1 * torch.randn_like(p))
    h, w = spatial
    y = proc._to_conv(torch.randn(5, 1, h, w, channels), 1, channels)
    cond = torch.randn(5, channels, h, w)
    with torch.no_grad():
        z = y
        for layer in proc.coupling_layers():
            z, _ = layer.encode(z, cond)
        recon = z
        for layer in reversed(proc.coupling_layers()):
            recon = layer.decode(recon, cond)
    # atol=1e-3: with a non-zero scale bound the coupling actually scales, so the
    # float32 encode/decode round-trip carries ~2e-4 (a real non-bijection is O(1)).
    assert torch.allclose(recon, y, atol=1e-3), (recon - y).abs().max().item()


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_two_train_steps_decrease(spatial, channels):
    """Two optimiser steps on a fixed batch reduce the NLL (the loss is real)."""
    torch.manual_seed(0)
    proc = _processor(spatial, channels)
    batch = _batch(8, spatial, channels)
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


def test_requires_2d_spatial_shape():
    with pytest.raises(ValueError, match="2D spatial_shape"):
        ConvCouplingFlowProcessor(spatial_shape=(8, 8, 2))


def test_loss_rejects_mismatched_field_shape():
    """A field whose trailing dims don't match (n_steps, *spatial, n_channels)
    raises a clear error instead of a cryptic reshape failure."""
    proc = _processor((8, 8), 1)
    bad = EncodedBatch(
        encoded_inputs=torch.randn(4, 1, 8, 8, 1),
        encoded_output_fields=torch.randn(4, 1, 8, 8, 2),  # wrong channel count
        global_cond=None,
        encoded_info={},
    )
    with pytest.raises(ValueError, match="elements per sample but expected"):
        proc.loss(bad)


def test_scale_bound_initialised_nonzero():
    """A zero scale bound is a dead saddle: with s = tanh(s_raw) * bound and both
    s_raw (zero-init conv) and bound at zero, d s / d s_raw = bound = 0 AND
    d s / d bound = tanh(s_raw) = 0, so neither the scale conv nor the bound ever
    receive gradient -- the flow can only shift, never scale, freezing the
    predictive std at the unit base. The bound must start non-zero."""
    proc = _processor((40, 1), 1)
    for layer in proc.coupling_layers():
        assert float(layer.log_scale_bound.detach().abs().min()) > 0.0


def test_predictive_std_compresses_below_base():
    """The flow must be able to compress the unit base toward the data's small
    one-step residual scale. The old zero-bound saddle froze the scale path,
    locking the predictive std at the base std of ~1.0 regardless of the data;
    with the saddle broken the proper NLL learns to compress."""
    torch.manual_seed(0)
    resid_std = 0.1
    proc = _processor((8, 8), 1, transforms=8)
    x = torch.randn(512, 1, 8, 8, 1)
    y = x + resid_std * torch.randn_like(x)
    batch = EncodedBatch(
        encoded_inputs=x,
        encoded_output_fields=y,
        global_cond=None,
        encoded_info={},
    )
    opt = torch.optim.Adam(proc.parameters(), lr=2e-3)
    for _ in range(600):
        opt.zero_grad()
        proc.loss(batch).backward()
        opt.step()
    with torch.no_grad():
        members = torch.stack([proc.map(x[:256], None) for _ in range(32)])
    pred_std = members.std(dim=0).mean().item()
    # Frozen-saddle value was ~1.0; a live scale path compresses well below it.
    assert pred_std < 0.6


def test_encode_logdet_matches_numerical_jacobian():
    """The coupling stack's reported ``log|det dz/dy|`` equals the numerical
    Jacobian determinant of the data->latent map (the change-of-variables term
    the likelihood relies on). Run in float64 for a tight tolerance."""
    torch.manual_seed(0)
    proc = _processor((4, 4), 1, transforms=3).double().eval()
    with torch.no_grad():
        for p in proc.parameters():
            p.add_(0.2 * torch.randn_like(p))
    h, w = 4, 4
    cond = torch.randn(1, 1, h, w, dtype=torch.float64)
    y0 = proc._to_conv(torch.randn(1, 1, h, w, 1, dtype=torch.float64), 1, 1)

    def encode_all(y_cf):
        z = y_cf
        ladj = z.new_zeros(z.shape[0])
        for layer in proc.coupling_layers():
            z, dl = layer.encode(z, cond)
            ladj = ladj + dl
        return z, ladj

    _, ladj = encode_all(y0)

    def flat_encode(flat):
        return encode_all(flat.reshape(y0.shape))[0].reshape(-1)

    jac = torch.autograd.functional.jacobian(flat_encode, y0.reshape(-1))
    _, logabsdet = torch.linalg.slogdet(jac)
    assert torch.allclose(ladj.reshape(()), logabsdet, atol=1e-6), (
        ladj.item(),
        logabsdet.item(),
    )
