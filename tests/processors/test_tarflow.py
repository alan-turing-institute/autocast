from typing import cast

import pytest
import torch

from autocast.processors.tarflow import TarFlowProcessor, _MetaBlock
from autocast.types import EncodedBatch

# TarFlow targets the square 2D toy (gs: 8x8, 2 channels). Cover two patch sizes.
CASES = [(8, 2, 1), (8, 2, 2)]


def _batch(b, size, channels):
    shape = (b, 1, size, size, channels)
    return EncodedBatch(
        encoded_inputs=torch.randn(*shape),
        encoded_output_fields=torch.randn(*shape),
        global_cond=None,
        encoded_info={},
    )


def _processor(size, channels, patch_size, num_blocks=4):
    return TarFlowProcessor(
        n_channels_in=channels,
        n_channels_out=channels,
        spatial_shape=(size, size),
        patch_size=patch_size,
        channels=32,
        num_blocks=num_blocks,
        layers_per_block=1,
        head_dim=32,
    )


@pytest.mark.parametrize(("size", "channels", "patch"), CASES)
def test_loss_finite_and_backward(size, channels, patch):
    proc = _processor(size, channels, patch).train()
    loss = proc.loss(_batch(4, size, channels))
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in proc.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize(("size", "channels", "patch"), CASES)
def test_map_shape_and_finite(size, channels, patch):
    proc = _processor(size, channels, patch).eval()
    x = torch.randn(3, 1, size, size, channels)
    y = proc.map(x, None)
    assert y.shape == (3, 1, size, size, channels)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(("size", "channels", "patch"), CASES)
def test_bijective_roundtrip(size, channels, patch):
    """Encoding the field then inverting the blocks recovers it — the property
    that makes the autoregressive change-of-variables likelihood exact."""
    proc = _processor(size, channels, patch).eval()
    # Perturb away from the zero-init (identity) so the test is non-trivial.
    with torch.no_grad():
        for p in proc.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(4, 1, size, size, channels)
    cond = proc._cond_patches(x)
    y_img = proc._to_images(torch.randn(4, 1, size, size, channels), 1, channels)
    y = proc._patchify(y_img)
    with torch.no_grad():
        z, _ = proc._encode(y, cond)
        recon = z
        for block in reversed(proc.blocks):
            recon = cast(_MetaBlock, block).reverse(recon, cond)
    assert torch.allclose(recon, y, atol=1e-3), (recon - y).abs().max().item()


def test_two_train_steps_decrease():
    torch.manual_seed(0)
    proc = _processor(8, 2, 1).train()
    batch = _batch(8, 8, 2)
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


@pytest.mark.parametrize(("size", "channels", "patch"), CASES)
def test_map_batched_shape_and_independence(size, channels, patch):
    """The batched sampler returns the same per-member layout as looped map()
    (member axis moved to the end) and produces distinct, finite members."""
    proc = _processor(size, channels, patch).eval()
    x = torch.randn(3, 1, size, size, channels)
    members = 8
    ens = proc.map_batched(x, members).movedim(1, -1)
    assert ens.shape == (3, 1, size, size, channels, members)
    assert torch.isfinite(ens).all()
    # Members are independent draws -> spread across the member axis is nonzero.
    assert ens.var(dim=-1).mean() > 0


@pytest.mark.parametrize(("size", "channels", "patch"), CASES)
def test_map_batched_matches_looped_map(size, channels, patch):
    """One batched member is the SAME transform as a single looped map() draw.

    With ``members=1`` the batched path draws the same-shaped latent and the same
    conditioning as ``map``, so under a fixed seed the two must agree bit-for-bit
    -- guarding against a mis-wired reverse/reshape that the spread check alone
    would miss. Perturb off the zero-init so the transform is non-trivial."""
    proc = _processor(size, channels, patch).eval()
    with torch.no_grad():
        for p in proc.parameters():
            p.add_(0.02 * torch.randn_like(p))
    x = torch.randn(3, 1, size, size, channels)
    torch.manual_seed(0)
    looped = proc.map(x, None)
    torch.manual_seed(0)
    batched = proc.map_batched(x, 1)[:, 0]  # drop the singleton member axis
    assert batched.shape == looped.shape
    assert torch.allclose(batched, looped, atol=1e-5), (
        (batched - looped).abs().max().item()
    )


def test_requires_square_2d():
    with pytest.raises(ValueError, match="square 2D"):
        TarFlowProcessor(spatial_shape=(40, 1))


def test_encode_logdet_matches_numerical_jacobian():
    """The autoregressive blocks' reported ``log|det dz/dy|`` equals the
    numerical Jacobian determinant of the data->latent encode."""
    torch.manual_seed(0)
    proc = _processor(8, 1, patch_size=2, num_blocks=2).eval()
    with torch.no_grad():
        for p in proc.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(1, 1, 8, 8, 1)
    cond = proc._cond_patches(x)
    y_img = proc._to_images(torch.randn(1, 1, 8, 8, 1), 1, 1)
    y0 = proc._patchify(y_img)

    _, logdets = proc._encode(y0, cond)
    # TarFlow reports the change-of-variables term as a per-element mean (its
    # loss mean-reduces both the prior and the log-det together), whereas
    # ``slogdet`` returns the sum, so scale by the latent element count.
    n_elem = y0[0].numel()

    def flat_encode(flat):
        return proc._encode(flat.reshape(y0.shape), cond)[0].reshape(-1)

    jac = torch.autograd.functional.jacobian(flat_encode, y0.reshape(-1))
    _, logabsdet = torch.linalg.slogdet(jac)
    assert torch.allclose(logdets.reshape(()) * n_elem, logabsdet, atol=1e-2), (
        logdets.item() * n_elem,
        logabsdet.item(),
    )
