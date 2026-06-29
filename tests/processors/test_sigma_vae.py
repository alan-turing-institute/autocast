import pytest
import torch

from autocast.processors.sigma_vae import SigmaVAEProcessor
from autocast.types import EncodedBatch

# (spatial_shape, channels) spanning every toy archetype the model must cover:
#   scalar (ou/dw), vector (mvou), 1D (l96), 2D (gs).
SHAPES = [((1, 1), 1), ((1, 1), 3), ((40, 1), 1), ((8, 8), 2)]


def _batch(b, spatial, channels):
    shape = (b, 1, *spatial, channels)
    return EncodedBatch(
        encoded_inputs=torch.randn(*shape),
        encoded_output_fields=torch.randn(*shape),
        global_cond=None,
        encoded_info={},
    )


def _processor(spatial, channels):
    return SigmaVAEProcessor(
        n_channels_in=channels,
        n_channels_out=channels,
        spatial_shape=spatial,
        latent_dim=8,
        hidden_features=(32, 32),
    )


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_loss_finite_and_backward(spatial, channels):
    proc = _processor(spatial, channels)
    loss = proc.loss(_batch(4, spatial, channels))
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    # The learned output sigma must receive a gradient (it is being calibrated).
    assert proc.log_sigma.grad is not None
    assert torch.isfinite(proc.log_sigma.grad).all()


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_map_shape_and_finite(spatial, channels):
    proc = _processor(spatial, channels)
    x = torch.randn(3, 1, *spatial, channels)
    y = proc.map(x, None)
    assert y.shape == (3, 1, *spatial, channels)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(("spatial", "channels"), SHAPES)
def test_two_train_steps_decrease(spatial, channels):
    torch.manual_seed(0)
    proc = _processor(spatial, channels)
    batch = _batch(16, spatial, channels)
    opt = torch.optim.Adam(proc.parameters(), lr=1e-2)
    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = proc.loss(batch)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]


def test_auto_init_sets_sigma_to_residual_scale():
    """init_log_sigma='auto' (the default) sets the output sigma to the data's
    per-channel one-step residual std on the first training batch.

    The output sigma is a single scalar per channel, so a decaying LR cannot move
    it far within a budget: started at a fixed constant it ships at that constant
    (the 4-6x over-dispersion seen when init_log_sigma=0 -> sigma=1 never reached
    the ~0.1-0.3 residual scale). Data-driven init starts it calibrated."""
    torch.manual_seed(0)
    proc = SigmaVAEProcessor(
        n_channels_in=1,
        n_channels_out=1,
        spatial_shape=(4, 1),
        latent_dim=4,
        hidden_features=(32, 32),
    )
    assert not bool(proc._sigma_inited)
    resid_std = 0.1
    x = torch.randn(4096, 1, 4, 1, 1)
    y = x + resid_std * torch.randn_like(x)
    proc.loss(
        EncodedBatch(
            encoded_inputs=x,
            encoded_output_fields=y,
            global_cond=None,
            encoded_info={},
        )
    )
    assert bool(proc._sigma_inited)
    assert proc.log_sigma.exp().item() == pytest.approx(resid_std, rel=0.15)


def test_loss_under_inference_mode_does_not_init_sigma():
    """Lightning runs a validation sanity-check pass (eval mode, inference_mode)
    before training starts. The data-driven sigma init must not fire there: the
    in-place init would raise under inference_mode, and a validation batch is the
    wrong source for the calibration. The init is training-only."""
    proc = _processor((4, 1), 1).eval()
    assert not bool(proc._sigma_inited)
    with torch.inference_mode():
        loss = proc.loss(_batch(4, (4, 1), 1))
    assert torch.isfinite(loss)
    assert not bool(proc._sigma_inited)  # still deferred to the first training batch


def test_loss_rejects_mismatched_field_shape():
    """A field whose flattened size does not match the configured dims raises a
    clear error instead of a cryptic matmul failure inside the encoder."""
    proc = _processor((8, 8), 1)
    bad = EncodedBatch(
        encoded_inputs=torch.randn(4, 1, 8, 8, 1),
        encoded_output_fields=torch.randn(4, 1, 8, 8, 2),  # wrong channel count
        global_cond=None,
        encoded_info={},
    )
    with pytest.raises(ValueError, match="features per sample but expected"):
        proc.loss(bad)


def test_predictive_std_calibrates_to_residual_scale():
    """After a realistic-budget fit on near-identity dynamics with a known small
    one-step noise, the GENERATIVE predictive std matches that residual scale --
    not the 4-6x over-dispersion a fixed init_log_sigma=0 (sigma=1) produced."""
    torch.manual_seed(0)
    resid_std = 0.1
    proc = SigmaVAEProcessor(
        n_channels_in=1,
        n_channels_out=1,
        spatial_shape=(4, 1),
        latent_dim=4,
        hidden_features=(64, 64),
    )
    x = torch.randn(2048, 1, 4, 1, 1)
    y = x + resid_std * torch.randn_like(x)
    batch = EncodedBatch(
        encoded_inputs=x,
        encoded_output_fields=y,
        global_cond=None,
        encoded_info={},
    )
    opt = torch.optim.Adam(proc.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        proc.loss(batch).backward()
        opt.step()
    with torch.no_grad():
        members = torch.stack([proc.map(x[:512], None) for _ in range(64)])
    pred_std = members.std(dim=0).mean().item()
    # Tight enough to catch the 4-6x failure, loose enough to be non-flaky.
    assert 0.5 * resid_std <= pred_std <= 2.0 * resid_std
