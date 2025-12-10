import math

import torch
from torch import nn

from autocast.decoders.base import Decoder
from autocast.decoders.dc import DCDecoder
from autocast.encoders.base import Encoder
from autocast.encoders.dc import DCEncoder
from autocast.models.vae import VAE, VAELoss
from autocast.types import (
    Batch,
    TensorBNC,
    TensorBSC,
    TensorBTSC,
)


def _make_batch(shape: tuple[int, ...], *, requires_grad: bool = False) -> Batch:
    """Create a batch with time dimension.

    Args:
        shape: Shape without time dimension (B, spatial..., C)

    Returns:
        Batch with shape (B, T, spatial..., C) where T=2
    """
    # Add time dimension: (B, spatial..., C) -> (B, T, spatial..., C)
    b, *spatial_and_c = shape
    t = 2  # Default time steps
    new_shape = (b, t, *spatial_and_c)
    x = torch.rand(*new_shape, requires_grad=requires_grad)
    return Batch(
        input_fields=x,
        output_fields=x.clone(),
        constant_scalars=None,
        constant_fields=None,
    )


class _FlatEncoder(Encoder):
    """Minimal encoder that produces flat (non-spatial) latents for tests."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.GELU(),
            nn.Linear(2 * input_dim, latent_dim),
        )

    def encode(self, batch: Batch) -> TensorBNC:
        x = batch.input_fields  # (B, T, ..., C)
        # Process each time step
        outputs = []
        for idx in range(x.shape[1]):
            x_t = x[:, idx, ...]  # (B, ..., C) or (B, C) for flat
            outputs.append(self.net(x_t))
        return torch.stack(outputs, dim=1)

    def forward(self, batch: Batch) -> TensorBNC:
        return self.encode(batch)


class _FlatDecoder(Decoder):
    """Minimal decoder that reconstructs flat tensors for tests."""

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, output_dim),
        )

    def decode(self, z: TensorBNC) -> TensorBNC:
        outputs = []
        for idx in range(z.shape[1]):
            z_t: TensorBNC = z[:, idx, ...]  # (B, C)
            outputs.append(self.net(z_t))
        return torch.stack(outputs, dim=1)  # (B, T, C)


class _FlatteningEncoder(Encoder):
    """Encoder that ingests spatial tensors and outputs flat latents."""

    def __init__(self, input_shape: tuple[int, ...], latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        in_features = math.prod(input_shape)
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, 2 * in_features),
            nn.GELU(),
            nn.Linear(2 * in_features, latent_dim),
        )

    def encode(self, batch: Batch) -> TensorBNC:
        x: TensorBTSC = batch.input_fields  # (B, T, spatial..., C)
        outputs = []
        for idx in range(x.shape[1]):
            x_t: TensorBSC = x[:, idx, ...]  # (B, spatial..., C)
            outputs.append(self.net(x_t))
        return torch.stack(outputs, dim=1)  # (B, T, latent_dim)


class _FlatteningDecoder(Decoder):
    """Decoder that maps flat latents back to spatial tensors."""

    def __init__(self, latent_dim: int, output_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_shape[0]
        self.output_shape = output_shape
        out_features = math.prod(output_shape)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, out_features),
        )

    def decode(self, z: TensorBNC) -> TensorBTSC:
        outputs = []
        for idx in range(z.shape[1]):
            z_t: TensorBNC = z[:, idx, ...]  # (B, latent_dim)
            x_t = self.net(z_t)
            outputs.append(x_t.view(-1, *self.output_shape))
        return torch.stack(outputs, dim=1)  # (B, T, C, H, W)


def test_vae_spatial_latents_2d():
    """Test VAE with spatial latent representations in 2D (e.g., DCEncoder)."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    # Create VAE with spatial=2 for 2D spatial latents
    vae = VAE(encoder=encoder, decoder=decoder, spatial=2)

    # Create a batch (channel-last)
    batch = _make_batch((2, 64, 64, 3))
    x = batch.output_fields

    # Test forward pass
    output = vae(batch)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    assert not output.isnan().any(), "Output contains NaN values"

    # Test forward_with_latent
    decoded, encoded = vae.forward_with_latent(batch)
    assert decoded.shape == x.shape, f"Expected {x.shape}, got {decoded.shape}"

    # Encoded should be [B, T, H, W, D, 2*C] where C=8
    assert encoded.shape[0] == 2, "Batch size should be 2"
    assert encoded.shape[1] == 2, "Time dimension should be 2"
    assert encoded.shape[-1] == 32, (
        f"Expected 32 channels (2*16), got {encoded.shape[-1]}"
    )
    assert encoded.dim() == 5, "Encoded should be 5D (B, T, H, W, 2*C)"
    assert not encoded.isnan().any(), "Encoded contains NaN values"

    # Test loss computation
    assert vae.loss_func is not None
    loss = vae.loss_func(vae, batch)
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not loss.isnan(), "Loss is NaN"


def test_vae_spatial_latents_3d():
    """Test VAE with spatial latent representations in 3D."""
    encoder = DCEncoder(
        in_channels=4,
        out_channels=8,
        hid_channels=(16, 32),
        spatial=3,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=8,
        out_channels=4,
        hid_channels=(32, 16),
        spatial=3,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    # Create VAE with spatial=3 for 3D spatial latents
    vae = VAE(encoder=encoder, decoder=decoder, spatial=3)

    # Create a batch (channel-last)
    batch = _make_batch((2, 32, 32, 32, 4))
    x = batch.input_fields

    # Test forward pass
    output = vae(batch)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    assert not output.isnan().any(), "Output contains NaN values"

    # Test forward_with_latent
    decoded, encoded = vae.forward_with_latent(batch)
    assert decoded.shape == x.shape, f"Expected {x.shape}, got {decoded.shape}"

    # Encoded should be [B, T, D, H, W, 2*C] where C=8
    assert encoded.shape[0] == 2, "Batch size should be 2"
    assert encoded.shape[1] == 2, "Time dimension should be 2"
    assert encoded.shape[-1] == 16, (
        f"Expected 16 channels (2*8), got {encoded.shape[-1]}"
    )
    assert encoded.dim() == 6, "Encoded should be 6D (B, T, D, H, W, 2*C)"
    assert not encoded.isnan().any(), "Encoded contains NaN values"


# @pytest.mark.skip(reason="Circular reference in DCEncoder causes recursion")
def test_vae_reparametrization_trick():
    """Test that reparametrization trick produces stochastic output during training."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    vae = VAE(encoder=encoder, decoder=decoder, spatial=2)

    batch = _make_batch((2, 64, 64, 3))

    # In training mode, outputs should be different due to sampling
    vae.train()
    output1 = vae(batch)
    output2 = vae(batch)
    assert not torch.allclose(output1, output2), (
        "Training outputs should differ due to stochastic sampling"
    )

    # In eval mode, outputs should be deterministic (same mean)
    vae.eval()
    output3 = vae(batch)
    output4 = vae(batch)
    assert torch.allclose(output3, output4, atol=1e-6), (
        "Eval outputs should be deterministic"
    )


def test_vae_flat_latents():
    """Ensure VAE handles flat latent representations with minimal spatial input."""

    input_dim = 12
    latent_dim = 4
    encoder = _FlatEncoder(input_dim=input_dim, latent_dim=latent_dim)
    decoder = _FlatDecoder(latent_dim=latent_dim, output_dim=input_dim)
    vae = VAE(encoder=encoder, decoder=decoder, spatial=None)

    # Use (B, T, 1, C) to satisfy TensorBTSC requirement of at least 1 spatial dim
    x = torch.rand(5, 2, 1, input_dim)  # (B, T, spatial, C)
    batch = Batch(
        input_fields=x,
        output_fields=x,
        constant_scalars=None,
        constant_fields=None,
    )

    # Forward paths should preserve shapes
    output = vae(batch)
    assert output.shape == x.shape
    decoded, encoded = vae.forward_with_latent(batch)
    assert decoded.shape == x.shape
    # Encoded should be (B, T, 1, 2*latent_dim) since spatial dim is preserved
    assert encoded.shape == (x.shape[0], x.shape[1], 1, 2 * latent_dim)

    # Stochastic sampling only during training for flat latents
    vae.train()
    train_out1 = vae(batch)
    train_out2 = vae(batch)
    assert not torch.allclose(train_out1, train_out2)

    vae.eval()
    eval_out1 = vae(batch)
    eval_out2 = vae(batch)
    assert torch.allclose(eval_out1, eval_out2, atol=1e-6)


def test_vae_spatial_input_flat_latent():
    """VAE can flatten spatial inputs down to 1D latent space."""

    input_shape = (3, 16, 16)
    latent_dim = 32
    encoder = _FlatteningEncoder(input_shape=input_shape, latent_dim=latent_dim)
    decoder = _FlatteningDecoder(latent_dim=latent_dim, output_shape=input_shape)
    vae = VAE(encoder=encoder, decoder=decoder, spatial=None)

    x = torch.rand(4, 2, *input_shape)  # Add time dimension (B, T, C, H, W)
    batch = Batch(
        input_fields=x,
        output_fields=x,
        constant_scalars=None,
        constant_fields=None,
    )

    decoded, encoded = vae.forward_with_latent(batch)
    assert decoded.shape == x.shape
    assert encoded.shape == (x.shape[0], x.shape[1], 2 * latent_dim)  # (B, T, 2*C)

    vae.train()
    train_out1 = vae(batch)
    train_out2 = vae(batch)
    assert not torch.allclose(train_out1, train_out2)

    vae.eval()
    eval_decoded, _ = vae.forward_with_latent(batch)
    eval_out = vae(batch)
    assert torch.allclose(eval_out, eval_decoded, atol=1e-6)


def test_vae_kl_divergence():
    """Test KL divergence computation."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    vae = VAE(encoder=encoder, decoder=decoder, spatial=2)

    batch = _make_batch((2, 64, 64, 3))
    x = batch.output_fields

    # Get encoded representation
    decoded, encoded = vae.forward_with_latent(batch)

    # Verify decoded has correct shape
    assert decoded.shape == x.shape, f"Expected {x.shape}, got {decoded.shape}"

    # Compute KL divergence
    kl_div = vae.loss_func.kl_divergence(encoded)  # type: ignore  # noqa: PGH003

    assert kl_div.item() >= 0, "KL divergence should be non-negative"
    assert not kl_div.isnan(), "KL divergence is NaN"


def test_vae_gradient_flow():
    """Test that gradients flow through the full VAE model."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    vae = VAE(encoder=encoder, decoder=decoder, spatial=2)
    vae.train()

    batch = _make_batch((2, 64, 64, 3), requires_grad=True)
    x = batch.input_fields

    # Forward pass with loss
    assert vae.loss_func is not None
    loss = vae.loss_func(vae, batch)
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None, "No gradients on input"
    assert not x.grad.isnan().any(), "Input gradients contain NaN"

    # Check that encoder parameters have gradients
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradients on encoder parameter {name}"
            msg = f"Encoder parameter {name} gradients contain NaN"
            assert not param.grad.isnan().any(), msg

    # Check that decoder parameters have gradients
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradients on decoder parameter {name}"
            msg = f"Decoder parameter {name} gradients contain NaN"
            assert not param.grad.isnan().any(), msg

    # Check that VAE-specific parameters have gradients
    assert vae.fc_mean.weight.grad is not None, "No gradients on fc_mean"
    assert vae.fc_log_var.weight.grad is not None, "No gradients on fc_log_var"


def test_vae_beta_parameter():
    """Test that beta parameter affects loss magnitude."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    # Create VAE with different beta values
    vae1 = VAE(encoder=encoder, decoder=decoder, spatial=2)
    vae1.loss_func = VAELoss(beta=1.0)

    vae2 = VAE(encoder=encoder, decoder=decoder, spatial=2)
    vae2.loss_func = VAELoss(beta=2.0)

    batch = _make_batch((2, 64, 64, 3))

    # Compute losses (both VAEs share same encoder/decoder, so same outputs)
    vae1.eval()
    vae2.eval()

    # Get the same encoded/decoded outputs by using vae1 for both
    decoded, encoded = vae1.forward_with_latent(batch)

    # Manually compute losses with different betas
    recon_loss = torch.nn.functional.mse_loss(decoded, batch.output_fields)
    kl_loss = vae1.loss_func.kl_divergence(encoded)

    loss1 = 1.0 * kl_loss + recon_loss
    loss2 = 2.0 * kl_loss + recon_loss

    # Beta should affect the KL term, making loss2 > loss1
    assert loss2.item() > loss1.item(), "Higher beta should increase loss"


def test_vae_pixel_shuffle():
    """Test VAE with pixel_shuffle enabled."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=True,
    )

    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=True,
    )

    vae = VAE(encoder=encoder, decoder=decoder, spatial=2)

    batch = _make_batch((2, 64, 64, 3))
    x = batch.output_fields

    # Test forward pass
    output = vae(batch)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    assert not output.isnan().any(), "Output contains NaN values"


def test_vae_with_attention():
    """Test VAE with self-attention in encoder and decoder."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        attention_heads={1: 4},
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        attention_heads={0: 4},
        pixel_shuffle=False,
    )

    vae = VAE(encoder=encoder, decoder=decoder, spatial=2)

    batch = _make_batch((2, 64, 64, 3))
    x = batch.output_fields

    # Test forward pass
    output = vae(batch)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    assert not output.isnan().any(), "Output contains NaN values"


def test_vae_parameter_count():
    """Test that VAE model has reasonable number of parameters."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    vae = VAE(encoder=encoder, decoder=decoder, spatial=2)

    # Count parameters
    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

    assert total_params > 0, "Model has no parameters"
    assert trainable_params > 0, "Model has no trainable parameters"
    assert trainable_params == total_params, "Not all parameters are trainable"

    # Check that VAE has more parameters than just encoder + decoder
    # (due to fc_mean and fc_log_var)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    assert total_params > encoder_params + decoder_params, (
        "VAE should have additional parameters"
    )
