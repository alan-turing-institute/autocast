import torch
from torch import nn, rand

from autocast.decoders.dc import DCDecoder
from autocast.encoders.dc import DCEncoder
from autocast.models.autoencoder import AE, AELoss
from autocast.types import Batch


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
    x = rand(*new_shape, requires_grad=requires_grad)
    return Batch(
        input_fields=x,
        output_fields=x.clone(),
        constant_scalars=None,
        constant_fields=None,
    )


def test_ae_full_dcae_2d():
    """Test AE with full DCAE (DCEncoder + DCDecoder) in 2D."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64, 128),
        spatial=2,
        hid_blocks=(2, 2, 2),
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(128, 64, 32),  # Reversed from encoder
        spatial=2,
        hid_blocks=(2, 2, 2),
        pixel_shuffle=False,
    )

    # Test forward pass through encoder and decoder
    batch = _make_batch((2, 64, 64, 3))
    x = batch.input_fields
    z = encoder.encode(batch)
    x_recon = decoder.decode(z)

    assert x_recon.shape == x.shape, f"Expected {x.shape}, got {x_recon.shape}"
    assert z.shape == (2, 2, 16, 16, 16), f"Expected (2, 2, 16, 16, 16), got {z.shape}"
    assert not x_recon.isnan().any(), "Reconstruction contains NaN values"
    assert not z.isnan().any(), "Latent contains NaN values"


def test_ae_full_dcae_3d():
    """Test AE with full DCAE (DCEncoder + DCDecoder) in 3D."""
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
        hid_channels=(32, 16),  # Reversed from encoder
        spatial=3,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    # Test forward pass through encoder and decoder
    batch = _make_batch((2, 32, 32, 32, 4))
    x = batch.input_fields
    z = encoder.encode(batch)
    x_recon = decoder.decode(z)

    assert x_recon.shape == x.shape, f"Expected {x.shape}, got {x_recon.shape}"
    expected_latent_shape = (2, 2, 16, 16, 16, 8)  # (B, T, spatial..., C)
    assert z.shape == expected_latent_shape, (
        f"Expected {expected_latent_shape}, got {z.shape}"
    )
    assert not x_recon.isnan().any(), "Reconstruction contains NaN values"
    assert not z.isnan().any(), "Latent contains NaN values"


def test_ae_hybrid_unet_dc():
    """Test that DCDecoder can work with different input sizes."""
    # Note: UNetEncoder is actually a full UNet (encoder-decoder) from Azula,
    # so we just test that DCDecoder can handle different latent shapes
    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(128, 64, 32),
        spatial=2,
        hid_blocks=(2, 2, 2),
        pixel_shuffle=False,
    )

    # Test forward pass with different latent sizes
    z1 = rand(2, 16, 8, 8)
    x1 = decoder.forward(z1)
    assert x1.shape == (2, 32, 32, 3), f"Expected (2, 32, 32, 3), got {x1.shape}"

    z2 = rand(2, 16, 16, 16)
    x2 = decoder.forward(z2)
    assert x2.shape == (2, 64, 64, 3), f"Expected (2, 64, 64, 3), got {x2.shape}"

    assert not x1.isnan().any(), "Output contains NaN values"
    assert not x2.isnan().any(), "Output contains NaN values"


def test_ae_pixel_shuffle():
    """Test AE with pixel_shuffle enabled."""
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

    # Test forward pass through encoder and decoder
    batch = _make_batch((2, 64, 64, 3))
    x = batch.input_fields
    z = encoder.encode(batch)
    x_recon = decoder.decode(z)

    assert x_recon.shape == x.shape, f"Expected {x.shape}, got {x_recon.shape}"
    assert not x_recon.isnan().any(), "Reconstruction contains NaN values"


def test_ae_with_attention():
    """Test AE with self-attention in encoder and decoder."""
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

    # Test forward pass through encoder and decoder
    batch = _make_batch((2, 64, 64, 3))
    x = batch.input_fields
    z = encoder.encode(batch)
    x_recon = decoder.decode(z)

    assert x_recon.shape == x.shape, f"Expected {x.shape}, got {x_recon.shape}"
    assert not x_recon.isnan().any(), "Reconstruction contains NaN values"


def test_ae_reconstruction_loss():
    """Test that AE can compute reconstruction loss."""
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

    # Test forward pass and loss computation
    batch = _make_batch((2, 64, 64, 3))
    x = batch.input_fields
    z = encoder.encode(batch)
    x_recon = decoder.decode(z)

    # Compute MSE loss
    loss = nn.functional.mse_loss(x_recon, x)

    assert loss.item() >= 0, "Loss should be non-negative"
    assert not loss.isnan(), "Loss is NaN"


def test_ae_gradient_flow():
    """Test that gradients flow through the full AE model."""
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

    # Forward pass with loss
    batch = _make_batch((2, 64, 64, 3), requires_grad=True)
    x = batch.input_fields
    z = encoder.encode(batch)
    x_recon = decoder.decode(z)
    loss = nn.functional.mse_loss(x_recon, x)
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


def test_ae_parameter_count():
    """Test that AE model has reasonable number of parameters."""
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

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    n_params = encoder_params + decoder_params
    n_trainable = sum(
        p.numel()
        for p in list(encoder.parameters()) + list(decoder.parameters())
        if p.requires_grad
    )

    assert n_params > 0, "Model has no parameters"
    assert n_trainable > 0, "Model has no trainable parameters"
    assert n_trainable == n_params, "Not all parameters are trainable"


def test_ae_wrapper_forward_with_batch():
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

    model = AE(encoder=encoder, decoder=decoder)
    batch = _make_batch((2, 64, 64, 3))

    output = model(batch)
    assert output.shape == batch.output_fields.shape
    decoded, encoded = model.forward_with_latent(batch)
    assert torch.allclose(output, decoded)
    assert encoded.shape[0] == batch.input_fields.shape[0]
    assert encoded.shape[1] == batch.input_fields.shape[1]  # Time dimension
    assert encoded.shape[-1] == encoder.latent_dim  # Channel dimension


def test_ae_wrapper_loss_and_backward():
    encoder = DCEncoder(
        in_channels=3,
        out_channels=8,
        hid_channels=(16, 32),
        spatial=2,
        hid_blocks=(1, 1),
        pixel_shuffle=False,
    )

    decoder = DCDecoder(
        in_channels=8,
        out_channels=3,
        hid_channels=(32, 16),
        spatial=2,
        hid_blocks=(1, 1),
        pixel_shuffle=False,
    )

    model = AE(encoder=encoder, decoder=decoder, loss_func=AELoss())
    batch = _make_batch((2, 64, 64, 3))
    model.train()

    assert model.loss_func is not None
    loss = model.loss_func(model, batch)
    assert loss.item() >= 0
    loss.backward()

    grads_present = any(
        param.grad is not None and not param.grad.isnan().any()
        for param in model.parameters()
        if param.requires_grad
    )
    assert grads_present, "Expected gradients on AE parameters"
