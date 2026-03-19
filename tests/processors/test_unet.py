import lightning as L
import torch
from conftest import get_optimizer_config

from autocast.models.processor import ProcessorModel
from autocast.processors.unet import AzulaUNetProcessor, UNetProcessor

# ---------------------------------------------------------------------------
# Classic UNet Processor Tests
# ---------------------------------------------------------------------------


def test_unet_processor(encoded_batch, encoded_dummy_loader):
    """Test UNet processor with encoded batch data."""
    input_channels = encoded_batch.encoded_inputs.shape[1]
    output_channels = encoded_batch.encoded_output_fields.shape[1]
    spatial_shape = encoded_batch.encoded_inputs.shape[2:]
    n_spatial_dims = len(spatial_shape)

    processor = UNetProcessor(
        in_channels=input_channels,
        out_channels=output_channels,
        spatial_resolution=spatial_shape,
        n_spatial_dims=n_spatial_dims,
        init_features=16,  # Smaller for faster testing
    )
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(),
    )

    output = model.map(encoded_batch.encoded_inputs, None)
    assert output.shape == encoded_batch.encoded_output_fields.shape

    train_loss = model.training_step(encoded_batch, 0)
    assert train_loss.shape == ()
    train_loss.backward()

    # Run a full training loop
    L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        enable_model_summary=False,
        accelerator="cpu",
    ).fit(
        model,
        train_dataloaders=encoded_dummy_loader,
        val_dataloaders=encoded_dummy_loader,
    )


def test_unet_gradient_checkpointing(encoded_batch):
    """Test UNet processor with gradient checkpointing enabled."""
    input_channels = encoded_batch.encoded_inputs.shape[1]
    output_channels = encoded_batch.encoded_output_fields.shape[1]
    spatial_shape = encoded_batch.encoded_inputs.shape[2:]
    n_spatial_dims = len(spatial_shape)

    processor = UNetProcessor(
        in_channels=input_channels,
        out_channels=output_channels,
        spatial_resolution=spatial_shape,
        n_spatial_dims=n_spatial_dims,
        init_features=16,
        gradient_checkpointing=True,
    )
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(),
    )

    output = model.map(encoded_batch.encoded_inputs, None)
    assert output.shape == encoded_batch.encoded_output_fields.shape

    train_loss = model.training_step(encoded_batch, 0)
    assert train_loss.shape == ()
    train_loss.backward()


# ---------------------------------------------------------------------------
# Azula UNet Processor Tests
# ---------------------------------------------------------------------------


def test_azula_unet_processor(encoded_batch, encoded_dummy_loader):
    """Test Azula UNet processor with encoded batch data."""
    in_ch = encoded_batch.encoded_inputs.shape[1]
    out_ch = encoded_batch.encoded_output_fields.shape[1]

    processor = AzulaUNetProcessor(
        in_channels=in_ch,
        out_channels=out_ch,
        hid_channels=[32, 64],
        hid_blocks=[1, 1],
    )
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(),
    )

    output = model.map(encoded_batch.encoded_inputs, None)
    assert output.shape == encoded_batch.encoded_output_fields.shape

    train_loss = model.training_step(encoded_batch, 0)
    assert train_loss.shape == ()
    train_loss.backward()

    # Run a full training loop
    L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        enable_model_summary=False,
        accelerator="cpu",
    ).fit(
        model,
        train_dataloaders=encoded_dummy_loader,
        val_dataloaders=encoded_dummy_loader,
    )


def test_azula_unet_no_noise_is_deterministic(encoded_batch):
    """Test that without noise injection, model is deterministic."""
    in_ch = encoded_batch.encoded_inputs.shape[1]
    out_ch = encoded_batch.encoded_output_fields.shape[1]

    # Test with n_noise_channels=0
    processor_no_noise = AzulaUNetProcessor(
        in_channels=in_ch,
        out_channels=out_ch,
        hid_channels=[32, 64],
        hid_blocks=[1, 1],
        n_noise_channels=0,
    )

    x = encoded_batch.encoded_inputs
    pred1 = processor_no_noise(x, x_noise=None)
    pred2 = processor_no_noise(x, x_noise=None)
    assert torch.allclose(pred1, pred2, atol=1e-6)

    # Test with n_noise_channels=None
    processor_none = AzulaUNetProcessor(
        in_channels=in_ch,
        out_channels=out_ch,
        hid_channels=[32, 64],
        hid_blocks=[1, 1],
        n_noise_channels=None,
    )

    pred3 = processor_none(x, x_noise=None)
    pred4 = processor_none(x, x_noise=None)
    assert torch.allclose(pred3, pred4, atol=1e-6)


def test_azula_unet_noise_with_different_normalization(encoded_batch):
    """Test noise injection works with different normalization types."""
    in_ch = encoded_batch.encoded_inputs.shape[1]
    out_ch = encoded_batch.encoded_output_fields.shape[1]
    x = encoded_batch.encoded_inputs

    for norm_type in ["group", "layer"]:
        processor = AzulaUNetProcessor(
            in_channels=in_ch,
            out_channels=out_ch,
            hid_channels=[32, 64],
            hid_blocks=[1, 1],
            n_noise_channels=128,
            norm=norm_type,
        )

        # Multiple calls to map should produce different outputs (different noise)
        output1 = processor.map(x, None)
        output2 = processor.map(x, None)

        assert output1.shape == encoded_batch.encoded_output_fields.shape
        assert output2.shape == encoded_batch.encoded_output_fields.shape

        # Should be different due to different random noise
        assert not torch.allclose(output1, output2, atol=1e-6)
