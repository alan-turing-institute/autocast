import lightning as L
from conftest import get_optimizer_config

from autocast.models.processor import ProcessorModel
from autocast.processors.unet import UNetProcessor


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
