import lightning as L

from autocast.models.processor import ProcessorModel
from autocast.processors.vit import AViTProcessor


def test_vit_processor(encoded_batch, encoded_dummy_loader):
    input_channels = encoded_batch.encoded_inputs.shape[1]
    output_channels = encoded_batch.encoded_output_fields.shape[1]
    assert (
        encoded_batch.encoded_inputs.shape[2:]
        == encoded_batch.encoded_output_fields.shape[2:]
    )
    spatial_resolution = tuple(encoded_batch.encoded_output_fields.shape[2:])

    processor = AViTProcessor(
        in_channels=input_channels,
        out_channels=output_channels,
        spatial_resolution=spatial_resolution,
    )
    model = ProcessorModel(
        processor=processor,
    )

    output = model.map(encoded_batch.encoded_inputs)
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
