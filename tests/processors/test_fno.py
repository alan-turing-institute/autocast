import lightning as L
import torch
from conftest import get_optimizer_config

from autocast.decoders.channels_last import ChannelsLast
from autocast.encoders.permute_concat import PermuteConcat
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.models.processor import ProcessorModel
from autocast.nn.noise.noise_injector import ConcatenatedNoiseInjector
from autocast.processors.fno import FNOProcessor
from autocast.types import Batch


def test_fno_processor(encoded_batch, encoded_dummy_loader):
    input_channels = encoded_batch.encoded_inputs.shape[1]
    output_channels = encoded_batch.encoded_output_fields.shape[1]
    processor = FNOProcessor(
        in_channels=input_channels,
        out_channels=output_channels,
        n_modes=(4, 4),
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


def test_fno_epd_concat_noise_produces_distinct_ensemble_members():
    torch.manual_seed(0)
    batch_size = 2
    n_members = 3
    channels = 2
    n_steps_output = 4
    spatial_resolution = 16

    batch = Batch(
        input_fields=torch.randn(
            batch_size, 1, spatial_resolution, spatial_resolution, channels
        ),
        output_fields=torch.randn(
            batch_size,
            n_steps_output,
            spatial_resolution,
            spatial_resolution,
            channels,
        ),
        constant_scalars=None,
        constant_fields=None,
    )
    encoder_decoder = EncoderDecoder(
        encoder=PermuteConcat(
            in_channels=channels + 1,
            n_steps_input=1,
            with_constants=False,
        ),
        decoder=ChannelsLast(
            output_channels=channels,
            time_steps=n_steps_output,
        ),
    )
    processor = FNOProcessor(
        in_channels=channels + 1,
        out_channels=channels * n_steps_output,
        n_modes=(4, 4),
        hidden_channels=8,
        n_layers=2,
    )
    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=encoder_decoder,
        processor=processor,
        n_members=n_members,
        input_noise_injector=ConcatenatedNoiseInjector(n_channels=1),
    )

    output = model(batch)

    assert output.shape == (
        batch_size,
        n_steps_output,
        spatial_resolution,
        spatial_resolution,
        channels,
        n_members,
    )
    assert not torch.allclose(output[..., 0], output[..., 1])
