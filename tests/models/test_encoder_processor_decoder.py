import lightning as L
from torch import nn

from auto_cast.decoders.channels_last import ChannelsLast
from auto_cast.encoders.permute_concat import PermuteConcat
from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.models.encoder_processor_decoder import EncoderProcessorDecoder
from auto_cast.processors.base import Processor
from auto_cast.types import Tensor


class TinyProcessor(Processor):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def map(self, x: Tensor) -> Tensor:
        return self(x)


def test_encoder_processor_decoder_training_step_runs(make_toy_batch, dummy_loader):
    batch = make_toy_batch()
    output_channels = batch.output_fields.shape[-1]
    time_steps = batch.output_fields.shape[1]
    # Encoder merges C*T into single dimension
    merged_channels = output_channels * time_steps

    encoder = PermuteConcat(with_constants=False)
    decoder = ChannelsLast(output_channels=output_channels, time_steps=time_steps)
    loss = nn.MSELoss()
    encoder_decoder = EncoderDecoder.from_encoder_decoder(
        encoder=encoder, decoder=decoder, loss_func=loss
    )

    processor = TinyProcessor(in_channels=merged_channels)
    model = EncoderProcessorDecoder.from_encoder_processor_decoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=loss,
    )

    train_loss = model.training_step(batch, 0)

    assert train_loss.shape == ()
    train_loss.backward()

    L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        accelerator="cpu",
    ).fit(model, train_dataloaders=dummy_loader, val_dataloaders=dummy_loader)


def test_encoder_processor_decoder_rollout_is_mixin_backed(make_toy_batch):
    batch = make_toy_batch()
    output_channels = batch.output_fields.shape[-1]
    time_steps = batch.output_fields.shape[1]
    merged_channels = output_channels * time_steps

    encoder = PermuteConcat(with_constants=False)
    decoder = ChannelsLast(output_channels=output_channels, time_steps=time_steps)
    loss = nn.MSELoss()
    encoder_decoder = EncoderDecoder.from_encoder_decoder(
        encoder=encoder, decoder=decoder, loss_func=loss
    )
    processor = TinyProcessor(in_channels=merged_channels)
    model = EncoderProcessorDecoder.from_encoder_processor_decoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=loss,
        stride=1,
        max_rollout_steps=2,
        teacher_forcing_ratio=0.0,
    )

    batch = make_toy_batch()
    preds, gts = model.rollout(batch)

    assert preds.shape[0] == 2
    assert gts is not None
    assert gts.shape[0] == 2
