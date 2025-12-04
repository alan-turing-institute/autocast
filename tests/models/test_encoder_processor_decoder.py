import lightning as L
from torch import nn

from auto_cast.decoders.channels_last import ChannelsLast
from auto_cast.encoders.permute_concat import PermuteConcat
from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.models.encoder_processor_decoder import EncoderProcessorDecoder
from auto_cast.processors.base import Processor
from auto_cast.types import Tensor


class TinyProcessor(Processor):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def map(self, x: Tensor) -> Tensor:
        return self(x)


def test_encoder_processor_decoder_training_step_runs(make_toy_batch, dummy_loader):
    encoder = PermuteConcat(with_constants=False)
    decoder = ChannelsLast()
    loss = nn.MSELoss()
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder, loss_func=loss)

    processor = TinyProcessor()
    model = EncoderProcessorDecoder.from_encoder_processor_decoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=loss,
    )

    batch = make_toy_batch()
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
