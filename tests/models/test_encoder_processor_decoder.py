import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from auto_cast.decoders.channels_last import ChannelsLast
from auto_cast.encoders.permute_concat import PermuteConcat
from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.models.encoder_processor_decoder import EncoderProcessorDecoder
from auto_cast.processors.base import Processor
from auto_cast.types import Batch, Tensor


class TinyProcessor(Processor):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def map(self, x: Tensor) -> Tensor:
        return self(x)


def _toy_batch(
    batch_size: int = 2,
    t_in: int = 2,
    t_out: int | None = None,
    w: int = 4,
    h: int = 4,
    c: int = 1,
) -> Batch:
    t_out = t_out or t_in
    input_fields = torch.randn(batch_size, t_in, w, h, c)
    output_fields = torch.randn(batch_size, t_out, w, h, c)
    return Batch(
        input_fields=input_fields,
        output_fields=output_fields,
        constant_scalars=None,
        constant_fields=None,
    )


class _BatchDataset(Dataset):
    def __len__(self) -> int:
        return 2  # keep small

    def __getitem__(self, idx: int) -> Batch:
        return _toy_batch(batch_size=1)


dummy_loader = DataLoader(
    _BatchDataset(),
    batch_size=1,
    collate_fn=lambda items: items[0],
    num_workers=0,
)


def test_encoder_processor_decoder_training_step_runs():
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

    batch = _toy_batch()
    train_loss = model.training_step(batch, 0)

    assert train_loss.shape == ()
    train_loss.backward()

    trainer = L.Trainer(
        max_epochs=1, logger=False, enable_checkpointing=False, limit_train_batches=1
    )
    trainer.fit(model, train_dataloaders=dummy_loader, val_dataloaders=dummy_loader)


def test_encoder_processor_decoder_rollout_is_mixin_backed():
    encoder = PermuteConcat(with_constants=False)
    decoder = ChannelsLast()
    loss = nn.MSELoss()
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder, loss_func=loss)
    processor = TinyProcessor()
    model = EncoderProcessorDecoder.from_encoder_processor_decoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=loss,
        stride=1,
        max_rollout_steps=2,
        teacher_forcing_ratio=0.0,
    )

    batch = _toy_batch(t_in=2, t_out=2)
    preds, gts = model.rollout(batch)

    assert preds.shape[0] == 2
    assert gts is not None
    assert gts.shape[0] == 2
