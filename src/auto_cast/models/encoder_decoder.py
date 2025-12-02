from typing import Any

import lightning as L
import torch
from torch import nn

from auto_cast.decoders import Decoder
from auto_cast.encoders import Encoder
from auto_cast.types import Batch, Tensor


class EncoderDecoder(L.LightningModule):
    """Encoder-Decoder Model."""

    encoder: Encoder
    decoder: Decoder
    loss_func: nn.Module
    learning_rate: float = 1e-3

    def __init__(self):
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.decoder(self.encoder(*args, **kwargs))

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        output = self(batch)
        loss = self.loss_func(output, batch.output_fields)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        output = self(batch)
        loss = self.loss_func(output, batch.output_fields)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        return self(batch)

    def predict_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        return self(batch)

    def encode(self, batch: Batch) -> Tensor:
        return self.encoder.encode(batch)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder.decode(z)

    def configure_optimizers(self):
        """Configure optimizers for training."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
