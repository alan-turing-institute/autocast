from typing import Any

import lightning as L
import torch
from torch import nn

from auto_cast.decoders import Decoder
from auto_cast.encoders import Encoder
from auto_cast.processors.base import Preprocessor
from auto_cast.types import Batch, Tensor


class EncoderDecoder(L.LightningModule):
    """Encoder-Decoder Model."""

    encoder: Encoder
    decoder: Decoder
    preprocessor: Preprocessor
    loss_func: nn.Module

    def __init__(self):
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.decoder(self.encoder(*args, **kwargs))

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        x = self.preprocessor(batch)
        output = self(x)
        loss = self.loss_func(output, batch["output_fields"])
        return loss  # noqa: RET504

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def predict_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def encode(self, x: Batch) -> Tensor:
        x = self.preprocessor(x)
        return self.encoder(x)

    def configure_optmizers(self):
        pass


class VAE(EncoderDecoder):
    """Variational Autoencoder Model."""

    def forward(self, x: Tensor) -> Tensor:
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        x = self.decoder(z)
        return x  # noqa: RET504

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
