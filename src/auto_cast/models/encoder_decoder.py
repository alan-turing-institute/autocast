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
    loss_func: nn.Module | None

    def __init__(
        self, encoder: Encoder, decoder: Decoder, loss_func: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func

    def forward(self, batch: Batch) -> Tensor:
        return self.decoder(self.encoder(batch))

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        if self.loss_func is None:
            msg = "Loss function not defined for EncoderDecoder model."
            raise ValueError(msg)
        x = self.encode(batch)
        output = self.decoder(x)
        loss = self.loss_func(output, batch.output_fields)
        return loss  # noqa: RET504

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def predict_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def encode(self, x: Batch) -> Tensor:
        return self.encoder(x)

    def configure_optmizers(self):
        pass


class VAE(EncoderDecoder):
    """Variational Autoencoder Model."""

    def forward(self, batch: Batch) -> Tensor:
        mu, log_var = self.encoder(batch)
        z = self.reparametrize(mu, log_var)
        x = self.decoder(z)
        return x  # noqa: RET504

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
