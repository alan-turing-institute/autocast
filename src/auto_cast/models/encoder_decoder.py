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

    def __init__(self):
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.decoder(self.encoder(*args, **kwargs))

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        output = self.encode(batch)
        loss = self.loss_func(output, batch.output_fields)
        return loss  # noqa: RET504

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def predict_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def encode(self, batch: Batch) -> Tensor:
        return self.encoder.encode(batch)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder.decode(z)

    def configure_optmizers(self):
        pass


class VAE(EncoderDecoder):
    """Variational Autoencoder Model."""

    encoder: Encoder
    decoder: Decoder
    fc_mean: nn.Linear
    fc_log_var: nn.Linear

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc_mean = nn.Linear(encoder.latent_dim, encoder.latent_dim)
        self.fc_log_var = nn.Linear(encoder.latent_dim, encoder.latent_dim)

    def forward(self, batch: Batch) -> Tensor:
        encoded = self.encode(batch)
        if (
            isinstance(encoded, Tensor)
            and encoded.dim() == 2
            and encoded.size(1) != 2 * self.encoder.latent_dim
        ):
            msg = "encoded must be [B, 2 * latent_dim]"
            raise ValueError(msg)
        mean, log_var = encoded.chunk(2, dim=-1)
        z = self.reparametrize(mean, log_var)
        return self.decode(z)

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, batch: Batch) -> Tensor:
        h = super().encode(batch)  # [B, latent_dim]
        mean = self.fc_mean(h)  # [B, latent_dim]
        log_var = self.fc_log_var(h)  # [B, latent_dim]
        return torch.cat([mean, log_var], dim=-1)  # [B, 2*latent_dim]
