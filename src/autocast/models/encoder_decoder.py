from typing import Any

import lightning as L
import torch
from torch import nn
from torchmetrics import Metric

from autocast.decoders import Decoder
from autocast.encoders import Encoder
from autocast.metrics.utils import MetricsMixin
from autocast.types import Batch, Tensor, TensorBNC, TensorBTSC


class EncoderDecoder(L.LightningModule, MetricsMixin):
    """Encoder-Decoder Model."""

    encoder: Encoder
    decoder: Decoder
    loss_func: nn.Module | None
    learning_rate: float = 1e-3

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        loss_func: nn.Module | None = None,
        val_metrics: list[Metric] | None = None,
        test_metrics: list[Metric] | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func
        self.val_metrics = self._build_metrics(val_metrics, "val_")
        self.test_metrics = self._build_metrics(test_metrics, "test_")

    def forward(self, batch: Batch) -> TensorBTSC:
        return self.decoder(self.encoder(batch))

    def forward_with_latent(self, batch: Batch) -> tuple[TensorBTSC, TensorBNC]:
        encoded = self.encode(batch)
        decoded = self.decode(encoded)
        return decoded, encoded

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        if self.loss_func is None:
            msg = "Loss function not defined for EncoderDecoder model."
            raise ValueError(msg)
        x = self(batch)
        output = self.decoder(x)
        loss = self.loss_func(output, batch.output_fields)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        if self.loss_func is None:
            msg = "Loss function not defined for EncoderDecoder model."
            raise ValueError(msg)
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.val_metrics is not None:
            self.val_metrics.update(y_pred, y_true)
            self.log_dict(
                self.val_metrics,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch.input_fields.shape[0],
            )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        if self.loss_func is None:
            msg = "Loss function not defined for EncoderDecoder model."
            raise ValueError(msg)
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "test_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.test_metrics is not None:
            self.test_metrics.update(y_pred, y_true)
            self.log_dict(
                self.test_metrics,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch.input_fields.shape[0],
            )
        return loss

    def predict_step(self, batch: Batch, batch_idx: int) -> TensorBTSC:  # noqa: ARG002
        return self(batch)

    def encode(self, batch: Batch) -> TensorBNC:
        return self.encoder.encode(batch)

    def decode(self, z: TensorBNC) -> TensorBTSC:
        return self.decoder.decode(z)

    def configure_optimizers(self):
        """Configure optimizers for training."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


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
