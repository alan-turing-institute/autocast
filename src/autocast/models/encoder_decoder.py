from collections.abc import Sequence
from typing import Any, Literal

import lightning as L
import torch
from omegaconf import DictConfig
from the_well.data.normalization import ZScoreNormalization
from torch import nn
from torchmetrics import Metric

from autocast.data.multidataset import ListBatch
from autocast.decoders import Decoder
from autocast.encoders.base import Encoder, EncoderWithCond, GenericEncoder
from autocast.metrics.utils import MetricsMixin
from autocast.models.denorm_mixin import DenormMixin
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.types import Batch, Tensor, TensorBNC, TensorBTSC
from autocast.types.types import TensorDBM


class EncoderDecoder(DenormMixin, OptimizerMixin, L.LightningModule, MetricsMixin):
    """Encoder-Decoder Model."""

    encoder: EncoderWithCond
    decoder: Decoder
    loss_func: nn.Module | None
    optimizer_config: DictConfig | dict[str, Any] | None

    def __init__(
        self,
        encoder: EncoderWithCond,
        decoder: Decoder,
        loss_func: nn.Module | None = None,
        optimizer_config: DictConfig | dict[str, Any] | None = None,
        train_metrics: Sequence[Metric] | None = [],
        val_metrics: Sequence[Metric] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        norm: ZScoreNormalization | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func
        self.optimizer_config = optimizer_config
        self.train_metrics = self._build_metrics(train_metrics, "train_")
        self.val_metrics = self._build_metrics(val_metrics, "val_")
        self.test_metrics = self._build_metrics(test_metrics, "test_")
        self.norm = norm

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
        y_pred = self.decoder(x)
        y_true = batch.output_fields
        # y_pred = self.denormalize_tensor(y_pred)
        # y_true = self.denormalize_tensor(y_true)
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        self._update_and_log_metrics(
            self, self.train_metrics, y_pred, y_true, batch.input_fields.shape[0]
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
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        # Denormalize for metrics computation
        y_pred_denorm = self.denormalize_tensor(y_pred)
        y_true_denorm = self.denormalize_tensor(y_true)
        self._update_and_log_metrics(
            self,
            self.val_metrics,
            y_pred_denorm,
            y_true_denorm,
            batch.input_fields.shape[0],
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
            "test_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        # Denormalize for metrics computation
        y_pred_denorm = self.denormalize_tensor(y_pred)
        y_true_denorm = self.denormalize_tensor(y_true)
        self._update_and_log_metrics(
            self,
            self.test_metrics,
            y_pred_denorm,
            y_true_denorm,
            batch.input_fields.shape[0],
        )
        return loss

    def encode(self, batch: Batch) -> TensorBNC:
        return self.encoder.encode(batch)

    def decode(self, z: TensorBNC) -> TensorBTSC:
        return self.decoder.decode(z)


class VAE(EncoderDecoder):
    """Variational Autoencoder Model."""

    def forward(self, batch: Batch) -> Tensor:
        mu, log_var = self.encoder(batch)
        z = self.reparametrize(mu, log_var)
        x = self.decoder(z)
        return x

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class MultiEncoder(GenericEncoder[ListBatch, None]):

    def __init__(self, encoders: list[Encoder], attention: bool = False):

        self.encoders = encoders
        self.attention = attention

    def encode(self, batch: ListBatch) -> TensorBNC:
        mask: TensorDBM | None = batch.mask

        outs = []
        for idx, encoder in enumerate(self.encoders):
            outs.append(encoder(batch.inner[idx]))

        # TODO: maybe the stacking case without masking (like in icenet-mp)
        # https://github.com/alan-turing-institute/icenet-mp/blob/580727ad81141a8fd90531e7777aa8a4294472bc/icenet_mp/models/encode_process_decode.py#L88
        # should be handled in a separate class
        if not self.attention:
            # stack along the channel dim
            stacked_outputs = torch.cat(outs, dim=-1)
            if mask is not None:
                msg = "Mask cannot be applied without using attention as the fusion mechanism"
                raise ValueError(msg)
            return stacked_outputs

        # flatten each embedding to get [B, F] dim for each dataset
        flattened = [out.flatten(start_dim=1) for out in outs]
        # stack along dataset dim
        stacked = torch.stack(flattened, dim=1)  # [B, D, F]

        # TODO optionally apply attention -- flatten embeddings for this
