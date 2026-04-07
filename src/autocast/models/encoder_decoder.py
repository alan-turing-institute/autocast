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

    def __init__(self, encoders: list[Encoder], attention: bool = False, 
                 embed_dim: int | None = None, n_heads: int = 1, dropout: float = 0.2, 
                 n_transformer_blocks: int = 1):

        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.attention = attention
        self.attention_mixer = None

        if attention:
            if embed_dim is None:
                raise ValueError("embed_dim must be provided if attention is True.")
            from autocast.models.multifidelity_transformer import AttentionMixer
            #TODO: is this the right place for this?
            self.attention_mixer = AttentionMixer(
                embedding_dim=embed_dim,
                n_heads=n_heads,
                dropout=dropout,
                n_transformer_blocks=n_transformer_blocks
            )

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

        # Apply attention using the mixer
        if mask is None:
            # Create a dummy mask of False (everything is available)
            attn_mask = torch.zeros(stacked.shape[0], stacked.shape[1], dtype=torch.bool, device=stacked.device)
            return self.attention_mixer(stacked, attn_mask)
        else:

            #TODO: double-check the following

            # mask is TensorDBM: shape (D, B, M)
            B, D, F = stacked.shape
            M = mask.shape[2]
            
            # Map representations to (B*M, D, F)
            # 1. Add M dimension: (B, 1, D, F)
            # 2. Expand to (B, M, D, F)
            # 3. Flatten B and M: (B*M, D, F)
            stacked_expanded = stacked.unsqueeze(1).expand(B, M, D, F).reshape(B * M, D, F)
            
            # Map mask to (B*M, D)
            # 1. Permute (D, B, M) -> (B, M, D)
            # 2. Flatten B and M: (B*M, D)
            attn_mask = mask.permute(1, 2, 0).reshape(B * M, D)
            
            # Run AttentionMixer, output is (B*M, F)
            mixed = self.attention_mixer(stacked_expanded, attn_mask)
            
            # Reshape back to separate Batch and Ensemble dimensions -> (B, M, F)
            # You can adjust this to (B*M, F) or (B, F, M) if your decoder expects something specific!
            return mixed.reshape(B, M, F)
