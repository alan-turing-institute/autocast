from collections.abc import Sequence
from typing import Any

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
from autocast.models.multifidelity_transformer import AttentionMixer
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
        train_metrics: Sequence[Metric] | None = None,
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
    """Orchestrates multiple encoders with multifidelity attention mixing."""

    def __init__(
        self,
        encoders: list[Encoder],
        attention: bool = False,
        transformer_dim: int | None = None,
        n_heads: int = 1,
        dropout: float = 0.2,
        n_transformer_blocks: int = 1,
    ):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.attention = attention
        self.attention_mixer = None
        self.input_projs = nn.ModuleList()
        self.output_proj = None

        if attention:
            # We use the maximum latent dimension from all encoders as the
            # target base latent_dim so that lower dimensionals are expanded
            # to the biggest dimension without losing expressivity.
            target_latent_dim = max(enc.latent_channels for enc in encoders)
            t_dim = (
                transformer_dim if transformer_dim is not None else target_latent_dim
            )

            # Setup input projections for EACH encoder depending on its latent_channels
            # This allows each dataset to have a different dimension before
            # being transformed to the common t_dim.
            for enc in encoders:
                l_i = enc.latent_channels
                if l_i != t_dim:
                    self.input_projs.append(nn.Linear(l_i, t_dim))
                else:
                    self.input_projs.append(nn.Identity())

            # Since AttentionMixer operates sequence-to-sequence,
            # we restore it to the target latent dimension.
            if target_latent_dim != t_dim:
                self.output_proj = nn.Linear(t_dim, target_latent_dim)

            self.attention_mixer = AttentionMixer(
                embedding_dim=t_dim,
                n_heads=n_heads,
                dropout=dropout,
                n_transformer_blocks=n_transformer_blocks,
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
                msg = (
                    "Mask cannot be applied without using attention "
                    "as the fusion mechanism"
                )
                raise ValueError(msg)
            return stacked_outputs

        transformed_outs = []
        for i, out in enumerate(outs):
            # Transform to shape (..., C_i, L_i) implicitly via transpose
            # irrespective of spatial dimensions
            out_perm = out.transpose(-2, -1)

            # Project independently to reach the common `transformer_dim`
            if len(self.input_projs) > 0:
                out_perm = self.input_projs[i](out_perm)

            transformed_outs.append(out_perm)

        # Safely concatenate matching inner sequence dimensions along the channels
        stacked = torch.cat(transformed_outs, dim=-2)

        sum_Ci = stacked.shape[-2]
        batch_dims = list(stacked.shape[:-2])

        # Apply attention using the mixer
        assert self.attention_mixer is not None, (
            "AttentionMixer must be initialized to use attention"
        )

        if mask is None:
            stacked_flat = stacked.flatten(0, -3)
            attn_mask = torch.zeros(
                stacked_flat.shape[0], sum_Ci, dtype=torch.bool, device=stacked.device
            )
            mixed = self.attention_mixer(stacked_flat, attn_mask)
            mixed = mixed.unflatten(0, batch_dims)
        else:
            # mask is TensorDBM: shape (D, B, M)
            B_dim = batch_dims[0]
            extra_batch_dims = batch_dims[1:]
            M = mask.shape[2]

            # Map mask to channels
            chan_repeats = torch.tensor(
                [out.shape[-1] for out in outs], device=mask.device
            )
            mask_bmd = mask.permute(1, 2, 0)  # (B, M, D)
            attn_mask_base = torch.repeat_interleave(
                mask_bmd, chan_repeats, dim=2
            )  # (B, M, sum_Ci)

            # Expand representations with the M dimension
            # stacked: (*batch_dims, sum_Ci, -1) -> (*batch_dims, 1, sum_Ci, -1)
            #       -> (*batch_dims, M, sum_Ci, -1)
            stacked_expanded = stacked.unsqueeze(-2).expand(*batch_dims, M, sum_Ci, -1)
            stacked_flat = stacked_expanded.flatten(0, -3)

            # Expand mask to match data batch dimensions
            # attn_mask_base is (B, M, sum_Ci). We insert 1s for any
            # intermediate temporal batch_dims
            attn_mask = attn_mask_base.view(
                B_dim, *[1 for _ in extra_batch_dims], M, sum_Ci
            )
            attn_mask = attn_mask.expand(B_dim, *extra_batch_dims, M, sum_Ci)
            attn_mask = attn_mask.flatten(
                0, -2
            )  # flatten all batch dimensions into a single sequence

            mixed = self.attention_mixer(stacked_flat, attn_mask)
            mixed = mixed.unflatten(0, [*batch_dims, M])

        # Reduce expressivity back to target_latent_dim
        if self.output_proj is not None:
            mixed = self.output_proj(mixed)

        # Restore dimension parity with the original expected baseline: (*, L, sum_Ci)
        mixed = mixed.transpose(-2, -1)

        return mixed
