import warnings
from collections.abc import Sequence
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from the_well.data.normalization import ZScoreNormalization
from torch import nn
from torchmetrics import Metric

from autocast.decoders import Decoder
from autocast.encoders.base import Encoder, EncoderWithCond, GenericEncoder
from autocast.metrics.utils import MetricsMixin
from autocast.models.denorm_mixin import DenormMixin
from autocast.models.multifidelity_transformer import AttentionMixer
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.types import Batch, Tensor, TensorBNC, TensorBTSC
from autocast.types.batch import ListBatch
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
        y_pred = self(batch)
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
        self.target_spatial_shape = None
        self.target_latent_dim = None
        self.transformer_dim = None
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_transformer_blocks = n_transformer_blocks

        if attention:
            self.transformer_dim = transformer_dim
            if transformer_dim is not None:
                self.attention_mixer = AttentionMixer(
                    embedding_dim=transformer_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    n_transformer_blocks=n_transformer_blocks,
                )

    def encode_batch(self, batch: ListBatch, encoded_info: dict | None = None) -> None:
        msg = "MultiEncoder leverages encode() exclusively"
        raise NotImplementedError(msg)

    def encode(self, batch: ListBatch) -> TensorBNC:  # noqa: PLR0912, PLR0915
        mask: TensorDBM | None = batch.mask

        outs = []
        # Encode each dataset
        for idx, encoder in enumerate(self.encoders):
            outs.append(encoder(batch.inner[idx]))

        # TODO: maybe the stacking case without masking (like in icenet-mp)
        # https://github.com/alan-turing-institute/icenet-mp/blob/580727ad81141a8fd90531e7777aa8a4294472bc/icenet_mp/models/encode_process_decode.py#L88
        # should be handled in a separate class
        if not self.attention:
            # stack along the channel dim
            # TODO: this assumes that all encoders maps to the same latent dimension
            stacked_outputs = torch.cat(outs, dim=-1)  # (B, T, L, sum(C_i))
            if mask is not None:
                msg = (
                    "Mask cannot be applied without using attention "
                    "as the fusion mechanism"
                )
                raise ValueError(msg)
            return stacked_outputs

        flat_outs = []
        spatial_shapes: list[tuple[int, ...]] = []
        latent_dims: list[int] = []
        for out_i in outs:
            out = out_i[0] if isinstance(out_i, tuple) else out_i
            spatial_shape_i = tuple(out.shape[2:-1])
            out_flat = out.flatten(2, -2)
            flat_outs.append(out_flat)
            spatial_shapes.append(spatial_shape_i)
            latent_dims.append(out_flat.shape[-2])

        max_latent_dim = max(latent_dims)
        if self.transformer_dim is None:
            self.transformer_dim = max_latent_dim
        elif self.transformer_dim < max_latent_dim:
            warnings.warn(
                (
                    "Provided transformer_dim "
                    f"({self.transformer_dim}) is smaller than the maximum "
                    "flattened latent dimension "
                    f"({max_latent_dim}). This may cause loss of "
                    "information."
                ),
                UserWarning,
                stacklevel=2,
            )

        if len(self.input_projs) == 0:
            for latent_dim in latent_dims:
                if latent_dim == self.transformer_dim:
                    self.input_projs.append(nn.Identity())
                else:
                    self.input_projs.append(
                        nn.Linear(
                            latent_dim,
                            self.transformer_dim,
                            device=flat_outs[0].device,
                            dtype=flat_outs[0].dtype,
                        )
                    )

            if self.transformer_dim == max_latent_dim:
                self.output_proj = nn.Identity()
            else:
                self.output_proj = nn.Linear(
                    self.transformer_dim,
                    max_latent_dim,
                    device=flat_outs[0].device,
                    dtype=flat_outs[0].dtype,
                )

            idx_max = latent_dims.index(max_latent_dim)
            self.target_spatial_shape = spatial_shapes[idx_max]
            self.target_latent_dim = max_latent_dim

        if self.attention_mixer is None:
            self.attention_mixer = AttentionMixer(
                embedding_dim=self.transformer_dim,
                n_heads=self.n_heads,
                dropout=self.dropout,
                n_transformer_blocks=self.n_transformer_blocks,
            ).to(device=flat_outs[0].device, dtype=flat_outs[0].dtype)

        transformed_outs = []
        for i, out_flat in enumerate(flat_outs):
            # Project latent axis: (B, T, L_i, C_i) -> (B, T, C_i, L_i)
            out_perm = out_flat.transpose(-2, -1)
            transformed_outs.append(self.input_projs[i](out_perm))

        # Safely concatenate matching inner sequence dimensions along the channels
        # to obtain a Tensor of shape (..., tot_n_channels, transformer_dim)
        stacked = torch.cat(transformed_outs, dim=-2)

        tot_n_channels = stacked.shape[-2]

        # AttentionMixer expects (batch, tot_n_channels, transformer_dim).
        # Flatten all dimensions except the last two over the batch dimension,
        # such that this works for both spatio-temporal data
        # (B, T, tot_n_channels, transformer_dim)
        # and not (B, tot_n_channels, transformer_dim)
        batch_dims = list(stacked.shape[:-2])

        # Apply attention using the mixer
        assert self.attention_mixer is not None, (
            "AttentionMixer must be initialized to use attention"
        )

        if mask is None:
            # Mask is None, meaning we have perfectly unmasked dense data.
            stacked_flat = stacked.flatten(0, -3)
            # We directly pass None to AttentionMixer avoiding dense zeroes
            mixed = self.attention_mixer(stacked_flat, None)
            mixed = mixed.unflatten(0, batch_dims)
        else:
            # mask is TensorDBM: shape (Dataset, Batch, Ensemble)
            B_dim = batch_dims[0]
            extra_batch_dims = batch_dims[1:]
            M = mask.shape[2]

            # 1. Prepare Base Mask:
            # Align dataset masks with channel-level sequence tokens
            mask_bmd = mask.permute(1, 2, 0)  # (Batch, Ensemble, Dataset)
            chan_repeats = torch.tensor(
                [
                    out[0].shape[-1] if isinstance(out, tuple) else out.shape[-1]
                    for out in outs
                ],
                device=mask.device,
            )
            # Duplicate the dataset's boolean mask C_i times for each channel token
            attn_mask_base = torch.repeat_interleave(
                mask_bmd, chan_repeats, dim=-1
            )  # (Batch, Ensemble, tot_n_channels)

            # 2. Mask Broadcasting:
            # Expand the mask across any extra spatial/temporal batch dimensions
            broadcast_shape = (B_dim, *[1] * len(extra_batch_dims), M, tot_n_channels)
            attn_mask = attn_mask_base.view(broadcast_shape).expand(
                *batch_dims, M, tot_n_channels
            )

            # Flatten everything except channels into a single working Batch dimension
            attn_mask_flat = attn_mask.flatten(0, -2)  # (B*T*M, tot_n_channels)

            # 3. Prepare Data: Replicate data for every mask (M) in the ensemble
            assert self.transformer_dim is not None
            stacked_expanded = stacked.unsqueeze(-3).expand(
                *batch_dims, M, tot_n_channels, self.transformer_dim
            )
            stacked_flat = stacked_expanded.flatten(0, -3)

            # Run mixing sequence
            mixed = self.attention_mixer(stacked_flat, attn_mask_flat)
            mixed = mixed.unflatten(0, [*batch_dims, M])

            # Collapse dummy ensemble dimensionality before decoding
            # so strict 2D spatial convolution layers aren't tricked into 3D.
            if M == 1:
                mixed = mixed.squeeze(len(batch_dims))

        # Restore original latent dimension after attention.
        assert self.output_proj is not None
        mixed = self.output_proj(mixed)

        # Restore parity with expected baseline shape: (*, L, tot_n_channels).
        mixed = mixed.transpose(-2, -1)

        # Restore latent spatial structure expected by the decoder.
        assert self.target_spatial_shape is not None
        return mixed.unflatten(-2, self.target_spatial_shape)
