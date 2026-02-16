from collections.abc import Sequence

import torch
from einops import rearrange, repeat
from torch import nn

from autocast.processors.base import Processor
from autocast.processors.vit import AViT
from autocast.types.batch import EncodedBatch
from autocast.types.types import Tensor, TensorBC, TensorBNC


class AViTLatentProcessor(Processor[EncodedBatch]):
    """Vision Transformer Module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_cond_channels: int,
        include_global_cond: bool,
        n_steps_input: int,
        n_steps_output: int,
        spatial_resolution: Sequence[int],
        hidden_dim: int = 64,
        num_heads: int = 4,
        n_layers: int = 4,
        drop_path: float = 0.0,
        groups: int = 8,
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.global_cond_channels = global_cond_channels
        self.include_global_cond = include_global_cond
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output

        per_step_in_channels = (
            in_channels + global_cond_channels if include_global_cond else in_channels
        )

        self.model = AViT(
            dim_in=per_step_in_channels * n_steps_input,
            dim_out=out_channels * n_steps_output,
            n_spatial_dims=self.n_spatial_dims,
            spatial_resolution=spatial_resolution,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            processor_blocks=n_layers,
            drop_path=drop_path,
            groups=groups,
            n_noise_channels=n_noise_channels,
            patch_size=1,  # Always use patch size 1 as using pre-trained encoder
        )

        self.loss_func = loss_func or nn.MSELoss()
        self.n_noise_channels = n_noise_channels

    def _permute_concat(self, x: TensorBNC, global_cond: TensorBC) -> TensorBNC:
        """Combine the input tensor and global cond with permuting and concatenating.

        x: TensorBNC (B, T, S, C_latent)
        global_cond: TensorBC (B, C_cond)
        """
        _, t, *spatial, _c = x.shape
        if self.n_spatial_dims == 2:
            cond = repeat(
                global_cond, "b c -> b t w h c", t=t, w=spatial[0], h=spatial[1]
            )
        elif self.n_spatial_dims == 3:
            cond = repeat(
                global_cond,
                "b c -> b t w h d c",
                t=t,
                w=spatial[0],
                h=spatial[1],
                d=spatial[2],
            )
        else:
            raise ValueError(f"Unsupported n_spatial_dims={self.n_spatial_dims}")

        if self.include_global_cond:
            # Concat to: (B, T, S, C_latent + C_cond)
            x = torch.cat([x, cond], dim=-1)

        # Permute to (B, (T, C_latent + C_cond), S)
        x = rearrange(x, "b t ... c -> b (t c) ...")

        return x

    def _channels_last(self, output: TensorBNC) -> TensorBNC:
        return rearrange(
            output,
            "b (t c) ... -> b t ... c",
            t=self.n_steps_output,
            c=output.shape[1] // self.n_steps_output,
        )

    def forward(
        self, x: TensorBNC, global_cond: TensorBC, x_noise: Tensor | None = None
    ) -> Tensor:
        x = self._permute_concat(x, global_cond)
        y = self.model(x, x_noise)
        return self._channels_last(y)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        # Generate noise if needed for generating conditional layer norm outputs
        if self.n_noise_channels is None:
            noise = None
        else:
            noise = torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
        return self(x, global_cond, noise)

    def loss(self, batch: EncodedBatch) -> Tensor:
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)
