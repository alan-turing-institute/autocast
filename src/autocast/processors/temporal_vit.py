from collections.abc import Sequence

import torch
from einops import rearrange
from torch import Tensor, nn

from autocast.nn.vit import TemporalViTBackbone
from autocast.processors.base import Processor
from autocast.types import EncodedBatch


class TemporalViTProcessor(Processor[EncodedBatch]):
    """Wrapper for the internal TemporalViTBackbone used in Diffusion Models.

    Provides building blocks for modern generative architectures (e.g. DiT).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        hidden_dim: int = 768,
        num_heads: int = 12,
        n_layers: int = 6,
        patch_size: int = 4,
        temporal_method: str = "attention",
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        if self.n_spatial_dims != 2:
            msg = "Diffusion wrapper expects 2D spatial resolution inputs (W,H)"
            raise ValueError(msg)

        self.n_noise_channels = n_noise_channels
        self.loss_func = loss_func or nn.MSELoss()
        self.model = TemporalViTBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=0,
            n_steps_output=1,
            n_steps_input=1,
            mod_features=n_noise_channels or 256,
            global_cond_channels=None,
            include_global_cond=False,
            hid_channels=hidden_dim,
            hid_blocks=n_layers,
            attention_heads=num_heads,
            patch_size=patch_size,
            spatial=2,
            temporal_method=temporal_method,
            temporal_attention_heads=num_heads,
            temporal_attention_hidden_dim=hidden_dim // num_heads,
        )

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        x_in = rearrange(x, "b c h w -> b 1 h w c").contiguous()
        y = self.model(x_in, t=x_noise, cond=None, global_cond=None)
        return rearrange(y, "b 1 h w c -> b c h w").contiguous()

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002  # noqa: ARG002
        noise = (
            torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
            if self.n_noise_channels
            else torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        )
        return self(x, noise)

    def loss(self, batch: EncodedBatch) -> Tensor:
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)
