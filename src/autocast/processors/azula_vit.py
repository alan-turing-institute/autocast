from collections.abc import Sequence

import torch
from einops import rearrange
from torch import Tensor, nn

from autocast.nn.vit import TemporalViTBackbone
from autocast.processors.base import Processor
from autocast.types import EncodedBatch


class AzulaViTProcessor(Processor[EncodedBatch]):
    """Wrapper for the internal TemporalViTBackbone used in Diffusion Models.

    Provides building blocks for modern generative architectures (e.g. DiT).

    Shape convention:
    - Public processor boundary: channel-first, (B, C, H, W)
    - Internal backbone input/output: channels-last with time, (B, T, H, W, C)
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
        n_noise_input_channels: int | None = None,
        global_cond_channels: int | None = None,
        include_global_cond: bool = False,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        if self.n_spatial_dims != 2:
            msg = "Diffusion wrapper expects 2D spatial resolution inputs (H, W)"
            raise ValueError(msg)

        self.n_noise_channels = n_noise_channels
        self.n_noise_input_channels = n_noise_input_channels or n_noise_channels
        self.global_cond_channels = global_cond_channels
        self.include_global_cond = include_global_cond

        if self.n_noise_channels is None and n_noise_input_channels is not None:
            msg = (
                "n_noise_input_channels requires n_noise_channels to be set "
                "for modulation."
            )
            raise ValueError(msg)

        self.modulation_proj = None
        if (
            self.n_noise_channels
            and self.n_noise_input_channels
            and self.n_noise_input_channels != self.n_noise_channels
        ):
            self.modulation_proj = nn.Linear(
                self.n_noise_input_channels, self.n_noise_channels
            )

        self.loss_func = loss_func or nn.MSELoss()
        self.model = TemporalViTBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=0,
            n_steps_output=1,
            n_steps_input=1,
            mod_features=n_noise_channels or 256,
            global_cond_channels=global_cond_channels,
            include_global_cond=include_global_cond,
            hid_channels=hidden_dim,
            hid_blocks=n_layers,
            attention_heads=num_heads,
            patch_size=patch_size,
            spatial=2,
            temporal_method=temporal_method,
            temporal_attention_heads=num_heads,
            temporal_attention_hidden_dim=hidden_dim // num_heads,
            checkpointing=checkpointing,
            use_precomputed_modulation=True,
        )

    def forward(
        self,
        x: Tensor,
        x_noise: Tensor | None = None,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        """Run TemporalViT with channel-first or channels-last-with-time inputs.

        Accepts both shapes so the same processor works in ambient mode (with
        encoders like ``PermuteConcat`` that fold T into C) and in latent mode
        (cached latents that keep an explicit T dim):

        Args:
            x: Input tensor with shape (B, C, H, W) or (B, T, H, W, C).
            x_noise: Optional noise/modulation tensor.
            global_cond: Optional global conditioning tensor with shape
                (B, C_global). Used only when include_global_cond=True.

        Returns
        -------
            Output tensor with the same rank as ``x``: (B, C, H, W) if ``x`` was
            4D, (B, T, H, W, C) otherwise.
        """
        if x_noise is not None and self.modulation_proj is not None:
            x_noise = self.modulation_proj(x_noise)

        if (
            self.n_noise_channels
            and x_noise is not None
            and x_noise.shape[-1] != self.n_noise_channels
        ):
            msg = (
                f"Expected x_noise with last dim {self.n_noise_channels}, "
                f"got {x_noise.shape[-1]}."
            )
            raise ValueError(msg)

        if (
            not self.n_noise_channels
            and x_noise is not None
            and x_noise.shape[-1] != self.model.mod_features
        ):
            msg = (
                f"Expected x_noise with last dim {self.model.mod_features}, "
                f"got {x_noise.shape[-1]}."
            )
            raise ValueError(msg)

        model_global_cond = None
        if self.include_global_cond:
            if global_cond is None:
                msg = "global_cond must be provided when include_global_cond=True."
                raise ValueError(msg)
            if global_cond.shape[-1] != self.global_cond_channels:
                msg = (
                    f"Expected global_cond with last dim "
                    f"{self.global_cond_channels}, got "
                    f"{global_cond.shape[-1]}."
                )
                raise ValueError(msg)
            model_global_cond = global_cond

        is_channel_first = x.ndim == 4
        if is_channel_first:
            x_in = rearrange(x, "b c h w -> b 1 h w c").contiguous()
        elif x.ndim == 5:
            x_in = x.contiguous()
        else:
            msg = (
                f"Expected x with 4 dims (B, C, H, W) or 5 dims (B, T, H, W, C), "
                f"got shape {tuple(x.shape)}."
            )
            raise ValueError(msg)

        y = self.model(x_in, t=x_noise, cond=None, global_cond=model_global_cond)

        if is_channel_first:
            return rearrange(y, "b 1 h w c -> b c h w").contiguous()
        return y.contiguous()

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        noise_channels = self.n_noise_input_channels or self.n_noise_channels
        if noise_channels is None:
            noise_channels = self.model.mod_features

        if self.n_noise_channels:
            noise = torch.randn(
                x.shape[0], noise_channels, dtype=x.dtype, device=x.device
            )
        else:
            noise = torch.zeros(
                x.shape[0], noise_channels, dtype=x.dtype, device=x.device
            )
        return self(x, noise, global_cond=global_cond)

    def loss(self, batch: EncodedBatch) -> Tensor:
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)
