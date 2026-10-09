from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import Enum

import torch
from einops import rearrange
from torch import Tensor, nn

from autocast.nn.vit import TemporalViTBackbone
from autocast.processors.base import Processor
from autocast.types import EncodedBatch


class AdaLNNoiseMode(str, Enum):
    """Whether AdaLN noise is shared globally or independent per patch token."""

    GLOBAL = "global"
    SPATIAL = "spatial"


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
        patch_size: int | Sequence[int] = 4,
        temporal_method: str = "attention",
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
        n_noise_input_channels: int | None = None,
        global_cond_channels: int | None = None,
        include_global_cond: bool = False,
        checkpointing: bool = False,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        dropout: float = 0.0,
        ffn_factor: int = 4,
        qk_norm: bool = True,
        rope: bool = False,
        rpb: bool = True,
        noise_mode: AdaLNNoiseMode | str = AdaLNNoiseMode.GLOBAL,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        if self.n_spatial_dims != 2:
            msg = "Diffusion wrapper expects 2D spatial resolution inputs (H, W)"
            raise ValueError(msg)

        self.noise_mode = AdaLNNoiseMode(noise_mode)
        self.patch_size = (
            (patch_size, patch_size)
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )
        if self.noise_mode == AdaLNNoiseMode.SPATIAL:
            if n_noise_channels is None or n_noise_channels <= 0:
                msg = "Spatial AdaLN requires positive n_noise_channels."
                raise ValueError(msg)
            if n_noise_input_channels is not None and n_noise_input_channels <= 0:
                msg = "Spatial AdaLN requires positive n_noise_input_channels."
                raise ValueError(msg)
            if len(self.patch_size) != 2 or any(p <= 0 for p in self.patch_size):
                msg = "Spatial AdaLN requires two positive patch_size dimensions."
                raise ValueError(msg)

        self.n_noise_channels = n_noise_channels
        self.n_noise_input_channels = n_noise_input_channels or n_noise_channels
        self.global_cond_channels = global_cond_channels
        self.include_global_cond = include_global_cond
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.dropout = dropout

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
        # Absorb input/output T into channel count so the backbone always runs
        # with a single effective time token. Forward() folds T_in into C on
        # 5D inputs and unfolds T_out from C on outputs; for 4D inputs the
        # encoder has already done the fold, so n_steps_input/output=1 and
        # this scaling is a no-op.
        self.model = TemporalViTBackbone(
            in_channels=in_channels * n_steps_input,
            out_channels=out_channels * n_steps_output,
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
            dropout=dropout,
            ffn_factor=ffn_factor,
            qk_norm=qk_norm,
            rope=rope,
            rpb=rpb,
            checkpointing=checkpointing,
            use_precomputed_modulation=True,
            spatial_modulation=self.noise_mode == AdaLNNoiseMode.SPATIAL,
        )

    def _noise_shape(self, x: Tensor) -> tuple[int, ...]:
        """Return the noise shape on the processor's actual input patch grid."""
        channels = self.n_noise_input_channels or self.model.mod_features
        if self.noise_mode == AdaLNNoiseMode.GLOBAL:
            return (x.shape[0], channels)
        if x.ndim not in (4, 5):
            msg = "Spatial AdaLN expects 4D or 5D input fields."
            raise ValueError(msg)
        height, width = x.shape[-2:] if x.ndim == 4 else x.shape[2:4]
        ph, pw = self.patch_size
        if height % ph or width % pw:
            msg = (
                f"Input grid {(height, width)} must be divisible by patch_size "
                f"{self.patch_size}."
            )
            raise ValueError(msg)
        return (x.shape[0], (height // ph) * (width // pw), channels)

    def forward(
        self,
        x: Tensor,
        x_noise: Tensor | None = None,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        """Run TemporalViT with channel-first or channels-last-with-time inputs.

        Accepts both shapes so the same processor works in ambient mode (with
        encoders like ``PermuteConcat`` that fold T into C) and in latent mode
        (cached latents that keep an explicit T dim). In latent mode, T_in is
        folded into C before the backbone and T_out is unfolded afterward, so
        the backbone itself always runs with a single effective time token.

        Args:
            x: Input tensor with shape (B, C, H, W) or
                (B, T=n_steps_input, H, W, C).
            x_noise: Noise of shape (B, D) in global mode, or (B, N, D) in
                spatial mode. D is n_noise_input_channels (defaulting to
                n_noise_channels); N is the number of input patch tokens,
                ordered with width varying fastest. Use map() to draw noise.
            global_cond: Optional global conditioning tensor with shape
                (B, C_global). Used only when include_global_cond=True.

        Returns:
            Output tensor with the same rank as ``x``: (B, C, H, W) if ``x`` was
            4D, (B, T=n_steps_output, H, W, C) otherwise.
        """
        expected_shape = self._noise_shape(x)
        if x_noise is None or tuple(x_noise.shape) != expected_shape:
            received = None if x_noise is None else tuple(x_noise.shape)
            msg = f"Expected x_noise with shape {expected_shape}, got {received}."
            raise ValueError(msg)
        if self.modulation_proj is not None:
            x_noise = self.modulation_proj(x_noise)

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
            x_in = rearrange(x, "b t h w c -> b 1 h w (t c)").contiguous()
        else:
            msg = (
                f"Expected x with 4 dims (B, C, H, W) or 5 dims (B, T, H, W, C), "
                f"got shape {tuple(x.shape)}."
            )
            raise ValueError(msg)

        y = self.model(x_in, t=x_noise, cond=None, global_cond=model_global_cond)

        if is_channel_first:
            return rearrange(y, "b 1 h w c -> b c h w").contiguous()
        return rearrange(
            y, "b 1 h w (t c) -> b t h w c", t=self.n_steps_output
        ).contiguous()

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        # One draw per member/forecast step, reused across all transformer blocks.
        noise_shape = self._noise_shape(x)
        if self.n_noise_channels:
            noise = torch.randn(noise_shape, dtype=x.dtype, device=x.device)
        else:
            noise = torch.zeros(noise_shape, dtype=x.dtype, device=x.device)
        return self(x, noise, global_cond=global_cond)

    def loss(self, batch: EncodedBatch) -> Tensor:
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)


class MCDropoutAzulaViTProcessor(AzulaViTProcessor):
    """Azula ViT using Monte Carlo dropout as its stochasticity source.

    Unlike :class:`AzulaViTProcessor`, this processor does not sample a random
    modulation vector. It passes a deterministic zero vector through the
    modulation path, preserving optional physical ``global_cond`` conditioning,
    and keeps the ViT's dropout active during inference.

    Dropout masks are sampled independently on every forward pass, including
    successive autoregressive rollout steps.
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
        dropout: float = 0.1,
        checkpointing: bool = False,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
    ):
        if n_noise_channels is not None or n_noise_input_channels is not None:
            msg = (
                "MCDropoutAzulaViTProcessor uses dropout instead of stochastic "
                "modulation; n_noise_channels and n_noise_input_channels must be None."
            )
            raise ValueError(msg)
        if not 0.0 <= dropout < 1.0:
            msg = f"dropout must be in [0, 1), got {dropout}."
            raise ValueError(msg)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=spatial_resolution,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            n_layers=n_layers,
            patch_size=patch_size,
            temporal_method=temporal_method,
            loss_func=loss_func,
            n_noise_channels=None,
            n_noise_input_channels=None,
            global_cond_channels=global_cond_channels,
            include_global_cond=include_global_cond,
            dropout=dropout,
            checkpointing=checkpointing,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
        )

    @contextmanager
    def _inference_dropout(self) -> Iterator[None]:
        """Temporarily enable only the Azula ViT dropout mechanisms."""
        stochastic_modules = [
            module for module in self.model.modules() if isinstance(module, nn.Dropout)
        ]
        training_states = [module.training for module in stochastic_modules]
        try:
            for module in stochastic_modules:
                module.train()
            yield
        finally:
            for module, training in zip(
                stochastic_modules, training_states, strict=True
            ):
                module.train(training)

    def forward(
        self,
        x: Tensor,
        x_noise: Tensor | None = None,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        """Run with dropout active in both training and inference modes."""
        if self.training:
            return super().forward(x, x_noise=x_noise, global_cond=global_cond)

        with self._inference_dropout():
            return super().forward(x, x_noise=x_noise, global_cond=global_cond)
