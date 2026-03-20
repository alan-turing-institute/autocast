"""Base class for temporal backbone architectures."""

from abc import ABC, abstractmethod

from azula.nn.embedding import SineEncoding
from einops import rearrange
from torch import nn

from autocast.nn.temporal_modules import (
    TemporalAttention,
    TemporalConvNet,
)
from autocast.types import Tensor, TensorBTSC


class TemporalBackboneBase(nn.Module, ABC):
    """Base class for temporal backbone architectures.

    Provides common functionality for:
    - Time embedding for diffusion timesteps
    - Temporal processing method selection and initialization
    - Shared temporal processing application
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        n_steps_output: int,
        n_steps_input: int,
        global_cond_channels: int | None,
        include_global_cond: bool,
        mod_features: int = 256,
        temporal_method: str = "none",
        temporal_attention_heads: int = 8,
        temporal_attention_hidden_dim: int = 64,
        # TCN parameters
        tcn_kernel_size: int = 3,
        tcn_num_layers: int = 2,
    ):
        """Initialize Temporal Backbone Base.

        Args:
            in_channels: Number of input channels per timestep
            out_channels: Number of output channels per timestep
            cond_channels: Number of conditioning channels per timestep
            n_steps_output: Number of output timesteps to predict
            n_steps_input: Number of input timesteps for conditioning
            mod_features: Dimension for time embedding (diffusion timestep)
            global_cond_channels: Dimension for optional conditioning/modulation
            include_global_cond: Whether to include global conditioning
            temporal_method: Method for temporal processing. Options:
                - "attention": Multi-head self-attention over time
                - "tcn": Temporal convolutional network
                - "none": No temporal processing (identity)
            temporal_attention_heads: Number of heads for attention methods
            temporal_attention_hidden_dim: Hidden dimension for attention methods
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.temporal_method = temporal_method
        self.n_steps_output = n_steps_output
        self.n_steps_input = n_steps_input
        self.mod_features = mod_features

        # Validate global conditioning configuration
        if include_global_cond and (
            global_cond_channels is None or global_cond_channels <= 0
        ):
            msg = "`include_global_cond` is True but global_cond_channels <= 0"
            raise ValueError(msg)
        self.global_cond_channels = global_cond_channels
        self.include_global_cond = include_global_cond

        # Time embedding for diffusion timestep
        self.time_embedding = nn.Sequential(
            SineEncoding(mod_features),
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )

        self.global_cond_embedding = (
            nn.Sequential(
                nn.Linear(global_cond_channels, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, mod_features),
            )
            if global_cond_channels is not None
            and global_cond_channels > 0
            and include_global_cond
            else None
        )

        # Initialize temporal processing modules
        self.temporal_proc_input = self._create_temporal_module(
            channels=in_channels,
            temporal_method=temporal_method,
            temporal_attention_heads=temporal_attention_heads,
            temporal_attention_hidden_dim=temporal_attention_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
        )

        self.temporal_proc_cond = self._create_temporal_module(
            channels=cond_channels,
            temporal_method=temporal_method,
            temporal_attention_heads=temporal_attention_heads,
            temporal_attention_hidden_dim=temporal_attention_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
        )

    def _create_temporal_module(
        self,
        channels: int,
        temporal_method: str,
        temporal_attention_heads: int,
        temporal_attention_hidden_dim: int,
        tcn_kernel_size: int,
        tcn_num_layers: int,
    ) -> nn.Module:
        """Create temporal processing module based on method selection.

        Args:
            channels: Number of channels for this module
            temporal_method: Method name
            temporal_attention_heads: Number of heads for attention
            temporal_attention_hidden_dim: Hidden dimension for attention
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers

        Returns
        -------
            Temporal processing module
        """
        if temporal_method == "attention":
            return TemporalAttention(
                channels=channels,
                attention_heads=temporal_attention_heads,
                hidden_dim=temporal_attention_hidden_dim,
            )
        if temporal_method == "tcn":
            return TemporalConvNet(
                channels=channels,
                kernel_size=tcn_kernel_size,
                num_layers=tcn_num_layers,
            )
        if temporal_method == "none":
            return nn.Identity()

        raise ValueError(
            f"Unknown temporal_method: {temporal_method}. "
            f"Choose from: attention, tcn, none"
        )

    def apply_temporal_processing(
        self, x_t: TensorBTSC, cond: TensorBTSC | None
    ) -> tuple[TensorBTSC, TensorBTSC | None]:
        """Apply temporal processing to input and conditioning.

        Args:
            x_t: Input tensor (B, T, W, H, C)
            cond: Conditioning tensor (B, T_cond, W, H, C), or None

        Returns
        -------
            Tuple of (processed_input, processed_cond)
        """
        x_t_temporal = self.temporal_proc_input(x_t)
        cond_temporal = self.temporal_proc_cond(cond) if cond is not None else None
        return x_t_temporal, cond_temporal

    @abstractmethod
    def _build_backbone(self, **kwargs) -> nn.Module:
        """Build the underlying backbone architecture (UNet, ViT, etc.).

        This method should be implemented by subclasses to instantiate
        their specific backbone architecture.

        Returns
        -------
            The backbone module (e.g., UNet or ViT)
        """

    @property
    @abstractmethod
    def backbone(self) -> nn.Module:
        """Return the backbone module.

        Subclasses should define this as a property that returns
        their backbone (e.g., self.unet or self.vit).
        """

    def forward(
        self,
        x_t: TensorBTSC,
        t: Tensor,
        cond: TensorBTSC,
        global_cond: Tensor | None = None,
    ) -> TensorBTSC:
        """Forward pass of the temporal backbone.

        Args:
            x_t: Noisy data (B, T, W, H, C) - spatial dims before channels
            t: Diffusion time steps (B,)
            cond: Conditioning input (B, T_cond, W, H, C)
            global_cond: Optional global conditioning/modulation vector (B, D)

        Returns
        -------
            Denoised output (B, T, W, H, C)
        """
        _, T_out, _, _, C = x_t.shape

        # Embed diffusion timestep
        t_emb = self.time_embedding(t)

        # Combine with global conditioning embedding if provided
        if self.global_cond_embedding is not None:
            if global_cond is None:
                msg = "Model init with global_cond_channels but no global_cond provided"
                raise ValueError(msg)
            t_emb = t_emb + self.global_cond_embedding(global_cond)

        # Apply temporal processing
        x_t_temporal, cond_temporal = self.apply_temporal_processing(x_t, cond)

        # Convert to channels-first format: (B, T, W, H, C) -> (B, T*C, W, H)
        x_t_cf = rearrange(x_t_temporal, "b t w h c -> b (t c) w h")
        x_cond_cf = (
            rearrange(cond_temporal, "b t w h c -> b (t c) w h")
            if cond_temporal is not None
            else None
        )

        # Backbone forward: (B, T*C, W, H) -> (B, T*out_channels, W, H)
        output = self.backbone(x=x_t_cf, mod=t_emb, cond=x_cond_cf)

        # Convert back to channels-last format: (B, T*C, W, H) -> (B, T, W, H, C)
        return rearrange(output, "b (t c) w h -> b t w h c", t=T_out, c=C)
