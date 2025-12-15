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
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        n_steps_output: int = 4,
        n_steps_input: int = 1,
        mod_features: int = 256,
        temporal_method: str = "attention",
        num_attention_heads: int = 8,
        attention_hidden_dim: int = 64,
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
            temporal_method: Method for temporal processing. Options:
                - "attention": Multi-head self-attention over time
                - "tcn": Temporal convolutional network
                - "none": No temporal processing (identity)
            num_attention_heads: Number of heads for attention methods
            attention_hidden_dim: Hidden dimension for attention methods
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

        # Time embedding for diffusion timestep
        self.time_embedding = nn.Sequential(
            SineEncoding(mod_features),
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )

        # Initialize temporal processing modules
        self.temporal_proc_input = self._create_temporal_module(
            channels=in_channels,
            temporal_method=temporal_method,
            num_attention_heads=num_attention_heads,
            attention_hidden_dim=attention_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
        )

        self.temporal_proc_cond = self._create_temporal_module(
            channels=cond_channels,
            temporal_method=temporal_method,
            num_attention_heads=num_attention_heads,
            attention_hidden_dim=attention_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
        )

    def _create_temporal_module(
        self,
        channels: int,
        temporal_method: str,
        num_attention_heads: int,
        attention_hidden_dim: int,
        tcn_kernel_size: int,
        tcn_num_layers: int,
    ) -> nn.Module:
        """Create temporal processing module based on method selection.

        Args:
            channels: Number of channels for this module
            temporal_method: Method name
            num_attention_heads: Number of heads for attention
            attention_hidden_dim: Hidden dimension for attention
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers

        Returns
        -------
            Temporal processing module
        """
        if temporal_method == "attention":
            return TemporalAttention(
                channels=channels,
                attention_heads=num_attention_heads,
                hidden_dim=attention_hidden_dim,
            )
        elif temporal_method == "tcn":
            return TemporalConvNet(
                channels=channels,
                kernel_size=tcn_kernel_size,
                num_layers=tcn_num_layers,
            )
        elif temporal_method == "none":
            return nn.Identity()
        else:
            raise ValueError(
                f"Unknown temporal_method: {temporal_method}. "
                f"Choose from: attention, tcn, none"
            )

    def apply_temporal_processing(
        self, x_t: TensorBTSC, cond: TensorBTSC
    ) -> tuple[TensorBTSC, TensorBTSC]:
        """Apply temporal processing to input and conditioning.

        Args:
            x_t: Input tensor (B, T, H, W, C)
            cond: Conditioning tensor (B, T_cond, H, W, C)

        Returns
        -------
            Tuple of (processed_input, processed_cond)
        """
        x_t_temporal = self.temporal_proc_input(x_t)
        cond_temporal = self.temporal_proc_cond(cond)
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


    def forward(self, x_t: TensorBTSC, t: Tensor, cond: TensorBTSC) -> TensorBTSC:
        """Forward pass of the temporal backbone.

        Args:
            x_t: Noisy data (B, T, H, W, C) - spatial dims before channels
            t: Diffusion time steps (B,)
            cond: Conditioning input (B, T_cond, H, W, C)

        Returns
        -------
            Denoised output (B, T, H, W, C)
        """
        _, T_out, _, _, C = x_t.shape

        # Embed diffusion timestep
        t_emb = self.time_embedding(t)

        # Apply temporal processing
        x_t_temporal, cond_temporal = self.apply_temporal_processing(x_t, cond)

        # Convert to channels-first format: (B, T, H, W, C) -> (B, T*C, H, W)
        x_t_cf = rearrange(x_t_temporal, "b t h w c -> b (t c) h w")
        x_cond_cf = rearrange(cond_temporal, "b t h w c -> b (t c) h w")

        # Backbone forward: (B, T*C, H, W) -> (B, T*out_channels, H, W)
        output = self.backbone(x=x_t_cf, mod=t_emb, cond=x_cond_cf)

        # Convert back to channels-last format: (B, T*C, H, W) -> (B, T, H, W, C)
        return rearrange(output, "b (t c) h w -> b t h w c", t=T_out, c=C)
