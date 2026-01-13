"""Temporal ViT backbone with flexible temporal processing methods."""

from azula.nn.vit import ViT
from torch import nn

from autocast.nn.base import TemporalBackboneBase


class TemporalViTBackbone(TemporalBackboneBase):
    """Azula ViT with flexible temporal encoding options.

    Wraps the Azula Vision Transformer to handle temporal sequences in the
    format (B, T, W, H, C) for diffusion models or temporal forecasting.
    Supports multiple temporal processing methods that can be selected via
    the `temporal_method` parameter.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        n_steps_output: int,
        n_steps_input: int,
        mod_features: int = 256,
        global_cond_features: int | None = None,
        hid_channels: int = 768,
        hid_blocks: int = 12,
        attention_heads: int = 12,
        patch_size: int = 4,
        spatial: int = 2,
        temporal_method: str = "none",
        temporal_attention_heads: int = 8,
        temporal_attention_hidden_dim: int = 64,
        # TCN parameters
        tcn_kernel_size: int = 3,
        tcn_num_layers: int = 2,
        # ViT parameters
        dropout: float = 0.0,
        ffn_factor: int = 4,
        checkpointing: bool = False,
    ):
        """Initialize Temporal ViT Backbone.

        Args:
            in_channels: Number of input channels per timestep
            out_channels: Number of output channels per timestep
            cond_channels: Number of conditioning channels per timestep
            n_steps_output: Number of output timesteps to predict
            n_steps_input: Number of input timesteps for conditioning
            mod_features: Dimension for time embedding (diffusion timestep)
            global_cond_features: Dimension for optional conditioning/modulation
            hid_channels: Hidden dimension for ViT transformer
            hid_blocks: Number of transformer blocks
            attention_heads: Number of attention heads in ViT
            patch_size: Size of patches for ViT
            spatial: Spatial dimensionality (2 for 2D)
            temporal_method: Method for temporal processing. Options:
                - "attention": Multi-head self-attention over time
                - "tcn": Temporal convolutional network
                - "none": No temporal processing (identity)
            temporal_attention_heads: Number of heads for attention methods
            temporal_attention_hidden_dim: Hidden dimension for attention methods
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers
            dropout: Dropout rate in ViT blocks
            ffn_factor: Feedforward network expansion factor in ViT blocks
            checkpointing: Whether to use gradient checkpointing in ViT
        """
        # Initialize base class with common parameters
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            n_steps_output=n_steps_output,
            n_steps_input=n_steps_input,
            mod_features=mod_features,
            global_cond_features=global_cond_features,
            temporal_method=temporal_method,
            temporal_attention_heads=temporal_attention_heads,
            temporal_attention_hidden_dim=temporal_attention_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
        )

        self.patch_size = patch_size

        # Build ViT backbone
        self.vit = self._build_backbone(
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            attention_heads=attention_heads,
            patch_size=patch_size,
            spatial=spatial,
            ffn_factor=ffn_factor,
            dropout=dropout,
            checkpointing=checkpointing,
        )

    @property
    def backbone(self) -> nn.Module:
        """Return the ViT backbone."""
        return self.vit

    def _build_backbone(self, **kwargs) -> nn.Module:
        """Build the ViT backbone.

        Args:
            **kwargs: Keyword arguments including:
                - hid_channels: Hidden dimension for ViT transformer
                - hid_blocks: Number of transformer blocks
                - attention_heads: Number of attention heads in ViT
                - patch_size: Size of patches for ViT
                - spatial: Spatial dimensionality (2 for 2D)
                - ffn_factor: The channel factor in the FFN.
                - dropout: The dropout rate in :math:`[0, 1]`.
                - checkpointing: Whether to use gradient checkpointing or not.

        Returns
        -------
            ViT module
        """
        return ViT(
            in_channels=self.in_channels * self.n_steps_output,
            out_channels=self.out_channels * self.n_steps_output,
            cond_channels=self.cond_channels * self.n_steps_input,
            mod_features=self.mod_features,
            hid_channels=kwargs["hid_channels"],
            hid_blocks=kwargs["hid_blocks"],
            attention_heads=kwargs["attention_heads"],
            patch_size=kwargs["patch_size"],
            spatial=kwargs["spatial"],
            ffn_factor=kwargs.get("ffn_factor", 4),
            dropout=kwargs.get("dropout", 0.0),
            checkpointing=kwargs.get("checkpointing", False),
        )
