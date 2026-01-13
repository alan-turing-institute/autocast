"""Temporal UNet backbone with flexible temporal processing methods."""

from collections.abc import Mapping, Sequence

from azula.nn.unet import UNet
from torch import nn

from autocast.nn.base import TemporalBackboneBase


class TemporalUNetBackbone(TemporalBackboneBase):
    """Azula UNet with flexible temporal encoding options.

    Wraps the Azula UNet to handle temporal sequences in the format (B, T, W, H, C)
    for diffusion models or temporal forecasting. Supports multiple temporal
    processing methods that can be selected via the `temporal_method` parameter.
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
        hid_channels: Sequence[int] = (32, 64, 128),
        hid_blocks: Sequence[int] = (2, 2, 2),
        spatial: int = 2,
        kernel_size: int = 3,
        periodic: bool = False,
        temporal_method: str = "none",
        temporal_attention_heads: int = 8,
        temporal_attention_hidden_dim: int = 64,
        # TCN parameters
        tcn_kernel_size: int = 3,
        tcn_num_layers: int = 2,
        # UNet parameters
        stride: int | Sequence[int] = 2,
        dropout: float = 0.0,
        ffn_factor: int = 1,
        identity_init: bool = False,
        attention_heads: Mapping[int, int] | None = None,
        checkpointing: bool = False,
    ):
        """Initialize Temporal UNet Backbone.

        Args:
            in_channels: Number of input channels per timestep
            out_channels: Number of output channels per timestep
            cond_channels: Number of conditioning channels per timestep
            n_steps_output: Number of output timesteps to predict
            n_steps_input: Number of input timesteps for conditioning
            mod_features: Dimension for time embedding (diffusion timestep)
            global_cond_features: Dimension for optional conditioning/modulation
            hid_channels: Tuple of hidden channels for UNet levels
            hid_blocks: Tuple of number of blocks per UNet level
            spatial: Spatial dimensionality (2 for 2D)
            kernel_size: Kernel size for UNet convolutions
            periodic: Whether to use periodic boundary conditions
            temporal_method: Method for temporal processing. Options:
                - "attention": Multi-head self-attention over time
                - "tcn": Temporal convolutional network
                - "none": No temporal processing (identity)
            temporal_attention_heads: Number of heads for attention methods
            temporal_attention_hidden_dim: Hidden dimension for attention methods
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers
            stride: Stride for UNet downsampling/upsampling
            dropout: Dropout rate in UNet blocks
            ffn_factor: Feedforward network expansion factor in UNet blocks
            identity_init: Whether to use identity initialization in UNet blocks
            attention_heads: Mapping of UNet levels to number of attention heads
            checkpointing: Whether to use gradient checkpointing in UNet
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

        # Build UNet backbone
        self.unet = self._build_backbone(
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            stride=stride,
            spatial=spatial,
            periodic=periodic,
            dropout=dropout,
            ffn_factor=ffn_factor,
            identity_init=identity_init,
            attention_heads=attention_heads or {},
            checkpointing=checkpointing,
        )

    @property
    def backbone(self) -> nn.Module:
        """Return the UNet backbone."""
        return self.unet

    def _build_backbone(self, **kwargs) -> nn.Module:
        """Build the UNet backbone.

        Args:
            **kwargs: Keyword arguments including:
                - hid_channels: Tuple of hidden channels for UNet levels
                - hid_blocks: Tuple of number of blocks per UNet level
                - spatial: Spatial dimensionality (2 for 2D)
                - periodic: Whether to use periodic boundary conditions

        Returns
        -------
            UNet module
        """
        hid_channels = kwargs.pop("hid_channels")
        hid_blocks = kwargs.pop("hid_blocks")
        spatial = kwargs.pop("spatial")
        periodic = kwargs.pop("periodic")
        kernel_size = kwargs.pop("kernel_size", 3)
        stride = kwargs.pop("stride", 2)
        dropout = kwargs.pop("dropout", 0.0)
        ffn_factor = kwargs.pop("ffn_factor", 4)
        identity_init = kwargs.pop("identity_init", False)
        attention_heads = kwargs.pop("attention_heads", {})
        checkpointing = kwargs.pop("checkpointing", False)

        return UNet(
            in_channels=self.in_channels * self.n_steps_output,
            out_channels=self.out_channels * self.n_steps_output,
            cond_channels=self.cond_channels * self.n_steps_input,
            mod_features=self.mod_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            stride=stride,
            spatial=spatial,
            periodic=periodic,
            dropout=dropout,
            ffn_factor=ffn_factor,
            identity_init=identity_init,
            attention_heads=attention_heads,
            checkpointing=checkpointing,
        )
