"""Temporal UNet backbone with flexible temporal processing methods."""

from collections.abc import Sequence

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
        hid_channels: Sequence[int] = (32, 64, 128),
        hid_blocks: Sequence[int] = (2, 2, 2),
        spatial: int = 2,
        periodic: bool = False,
        temporal_method: str = "none",
        num_attention_heads: int = 8,
        attention_hidden_dim: int = 64,
        # TCN parameters
        tcn_kernel_size: int = 3,
        tcn_num_layers: int = 2,
        **kwargs,
    ):
        """Initialize Temporal UNet Backbone.

        Args:
            in_channels: Number of input channels per timestep
            out_channels: Number of output channels per timestep
            cond_channels: Number of conditioning channels per timestep
            n_steps_output: Number of output timesteps to predict
            n_steps_input: Number of input timesteps for conditioning
            mod_features: Dimension for time embedding (diffusion timestep)
            hid_channels: Tuple of hidden channels for UNet levels
            hid_blocks: Tuple of number of blocks per UNet level
            spatial: Spatial dimensionality (2 for 2D)
            periodic: Whether to use periodic boundary conditions
            temporal_method: Method for temporal processing. Options:
                - "attention": Multi-head self-attention over time
                - "tcn": Temporal convolutional network
                - "none": No temporal processing (identity)
            num_attention_heads: Number of heads for attention methods
            attention_hidden_dim: Hidden dimension for attention methods
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers
        """
        # Initialize base class with common parameters
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            n_steps_output=n_steps_output,
            n_steps_input=n_steps_input,
            mod_features=mod_features,
            temporal_method=temporal_method,
            num_attention_heads=num_attention_heads,
            attention_hidden_dim=attention_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
            **kwargs,
        )

        # Build UNet backbone
        self.unet = self._build_backbone(
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            spatial=spatial,
            periodic=periodic,
            **kwargs,
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
        # Extract required parameters and remove from kwargs to avoid duplicates
        hid_channels = kwargs.pop("hid_channels")
        hid_blocks = kwargs.pop("hid_blocks")
        spatial = kwargs.pop("spatial")
        periodic = kwargs.pop("periodic")

        return UNet(
            in_channels=self.in_channels * self.n_steps_output,
            out_channels=self.out_channels * self.n_steps_output,
            cond_channels=self.cond_channels * self.n_steps_input,
            mod_features=self.mod_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=3,
            stride=2,
            spatial=spatial,
            periodic=periodic,
            **kwargs,
        )
