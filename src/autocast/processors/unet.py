from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from azula.nn.unet import UNet
from torch import nn
from torch.utils.checkpoint import checkpoint

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor

# Map spatial dimensions to corresponding PyTorch modules
conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}
pool_modules = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
norm_modules = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}


class UNetClassic(nn.Module):
    """Classic U-Net architecture for spatiotemporal prediction.

    Adapted from:
        Takamoto et al. 2022, PDEBENCH: An Extensive Benchmark
        for Scientific Machine Learning
        Source:
        github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py

        Via the_well repository:
        github.com/PolymathicAI/the_well/blob/master/
        the_well/benchmark/models/unet_classic/__init__.py

    If you use this implementation, please cite the original work above.

    Parameters
    ----------
    dim_in : int
        Number of input channels.
    dim_out : int
        Number of output channels.
    n_spatial_dims : int
        Number of spatial dimensions (1, 2, or 3).
    spatial_resolution : Sequence[int]
        Spatial resolution of the input data.
    init_features : int, optional
        Number of features in the first encoder block. Default is 32.
    gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing to reduce memory usage.
        Default is False.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: Sequence[int],
        init_features: int = 32,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.spatial_resolution = spatial_resolution
        self.gradient_checkpointing = gradient_checkpointing

        features = init_features

        # Encoder blocks
        self.encoder1 = self._block(dim_in, features, name="enc1")
        self.pool1 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        # Decoder blocks
        self.upconv4 = conv_transpose_modules[n_spatial_dims](
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = conv_transpose_modules[n_spatial_dims](
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = conv_transpose_modules[n_spatial_dims](
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = conv_transpose_modules[n_spatial_dims](
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        # Final output convolution
        self.conv = conv_modules[n_spatial_dims](
            in_channels=features, out_channels=dim_out, kernel_size=1
        )

    def optional_checkpointing(
        self,
        layer: Callable[..., Tensor],
        *inputs: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing:
            return cast(
                Tensor,
                checkpoint(layer, *inputs, use_reentrant=False, **kwargs),
            )
        return layer(*inputs, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the U-Net.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C_in, *spatial_dims).

        Returns
        -------
        Tensor
            Output tensor of shape (B, C_out, *spatial_dims).
        """
        # Encoder path with skip connections
        enc1 = self.optional_checkpointing(self.encoder1, x)
        enc2 = self.optional_checkpointing(self.encoder2, self.pool1(enc1))
        enc3 = self.optional_checkpointing(self.encoder3, self.pool2(enc2))
        enc4 = self.optional_checkpointing(self.encoder4, self.pool3(enc3))

        # Bottleneck
        bottleneck = self.optional_checkpointing(self.bottleneck, self.pool4(enc4))

        # Decoder path with skip connections
        dec4 = self.optional_checkpointing(self.upconv4, bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.optional_checkpointing(self.decoder4, dec4)

        dec3 = self.optional_checkpointing(self.upconv3, dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.optional_checkpointing(self.decoder3, dec3)

        dec2 = self.optional_checkpointing(self.upconv2, dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.optional_checkpointing(self.decoder2, dec2)

        dec1 = self.optional_checkpointing(self.upconv1, dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.optional_checkpointing(self.decoder1, dec1)

        return self.conv(dec1)

    def _block(self, in_channels: int, features: int, name: str) -> nn.Sequential:
        """Create a U-Net convolutional block.

        Each block consists of:
        - Conv -> BatchNorm -> Tanh
        - Conv -> BatchNorm -> Tanh

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        features : int
            Number of output channels.
        name : str
            Name prefix for the layers in this block.

        Returns
        -------
        nn.Sequential
            Sequential module containing the block layers.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        conv_modules[self.n_spatial_dims](
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        name + "norm1",
                        norm_modules[self.n_spatial_dims](num_features=features),
                    ),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        conv_modules[self.n_spatial_dims](
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        name + "norm2",
                        norm_modules[self.n_spatial_dims](num_features=features),
                    ),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


class UNetProcessor(Processor[EncodedBatch]):
    """UNet Processor for spatiotemporal prediction.

    This processor wraps the classic U-Net architecture for learning
    mappings between spatiotemporal fields.

    Adapted from:
        Takamoto et al. 2022, PDEBENCH: An Extensive Benchmark
        for Scientific Machine Learning
        Source:
        github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py

        Via the_well repository:
        github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/models/unet_classic/__init__.py

    If you use this implementation, please cite the original work above.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    spatial_resolution : Sequence[int]
        Spatial resolution of the input data (e.g., [64, 64] for 2D).
    n_spatial_dims : int, optional
        Number of spatial dimensions (1, 2, or 3). Default is 2.
    init_features : int, optional
        Number of features in the first encoder block. Default is 32.
    gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing to reduce memory usage.
        Default is False.
    loss_func : nn.Module, optional
        Loss function. Defaults to MSELoss.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        n_spatial_dims: int = 2,
        init_features: int = 32,
        gradient_checkpointing: bool = False,
        loss_func: nn.Module | None = None,
    ):
        super().__init__()

        self.model = UNetClassic(
            dim_in=in_channels,
            dim_out=out_channels,
            n_spatial_dims=n_spatial_dims,
            spatial_resolution=spatial_resolution,
            init_features=init_features,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.loss_func = loss_func or nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the UNet."""
        return self.model(x)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Map input states to output states.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, T_in, *spatial_dims).
        global_cond : Tensor | None
            Optional conditioning tensor (currently unused).

        Returns
        -------
        Tensor
            Output tensor of shape (B, T_out, *spatial_dims).
        """
        _ = global_cond  # Unused global_cond within UNet currently
        return self(x)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute loss between output and target.

        Parameters
        ----------
        batch : EncodedBatch
            Batch containing encoded inputs and output fields.

        Returns
        -------
        Tensor
            Loss value.
        """
        output = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(output, batch.encoded_output_fields)


# TODO: Look into modulation feature handling
# TODO: should this really be a backbone rather than a processor?
class AzulaUNetProcessor(Processor[EncodedBatch]):
    """UNet Processor using Azula's modern UNet architecture.

    This processor wraps the Azula UNet implementation which includes
    additional features like residual connections, flexible normalization,
    and optional modulation support.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hid_channels : Sequence[int], optional
        Hidden channel dimensions at each level.
        Default is [64, 128, 256, 512].
    hid_blocks : Sequence[int], optional
        Number of residual blocks at each level.
        Default is [2, 2, 2, 2].
    norm : str, optional
        Normalization type: 'batch', 'group', 'layer', or 'rms'.
        Default is 'group'.
    groups : int, optional
        Number of groups for GroupNorm. Default is 8.
    ffn_factor : int, optional
        Feed-forward network expansion factor. Default is 2.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    periodic : bool, optional
        Whether to use periodic boundary conditions. Default is False.
    mod_features : int, optional
        Number of modulation features for FiLM conditioning. Only used
        when UNet is part of a generative model (e.g., as a backbone for
        diffusion). For standard deterministic mapping, keep at 0.
        Default is 0 (no modulation).
    cond_channels : int, optional
        Number of spatial conditioning channels (concatenated to input).
        Default is 0.
    gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing. Default is False.
    loss_func : nn.Module, optional
        Loss function. Defaults to MSELoss.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256, 512),
        hid_blocks: Sequence[int] = (2, 2, 2, 2),
        norm: str = "group",
        groups: int = 8,
        ffn_factor: int = 2,
        dropout: float = 0.0,
        periodic: bool = False,
        mod_features: int = 0,
        cond_channels: int = 0,
        gradient_checkpointing: bool = False,
        loss_func: nn.Module | None = None,
    ):
        super().__init__()

        # instantiate Azula UNet
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            norm=norm,
            groups=groups,
            ffn_factor=ffn_factor,
            dropout=dropout,
            periodic=periodic,
            mod_features=mod_features,
            cond_channels=cond_channels,
            checkpointing=gradient_checkpointing,
        )

        # Azula saves this as self.checkpointing
        self.gradient_checkpointing = gradient_checkpointing
        self.mod_features = mod_features
        self.loss_func = loss_func or nn.MSELoss()

    def forward(self, x: Tensor, mod: Tensor | None = None) -> Tensor:
        """Forward pass through the Azula UNet.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C_in, *spatial_dims).
        mod : Tensor | None, optional
            Modulation tensor for conditional generation.

        Returns
        -------
        Tensor
            Output tensor of shape (B, C_out, *spatial_dims).
        """
        return self.model(x, mod=mod)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Map input states to output states.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C_in, *spatial_dims).
        global_cond : Tensor | None
            Optional conditioning tensor. Only used if mod_features > 0.
            When mod_features=0 (default), global_cond is ignored for
            consistency with other deterministic processors (Classic UNet,
            FNO, ViT).

        Returns
        -------
        Tensor
            Output tensor of shape (B, C_out, *spatial_dims).
        """
        # Use modulation only if explicitly configured
        mod = None

        # TODO: remove this?
        if self.mod_features > 0:
            if global_cond is None:
                # Create zero modulation tensor when features expected but not provided
                batch_size = x.shape[0]
                mod = torch.zeros(
                    batch_size,
                    self.mod_features,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                mod = global_cond

        return self(x, mod=mod)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute loss between output and target.

        Parameters
        ----------
        batch : EncodedBatch
            Batch containing encoded inputs and output fields.

        Returns
        -------
        Tensor
            Loss value.
        """
        output = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(output, batch.encoded_output_fields)
