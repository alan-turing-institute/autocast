from collections.abc import Sequence

from azula.nn.unet import UNet
from torch import nn

from autocast.encoders.base import Encoder
from autocast.types import Batch, Tensor


class UNetEncoder(Encoder):
    """Base encoder."""

    encoder_model: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 2,
        attention_heads: dict[int, int] | None = None,
        spatial: int = 2,
        periodic: bool = False,
        identity_init: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        attention_heads = attention_heads or {}
        self.latent_dim = out_channels
        self.input_channels = in_channels

        self.encoder_model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            stride=stride,
            attention_heads=attention_heads,
            spatial=spatial,
            periodic=periodic,
            identity_init=identity_init,
            **kwargs,
        )

    def encode(self, batch: Batch) -> Tensor:
        # TODO: implement more sophisticated encoding combining fields if needed
        x = batch.input_fields
        return self.encoder_model(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder_model(x)

    def __call__(self, batch: Batch) -> Tensor:
        return self.encode(batch)
