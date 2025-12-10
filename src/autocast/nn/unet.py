from azula.nn.embedding import SineEncoding
from azula.nn.unet import UNet
from einops import rearrange
from torch import nn

from autocast.types import Tensor, TensorBTSC


class TemporalUNetBackbone(nn.Module):
    """Azula UNet with proper time embedding."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        mod_features: int = 256,
        hid_channels: tuple = (32, 64, 128),
        hid_blocks: tuple = (2, 2, 2),
        spatial: int = 2,
        periodic: bool = False,
    ):
        super().__init__()

        # Time embedding
        self.time_embedding = nn.Sequential(
            SineEncoding(mod_features),
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )

        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            mod_features=mod_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=3,
            stride=2,
            spatial=spatial,
            periodic=periodic,
        )

    def forward(self, x_t: TensorBTSC, t: Tensor, cond: TensorBTSC) -> TensorBTSC:
        """Forward pass of the Temporal UNet.

        Args:
            x_out: Noisy data (B, T, C, H, W) - channels first from Azula
            t: Time steps (B,) # TODO: define a type for this
            cond: Conditioning input (B, T_cond, C, H, W) - channels first
        Returns:
            Denoised output (B, T, C, H, W)
        """
        _, T_out, _, _, C = x_t.shape
        t_emb = self.time_embedding(t)
        x_t_cf = rearrange(x_t, "b t w h c -> b (t c) w h")
        x_cond_cf = rearrange(cond, "b t w h c -> b (t c) w h")

        # unet.forward(TensorBCLPlus, TensorBD, TensorBCLPlus) -> TensorBCLPlus
        output = self.unet(x=x_t_cf, mod=t_emb, cond=x_cond_cf)

        return rearrange(output, "b (t c) w h -> b t w h c", t=T_out, c=C)
