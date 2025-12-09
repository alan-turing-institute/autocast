import torch
from azula.nn.embedding import SineEncoding
from azula.nn.unet import UNet
from einops import rearrange
from torch import nn

from auto_cast.types import Tensor, TensorBTSC


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
        # B, T, W, H, C = x_out.shape
        # _, T_cond, W_cond, H_cond , C_cond = cond.shape
        # assert W == W_cond and H == H_cond
        # print("x_out.shape", x_out.shape)
        # print("cond.shape", cond.shape)
        # # Embed time (once per batch)
        # t_emb = self.time_embedding(t)  # (B, mod_features)
        # mod_for_unet = t_emb
        # print(t_emb.shape)
        # t_emb = rearrange(t_emb, "b m -> b  1 1 1 m")
        # t_emb = t_emb.expand(B, T_cond, W, H, -1)  # (B, mod_features, H, W)

        # print("t_emb.shape", t_emb.shape)
        # # Concatenate along channel dimension
        # x_cond = torch.cat([cond, t_emb], dim=-1)  # (B, T, C+C_cond, H, W)
        # print("x_cond.shape", x_cond.shape)

        # x_cond = rearrange(x_cond, "b t w h c -> b (t c) w h")
        # print("x_cond reshaped", x_cond.shape)
        # # Process through UNet
        # out_flat = self.unet(x_cond, mod=mod_for_unet)
        # print("out",out_flat.shape)
        # # Reshape back to (B, T, C, H, W)
        # return out_flat.reshape(B, T, W, H, C)

        _, T_out, _, _, C = x_t.shape
        t_emb = self.time_embedding(t)
        x_t_cf = rearrange(x_t, "b t w h c -> b (t c) w h")
        x_cond_cf = rearrange(cond, "b t w h c -> b (t c) w h")

        # unet.forward(TensorBCLPlus, TensorBD, TensorBCLPlus) -> TensorBCLPlus
        output = self.unet(x=x_t_cf, mod=t_emb, cond=x_cond_cf)

        return rearrange(output, "b (t c) w h -> b t w h c", t=T_out, c=C)


class SimpleUNet(nn.Module):
    """Simple UNet for diffusion with time and conditioning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        mod_features: int = 128,
    ):
        super().__init__()
        self.depth = depth

        # Time embedding (sinusoidal + linear)
        self.time_embed = nn.Sequential(
            nn.Linear(mod_features, mod_features * 4),
            nn.SiLU(),
            nn.Linear(mod_features * 4, mod_features),
        )
        self.mod_features = mod_features

        # Encoder layers
        self.encoders = nn.ModuleList()
        in_ch = in_channels + cond_channels  # Concatenate input + conditioning

        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.encoders.append(self._conv_block(in_ch, out_ch))
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = base_channels * (2**depth)
        self.bottleneck = self._conv_block(in_ch, bottleneck_ch)

        # Decoder layers
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2**i)
            in_ch_up = (
                base_channels * (2 ** (i + 1)) if i < depth - 1 else bottleneck_ch
            )

            self.upconvs.append(
                nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(self._conv_block(out_ch * 2, out_ch))

        # Final output
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch, out_ch):
        """Compute basic conv block: Conv -> ReLU -> Conv -> ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def _timestep_embedding(self, t, dim):
        """Create sinusoidal timestep embeddings.

        Args:
            t: (B,) tensor of timesteps in [0, 1]
            dim: embedding dimension
        Returns:
            (B, dim) embeddings
        """
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # noqa: RET504

    def forward(self, x_t, t, cond):
        """Call model forward.

        Args:
            x_t: Noisy output frames (B, T_out, W, H, C)
            t: Diffusion timesteps (B,) in [0, 1]
            cond: Conditioning input frames (B, T_cond, W, H, C_cond)

        Returns
        -------
            Denoised output frames (B, T_out, W, H, C)
        """
        _, T_out, _, _, C = x_t.shape

        # Get time embeddings
        t_emb = self._timestep_embedding(t, self.mod_features)
        t_emb = self.time_embed(t_emb)  # (B, mod_features)

        # Convert to channels-first using einops
        x_t_cf = rearrange(x_t, "b t w h c -> b (t c) w h")
        cond_cf = rearrange(cond, "b t w h c -> b (t c) w h")

        # Concatenate input with conditioning
        x_in = torch.cat([x_t_cf, cond_cf], dim=1)

        # Encoder with skip connections
        skip_connections = []
        x = x_in

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        skip_connections = skip_connections[::-1]

        for i, (upconv, decoder) in enumerate(
            zip(self.upconvs, self.decoders, strict=True)
        ):
            x = upconv(x)
            # Match spatial dimensions if needed
            if x.shape[2:] != skip_connections[i].shape[2:]:
                x = torch.nn.functional.interpolate(
                    x,
                    size=skip_connections[i].shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = decoder(x)

        # Output
        out = self.out_conv(x)  # (B, T_out*C, W, H)

        # Convert back to channels-last using einops
        return rearrange(out, "b (t c) w h -> b t w h c", t=T_out, c=C)
