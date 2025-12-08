import torch
import torch.nn as nn
from azula.nn.unet import UNet
from azula.nn.embedding import SineEncoding
from einops import rearrange


import torch
import torch.nn as nn
from azula.nn.unet import UNet
from azula.nn.embedding import SineEncoding

from auto_cast.types import Tensor, TensorBTSPlusC

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

    def forward(self, x_t: TensorBTSPlusC, t: Tensor, cond: TensorBTSPlusC) -> TensorBTSPlusC:
        """
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
        
        B, T_out, W, H, C = x_t.shape
        _, T_cond, W_cond, H_cond, C_cond = cond.shape
        t_emb = self.time_embedding(t)
        x_t_cf = rearrange(x_t, "b t w h c -> b (t c) w h")
        x_cond_cf = rearrange(cond, "b t w h c -> b (t c) w h")
        
        # unet.forward(TensorBCLPlus, TensorBD, TensorBCLPlus) -> TensorBCLPlus
        output = self.unet(x = x_t_cf, mod=t_emb, cond=x_cond_cf)

        return rearrange(output, "b (t c) w h -> b t w h c", t=T_out, c=C)


class SimpleUNet(nn.Module):
    """Simple UNet for diffusion with time and conditioning."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        
        self.depth = depth
        
        # Time embedding (simple linear projection)
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )
        
        # Encoder layers
        self.encoders = nn.ModuleList()
        in_ch = in_channels * 2  # *2 for conditioning
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(self.conv_block(in_ch, out_ch))
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = self.conv_block(in_ch, bottleneck_ch)
        
        # Decoder layers
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2 ** i)
            in_ch = out_ch * 2  # For upconv
            
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(
                self.conv_block(in_ch, out_ch)
            )
        
        # Final output
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_ch, out_ch):
        """Basic conv block: Conv -> ReLU -> Conv -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy output frames (B, T_out, C, H, W)
            t: Diffusion time (B,) - we'll mostly ignore this for simplicity
            cond: Conditioning input frames (B, T_in, C, H, W)
        Returns:
            Denoised output frames (B, T_out, C, H, W)
        """
        B, T, C, H, W = x.shape
        _, T_cond, _, _, _ = cond.shape
        
        # Flatten temporal dimension
        x_flat = x.reshape(B * T, C, H, W)
        cond_flat = cond.reshape(B * T_cond, C, H, W)
        cond_flat = cond_flat.repeat_interleave(T // T_cond, dim=0)
        
        # Concatenate input with conditioning
        x_cond = torch.cat([x_flat, cond_flat], dim=1)  # (B*T, 2*C, H, W)
        
        # Note: We're ignoring the time embedding `t` for simplicity
        # In a full implementation, you'd modulate the features with it
        
        # Encoder with skip connections
        skip_connections = []
        x = x_cond
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = decoder(x)
        
        # Output
        out = self.out_conv(x)  # (B*T, C, H, W)
        
        # Reshape back
        return out.reshape(B, T, C, H, W)