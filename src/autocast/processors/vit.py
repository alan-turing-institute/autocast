import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from the_well.benchmark.models.common import BaseModel
from timm.layers.drop import DropPath
from torch import nn

from autocast.nn.conditional_layer_norm import ConditionalLayerNorm
from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


def _largest_divisor_leq(n: int, max_divisor: int) -> int:
    d = min(n, max_divisor)
    while d > 1 and n % d != 0:
        d -= 1
    return max(d, 1)


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int,
        groups: int = 12,
        n_spatial_dims: int = 2,
    ):
        super().__init__()
        if n_spatial_dims == 1:
            conv = nn.Conv1d
        elif n_spatial_dims == 2:
            conv = nn.Conv2d
        elif n_spatial_dims == 3:
            conv = nn.Conv3d
        else:
            raise ValueError(f"Unsupported n_spatial_dims={n_spatial_dims}")

        self.n_spatial_dims = n_spatial_dims

        # Make GroupNorm robust for small hidden_dim
        g_quarter = _largest_divisor_leq(hidden_dim // 4, groups)
        g_full = _largest_divisor_leq(hidden_dim, groups)

        self.in_projs = nn.Sequential(
            conv(dim_in, hidden_dim // 4, kernel_size=4, stride=4, bias=False),
            nn.GroupNorm(g_quarter, hidden_dim // 4, affine=True),
            nn.GELU(),
            conv(hidden_dim // 4, hidden_dim // 4, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(g_quarter, hidden_dim // 4, affine=True),
            nn.GELU(),
            conv(hidden_dim // 4, hidden_dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(g_full, hidden_dim, affine=True),
        )

    def forward(self, x):
        return self.in_projs(x)


class PatchUnembedding(nn.Module):
    """Patch to Image Unembedding."""

    def __init__(
        self,
        dim_out: int,
        hidden_dim: int = 768,
        groups: int = 12,
        n_spatial_dims: int = 2,
    ):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        if n_spatial_dims == 1:
            conv = nn.ConvTranspose1d
        elif n_spatial_dims == 2:
            conv = nn.ConvTranspose2d
        elif n_spatial_dims == 3:
            conv = nn.ConvTranspose3d
        else:
            raise ValueError(f"Unsupported n_spatial_dims={n_spatial_dims}")

        g_quarter = _largest_divisor_leq(hidden_dim // 4, groups)

        self.out_proj = nn.Sequential(
            conv(hidden_dim, hidden_dim // 4, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(g_quarter, hidden_dim // 4, affine=True),
            nn.GELU(),
            conv(hidden_dim // 4, hidden_dim // 4, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(g_quarter, hidden_dim // 4, affine=True),
            nn.GELU(),
            conv(hidden_dim // 4, dim_out, kernel_size=4, stride=4, bias=False),
        )

    def forward(self, x):
        return self.out_proj(x)


class AxialAttentionBlock(nn.Module):
    """Axial attention block for multi-dimensional feature processing.

    This module performs scaled dot-product attention over spatial axes,
    enabling efficient attention computation for multi-dimensional inputs.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        n_spatial_dims: int = 2,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        n_noise_channels: int | None = None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.n_spatial_dims = n_spatial_dims

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(hidden_dim), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm = ConditionalLayerNorm(
            hidden_dim, n_noise_channels, elementwise_affine=False
        )

        self.fused_heads = [hidden_dim, hidden_dim, hidden_dim, 4 * hidden_dim]
        self.fused_projection = nn.Linear(hidden_dim, sum(self.fused_heads))

        head_dim = hidden_dim // num_heads
        self.qnorm = ConditionalLayerNorm(
            head_dim, n_noise_channels, elementwise_affine=False
        )
        self.knorm = ConditionalLayerNorm(
            head_dim, n_noise_channels, elementwise_affine=False
        )

        self.output_head = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_remaining = nn.Sequential(
            nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim)
        )

        if n_spatial_dims == 2:
            self.head_split = "b h w (he c) -> b h w he c"
            self.spatial_permutations = [
                # attend over W
                ("b h w he c -> b h he w c", "b h he w c -> b h w (he c)"),
                # attend over H
                ("b h w he c -> b w he h c", "b w he h c -> b h w (he c)"),
            ]
        elif n_spatial_dims == 3:
            self.head_split = "b h w d (he c) -> b h w d he c"
            self.spatial_permutations = [
                # attend over D
                ("b h w d he c -> b h w he d c", "b h w he d c -> b h w d (he c)"),
                # attend over W
                ("b h w d he c -> b h d he w c", "b h d he w c -> b h w d (he c)"),
                # attend over H
                ("b h w d he c -> b w d he h c", "b w d he h c -> b h w d (he c)"),
            ]
        else:
            raise ValueError(f"Unsupported n_spatial_dims={n_spatial_dims}")

    def forward(self, x, x_noise):
        residual = x
        x = self.norm(x, x_noise=x_noise)

        q, k, v, ff = self.fused_projection(x).split(self.fused_heads, dim=-1)

        # base layout + make contiguous
        q, k, v = (
            rearrange(t, self.head_split, he=self.num_heads).contiguous()
            for t in (q, k, v)
        )
        q, k = self.qnorm(q, x_noise=x_noise), self.knorm(k, x_noise=x_noise)

        out = torch.zeros_like(x)

        for in_perm, out_perm in self.spatial_permutations:
            q1, k1, v1 = (rearrange(t, in_perm).contiguous() for t in (q, k, v))
            ax_out = F.scaled_dot_product_attention(q1, k1, v1)
            ax_out = rearrange(ax_out, out_perm).contiguous()
            out = out + ax_out

        x = self.output_head(out) + self.mlp_remaining(ff)

        if self.gamma is not None:
            x = self.gamma * x

        return residual + self.drop_path(x)


class AViT(BaseModel):
    """Uses axial attention to predict forward dynamics.

    This simplified version just stacks time in channels.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        hidden_dim: int = 768,
        num_heads: int = 12,
        processor_blocks: int = 8,
        drop_path: float = 0.0,
        groups: int = 12,
        n_noise_channels: int | None = None,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)

        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)

        self.patch_size = 16
        for k in self.spatial_resolution:
            if k % self.patch_size != 0:
                raise ValueError(
                    f"spatial_resolution {self.spatial_resolution} must be "
                    f"divisible by {self.patch_size}"
                )

        pe_size = (
            *tuple(int(k / self.patch_size) for k in self.spatial_resolution),
            hidden_dim,
        )
        self.absolute_pe = nn.Parameter(0.02 * torch.randn(*pe_size))

        self.embed = PatchEmbedding(
            dim_in=dim_in,
            hidden_dim=hidden_dim,
            groups=groups,
            n_spatial_dims=self.n_spatial_dims,
        )

        self.blocks = nn.ModuleList(
            [
                AxialAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    n_spatial_dims=self.n_spatial_dims,
                    drop_path=float(self.dp[i]),
                    n_noise_channels=n_noise_channels,
                )
                for i in range(processor_blocks)
            ]
        )

        self.debed = PatchUnembedding(
            hidden_dim=hidden_dim,
            dim_out=dim_out,
            groups=groups,
            n_spatial_dims=self.n_spatial_dims,
        )

        if self.n_spatial_dims == 2:
            self.embed_reshapes = ["b h w c -> b c h w", "b c h w -> b h w c"]
        elif self.n_spatial_dims == 3:
            self.embed_reshapes = ["b h w d c -> b c h w d", "b c h w d -> b h w d c"]
        else:
            raise ValueError(f"Unsupported n_spatial_dims={self.n_spatial_dims}")

    def forward(self, x, x_noise: Tensor | None = None) -> Tensor:
        x = rearrange(x, self.embed_reshapes[0])
        x = self.embed(x)
        x = rearrange(x, self.embed_reshapes[1])

        x = x + self.absolute_pe
        for blk in self.blocks:
            x = blk(x, x_noise=x_noise)

        x = rearrange(x, self.embed_reshapes[0])
        x = self.debed(x)
        return rearrange(x, self.embed_reshapes[1])


class AViTProcessor(Processor[EncodedBatch]):
    """Vision Transformer Module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: tuple[int, ...],
        hidden_dim: int = 64,
        num_heads: int = 4,
        n_layers: int = 4,
        drop_path: float = 0.0,
        groups: int = 8,
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)

        self.model = AViT(
            dim_in=in_channels,
            dim_out=out_channels,
            n_spatial_dims=self.n_spatial_dims,
            spatial_resolution=spatial_resolution,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            processor_blocks=n_layers,
            drop_path=drop_path,
            groups=groups,
            n_noise_channels=n_noise_channels,
        )

        self.loss_func = loss_func or nn.MSELoss()
        self.n_noise_channels = n_noise_channels
        # self.learning_rate = learning_rate

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        if self.n_spatial_dims == 2:
            x = x.permute(0, 2, 3, 1).contiguous()  # (B,W,H,C)
            y = self.model(x, x_noise)  # (B,W,H,Cout)
            return y.permute(0, 3, 1, 2).contiguous()

        if self.n_spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B,W,H,D,C)
            y = self.model(x, x_noise)  # (B,W,H,D,Cout)
            return y.permute(0, 4, 1, 2, 3).contiguous()

        raise ValueError(f"Unsupported n_spatial_dims={self.n_spatial_dims}")

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        _ = global_cond  # Unused global_cond within AViT currently
        # Generate noise if needed for generating conditional layer norm outputs
        if self.n_noise_channels is None:
            noise = None
        else:
            noise = torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
        return self(x, noise)

    def loss(self, batch: EncodedBatch) -> Tensor:
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)
