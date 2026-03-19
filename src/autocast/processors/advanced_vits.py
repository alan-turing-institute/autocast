from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers.drop import DropPath
from torch import nn

from autocast.nn.noise.conditional_layer_norm import ConditionalLayerNorm
from autocast.processors.base import Processor
from autocast.processors.vit import PatchEmbedding, PatchUnembedding
from autocast.types import EncodedBatch, Tensor


def modulate(x: Tensor, shift: Tensor | None, scale: Tensor | None) -> Tensor:
    if shift is None or scale is None:
        return x
    shape = [-1] + [1] * (x.ndim - 2) + [x.shape[-1]]
    return x * (1 + scale.view(*shape)) + shift.view(*shape)


def apply_gate(x: Tensor, gate: Tensor | None) -> Tensor:
    if gate is None:
        return x
    shape = [-1] + [1] * (x.ndim - 2) + [x.shape[-1]]
    return x * gate.view(*shape)


class AdaLNZeroGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_noise_channels: int | None,
        num_chunks: int,
        use_ada_ln_zero: bool = True,
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.use_ada_ln_zero = use_ada_ln_zero
        if n_noise_channels is not None and n_noise_channels > 0 and use_ada_ln_zero:
            self.net = nn.Sequential(
                nn.SiLU(), nn.Linear(n_noise_channels, num_chunks * hidden_dim)
            )
            nn.init.zeros_(self.net[1].weight)
            nn.init.zeros_(self.net[1].bias)
        else:
            self.net = None

    def forward(self, x_noise: Tensor | None = None) -> tuple[Tensor | None, ...]:
        if self.net is None or x_noise is None:
            return tuple(None for _ in range(self.num_chunks))
        noise = x_noise.flatten(start_dim=1)
        params = self.net(noise)
        return params.chunk(self.num_chunks, dim=-1)


class DiffusionBackboneViTProcessor(Processor[EncodedBatch]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        hidden_dim: int = 768,
        num_heads: int = 12,
        n_layers: int = 6,
        patch_size: int = 4,
        temporal_method: str = "attention",
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        if self.n_spatial_dims != 2:
            raise ValueError(
                "Diffusion backbone wrapper expects 2D spatial resolution inputs (W,H)"
            )
        from autocast.nn.vit import TemporalViTBackbone

        self.n_noise_channels = n_noise_channels
        self.loss_func = loss_func or nn.MSELoss()
        self.model = TemporalViTBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=0,
            n_steps_output=1,
            n_steps_input=1,
            mod_features=n_noise_channels or 256,
            global_cond_channels=None,
            include_global_cond=False,
            hid_channels=hidden_dim,
            hid_blocks=n_layers,
            attention_heads=num_heads,
            patch_size=patch_size,
            spatial=2,
            temporal_method=temporal_method,
            temporal_attention_heads=num_heads,
            temporal_attention_hidden_dim=hidden_dim // num_heads,
        )

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        x_in = x.unsqueeze(1).permute(0, 1, 3, 4, 2).contiguous()
        y = self.model(x_in, t=x_noise, cond=None, global_cond=None)
        return y.squeeze(1).permute(0, 3, 1, 2).contiguous()

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        noise = (
            torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
            if self.n_noise_channels
            else torch.zeros(x.shape[0], device=x.device)
        )
        return self(x, noise)

    def loss(self, batch: EncodedBatch) -> Tensor:
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)


class FactorizedViTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_spatial_dims: int,
        n_noise_channels: int | None,
        drop_path: float = 0.0,
        use_ada_ln_zero: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.n_spatial_dims = n_spatial_dims
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ada_generator = AdaLNZeroGenerator(
            hidden_dim, n_noise_channels, 6, use_ada_ln_zero
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln_zero)
        self.fused_heads = [hidden_dim, hidden_dim, hidden_dim]
        self.qkv_proj = nn.Linear(hidden_dim, sum(self.fused_heads))
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln_zero)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        if n_spatial_dims == 2:
            self.head_split = "b h w (he c) -> b h w he c"
            self.spatial_permutations = [
                ("b h w he c -> b h he w c", "b h he w c -> b h w (he c)"),
                ("b h w he c -> b w he h c", "b w he h c -> b h w (he c)"),
            ]
        elif n_spatial_dims == 3:
            self.head_split = "b h w d (he c) -> b h w d he c"
            self.spatial_permutations = [
                ("b h w d he c -> b h w he d c", "b h w he d c -> b h w d (he c)"),
                ("b h w d he c -> b h d he w c", "b h d he w c -> b h w d (he c)"),
                ("b h w d he c -> b w d he h c", "b w d he h c -> b h w d (he c)"),
            ]

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_generator(x_noise)
        norm_x = modulate(self.norm1(x), shift1, scale1)
        qkv = self.qkv_proj(norm_x)
        q, k, v = qkv.split(self.fused_heads, dim=-1)
        q, k, v = (
            rearrange(t, self.head_split, he=self.num_heads).contiguous()
            for t in (q, k, v)
        )
        out = torch.zeros_like(norm_x)
        for in_perm, out_perm in self.spatial_permutations:
            q1, k1, v1 = (rearrange(t, in_perm).contiguous() for t in (q, k, v))
            ax_out = F.scaled_dot_product_attention(q1, k1, v1)
            ax_out = rearrange(ax_out, out_perm).contiguous()
            out = out + ax_out
        out = apply_gate(self.output_proj(out), gate1)
        x = x + self.drop_path(out)
        mlp_out = apply_gate(self.mlp(modulate(self.norm2(x), shift2, scale2)), gate2)
        return x + self.drop_path(mlp_out)


class FactorizedViTProcessor(Processor[EncodedBatch]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        hidden_dim: int = 768,
        num_heads: int = 12,
        n_layers: int = 8,
        drop_path: float = 0.0,
        groups: int = 12,
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
        patch_size: int | None = None,
        use_ada_ln_zero: bool = True,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.n_noise_channels = n_noise_channels
        self.loss_func = loss_func or nn.MSELoss()
        self.embed = PatchEmbedding(
            in_channels, hidden_dim, groups, self.n_spatial_dims, patch_size
        )
        dp_rates = torch.linspace(0, drop_path, n_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                FactorizedViTBlock(
                    hidden_dim,
                    num_heads,
                    self.n_spatial_dims,
                    n_noise_channels,
                    drop_path=dp_rates[i],
                    use_ada_ln_zero=use_ada_ln_zero,
                )
                for i in range(n_layers)
            ]
        )
        self.debed = PatchUnembedding(
            out_channels, hidden_dim, groups, self.n_spatial_dims, patch_size
        )
        self.embed_reshapes = (
            ["b h w c -> b c h w", "b c h w -> b h w c"]
            if self.n_spatial_dims == 2
            else ["b h w d c -> b c h w d", "b c h w d -> b h w d c"]
        )

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        x = rearrange(
            self.embed(rearrange(x, self.embed_reshapes[0])), self.embed_reshapes[1]
        )
        for blk in self.blocks:
            x = blk(x, x_noise=x_noise)
        return rearrange(
            self.debed(rearrange(x, self.embed_reshapes[0])), self.embed_reshapes[1]
        )

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        noise = (
            torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
            if self.n_noise_channels
            else None
        )
        if self.n_spatial_dims == 2:
            return (
                self(x.permute(0, 2, 3, 1).contiguous(), noise)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        return (
            self(x.permute(0, 2, 3, 4, 1).contiguous(), noise)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    def loss(self, batch: EncodedBatch) -> Tensor:
        return self.loss_func(
            self.map(batch.encoded_inputs, batch.global_cond),
            batch.encoded_output_fields,
        )


class FullAttentionViTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_spatial_dims: int,
        n_noise_channels: int | None,
        drop_path: float = 0.0,
        use_ada_ln_zero: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.n_spatial_dims = n_spatial_dims
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ada_generator = AdaLNZeroGenerator(
            hidden_dim, n_noise_channels, 6, use_ada_ln_zero
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln_zero)
        self.fused_heads = [hidden_dim, hidden_dim, hidden_dim]
        self.qkv_proj = nn.Linear(hidden_dim, sum(self.fused_heads))
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln_zero)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        if n_spatial_dims == 2:
            self.head_split, self.head_out = (
                "b h w (he c) -> b (h w) he c",
                "b (h w) he c -> b h w (he c)",
            )
        elif n_spatial_dims == 3:
            self.head_split, self.head_out = (
                "b h w d (he c) -> b (h w d) he c",
                "b (h w d) he c -> b h w d (he c)",
            )
        else:
            raise ValueError(f"Unsupported n_spatial_dims={n_spatial_dims}")

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        shape_info = {"h": x.shape[1], "w": x.shape[2]}
        if self.n_spatial_dims == 3:
            shape_info["d"] = x.shape[3]
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_generator(x_noise)
        qkv = self.qkv_proj(modulate(self.norm1(x), shift1, scale1))
        q, k, v = qkv.split(self.fused_heads, dim=-1)
        q, k, v = (
            rearrange(t, self.head_split, he=self.num_heads).transpose(1, 2)
            for t in (q, k, v)
        )
        out = rearrange(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2),
            self.head_out,
            **shape_info,
        )
        x = x + self.drop_path(apply_gate(self.output_proj(out), gate1))
        return x + self.drop_path(
            apply_gate(self.mlp(modulate(self.norm2(x), shift2, scale2)), gate2)
        )


class FullAttentionViTProcessor(Processor[EncodedBatch]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        hidden_dim: int = 768,
        num_heads: int = 12,
        n_layers: int = 8,
        drop_path: float = 0.0,
        groups: int = 12,
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
        patch_size: int | None = None,
        use_ada_ln_zero: bool = True,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.n_noise_channels = n_noise_channels
        self.loss_func = loss_func or nn.MSELoss()
        self.embed = PatchEmbedding(
            in_channels, hidden_dim, groups, self.n_spatial_dims, patch_size
        )
        dp_rates = torch.linspace(0, drop_path, n_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                FullAttentionViTBlock(
                    hidden_dim,
                    num_heads,
                    self.n_spatial_dims,
                    n_noise_channels,
                    drop_path=dp_rates[i],
                    use_ada_ln_zero=use_ada_ln_zero,
                )
                for i in range(n_layers)
            ]
        )
        self.debed = PatchUnembedding(
            out_channels, hidden_dim, groups, self.n_spatial_dims, patch_size
        )
        self.embed_reshapes = (
            ["b h w c -> b c h w", "b c h w -> b h w c"]
            if self.n_spatial_dims == 2
            else ["b h w d c -> b c h w d", "b c h w d -> b h w d c"]
        )

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        x = rearrange(
            self.embed(rearrange(x, self.embed_reshapes[0])), self.embed_reshapes[1]
        )
        for blk in self.blocks:
            x = blk(x, x_noise=x_noise)
        return rearrange(
            self.debed(rearrange(x, self.embed_reshapes[0])), self.embed_reshapes[1]
        )

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        noise = (
            torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
            if self.n_noise_channels
            else None
        )
        if self.n_spatial_dims == 2:
            return (
                self(x.permute(0, 2, 3, 1).contiguous(), noise)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        return (
            self(x.permute(0, 2, 3, 4, 1).contiguous(), noise)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    def loss(self, batch: EncodedBatch) -> Tensor:
        return self.loss_func(
            self.map(batch.encoded_inputs, batch.global_cond),
            batch.encoded_output_fields,
        )


class Swin3DViTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_spatial_dims: int,
        n_noise_channels: int | None,
        window_size: Sequence[int],
        drop_path: float = 0.0,
        use_ada_ln_zero: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.n_spatial_dims = n_spatial_dims
        self.window_size = window_size
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ada_generator = AdaLNZeroGenerator(
            hidden_dim, n_noise_channels, 6, use_ada_ln_zero
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln_zero)
        self.fused_heads = [hidden_dim, hidden_dim, hidden_dim]
        self.qkv_proj = nn.Linear(hidden_dim, sum(self.fused_heads))
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln_zero)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_generator(x_noise)
        qkv = self.qkv_proj(modulate(self.norm1(x), shift1, scale1))
        q, k, v = qkv.split(self.fused_heads, dim=-1)
        B, H, W = (*q.shape[:3],) if self.n_spatial_dims == 2 else (*q.shape[:3],)
        D = q.shape[3] if self.n_spatial_dims == 3 else 1
        wh, ww = self.window_size[0], self.window_size[1]
        wd = self.window_size[2] if len(self.window_size) > 2 else 1

        if self.n_spatial_dims == 3:
            q = rearrange(
                q,
                "b (h wh) (w ww) (d wd) (he c) -> (b h w d) he (wh ww wd) c",
                wh=wh,
                ww=ww,
                wd=wd,
                he=self.num_heads,
            )
            k = rearrange(
                k,
                "b (h wh) (w ww) (d wd) (he c) -> (b h w d) he (wh ww wd) c",
                wh=wh,
                ww=ww,
                wd=wd,
                he=self.num_heads,
            )
            v = rearrange(
                v,
                "b (h wh) (w ww) (d wd) (he c) -> (b h w d) he (wh ww wd) c",
                wh=wh,
                ww=ww,
                wd=wd,
                he=self.num_heads,
            )
        else:
            q = rearrange(
                q,
                "b (h wh) (w ww) (he c) -> (b h w) he (wh ww) c",
                wh=wh,
                ww=ww,
                he=self.num_heads,
            )
            k = rearrange(
                k,
                "b (h wh) (w ww) (he c) -> (b h w) he (wh ww) c",
                wh=wh,
                ww=ww,
                he=self.num_heads,
            )
            v = rearrange(
                v,
                "b (h wh) (w ww) (he c) -> (b h w) he (wh ww) c",
                wh=wh,
                ww=ww,
                he=self.num_heads,
            )

        out = F.scaled_dot_product_attention(q, k, v)

        if self.n_spatial_dims == 3:
            out = rearrange(
                out,
                "(b h w d) he (wh ww wd) c -> b (h wh) (w ww) (d wd) (he c)",
                b=B,
                h=H // wh,
                w=W // ww,
                d=D // wd,
                wh=wh,
                ww=ww,
                wd=wd,
            )
        else:
            out = rearrange(
                out,
                "(b h w) he (wh ww) c -> b (h wh) (w ww) (he c)",
                b=B,
                h=H // wh,
                w=W // ww,
                wh=wh,
                ww=ww,
            )

        x = x + self.drop_path(apply_gate(self.output_proj(out), gate1))
        return x + self.drop_path(
            apply_gate(self.mlp(modulate(self.norm2(x), shift2, scale2)), gate2)
        )


class Swin3DViTProcessor(Processor[EncodedBatch]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        window_size: Sequence[int] = (4, 4, 4),
        hidden_dim: int = 768,
        num_heads: int = 12,
        n_layers: int = 8,
        drop_path: float = 0.0,
        groups: int = 12,
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
        patch_size: int | None = None,
        use_ada_ln_zero: bool = True,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.n_noise_channels = n_noise_channels
        self.loss_func = loss_func or nn.MSELoss()
        self.embed = PatchEmbedding(
            in_channels, hidden_dim, groups, self.n_spatial_dims, patch_size
        )
        dp_rates = torch.linspace(0, drop_path, n_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                Swin3DViTBlock(
                    hidden_dim,
                    num_heads,
                    self.n_spatial_dims,
                    n_noise_channels,
                    window_size,
                    drop_path=dp_rates[i],
                    use_ada_ln_zero=use_ada_ln_zero,
                )
                for i in range(n_layers)
            ]
        )
        self.debed = PatchUnembedding(
            out_channels, hidden_dim, groups, self.n_spatial_dims, patch_size
        )
        self.embed_reshapes = (
            ["b h w c -> b c h w", "b c h w -> b h w c"]
            if self.n_spatial_dims == 2
            else ["b h w d c -> b c h w d", "b c h w d -> b h w d c"]
        )

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        x = rearrange(
            self.embed(rearrange(x, self.embed_reshapes[0])), self.embed_reshapes[1]
        )
        for blk in self.blocks:
            x = blk(x, x_noise=x_noise)
        return rearrange(
            self.debed(rearrange(x, self.embed_reshapes[0])), self.embed_reshapes[1]
        )

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        noise = (
            torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
            if self.n_noise_channels
            else None
        )
        if self.n_spatial_dims == 2:
            return (
                self(x.permute(0, 2, 3, 1).contiguous(), noise)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        return (
            self(x.permute(0, 2, 3, 4, 1).contiguous(), noise)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    def loss(self, batch: EncodedBatch) -> Tensor:
        return self.loss_func(
            self.map(batch.encoded_inputs, batch.global_cond),
            batch.encoded_output_fields,
        )
