from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers.drop import DropPath
from torch import nn

from autocast.nn.vit import TemporalViTBackbone
from autocast.processors.base import Processor
from autocast.processors.vit import PatchEmbedding, PatchUnembedding
from autocast.types import EncodedBatch, Tensor


def modulate(x: Tensor, shift: Tensor | None, scale: Tensor | None) -> Tensor:
    """Modulate the input tensor with shift and scale parameters."""
    if shift is None or scale is None:
        return x
    shape = [-1] + [1] * (x.ndim - 2) + [x.shape[-1]]
    return x * (1 + scale.view(*shape)) + shift.view(*shape)


def apply_gate(x: Tensor, gate: Tensor | None) -> Tensor:
    """Apply a gating mechanism to the input tensor."""
    if gate is None:
        return x
    shape = [-1] + [1] * (x.ndim - 2) + [x.shape[-1]]
    return x * gate.view(*shape)


class AdaLNGenerator(nn.Module):
    """Generate Adaptive Layer Norm parameters from noise embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        n_noise_channels: int | None,
        num_chunks: int,
        use_ada_ln: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.use_ada_ln = use_ada_ln
        self.zero_init = zero_init
        if n_noise_channels is not None and n_noise_channels > 0 and use_ada_ln:
            self.net = nn.Sequential(
                nn.SiLU(), nn.Linear(n_noise_channels, num_chunks * hidden_dim)
            )
            if zero_init:
                nn.init.zeros_(self.net[1].weight)  # type: ignore[arg-type]
                nn.init.zeros_(self.net[1].bias)  # type: ignore[arg-type]
        else:
            self.net = None

    def forward(self, x_noise: Tensor | None = None) -> tuple[Tensor | None, ...]:
        if self.net is None or x_noise is None:
            return tuple(None for _ in range(self.num_chunks))
        noise = x_noise.flatten(start_dim=1)
        params = self.net(noise)
        return params.chunk(self.num_chunks, dim=-1)


class DiffusionBackboneViTProcessor(Processor[EncodedBatch]):
    """Wrapper for the internal TemporalViTBackbone used in Diffusion Models.

    Provides building blocks for modern generative architectures (e.g. DiT).
    """

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
            msg = "Diffusion wrapper expects 2D spatial resolution inputs (W,H)"
            raise ValueError(msg)

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
        x_in = rearrange(x, "b c h w -> b 1 h w c").contiguous()
        y = self.model(x_in, t=x_noise, cond=None, global_cond=None)
        return rearrange(y, "b 1 h w c -> b c h w").contiguous()

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002  # noqa: ARG002
        noise = (
            torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
            if self.n_noise_channels
            else torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        )
        return self(x, noise)

    def loss(self, batch: EncodedBatch) -> Tensor:
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)


class FactorizedViTBlock(nn.Module):
    """Block for Factorized ViT Processor."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_spatial_dims: int,
        n_noise_channels: int | None,
        drop_path: float = 0.0,
        use_ada_ln: bool = True,
        zero_init: bool = True,
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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ada_generator = AdaLNGenerator(
            hidden_dim, n_noise_channels, 6, use_ada_ln=use_ada_ln, zero_init=zero_init
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln)
        self.fused_heads = [hidden_dim, hidden_dim, hidden_dim]
        self.qkv_proj = nn.Linear(hidden_dim, sum(self.fused_heads))
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln)
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
    """
    ViT Processor using Factorized Spatiotemporal Attention and Ada-LN.

    References
    ----------
    - VivIT: A Video Vision Transformer (https://arxiv.org/abs/2103.15691)
    - MetNet-3: Deep Learning for Day Ahead Global Weather Forecasting (https://arxiv.org/abs/2306.06079)
    - AViT: An Efficient and Scalable Attention (Internal or Relevant Paper)
    """

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
        use_ada_ln: bool = True,
        zero_init: bool = True,
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
                    use_ada_ln=use_ada_ln,
                    zero_init=zero_init,
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

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002
        noise = (
            torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
            if self.n_noise_channels
            else None
        )
        if self.n_spatial_dims == 2:
            out = self(rearrange(x, "b c h w -> b h w c").contiguous(), noise)
            return rearrange(out, "b h w c -> b c h w").contiguous()
        out = self(rearrange(x, "b c d h w -> b d h w c").contiguous(), noise)
        return rearrange(out, "b d h w c -> b c d h w").contiguous()

    def loss(self, batch: EncodedBatch) -> Tensor:
        return self.loss_func(
            self.map(batch.encoded_inputs, batch.global_cond),
            batch.encoded_output_fields,
        )


class FullAttentionViTBlock(nn.Module):
    """Block for Full Attention ViT Processor."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_spatial_dims: int,
        n_noise_channels: int | None,
        drop_path: float = 0.0,
        use_ada_ln: bool = True,
        zero_init: bool = True,
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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ada_generator = AdaLNGenerator(
            hidden_dim, n_noise_channels, 6, use_ada_ln=use_ada_ln, zero_init=zero_init
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln)
        self.fused_heads = [hidden_dim, hidden_dim, hidden_dim]
        self.qkv_proj = nn.Linear(hidden_dim, sum(self.fused_heads))
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln)
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
    """
    ViT Processor using Full Spatiotemporal Attention and Ada-LN.

    References
    ----------
    - EarthPT: A Foundation Model for Earth Observation (https://arxiv.org/abs/2309.07207)
    - ClimaX: A foundation model for weather and climate (https://arxiv.org/abs/2301.10343)
    - Aardvark Weather (https://www.nature.com/articles/s41586-025-08897-0)
    """

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
        use_ada_ln: bool = True,
        zero_init: bool = True,
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
                    use_ada_ln=use_ada_ln,
                    zero_init=zero_init,
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

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002
        noise = (
            torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
            if self.n_noise_channels
            else None
        )
        if self.n_spatial_dims == 2:
            out = self(rearrange(x, "b c h w -> b h w c").contiguous(), noise)
            return rearrange(out, "b h w c -> b c h w").contiguous()
        out = self(rearrange(x, "b c d h w -> b d h w c").contiguous(), noise)
        return rearrange(out, "b d h w c -> b c d h w").contiguous()

    def loss(self, batch: EncodedBatch) -> Tensor:
        return self.loss_func(
            self.map(batch.encoded_inputs, batch.global_cond),
            batch.encoded_output_fields,
        )


class SwinViTBlock(nn.Module):
    """Block for Swin ViT Processor."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_spatial_dims: int,
        n_noise_channels: int | None,
        window_size: Sequence[int],
        shift: bool = False,
        drop_path: float = 0.0,
        use_ada_ln: bool = True,
        zero_init: bool = True,
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
        self.window_size = window_size
        self.shift = shift
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ada_generator = AdaLNGenerator(
            hidden_dim, n_noise_channels, 6, use_ada_ln=use_ada_ln, zero_init=zero_init
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln)
        self.fused_heads = [hidden_dim, hidden_dim, hidden_dim]
        self.qkv_proj = nn.Linear(hidden_dim, sum(self.fused_heads))
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=not use_ada_ln)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def _compute_attn_mask(self, H: int, W: int, device: torch.device) -> Tensor | None:
        """Attention mask for cyclic-shifted 2D windows."""
        if not self.shift:
            return None
        wh, ww = self.window_size[0], self.window_size[1]
        shift_h, shift_w = wh // 2, ww // 2
        img_mask = torch.zeros(1, H, W, 1, device=device)
        cnt = 0
        for h_s in (slice(0, -wh), slice(-wh, -shift_h), slice(-shift_h, None)):
            for w_s in (slice(0, -ww), slice(-ww, -shift_w), slice(-shift_w, None)):
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1
        mask_w = rearrange(
            img_mask, "1 (nh wh) (nw ww) 1 -> (nh nw) (wh ww)", wh=wh, ww=ww
        )
        diff = mask_w.unsqueeze(2) - mask_w.unsqueeze(1)
        # (num_windows, 1, wh*ww, wh*ww)
        return (
            torch.zeros_like(diff).masked_fill_(diff != 0, float("-inf")).unsqueeze(1)
        )

    def _compute_attn_mask_3d(
        self, H: int, W: int, D: int, device: torch.device
    ) -> Tensor | None:
        """Attention mask for cyclic-shifted 3D windows."""
        if not self.shift:
            return None
        wh, ww = self.window_size[0], self.window_size[1]
        wd = self.window_size[2] if len(self.window_size) > 2 else 1
        shift_h, shift_w, shift_d = wh // 2, ww // 2, wd // 2
        img_mask = torch.zeros(1, H, W, D, 1, device=device)
        cnt = 0
        for h_s in (slice(0, -wh), slice(-wh, -shift_h), slice(-shift_h, None)):
            for w_s in (slice(0, -ww), slice(-ww, -shift_w), slice(-shift_w, None)):
                for d_s in (slice(0, -wd), slice(-wd, -shift_d), slice(-shift_d, None)):
                    img_mask[:, h_s, w_s, d_s, :] = cnt
                    cnt += 1
        mask_w = rearrange(
            img_mask,
            "1 (nh wh) (nw ww) (nd wd) 1 -> (nh nw nd) (wh ww wd)",
            wh=wh,
            ww=ww,
            wd=wd,
        )
        diff = mask_w.unsqueeze(2) - mask_w.unsqueeze(1)
        # (num_windows, 1, wh*ww*wd, wh*ww*wd)
        return (
            torch.zeros_like(diff).masked_fill_(diff != 0, float("-inf")).unsqueeze(1)
        )

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_generator(x_noise)
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        wh, ww = self.window_size[0], self.window_size[1]
        shift_h, shift_w = wh // 2, ww // 2

        # Cyclic shift before window partitioning
        if self.shift:
            xs = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
        else:
            xs = x

        qkv = self.qkv_proj(modulate(self.norm1(xs), shift1, scale1))
        q, k, v = qkv.split(self.fused_heads, dim=-1)

        if self.n_spatial_dims == 3:
            D = xs.shape[3]
            wd = self.window_size[2] if len(self.window_size) > 2 else 1
            attn_mask = self._compute_attn_mask_3d(H, W, D, x.device)
            q, k, v = (
                rearrange(
                    t,
                    "b (h wh) (w ww) (d wd) (he c) -> (b h w d) he (wh ww wd) c",
                    wh=wh,
                    ww=ww,
                    wd=wd,
                    he=self.num_heads,
                )
                for t in (q, k, v)
            )
            if attn_mask is not None:
                attn_mask = attn_mask.repeat(B, 1, 1, 1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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
            attn_mask = self._compute_attn_mask(H, W, x.device)
            q, k, v = (
                rearrange(
                    t,
                    "b (h wh) (w ww) (he c) -> (b h w) he (wh ww) c",
                    wh=wh,
                    ww=ww,
                    he=self.num_heads,
                )
                for t in (q, k, v)
            )
            if attn_mask is not None:
                attn_mask = attn_mask.repeat(B, 1, 1, 1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            out = rearrange(
                out,
                "(b h w) he (wh ww) c -> b (h wh) (w ww) (he c)",
                b=B,
                h=H // wh,
                w=W // ww,
                wh=wh,
                ww=ww,
            )

        # Reverse cyclic shift
        if self.shift:
            out = torch.roll(out, shifts=(shift_h, shift_w), dims=(1, 2))

        x = x + self.drop_path(apply_gate(self.output_proj(out), gate1))
        return x + self.drop_path(
            apply_gate(self.mlp(modulate(self.norm2(x), shift2, scale2)), gate2)
        )


class SwinViTProcessor(Processor[EncodedBatch]):
    """
    ViT Processor using 2D/3D Shifted-Window Attention (Swin) and Ada-LN.

    References
    ----------
    - Aurora: A Foundation Model of the Atmosphere (https://arxiv.org/abs/2405.13063)
    - Video Swin Transformer (https://arxiv.org/abs/2106.13230)
    """

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
        use_ada_ln: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        for i, (res, ws) in enumerate(
            zip(spatial_resolution, window_size, strict=False)
        ):
            if res % ws != 0:
                raise ValueError(
                    f"spatial_resolution[{i}]={res} must be divisible "
                    f"by window_size[{i}]={ws}"
                )
        self.n_noise_channels = n_noise_channels
        self.loss_func = loss_func or nn.MSELoss()
        self.embed = PatchEmbedding(
            in_channels, hidden_dim, groups, self.n_spatial_dims, patch_size
        )
        dp_rates = torch.linspace(0, drop_path, n_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                SwinViTBlock(
                    hidden_dim,
                    num_heads,
                    self.n_spatial_dims,
                    n_noise_channels,
                    window_size,
                    shift=(i % 2 == 1),
                    drop_path=dp_rates[i],
                    use_ada_ln=use_ada_ln,
                    zero_init=zero_init,
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

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002
        noise = (
            torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
            if self.n_noise_channels
            else None
        )
        if self.n_spatial_dims == 2:
            out = self(rearrange(x, "b c h w -> b h w c").contiguous(), noise)
            return rearrange(out, "b h w c -> b c h w").contiguous()
        out = self(rearrange(x, "b c d h w -> b d h w c").contiguous(), noise)
        return rearrange(out, "b d h w c -> b c d h w").contiguous()

    def loss(self, batch: EncodedBatch) -> Tensor:
        return self.loss_func(
            self.map(batch.encoded_inputs, batch.global_cond),
            batch.encoded_output_fields,
        )
