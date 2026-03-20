import itertools
from collections.abc import Sequence
from functools import lru_cache

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


def _adjust_window_and_shift(
    window_size: tuple[int, ...],
    shift_size: tuple[int, ...],
    spatial_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Clamp windows to the input shape and disable shifts on clamped axes."""
    ws = tuple(min(w, s) for w, s in zip(window_size, spatial_shape, strict=False))
    ss = tuple(
        0 if w >= s else sh
        for w, sh, s in zip(window_size, shift_size, spatial_shape, strict=False)
    )
    return ws, ss


def _get_two_sided_padding(h_pad: int, w_pad: int) -> tuple[int, int, int, int]:
    """Return left, right, top, and bottom padding."""
    if h_pad:
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
    else:
        pad_top = pad_bottom = 0

    if w_pad:
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
    else:
        pad_left = pad_right = 0

    return pad_left, pad_right, pad_top, pad_bottom


def _get_three_sided_padding(
    d_pad: int, h_pad: int, w_pad: int
) -> tuple[int, int, int, int, int, int]:
    """Return left, right, top, bottom, front, and back padding."""
    if d_pad:
        pad_front = d_pad // 2
        pad_back = d_pad - pad_front
    else:
        pad_front = pad_back = 0
    return (*_get_two_sided_padding(h_pad, w_pad), pad_front, pad_back)


def _pad_2d(x: Tensor, pad_size: tuple[int, int], value: float = 0.0) -> Tensor:
    """Pad 2D spatial dimensions for tensors shaped (B, H, W, C)."""
    return F.pad(x, (0, 0, *_get_two_sided_padding(*pad_size)), value=value)


def _crop_2d(x: Tensor, pad_size: tuple[int, int]) -> Tensor:
    """Undo _pad_2d by symmetric cropping."""
    _, h, w, _ = x.shape
    hp, wp = pad_size
    pleft, pright, ptop, pbottom = _get_two_sided_padding(hp, wp)
    return x[:, ptop : h - pbottom, pleft : w - pright, :]


def _pad_3d(x: Tensor, pad_size: tuple[int, int, int], value: float = 0.0) -> Tensor:
    """Pad 3D spatial dimensions for tensors shaped (B, H, W, D, C)."""
    return F.pad(x, (0, 0, *_get_three_sided_padding(*pad_size)), value=value)


def _crop_3d(x: Tensor, pad_size: tuple[int, int, int]) -> Tensor:
    """Undo _pad_3d by symmetric cropping."""
    _, h, w, d, _ = x.shape
    hp, wp, dp = pad_size
    pleft, pright, ptop, pbottom, pfront, pback = _get_three_sided_padding(hp, wp, dp)
    return x[:, ptop : h - pbottom, pleft : w - pright, pfront : d - pback, :]


def _window_partition_2d(x: Tensor, ws: tuple[int, int]) -> Tensor:
    """Partition (B, H, W, C) into local 2D windows."""
    b, h, w, c = x.shape
    wh, ww = ws
    x = x.view(b, h // wh, wh, w // ww, ww, c)
    return rearrange(x, "b h1 wh w1 ww c -> (b h1 w1) wh ww c")


def _window_reverse_2d(windows: Tensor, ws: tuple[int, int], h: int, w: int) -> Tensor:
    """Undo _window_partition_2d."""
    wh, ww = ws
    h1, w1 = h // wh, w // ww
    b = int(windows.shape[0] / (h1 * w1))
    return rearrange(
        windows,
        "(b h1 w1) wh ww c -> b (h1 wh) (w1 ww) c",
        b=b,
        h1=h1,
        w1=w1,
        wh=wh,
        ww=ww,
    )


def _window_partition_3d(x: Tensor, ws: tuple[int, int, int]) -> Tensor:
    """Partition (B, H, W, D, C) into local 3D windows."""
    b, h, w, d, c = x.shape
    wh, ww, wd = ws
    x = x.view(b, h // wh, wh, w // ww, ww, d // wd, wd, c)
    return rearrange(x, "b h1 wh w1 ww d1 wd c -> (b h1 w1 d1) wh ww wd c")


def _window_reverse_3d(
    windows: Tensor, ws: tuple[int, int, int], h: int, w: int, d: int
) -> Tensor:
    """Undo _window_partition_3d."""
    wh, ww, wd = ws
    h1, w1, d1 = h // wh, w // ww, d // wd
    b = int(windows.shape[0] / (h1 * w1 * d1))
    return rearrange(
        windows,
        "(b h1 w1 d1) wh ww wd c -> b (h1 wh) (w1 ww) (d1 wd) c",
        b=b,
        h1=h1,
        w1=w1,
        d1=d1,
        wh=wh,
        ww=ww,
        wd=wd,
    )


@lru_cache
def _compute_shifted_window_mask_2d(
    h: int,
    w: int,
    ws: tuple[int, int],
    ss: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Compute Aurora-style shifted-window attention mask for 2D."""
    img_mask = torch.zeros((1, h, w, 1), device=device, dtype=dtype)
    h_slices = (slice(0, -ws[0]), slice(-ws[0], -ss[0]), slice(-ss[0], None))
    w_slices = (slice(0, -ws[1]), slice(-ws[1], -ss[1]), slice(-ss[1], None))

    cnt = 0
    for hs, wslice in itertools.product(h_slices, w_slices):
        img_mask[:, hs, wslice, :] = cnt
        cnt += 1

    pad_size = ((-h) % ws[0], (-w) % ws[1])
    img_mask = _pad_2d(img_mask, pad_size, value=cnt)

    mask_windows = _window_partition_2d(img_mask, ws)
    mask_windows = mask_windows.view(-1, ws[0] * ws[1])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )


@lru_cache
def _compute_shifted_window_mask_3d(
    h: int,
    w: int,
    d: int,
    ws: tuple[int, int, int],
    ss: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Compute Aurora-style shifted-window attention mask for 3D."""
    img_mask = torch.zeros((1, h, w, d, 1), device=device, dtype=dtype)
    h_slices = (slice(0, -ws[0]), slice(-ws[0], -ss[0]), slice(-ss[0], None))
    w_slices = (slice(0, -ws[1]), slice(-ws[1], -ss[1]), slice(-ss[1], None))
    d_slices = (slice(0, -ws[2]), slice(-ws[2], -ss[2]), slice(-ss[2], None))

    cnt = 0
    for hs, wslice, ds in itertools.product(h_slices, w_slices, d_slices):
        img_mask[:, hs, wslice, ds, :] = cnt
        cnt += 1

    pad_size = ((-h) % ws[0], (-w) % ws[1], (-d) % ws[2])
    img_mask = _pad_3d(img_mask, pad_size, value=cnt)

    mask_windows = _window_partition_3d(img_mask, ws)
    mask_windows = mask_windows.view(-1, ws[0] * ws[1] * ws[2])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )


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
        if len(window_size) < n_spatial_dims:
            raise ValueError(
                "window_size has "
                f"{len(window_size)} dims but n_spatial_dims={n_spatial_dims}"
            )
        self.window_size = tuple(window_size[:n_spatial_dims])
        self.shift_size = (
            tuple(ws // 2 for ws in self.window_size)
            if shift
            else tuple(0 for _ in range(n_spatial_dims))
        )
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

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:  # noqa: PLR0915
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_generator(x_noise)
        spatial_shape = tuple(x.shape[1 : 1 + self.n_spatial_dims])
        ws, ss = _adjust_window_and_shift(
            self.window_size, self.shift_size, spatial_shape
        )
        do_shift = any(s > 0 for s in ss)
        hp = wp = dp = 0

        xs = modulate(self.norm1(x), shift1, scale1)
        if do_shift:
            xs = torch.roll(
                xs,
                shifts=tuple(-s for s in ss),
                dims=tuple(range(1, 1 + self.n_spatial_dims)),
            )

        if self.n_spatial_dims == 2:
            h, w = spatial_shape
            ws2 = (ws[0], ws[1])
            ss2 = (ss[0], ss[1])
            pad_size_2d = ((-h) % ws2[0], (-w) % ws2[1])
            xs = _pad_2d(xs, pad_size_2d)
            _, hp, wp, _ = xs.shape

            windows = _window_partition_2d(xs, ws2).view(
                -1, ws2[0] * ws2[1], self.hidden_dim
            )
            attn_mask = (
                _compute_shifted_window_mask_2d(h, w, ws2, ss2, x.device, x.dtype)
                if do_shift
                else None
            )
        else:
            h, w, d = spatial_shape
            ws3 = (ws[0], ws[1], ws[2])
            ss3 = (ss[0], ss[1], ss[2])
            pad_size_3d = ((-h) % ws3[0], (-w) % ws3[1], (-d) % ws3[2])
            xs = _pad_3d(xs, pad_size_3d)
            _, hp, wp, dp, _ = xs.shape

            windows = _window_partition_3d(xs, ws3).view(
                -1, ws3[0] * ws3[1] * ws3[2], self.hidden_dim
            )
            attn_mask = (
                _compute_shifted_window_mask_3d(h, w, d, ws3, ss3, x.device, x.dtype)
                if do_shift
                else None
            )

        qkv = self.qkv_proj(windows)
        q, k, v = qkv.split(self.fused_heads, dim=-1)
        q, k, v = (
            rearrange(t, "b n (he c) -> b he n c", he=self.num_heads) for t in (q, k, v)
        )

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1).unsqueeze(0)
            batch_size = q.shape[0] // mask.shape[1]
            mask = mask.repeat(batch_size, 1, 1, 1, 1).reshape(-1, *mask.shape[2:])
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = rearrange(attn_out, "b he n c -> b n (he c)")

        if self.n_spatial_dims == 2:
            ws2 = (ws[0], ws[1])
            pad_size_2d = ((-spatial_shape[0]) % ws2[0], (-spatial_shape[1]) % ws2[1])
            out_windows = attn_out.view(-1, ws2[0], ws2[1], self.hidden_dim)
            out = _window_reverse_2d(out_windows, ws2, hp, wp)
            out = _crop_2d(out, pad_size_2d)
        else:
            ws3 = (ws[0], ws[1], ws[2])
            pad_size_3d = (
                (-spatial_shape[0]) % ws3[0],
                (-spatial_shape[1]) % ws3[1],
                (-spatial_shape[2]) % ws3[2],
            )
            out_windows = attn_out.view(-1, ws3[0], ws3[1], ws3[2], self.hidden_dim)
            out = _window_reverse_3d(out_windows, ws3, hp, wp, dp)
            out = _crop_3d(out, pad_size_3d)

        if do_shift:
            out = torch.roll(
                out,
                shifts=ss,
                dims=tuple(range(1, 1 + self.n_spatial_dims)),
            )

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
        if len(window_size) < self.n_spatial_dims:
            raise ValueError(
                "window_size has "
                f"{len(window_size)} dims but n_spatial_dims={self.n_spatial_dims}"
            )
        self.window_size = tuple(window_size[: self.n_spatial_dims])
        if any(ws <= 0 for ws in self.window_size):
            raise ValueError(
                f"window_size values must be positive, got {self.window_size}"
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
                    self.window_size,
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
