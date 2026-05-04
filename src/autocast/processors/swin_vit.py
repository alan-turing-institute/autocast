import itertools
from collections.abc import Sequence
from functools import lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers.drop import DropPath
from torch import nn

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
    hp, wp, dp = pad_size
    if hp:
        h_front = hp // 2
        h_back = hp - h_front
    else:
        h_front = h_back = 0
    if wp:
        w_left = wp // 2
        w_right = wp - w_left
    else:
        w_left = w_right = 0
    if dp:
        d_left = dp // 2
        d_right = dp - d_left
    else:
        d_left = d_right = 0

    # For (B, H, W, D, C), F.pad expects (C_l, C_r, D_l, D_r, W_l, W_r, H_l, H_r).
    return F.pad(
        x,
        (0, 0, d_left, d_right, w_left, w_right, h_front, h_back),
        value=value,
    )


def _crop_3d(x: Tensor, pad_size: tuple[int, int, int]) -> Tensor:
    """Undo _pad_3d by symmetric cropping."""
    _, h, w, d, _ = x.shape
    hp, wp, dp = pad_size
    if hp:
        h_front = hp // 2
        h_back = hp - h_front
    else:
        h_front = h_back = 0
    if wp:
        w_left = wp // 2
        w_right = wp - w_left
    else:
        w_left = w_right = 0
    if dp:
        d_left = dp // 2
        d_right = dp - d_left
    else:
        d_left = d_right = 0
    return x[
        :,
        h_front : h - h_back,
        w_left : w - w_right,
        d_left : d - d_right,
        :,
    ]


def _window_partition_2d(x: Tensor, ws: tuple[int, int]) -> Tensor:
    """Partition (B, H, W, C) into local 2D windows."""
    wh, ww = ws
    return rearrange(x, "b (h1 wh) (w1 ww) c -> (b h1 w1) wh ww c", wh=wh, ww=ww)


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
    wh, ww, wd = ws
    return rearrange(
        x, "b (h1 wh) (w1 ww) (d1 wd) c -> (b h1 w1 d1) wh ww wd c", wh=wh, ww=ww, wd=wd
    )


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
    mask_windows = rearrange(mask_windows, "nw wh ww 1 -> nw (wh ww)")
    attn_mask = rearrange(mask_windows, "nw w -> nw 1 w") - rearrange(
        mask_windows, "nw w -> nw w 1"
    )
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
    mask_windows = rearrange(mask_windows, "nw wh ww wd 1 -> nw (wh ww wd)")
    attn_mask = rearrange(mask_windows, "nw w -> nw 1 w") - rearrange(
        mask_windows, "nw w -> nw w 1"
    )
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

            windows = _window_partition_2d(xs, ws2)
            windows = rearrange(windows, "nw wh ww c -> nw (wh ww) c")
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

            windows = _window_partition_3d(xs, ws3)
            windows = rearrange(windows, "nw wh ww wd c -> nw (wh ww wd) c")
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
            b_q = q.shape[0] // attn_mask.shape[0]
            mask = repeat(attn_mask, "nw n1 n2 -> (b nw) 1 n1 n2", b=b_q)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = rearrange(attn_out, "b he n c -> b n (he c)")

        if self.n_spatial_dims == 2:
            ws2 = (ws[0], ws[1])
            pad_size_2d = ((-spatial_shape[0]) % ws2[0], (-spatial_shape[1]) % ws2[1])
            out_windows = rearrange(
                attn_out, "nw (wh ww) c -> nw wh ww c", wh=ws2[0], ww=ws2[1]
            )
            out = _window_reverse_2d(out_windows, ws2, hp, wp)
            out = _crop_2d(out, pad_size_2d)
        else:
            ws3 = (ws[0], ws[1], ws[2])
            pad_size_3d = (
                (-spatial_shape[0]) % ws3[0],
                (-spatial_shape[1]) % ws3[1],
                (-spatial_shape[2]) % ws3[2],
            )
            out_windows = rearrange(
                attn_out,
                "nw (wh ww wd) c -> nw wh ww wd c",
                wh=ws3[0],
                ww=ws3[1],
                wd=ws3[2],
            )
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


class PatchMerging(nn.Module):
    """Patch merging layer."""

    def __init__(self, dim: int, n_spatial_dims: int = 2):
        super().__init__()
        self.dim = dim
        self.n_spatial_dims = n_spatial_dims
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: Tensor, spatial_shape: tuple[int, ...]) -> Tensor:
        if self.n_spatial_dims == 2:
            H, W = spatial_shape
            x = rearrange(x, "b (h w) d -> b h w d", h=H, w=W)
            pad_H, pad_W = H % 2, W % 2
            if pad_H > 0 or pad_W > 0:
                x = F.pad(x, (0, 0, 0, pad_W, 0, pad_H))
            x = rearrange(x, "b (h h2) (w w2) d -> b (h w) (w2 h2 d)", h2=2, w2=2)
        elif self.n_spatial_dims == 3:
            H, W, depth = spatial_shape
            x = rearrange(x, "b (h w dp) d -> b h w dp d", h=H, w=W, dp=depth)
            pad_H, pad_W = H % 2, W % 2
            if pad_H > 0 or pad_W > 0:
                x = F.pad(x, (0, 0, 0, 0, 0, pad_W, 0, pad_H))
            x = rearrange(x, "b (h h2) (w w2) dp d -> b (h w dp) (w2 h2 d)", h2=2, w2=2)
        else:
            raise ValueError(f"Unsupported n_spatial_dims {self.n_spatial_dims}")

        x = self.norm(x)
        return self.reduction(x)


class PatchSplitting(nn.Module):
    """Patch splitting layer."""

    def __init__(self, dim: int, n_spatial_dims: int = 2):
        super().__init__()
        self.dim = dim
        self.n_spatial_dims = n_spatial_dims
        self.lin1 = nn.Linear(dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)
        self.lin2 = nn.Linear(dim // 2, dim // 2, bias=False)

    def forward(
        self,
        x: Tensor,
        spatial_shape: tuple[int, ...],
        crop: tuple[int, ...] | None = None,
    ) -> Tensor:
        x = self.lin1(x)
        if self.n_spatial_dims == 2:
            H, W = spatial_shape
            x = rearrange(
                x, "b (h w) (w2 h2 d) -> b (h h2) (w w2) d", h=H, w=W, h2=2, w2=2
            )
            if crop is not None and any(c > 0 for c in crop):
                x = x[:, : x.shape[1] - crop[0], : x.shape[2] - crop[1], :]
            x = rearrange(x, "b h w d -> b (h w) d")
        elif self.n_spatial_dims == 3:
            H, W, depth = spatial_shape
            x = rearrange(
                x,
                "b (h w dp) (w2 h2 d) -> b (h h2) (w w2) dp d",
                h=H,
                w=W,
                dp=depth,
                h2=2,
                w2=2,
            )
            if crop is not None and any(c > 0 for c in crop):
                x = x[
                    :,
                    : x.shape[1] - crop[0],
                    : x.shape[2] - crop[1],
                    : x.shape[3] - crop[2],
                    :,
                ]
            x = rearrange(x, "b h w dp d -> b (h w dp) d")
        else:
            raise ValueError(f"Unsupported n_spatial_dims {self.n_spatial_dims}")

        x = self.norm(x)
        return self.lin2(x)


class BasicSwinLayer(nn.Module):
    """A basic Swin layer comprising multiple blocks and an optional down/upsample."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        n_spatial_dims: int,
        n_noise_channels: int | None,
        window_size: tuple[int, ...],
        drop_path: list[float],
        downsample: type[nn.Module] | None = None,
        upsample: type[nn.Module] | None = None,
        use_ada_ln: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                SwinViTBlock(
                    dim,
                    num_heads,
                    n_spatial_dims,
                    n_noise_channels,
                    window_size,
                    shift=(i % 2 == 1),
                    drop_path=drop_path[i],
                    use_ada_ln=use_ada_ln,
                    zero_init=zero_init,
                )
                for i in range(depth)
            ]
        )
        if downsample is not None:
            self.downsample = downsample(dim=dim, n_spatial_dims=n_spatial_dims)
        else:
            self.downsample = None

        if upsample is not None:
            self.upsample = upsample(dim=dim, n_spatial_dims=n_spatial_dims)
        else:
            self.upsample = None

    def forward(  # noqa: PLR0912
        self,
        x: Tensor,
        x_noise: Tensor | None,
        spatial_shape: tuple[int, ...],
        crop: tuple[int, ...] | None = None,
    ) -> tuple[Tensor, Tensor | None, tuple[int, ...], tuple[int, ...] | None]:
        if len(spatial_shape) == 2:
            if x.ndim == 3:
                x = rearrange(
                    x,
                    "b (h w) c -> b h w c",
                    h=spatial_shape[0],
                    w=spatial_shape[1],
                )
            elif x.ndim != 4:
                raise ValueError(
                    "Expected 3D or 4D tensor for 2D Swin layer, "
                    f"got shape {tuple(x.shape)}"
                )
        elif len(spatial_shape) == 3:
            if x.ndim == 3:
                x = rearrange(
                    x,
                    "b (h w d) c -> b h w d c",
                    h=spatial_shape[0],
                    w=spatial_shape[1],
                    d=spatial_shape[2],
                )
            elif x.ndim != 5:
                raise ValueError(
                    "Expected 3D or 5D tensor for 3D Swin layer, "
                    f"got shape {tuple(x.shape)}"
                )
        else:
            raise ValueError(
                f"Unsupported spatial_shape with {len(spatial_shape)} dims"
            )

        for blk in self.blocks:
            x = blk(x, x_noise)

        if len(spatial_shape) == 2:
            x = rearrange(x, "b h w c -> b (h w) c")
        else:
            x = rearrange(x, "b h w d c -> b (h w d) c")
        out_shape = spatial_shape

        if self.downsample is not None:
            x_down = self.downsample(x, spatial_shape)
            padding = (spatial_shape[0] % 2, spatial_shape[1] % 2)
            if len(spatial_shape) == 3:
                out_shape = (
                    (spatial_shape[0] + padding[0]) // 2,
                    (spatial_shape[1] + padding[1]) // 2,
                    spatial_shape[2],
                )
                padding_out = (*padding, 0)
            else:
                out_shape = (
                    (spatial_shape[0] + padding[0]) // 2,
                    (spatial_shape[1] + padding[1]) // 2,
                )
                padding_out = padding
            return x_down, x, out_shape, padding_out

        if self.upsample is not None:
            x_up = self.upsample(x, spatial_shape, crop=crop)
            if len(spatial_shape) == 3:
                out_shape = (
                    spatial_shape[0] * 2,
                    spatial_shape[1] * 2,
                    spatial_shape[2],
                )
            else:
                out_shape = (spatial_shape[0] * 2, spatial_shape[1] * 2)
            if crop:
                out_shape = tuple(o - c for o, c in zip(out_shape, crop, strict=False))
            return x_up, x, out_shape, None

        return x, x, out_shape, None


class SwinViTProcessor(Processor[EncodedBatch]):
    """ViT Processor using 2D/3D Shifted-Window Attention (Swin) and Ada-LN.

    Constructs a U-Net style encoder-decoder architecture.

    Features:
    - Shifted-Window Multi-Head Self Attention (SW-MSA).
    - Patch Merging for hierarchical staging (encoder downsampling).
    - Patch Splitting for hierarchical expansion (decoder upsampling).
    - Ada-LN conditioning from noise embeddings (post-norm or per-block scaled).

    References
    ----------
    - Liu, Z. et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted
      Windows." ICCV 2021.
    - Liu, Z. et al. "Video Swin Transformer." CVPR 2022.
    - Microsoft Aurora Swin3D implementation:
      https://github.com/microsoft/aurora/blob/main/aurora/model/swin3d.py

        Shape convention:
        - Public processor boundary: channel-first
            - 2D: (B, C, H, W)
            - 3D: (B, C, H, W, D)
        - Internal Swin blocks: channels-last
            - 2D: (B, H, W, C)
            - 3D: (B, H, W, D, C)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        window_size: Sequence[int] = (4, 4, 4),
        hidden_dim: int = 64,
        encoder_depths: Sequence[int] = (2, 2, 2),
        encoder_num_heads: Sequence[int] = (3, 6, 12),
        decoder_depths: Sequence[int] = (2, 2, 2),
        decoder_num_heads: Sequence[int] = (12, 6, 3),
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

        self.num_encoder_layers = len(encoder_depths)
        self.num_decoder_layers = len(decoder_depths)
        assert self.num_encoder_layers == self.num_decoder_layers

        dpr = torch.linspace(0, drop_path, sum(encoder_depths)).tolist()

        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_encoder_layers):
            layer = BasicSwinLayer(
                dim=int(hidden_dim * 2**i_layer),
                depth=encoder_depths[i_layer],
                num_heads=encoder_num_heads[i_layer],
                n_spatial_dims=self.n_spatial_dims,
                n_noise_channels=n_noise_channels,
                window_size=self.window_size,
                drop_path=dpr[
                    sum(encoder_depths[:i_layer]) : sum(encoder_depths[: i_layer + 1])
                ],
                downsample=PatchMerging
                if (i_layer < self.num_encoder_layers - 1)
                else None,
                use_ada_ln=use_ada_ln,
                zero_init=zero_init,
            )
            self.encoder_layers.append(layer)

        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_decoder_layers):
            exponent = self.num_decoder_layers - i_layer - 1
            layer = BasicSwinLayer(
                dim=int(hidden_dim * 2**exponent),
                depth=decoder_depths[i_layer],
                num_heads=decoder_num_heads[i_layer],
                n_spatial_dims=self.n_spatial_dims,
                n_noise_channels=n_noise_channels,
                window_size=self.window_size,
                drop_path=dpr[
                    sum(decoder_depths[:i_layer]) : sum(decoder_depths[: i_layer + 1])
                ],
                upsample=PatchSplitting
                if (i_layer < self.num_decoder_layers - 1)
                else None,
                use_ada_ln=use_ada_ln,
                zero_init=zero_init,
            )
            self.decoder_layers.append(layer)

        self.debed = PatchUnembedding(
            out_channels, hidden_dim, groups, self.n_spatial_dims, patch_size
        )
        self.embed_reshapes = (
            ["b h w c -> b c h w", "b c h w -> b h w c"]
            if self.n_spatial_dims == 2
            else ["b h w d c -> b c h w d", "b c h w d -> b h w d c"]
        )

    def get_encoder_specs(
        self, patch_res: tuple[int, ...]
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...] | None]]:
        all_res = [patch_res]
        padded_outs = []
        for _ in range(1, self.num_encoder_layers):
            res = all_res[-1]
            pad_H, pad_W = res[0] % 2, res[1] % 2
            if len(res) == 2:
                padded_outs.append((pad_H, pad_W))
                all_res.append(((res[0] + pad_H) // 2, (res[1] + pad_W) // 2))
            else:
                padded_outs.append((pad_H, pad_W, 0))
                all_res.append(((res[0] + pad_H) // 2, (res[1] + pad_W) // 2, res[2]))
        padded_outs.append(None)
        return all_res, padded_outs

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        x = rearrange(
            self.embed(rearrange(x, self.embed_reshapes[0])), self.embed_reshapes[1]
        )

        spatial_shape = tuple(x.shape[1 : 1 + self.n_spatial_dims])
        all_enc_res, padded_outs = self.get_encoder_specs(spatial_shape)

        skips = []
        for i, layer in enumerate(self.encoder_layers):
            x, x_unscaled, _, _ = layer(x, x_noise, all_enc_res[i])
            skips.append(x_unscaled)

        for i, layer in enumerate(self.decoder_layers):
            index = self.num_decoder_layers - i - 1
            x, _, _, _ = layer(
                x,
                x_noise,
                all_enc_res[index],
                crop=padded_outs[index - 1] if index > 0 else None,
            )

            if index > 0:
                x = x + skips[index - 1]

        # Decoder layers may return flattened tokens; restore spatial layout
        # expected by PatchUnembedding.
        if self.n_spatial_dims == 2 and x.ndim == 3:
            x = rearrange(
                x,
                "b (h w) c -> b h w c",
                h=spatial_shape[0],
                w=spatial_shape[1],
            )
        elif self.n_spatial_dims == 3 and x.ndim == 3:
            x = rearrange(
                x,
                "b (h w d) c -> b h w d c",
                h=spatial_shape[0],
                w=spatial_shape[1],
                d=spatial_shape[2],
            )

        return rearrange(
            self.debed(rearrange(x, self.embed_reshapes[0])), self.embed_reshapes[1]
        )

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002
        """Map encoded inputs through Swin while preserving channel-first I/O."""
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
        if self.n_spatial_dims == 3:
            out = self(rearrange(x, "b c h w d -> b h w d c").contiguous(), noise)
            return rearrange(out, "b h w d c -> b c h w d").contiguous()
        raise ValueError(f"Unsupported n_spatial_dims={self.n_spatial_dims}")

    def loss(self, batch: EncodedBatch) -> Tensor:
        return self.loss_func(
            self.map(batch.encoded_inputs, batch.global_cond),
            batch.encoded_output_fields,
        )
