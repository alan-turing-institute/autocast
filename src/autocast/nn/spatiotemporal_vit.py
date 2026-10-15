"""LOLA-compatible global spatiotemporal attention using Azula components.

Architecture reference (MIT, Copyright 2025 François Rozet):
https://github.com/francois-rozet/lola/tree/21a4354b327e6e5ee06da5075ba3bd1dd88c61f1/lola/nn
See LICENSE.lola for the upstream license.
"""

from collections.abc import Sequence
from math import prod
from typing import cast

import torch
from azula.nn.embedding import SineEncoding
from azula.nn.layers import Patchify, Unpatchify
from azula.nn.vit import ViTBlock
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from autocast.nn.base import TemporalBackboneBase


class _SpatiotemporalBlock(nn.Module):
    """Azula attention/FFN with LOLA's fourth AdaLN gate and input skip."""

    def __init__(
        self, channels: int, mod_features: int, attention_heads: int, **kwargs
    ) -> None:
        super().__init__()
        block = ViTBlock(
            channels,
            mod_features=mod_features,
            attention_heads=attention_heads,
            **kwargs,
        )
        self.norm, self.msa, self.ffn = block.norm, block.msa, block.ffn
        self.checkpointing = block.checkpointing
        gates = nn.Linear(mod_features, 4 * channels)
        gates.weight.data.mul_(1e-2)
        self.ada_zero = nn.Sequential(
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            gates,
            Rearrange("b (n c) -> n b 1 c", n=4),
        )
        # Azula 0.7 uses the opposite projection bias convention and a
        # different RoPE initialization; retain its attention computation.
        self.msa.qkv_proj = nn.Linear(channels, 3 * channels, bias=False)
        self.msa.y_proj = nn.Linear(channels, channels)
        self.msa.theta_proj = nn.Linear(3, channels // 2, bias=False)
        amplitude = 1e2 ** -torch.rand(channels // 2)
        direction = nn.functional.normalize(torch.randn(3, channels // 2), dim=0)
        self.msa.theta_proj.weight.data.copy_((amplitude * direction).T)

    def _forward(self, x: Tensor, mod: Tensor, pos: Tensor, skip: Tensor) -> Tensor:
        a, b, c, d = self.ada_zero(mod)
        y = (a + 1) * self.norm(x) + b
        y = y + self.msa(y, pos)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)
        return (y + d * skip) * torch.rsqrt(1 + d * d)

    def forward(self, x: Tensor, mod: Tensor, pos: Tensor, skip: Tensor) -> Tensor:
        if self.checkpointing:
            return cast(
                Tensor,
                checkpoint(self._forward, x, mod, pos, skip, use_reentrant=False),
            )
        return self._forward(x, mod, pos, skip)


class _SpatiotemporalViT(nn.Module):
    """Patch three axes and attend jointly over all frame/patch tokens."""

    def __init__(
        self,
        channels: int,
        mod_features: int,
        hid_channels: int,
        hid_blocks: int,
        attention_heads: int,
        patch_size: Sequence[int],
        **kwargs,
    ) -> None:
        super().__init__()
        self.patch = Patchify(patch_size, channel_last=True)
        self.unpatch = Unpatchify(patch_size, channel_last=True)
        patch_channels = channels * prod(patch_size)
        self.in_proj = nn.Linear(2 * patch_channels, hid_channels)
        self.out_proj = nn.Linear(hid_channels, patch_channels)
        self.pos_embedding = nn.Sequential(
            SineEncoding(hid_channels, omega=1e3),
            Rearrange("... n c -> ... (n c)"),
            nn.Linear(3 * hid_channels, hid_channels),
        )
        self.blocks = nn.ModuleList(
            _SpatiotemporalBlock(hid_channels, mod_features, attention_heads, **kwargs)
            for _ in range(hid_blocks)
        )

    def forward(self, x: Tensor, mod: Tensor, cond: Tensor) -> Tensor:
        x = self.in_proj(self.patch(torch.cat((x, cond), dim=1)))
        shape = x.shape[1:-1]
        axes = [torch.arange(n, dtype=x.dtype, device=x.device) for n in shape]
        pos = torch.cartesian_prod(*axes)
        skip = rearrange(x, "b t h w c -> b (t h w) c")
        x = skip + self.pos_embedding(pos)
        for block in self.blocks:
            x = block(x, mod, pos, skip)
        x = rearrange(x, "b (t h w) c -> b t h w c", t=shape[0], h=shape[1], w=shape[2])
        return self.unpatch(self.out_proj(x))


class SpatiotemporalViTBackbone(TemporalBackboneBase):
    """LOLA's joint spatiotemporal transformer, independent of the objective.

    ``conditioning='frames'`` joins clean context frames to the forecast slots
    and returns only the forecast slice (ordinary FM, diffusion or CRPS).
    ``conditioning='mask'`` accepts a full window and a same-shaped observation
    mask, returning the whole window (masked-window processors).

    Inputs/outputs use (B, T, H, W, C). A scalar spatial patch size ``p``
    means (1, p, p); triples specify (time, height, width), with time size one.
    Only global attention with RoPE and no relative bias is supported.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        n_steps_output: int,
        n_steps_input: int,
        global_cond_channels: int | None = None,
        include_global_cond: bool = False,
        mod_features: int = 256,
        hid_channels: int = 1024,
        hid_blocks: int = 16,
        attention_heads: int = 8,
        patch_size: int | Sequence[int] = (1, 1, 1),
        conditioning: str = "frames",
        dropout: float = 0.05,
        ffn_factor: int = 4,
        qk_norm: bool = True,
        checkpointing: bool = False,
        use_precomputed_modulation: bool = False,
        rope: bool = True,
        rpb: bool = False,
        window_size: int | Sequence[int] | None = None,
    ) -> None:
        if not rope or rpb or window_size is not None:
            msg = "LOLA attention requires rope=True, rpb=False and window_size=None."
            raise ValueError(msg)
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        if conditioning not in {"frames", "mask"}:
            msg = "conditioning must be 'frames' or 'mask'."
            raise ValueError(msg)
        if (
            min(in_channels, n_steps_input, n_steps_output, mod_features, hid_blocks)
            <= 0
        ):
            msg = (
                "Channels, temporal steps, modulation width and depth must be positive."
            )
            raise ValueError(msg)
        if in_channels != out_channels or cond_channels != in_channels:
            msg = "Fields and conditioning must have matching channel counts."
            raise ValueError(msg)
        if (
            len(patch_size) != 3
            or patch_size[0] != 1
            or any(p <= 0 for p in patch_size)
        ):
            msg = "patch_size must be (1, positive height, positive width)."
            raise ValueError(msg)
        if conditioning == "mask" and n_steps_input != n_steps_output:
            msg = "Mask conditioning requires equal full-window input/output steps."
            raise ValueError(msg)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            global_cond_channels=global_cond_channels,
            include_global_cond=include_global_cond,
            mod_features=mod_features,
            use_precomputed_modulation=use_precomputed_modulation,
        )
        if self.time_embedding is not None:
            self.time_embedding[0] = SineEncoding(mod_features, omega=1e3)
        self.conditioning = conditioning
        self.patch_size = tuple(patch_size)
        self.vit = self._build_backbone(
            channels=in_channels,
            mod_features=mod_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            attention_heads=attention_heads,
            patch_size=patch_size,
            dropout=dropout,
            ffn_factor=ffn_factor,
            qk_norm=qk_norm,
            checkpointing=checkpointing,
            rpb=False,
            rope=False,  # The block installs LOLA's initialized RoPE projection.
        )

    def _build_backbone(self, **kwargs) -> _SpatiotemporalViT:
        return _SpatiotemporalViT(**kwargs)

    @property
    def backbone(self) -> nn.Module:
        """Return the joint spatiotemporal transformer."""
        return self.vit

    def _forward_fields(
        self, x_t: Tensor, t_emb: Tensor, cond: Tensor | None
    ) -> Tensor:
        if (
            x_t.ndim != 5
            or cond is None
            or cond.ndim != 5
            or x_t.shape[1] != self.n_steps_output
            or cond.shape[1] != self.n_steps_input
            or x_t.shape[0] != cond.shape[0]
            or x_t.shape[2:] != cond.shape[2:]
            or x_t.shape[-1] != self.in_channels
        ):
            msg = "Expected compatible (B, T, H, W, C) fields and conditioning."
            raise ValueError(msg)
        if any(n % p for n, p in zip(x_t.shape[1:4], self.patch_size, strict=True)):
            msg = "Field dimensions must be divisible by patch_size."
            raise ValueError(msg)
        if self.conditioning == "frames":
            x_t = torch.cat((cond, x_t), dim=1)
            cond = torch.zeros_like(x_t)
            cond[:, : self.n_steps_input] = 1
        y = self.vit(
            rearrange(x_t, "b t h w c -> b c t h w"),
            mod=t_emb,
            cond=rearrange(cond, "b t h w c -> b c t h w"),
        )
        y = rearrange(y, "b c t h w -> b t h w c")
        return y[:, self.n_steps_input :] if self.conditioning == "frames" else y
