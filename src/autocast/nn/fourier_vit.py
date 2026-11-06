"""Optional Fourier conditioning at the existing Azula ViT modulation site."""

import torch
from azula.nn.utils import checkpoint
from azula.nn.vit import ViT, ViTBlock
from einops import rearrange
from torch import Tensor, nn

from autocast.nn.fourier_modulation import FourierModulation2d


class FourierViTBlock(nn.Module):
    """Reuse an Azula block, adding identity-initialized spectral modulation.

    Original modules retain their names and parameters. The spectral filter acts
    on normalized spatial features before the existing scale/shift, attention,
    feed-forward network and residual gate. Its global conditioning vector is
    the same vector supplied to AdaLN, not an additional random draw.
    """

    def __init__(self, block: ViTBlock, mod_features: int, num_bands: int):
        super().__init__()
        ada_zero = block.ada_zero
        if not isinstance(ada_zero, nn.Sequential):
            msg = "Fourier modulation requires an Azula block with global AdaLN."
            raise ValueError(msg)
        self.norm = block.norm
        self.ada_zero = ada_zero
        self.msa = block.msa
        self.ffn = block.ffn
        self.checkpointing = block.checkpointing
        self.fourier = FourierModulation2d(mod_features, num_bands=num_bands)
        self.train(block.training)

    def _forward(
        self,
        x: Tensor,
        mod: Tensor,
        pos: Tensor | None,
        mask: Tensor | None,
        *,
        spatial_shape: tuple[int, int],
    ) -> Tensor:
        a, b, c = self.ada_zero(mod)
        height, width = spatial_shape
        y = rearrange(self.norm(x), "b (h w) c -> b h w c", h=height, w=width)
        y = self.fourier(y, mod)
        y = rearrange(y, "b h w c -> b (h w) c")
        # Match Azula 0.7's one-norm, one-gated-branch computation exactly.
        y = (a + 1) * y + b
        y = y + self.msa(y, pos, mask)
        y = self.ffn(y)
        return (x + c * y) * torch.rsqrt(1 + c * c)

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
        pos: Tensor | None = None,
        mask: Tensor | None = None,
        *,
        spatial_shape: tuple[int, int],
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, reentrant=not self.training)(
                x, mod, pos, mask, spatial_shape=spatial_shape
            )
        return self._forward(x, mod, pos, mask, spatial_shape=spatial_shape)


class FourierViT(nn.Module):
    """Adapt an initialized 2D ViT without rebuilding its learned modules.

    Only Fourier modulation adds parameters. Passing the actual token grid to
    each block supports rectangular inputs and resolution changes without
    mutable per-forward state, including during activation recomputation.
    """

    def __init__(self, vit: ViT, mod_features: int, num_bands: int = 4):
        super().__init__()
        if vit.spatial != 2 or mod_features <= 0:
            msg = "Fourier modulation requires a 2D ViT and positive mod_features."
            raise ValueError(msg)
        self.patch = vit.patch
        self.unpatch = vit.unpatch
        self.in_proj = vit.in_proj
        self.out_proj = vit.out_proj
        self.pos_embedding = vit.pos_embedding
        self.spatial = vit.spatial
        blocks = []
        for block in vit.blocks:
            if not isinstance(block, ViTBlock):
                msg = "Fourier modulation requires the installed Azula ViTBlock layout."
                raise TypeError(msg)
            blocks.append(FourierViTBlock(block, mod_features, num_bands))
        self.blocks = nn.ModuleList(blocks)
        self.train(vit.training)

    def forward(
        self, x: Tensor, mod: Tensor | None = None, cond: Tensor | None = None
    ) -> Tensor:
        if mod is None or mod.ndim != 2 or mod.shape[0] != x.shape[0]:
            msg = "Fourier modulation requires global conditioning of shape (B, D)."
            raise ValueError(msg)
        if cond is not None:
            x = torch.cat((x, cond), dim=1)
        x = self.in_proj(self.patch(x))
        height, width = x.shape[1:3]
        pos = torch.cartesian_prod(
            torch.arange(height, dtype=x.dtype, device=x.device),
            torch.arange(width, dtype=x.dtype, device=x.device),
        )
        x = rearrange(x, "b h w c -> b (h w) c")
        x = x + self.pos_embedding(pos)
        for block in self.blocks:
            x = block(x, mod, pos=pos, spatial_shape=(height, width))
        x = rearrange(x, "b (h w) c -> b h w c", h=height, w=width)
        return self.unpatch(self.out_proj(x))
