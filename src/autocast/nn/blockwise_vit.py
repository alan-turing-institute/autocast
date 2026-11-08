"""Azula spatial ViT with an explicit modulation tensor for each block."""

import torch
from azula.nn.vit import ViT
from einops import rearrange
from torch import Tensor, nn


class BlockwiseViT(nn.Module):
    """Reuse initialized ViT modules with ``(B, L, N, D)`` spatial conditioning.

    The caller filters raw noise and projects its channels before this module.
    Each original block then receives its own token-wise AdaLN vectors. No
    activations are Fourier filtered and no global noise bypass is introduced.
    Existing module names, parameter shapes and checkpoint keys are retained.
    """

    def __init__(self, vit: ViT) -> None:
        super().__init__()
        if vit.spatial != 2:
            msg = "BlockwiseViT requires a two-dimensional spatial ViT."
            raise ValueError(msg)
        self.patch = vit.patch
        self.unpatch = vit.unpatch
        self.in_proj = vit.in_proj
        self.out_proj = vit.out_proj
        self.pos_embedding = vit.pos_embedding
        self.blocks = vit.blocks
        self.spatial = vit.spatial
        self.train(vit.training)

    def forward(
        self, x: Tensor, mod: Tensor | None = None, cond: Tensor | None = None
    ) -> Tensor:
        if cond is not None:
            x = torch.cat((x, cond), dim=1)
        x = self.in_proj(self.patch(x))
        height, width = x.shape[1:3]
        if (
            mod is None
            or mod.ndim != 4
            or mod.shape[:3] != (x.shape[0], len(self.blocks), height * width)
        ):
            msg = "BlockwiseViT requires modulation with shape (B, blocks, H*W, D)."
            raise ValueError(msg)
        pos = torch.cartesian_prod(
            torch.arange(height, dtype=x.dtype, device=x.device),
            torch.arange(width, dtype=x.dtype, device=x.device),
        )
        x = rearrange(x, "b h w c -> b (h w) c")
        x = x + self.pos_embedding(pos)
        for index, block in enumerate(self.blocks):
            x = block(x, mod[:, index], pos=pos)
        x = rearrange(x, "b (h w) c -> b h w c", h=height, w=width)
        return self.unpatch(self.out_proj(x))
