"""Azula ViT with attention over a periodic graph of patch tokens."""

import torch
from azula.nn.vit import ViT
from einops import rearrange
from torch import Tensor


def periodic_grid_neighbors(
    height: int, width: int, hops: int, *, device: torch.device
) -> Tensor:
    """Return the width-fastest neighbor indices for each patch on a torus.

    The input is a positive grid shape and a nonnegative hop count. Each row
    contains the patch itself and all patches within that many grid hops,
    counted once even when a neighborhood wraps around a short axis.
    """
    # A single representative per offset also handles the antipode on even axes.
    row_offsets = range(-min(hops, height // 2), min(hops, (height - 1) // 2) + 1)
    col_offsets = range(-min(hops, width // 2), min(hops, (width - 1) // 2) + 1)
    offsets = [
        (dy, dx)
        for dy in row_offsets
        for dx in col_offsets
        if abs(dy) + abs(dx) <= hops
    ]
    dy, dx = torch.tensor(offsets, device=device).unbind(dim=1)
    nodes = torch.arange(height * width, device=device)
    rows = (nodes[:, None] // width + dy) % height
    cols = (nodes[:, None] % width + dx) % width
    return rows * width + cols


class PeriodicGraphViT(ViT):
    """Reuse Azula blocks with masked dense attention on a 2D periodic graph."""

    def __init__(self, *, graph_hops: int, spatial: int = 2, **kwargs):
        if spatial != 2:
            msg = "Periodic graph attention requires a 2D patch grid."
            raise ValueError(msg)
        if type(graph_hops) is not int or graph_hops < 0:
            msg = "graph_hops must be a nonnegative integer."
            raise ValueError(msg)
        super().__init__(spatial=spatial, **kwargs)
        self.graph_hops = graph_hops

    def forward(
        self, x: Tensor, mod: Tensor | None = None, cond: Tensor | None = None
    ) -> Tensor:
        """Pass the graph mask through Azula's per-block attention interface."""
        if cond is not None:
            x = torch.cat((x, cond), dim=1)

        patches = self.in_proj(self.patch(x))
        height, width = patches.shape[1:3]
        pos = torch.cartesian_prod(
            torch.arange(height, dtype=patches.dtype, device=patches.device),
            torch.arange(width, dtype=patches.dtype, device=patches.device),
        )

        neighbors = periodic_grid_neighbors(
            height, width, self.graph_hops, device=x.device
        )
        n_patches = height * width
        mask = torch.zeros(n_patches, n_patches, dtype=torch.bool, device=x.device)
        mask.scatter_(dim=1, index=neighbors, value=True)

        tokens = rearrange(patches, "b h w c -> b (h w) c") + self.pos_embedding(pos)
        for block in self.blocks:
            tokens = block(tokens, mod, pos=pos, mask=mask)

        patches = rearrange(tokens, "b (h w) c -> b h w c", h=height, w=width)
        return self.unpatch(self.out_proj(patches))
