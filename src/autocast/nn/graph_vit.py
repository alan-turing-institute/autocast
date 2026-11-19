"""Azula ViT with attention over a periodic graph of patch tokens."""

from typing import Literal

import torch
import torch.nn.functional as F
from azula.nn.attention import MultiheadSelfAttention, apply_rope
from azula.nn.vit import ViT
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

GraphAttentionBackend = Literal["masked", "gathered"]


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


class GatheredGraphAttention(nn.Module):
    """Evaluate Azula self-attention only on a fixed-size list of neighbors."""

    checkpoint_gather: bool = True

    def __init__(
        self, attention: MultiheadSelfAttention, *, checkpoint_gather: bool = True
    ):
        super().__init__()
        self.qkv_proj = attention.qkv_proj
        self.y_proj = attention.y_proj
        self.qk_norm = attention.qk_norm
        self.bias_proj = attention.bias_proj
        self.theta_proj = attention.theta_proj
        self.heads = attention.heads
        self.dropout = attention.dropout
        self.checkpoint_gather = checkpoint_gather

    def _position_bias(self, pos: Tensor, neighbors: Tensor) -> Tensor | None:
        if self.bias_proj is None:
            return None

        p = F.linear(pos, self.bias_proj.weight)
        p = rearrange(p, "n (h d) -> h n d", h=self.heads)
        shift = self.bias_proj.bias
        assert shift is not None
        shifted = p + rearrange(shift, "(h d) -> h 1 d", h=self.heads)
        # Match Azula's squared-distance bias, evaluating it only on graph edges.
        query_norm = p.square().sum(dim=-1, keepdim=True)
        key_norm = shifted.square().sum(dim=-1)[:, neighbors]
        dot = torch.einsum("hnd,hnkd->hnk", p, shifted[:, neighbors])
        return rearrange(-(query_norm + key_norm - 2 * dot), "h n k -> 1 (h n) 1 k")

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor | None,
        neighbors: Tensor,
        dropout_p: float,
    ) -> Tensor:
        # Fold nodes into the head axis to keep one query per neighborhood and
        # allow a single position bias to broadcast across the batch.
        return F.scaled_dot_product_attention(
            query=rearrange(q, "b h n d -> b (h n) 1 d"),
            key=rearrange(k[:, :, neighbors], "b h n k d -> b (h n) k d"),
            value=rearrange(v[:, :, neighbors], "b h n k d -> b (h n) k d"),
            attn_mask=bias,
            dropout_p=dropout_p,
        )

    def forward(self, x: Tensor, pos: Tensor, neighbors: Tensor) -> Tensor:
        """Gather keys and values before calling scaled dot-product attention."""
        q, k, v = rearrange(
            self.qkv_proj(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )
        q, k = self.qk_norm(q), self.qk_norm(k)
        if self.theta_proj is not None:
            theta = rearrange(self.theta_proj(pos), "n (h d) -> h n d", h=self.heads)
            q, k = apply_rope(q, k, theta)

        dropout_p = self.dropout if self.training else 0.0
        attention_inputs = (
            q,
            k,
            v,
            self._position_bias(pos, neighbors),
            neighbors,
            dropout_p,
        )
        if self.checkpoint_gather and torch.is_grad_enabled():
            # Save the original projections and regenerate the edge-sized K/V.
            attended = checkpoint(self._attend, *attention_inputs, use_reentrant=False)
        else:
            attended = self._attend(*attention_inputs)
        return self.y_proj(
            rearrange(attended, "b (h n) 1 d -> b n (h d)", h=self.heads)
        )


class PeriodicGraphViT(ViT):
    """Reuse Azula blocks with a periodic graph of 2D patch tokens."""

    graph_attention_backend: GraphAttentionBackend = "masked"

    def __init__(
        self,
        *,
        graph_hops: int,
        graph_attention_backend: GraphAttentionBackend = "masked",
        spatial: int = 2,
        **kwargs,
    ):
        if spatial != 2:
            msg = "Periodic graph attention requires a 2D patch grid."
            raise ValueError(msg)
        if type(graph_hops) is not int or graph_hops < 0:
            msg = "graph_hops must be a nonnegative integer."
            raise ValueError(msg)
        if graph_attention_backend not in ("masked", "gathered"):
            msg = "graph_attention_backend must be 'masked' or 'gathered'."
            raise ValueError(msg)
        super().__init__(spatial=spatial, **kwargs)
        self.graph_hops = graph_hops
        self.graph_attention_backend = graph_attention_backend
        if graph_attention_backend == "gathered":
            for block in self.blocks:
                attention = block.get_submodule("msa")
                if not isinstance(attention, MultiheadSelfAttention):
                    msg = "Installed Azula has an unsupported self-attention module."
                    raise RuntimeError(msg)
                block.set_submodule(
                    "msa",
                    GatheredGraphAttention(
                        attention, checkpoint_gather=not block.checkpointing
                    ),
                )

    def forward(
        self, x: Tensor, mod: Tensor | None = None, cond: Tensor | None = None
    ) -> Tensor:
        """Pass the graph connectivity through Azula's block attention interface."""
        if cond is not None:
            x = torch.cat((x, cond), dim=1)

        patches = self.in_proj(self.patch(x))
        height, width = patches.shape[1:3]
        pos = torch.cartesian_prod(
            torch.arange(height, dtype=patches.dtype, device=patches.device),
            torch.arange(width, dtype=patches.dtype, device=patches.device),
        )

        attention_context = periodic_grid_neighbors(
            height, width, self.graph_hops, device=x.device
        )
        if self.graph_attention_backend == "masked":
            n_patches = height * width
            mask = x.new_zeros((n_patches, n_patches), dtype=torch.bool)
            attention_context = mask.scatter_(
                dim=1, index=attention_context, value=True
            )

        tokens = rearrange(patches, "b h w c -> b (h w) c") + self.pos_embedding(pos)
        for block in self.blocks:
            tokens = block(tokens, mod, pos=pos, mask=attention_context)

        patches = rearrange(tokens, "b (h w) c -> b h w c", h=height, w=width)
        return self.unpatch(self.out_proj(patches))
