"""Azula ViT with attention over a grid graph of patch tokens."""

from collections.abc import Sequence
from typing import Literal, get_args

import torch
import torch.nn.functional as F
from azula.nn.attention import MultiheadSelfAttention, apply_rope
from azula.nn.vit import ViT
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

GraphAttentionBackend = Literal["masked", "gathered"]
_CUDA_MAX_GRID_HEADS = 65_535


def validate_graph_attention_backend(backend: str) -> None:
    """Reject unknown graph attention backends before validating graph hops."""
    if backend not in get_args(GraphAttentionBackend):
        msg = "graph_attention_backend must be 'masked' or 'gathered'."
        raise ValueError(msg)


def periodic_grid_neighbors(
    height: int,
    width: int,
    hops: int,
    *,
    device: torch.device,
    periodic_axes: Sequence[bool] = (True, True),
) -> Tensor:
    """Return width-fastest neighbor indices with optional wrapping per axis.

    The input is a positive grid shape and a nonnegative hop count. Each row
    contains the patch itself and all patches within that many grid hops,
    counted once even when a neighborhood wraps around a short axis. Entries
    outside a bounded axis are padded with -1.
    """
    wrap_rows, wrap_cols = periodic_axes

    def axis_offsets(size: int, *, wrap: bool) -> range:
        # Keep one representative for an offset, including the even antipode.
        left, right = (size // 2, (size - 1) // 2) if wrap else (size - 1, size - 1)
        return range(-min(hops, left), min(hops, right) + 1)

    row_offsets = axis_offsets(height, wrap=wrap_rows)
    col_offsets = axis_offsets(width, wrap=wrap_cols)
    offsets = [
        (dy, dx)
        for dy in row_offsets
        for dx in col_offsets
        if abs(dy) + abs(dx) <= hops
    ]
    dy, dx = torch.tensor(offsets, device=device).unbind(dim=1)
    nodes = torch.arange(height * width, device=device)
    rows = nodes[:, None] // width + dy
    cols = nodes[:, None] % width + dx
    if wrap_rows:
        rows = rows.remainder(height)
    if wrap_cols:
        cols = cols.remainder(width)
    neighbors = rows * width + cols
    if wrap_rows and wrap_cols:
        return neighbors
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return neighbors.masked_fill(~valid, -1)


class GatheredGraphAttention(nn.Module):
    """Evaluate Azula self-attention only on a fixed-size list of neighbors."""

    checkpoint_gather: bool = True
    has_padding: bool = False

    def __init__(
        self,
        attention: MultiheadSelfAttention,
        *,
        checkpoint_gather: bool = True,
        has_padding: bool = False,
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
        self.has_padding = has_padding

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
        return -(query_norm + key_norm - 2 * dot)

    def _queries_per_call(self, q: Tensor) -> int:
        if not q.is_cuda:
            return q.shape[-2]
        # The CUDA memory-efficient SDPA launch uses packed heads as grid.y.
        limit = _CUDA_MAX_GRID_HEADS // self.heads
        if limit == 0:
            msg = "Gathered graph attention on CUDA supports at most 65,535 heads."
            raise ValueError(msg)
        return limit

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor | None,
        neighbors: Tensor,
        dropout_p: float,
    ) -> Tensor:
        if self.has_padding:
            valid = neighbors >= 0
            bias = (
                valid.expand(self.heads, -1, -1)
                if bias is None
                else bias.masked_fill(~valid, -torch.inf)
            )
        # Fold nodes into the head axis to keep one query per neighborhood and
        # allow a single position bias to broadcast across the batch.
        attended = F.scaled_dot_product_attention(
            query=rearrange(q, "b h n d -> b (h n) 1 d"),
            key=rearrange(k[:, :, neighbors], "b h n k d -> b (h n) k d"),
            value=rearrange(v[:, :, neighbors], "b h n k d -> b (h n) k d"),
            attn_mask=rearrange(bias, "h n k -> 1 (h n) 1 k")
            if bias is not None
            else None,
            dropout_p=dropout_p,
        )
        return rearrange(attended, "b (h n) 1 d -> b n (h d)", h=self.heads)

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
        bias = self._position_bias(pos, neighbors)
        queries_per_call = self._queries_per_call(q)
        checkpoint_gather = self.checkpoint_gather and torch.is_grad_enabled()
        outputs = []
        for start in range(0, neighbors.shape[0], queries_per_call):
            rows = slice(start, start + queries_per_call)
            chunk_bias = bias[:, rows] if bias is not None else None
            attention_inputs = (
                q[:, :, rows],
                k,
                v,
                chunk_bias,
                neighbors[rows],
                dropout_p,
            )
            if checkpoint_gather:
                # Save the projections and regenerate the edge-sized K/V per chunk.
                output = checkpoint(
                    self._attend, *attention_inputs, use_reentrant=False
                )
            else:
                output = self._attend(*attention_inputs)
            outputs.append(output)
        attended = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        return self.y_proj(attended)


class PeriodicGraphViT(ViT):
    """Reuse Azula blocks with a graph of 2D patch tokens."""

    graph_attention_backend: GraphAttentionBackend = "masked"

    def __init__(
        self,
        *,
        graph_hops: int,
        graph_attention_backend: GraphAttentionBackend = "masked",
        graph_periodic_axes: Sequence[bool] = (True, True),
        spatial: int = 2,
        **kwargs,
    ):
        if spatial != 2:
            msg = "Periodic graph attention requires a 2D patch grid."
            raise ValueError(msg)
        if type(graph_hops) is not int or graph_hops < 0:
            msg = "graph_hops must be a nonnegative integer."
            raise ValueError(msg)
        validate_graph_attention_backend(graph_attention_backend)
        if len(graph_periodic_axes) != 2 or any(
            type(axis) is not bool for axis in graph_periodic_axes
        ):
            msg = "graph_periodic_axes must specify two boolean spatial axes."
            raise ValueError(msg)
        super().__init__(spatial=spatial, **kwargs)
        self.graph_hops = graph_hops
        self.graph_attention_backend = graph_attention_backend
        self.graph_periodic_axes = tuple(graph_periodic_axes)
        if graph_attention_backend == "gathered":
            for block in self.blocks:
                attention = block.get_submodule("msa")
                if not isinstance(attention, MultiheadSelfAttention):
                    msg = "Installed Azula has an unsupported self-attention module."
                    raise RuntimeError(msg)
                block.set_submodule(
                    "msa",
                    GatheredGraphAttention(
                        attention,
                        checkpoint_gather=not block.checkpointing,
                        has_padding=not all(self.graph_periodic_axes),
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
            height,
            width,
            self.graph_hops,
            device=x.device,
            periodic_axes=self.graph_periodic_axes,
        )
        if self.graph_attention_backend == "masked":
            n_patches = height * width
            mask = x.new_zeros((n_patches, n_patches), dtype=torch.bool)
            if all(self.graph_periodic_axes):
                mask.scatter_(dim=1, index=attention_context, value=True)
            else:
                valid = attention_context >= 0
                queries = torch.arange(n_patches, device=x.device)[:, None]
                mask[
                    queries.expand_as(attention_context)[valid],
                    attention_context[valid],
                ] = True
            attention_context = mask

        tokens = rearrange(patches, "b h w c -> b (h w) c") + self.pos_embedding(pos)
        for block in self.blocks:
            tokens = block(tokens, mod, pos=pos, mask=attention_context)

        patches = rearrange(tokens, "b (h w) c -> b h w c", h=height, w=width)
        return self.unpatch(self.out_proj(patches))
