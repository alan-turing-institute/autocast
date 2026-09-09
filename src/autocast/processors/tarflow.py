from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor

# Adapted from Apple's TarFlow reference (apple/ml-tarflow, transformer_flow.py;
# permissive license). The transformer autoregressive-flow machinery is ported
# faithfully; the one substantive change is conditioning: the discrete
# class-embedding is replaced by a per-patch projection of the patchified current
# state x_t (injected into every block), and classifier-free guidance is dropped.


class _Permutation(nn.Module):
    def __init__(self, flip: bool) -> None:
        super().__init__()
        self.flip = flip

    def forward(self, x: Tensor, dim: int = 1, inverse: bool = False) -> Tensor:  # noqa: ARG002
        return x.flip(dims=[dim]) if self.flip else x


class _Attention(nn.Module):
    """Multi-head self-attention with optional causal KV-cached sampling."""

    def __init__(self, channels: int, head_channels: int) -> None:
        super().__init__()
        if channels % head_channels != 0:
            msg = "channels must be divisible by head_channels."
            raise ValueError(msg)
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.num_heads = channels // head_channels
        self.scale = head_channels ** (-0.5)
        self.sample = False
        self.k_cache: list[Tensor] = []
        self.v_cache: list[Tensor] = []

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, t, c = x.shape
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = (
            self.qkv(x)
            .reshape(b, t, 3 * self.num_heads, -1)
            .transpose(1, 2)
            .chunk(3, dim=1)
        )
        if self.sample:
            self.k_cache.append(k)
            self.v_cache.append(v)
            k = torch.cat(self.k_cache, dim=2)
            v = torch.cat(self.v_cache, dim=2)
        attn_mask = mask.bool() if mask is not None else None
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.scale
        )
        return self.proj(x.transpose(1, 2).reshape(b, t, c))


class _MLP(nn.Module):
    def __init__(self, channels: int, expansion: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.main = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class _AttentionBlock(nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int) -> None:
        super().__init__()
        self.attention = _Attention(channels, head_channels)
        self.mlp = _MLP(channels, expansion)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(x, mask)
        return x + self.mlp(x)


class _MetaBlock(nn.Module):
    """One causal-masked autoregressive affine transform over the patch sequence."""

    attn_mask: Tensor

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        channels: int,
        num_patches: int,
        flip: bool,
        num_layers: int,
        head_dim: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(in_channels, channels)
        self.cond_proj = nn.Linear(cond_channels, channels)
        self.pos_embed = nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        self.attn_blocks = nn.ModuleList(
            _AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)
        )
        self.proj_out = nn.Linear(channels, in_channels * 2)
        self.proj_out.weight.data.fill_(0.0)
        self.proj_out.bias.data.fill_(0.0)
        self.permutation = _Permutation(flip)
        self.register_buffer(
            "attn_mask", torch.tril(torch.ones(num_patches, num_patches))
        )

    def _embed(self, x: Tensor, pos_embed: Tensor, cond: Tensor) -> Tensor:
        return self.proj_in(x) + pos_embed + self.cond_proj(cond)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        x = self.permutation(x)
        cond = self.permutation(cond)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        h = self._embed(x, pos_embed, cond)
        for block in self.attn_blocks:
            h = block(h, self.attn_mask)
        h = self.proj_out(h)
        # Shift so patch i's transform params depend only on patches < i.
        h = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        xa, xb = h.chunk(2, dim=-1)
        scale = (-xa.float()).exp().type(xa.dtype)
        z = self.permutation((x_in - xb) * scale, inverse=True)
        return z, -xa.mean(dim=[1, 2])

    def _set_sample(self, flag: bool) -> None:
        for m in self.modules():
            if isinstance(m, _Attention):
                m.sample = flag
                m.k_cache = []
                m.v_cache = []

    def reverse(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.permutation(x)
        cond = self.permutation(cond)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        self._set_sample(True)
        for i in range(x.size(1) - 1):
            h = self._embed(x[:, i : i + 1], pos_embed[i : i + 1], cond[:, i : i + 1])
            for block in self.attn_blocks:
                h = block(h)
            h = self.proj_out(h)
            xa, xb = h.chunk(2, dim=-1)
            scale = xa[:, 0].float().exp().type(xa.dtype)
            x[:, i + 1] = x[:, i + 1] * scale + xb[:, 0]
        self._set_sample(False)
        return self.permutation(x, inverse=True)


class TarFlowProcessor(Processor):
    """Transformer autoregressive normalizing flow (TarFlow), conditioned on x_t.

    A patch-based transformer flow that models ``p(x_{t+1} | x_t)`` exactly: a
    stack of causal-masked autoregressive affine blocks (with alternating patch
    permutations) maps the next-state field to a Gaussian latent, conditioned on
    the patchified current state injected into every block. Likelihood is a
    single parallel pass; sampling is a sequential per-patch reverse with KV
    caching (one transformer pass per patch, no per-dimension unroll). Designed
    for small square 2D fields (e.g. ``[8, 8]``).

    The training loss is the exact change-of-variables negative log-likelihood
    under a unit ``N(0, I)`` latent prior. At sampling time the latents are
    rescaled by an exponential moving average of the encoded-latent variance
    (the ``var`` buffer) to match the marginal scale of the data. The only
    capacity choices are patch size, channels and depth; the transform scales
    are learned by the flow.
    """

    var: Tensor

    def __init__(
        self,
        *,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        spatial_shape: Sequence[int] = (1, 1),
        global_cond_features: int = 0,
        patch_size: int = 1,
        channels: int = 64,
        num_blocks: int = 8,
        layers_per_block: int = 2,
        head_dim: int = 64,
        expansion: int = 4,
        var_lr: float = 0.1,
    ) -> None:
        super().__init__()
        spatial = tuple(int(s) for s in spatial_shape)
        if len(spatial) != 2 or spatial[0] != spatial[1]:
            msg = (
                "TarFlowProcessor expects a square 2D spatial_shape (H == W); "
                f"got {spatial}. Use the flat/conv flows for non-square fields."
            )
            raise ValueError(msg)
        if global_cond_features:
            msg = "TarFlowProcessor does not support global_cond_features yet."
            raise ValueError(msg)
        h = spatial[0]
        if h % int(patch_size) != 0:
            msg = f"patch_size {patch_size} must divide the grid size {h}."
            raise ValueError(msg)

        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.n_channels_in = int(n_channels_in)
        self.n_channels_out = int(n_channels_out)
        self.spatial_shape = spatial
        self.img_size = h
        self.patch_size = int(patch_size)
        self.var_lr = float(var_lr)

        # Fold time into channels for the conv-style (N, C, H, W) layout.
        self.tgt_channels = self.n_steps_output * self.n_channels_out
        self.cond_channels = self.n_steps_input * self.n_channels_in
        self.num_patches = (h // self.patch_size) ** 2
        tgt_patch = self.tgt_channels * self.patch_size**2
        cond_patch = self.cond_channels * self.patch_size**2

        self.blocks = nn.ModuleList(
            _MetaBlock(
                tgt_patch,
                cond_patch,
                int(channels),
                self.num_patches,
                flip=bool(i % 2),
                num_layers=int(layers_per_block),
                head_dim=int(head_dim),
                expansion=int(expansion),
            )
            for i in range(int(num_blocks))
        )
        self.register_buffer("var", torch.ones(self.num_patches, tgt_patch))

    # -- patch + layout helpers ----------------------------------------------
    def _to_images(self, field: Tensor, steps: int, channels: int) -> Tensor:
        """(B, T, H, W, C) channels-last -> (B, T*C, H, W) channels-first."""
        h = self.img_size
        x = field.reshape(field.shape[0], steps, h, h, channels)
        return x.permute(0, 1, 4, 2, 3).reshape(field.shape[0], steps * channels, h, h)

    def _patchify(self, x: Tensor) -> Tensor:
        u = nn.functional.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1, 2)

    def _unpatchify(self, x: Tensor) -> Tensor:
        u = x.transpose(1, 2)
        return nn.functional.fold(
            u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size
        )

    def _cond_patches(self, x: Tensor) -> Tensor:
        return self._patchify(
            self._to_images(x, self.n_steps_input, self.n_channels_in)
        )

    # -- flow -----------------------------------------------------------------
    def _encode(self, y_patches: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        z = y_patches
        logdets = z.new_zeros(())
        for block in self.blocks:
            z, logdet = block(z, cond)
            logdets = logdets + logdet
        return z, logdets

    # -- Processor API --------------------------------------------------------
    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        return self.map(x, global_cond)

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002
        """Draw one member: Gaussian latent -> sequential reverse, given x_t."""
        cond = self._cond_patches(x)
        z = torch.randn(
            x.shape[0],
            self.num_patches,
            self.var.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        z = z * self.var.sqrt()
        for block in reversed(self.blocks):
            z = cast(_MetaBlock, block).reverse(z, cond)
        img = self._unpatchify(z)  # (B, T_out*C_out, H, W)
        h = self.img_size
        y = img.reshape(x.shape[0], self.n_steps_output, self.n_channels_out, h, h)
        return y.permute(0, 1, 3, 4, 2)

    @torch.no_grad()
    def map_batched(self, x: Tensor, members: int, max_rows: int = 8192) -> Tensor:
        """Draw ``members`` ensemble members in ONE sequential reverse.

        ``map`` runs a full per-patch sequential reverse per call, so an M-member
        ensemble built by looping ``map`` costs M reverses. Here the members are
        folded into the batch dimension and a single reverse produces the whole
        ensemble (the per-patch transformer pass is parallel over the batch), the
        key cost saving for the autoregressive sampler. Returns
        ``(B, members, T_out, *spatial, C_out)`` -- the per-member axis is dim 1,
        which is moved to the trailing ensemble axis at sampling time.

        The effective batch is ``B * members``. For a full eval set that is large
        (e.g. 6080 one-step inputs x 64 members = 389k rows), and the per-patch
        KV cache built during the reverse scales with it, so the reverse is run
        over the input batch in chunks capped at ``max_rows`` rows to bound peak
        memory. Chunking only changes the latent draw order, not the output
        distribution (each chunk draws its own latents); with ``B * members <=
        max_rows`` the path is unchunked and bit-identical to a single reverse.
        """
        b = x.shape[0]
        h = self.img_size
        rows_per_chunk = max(1, max_rows // max(1, members))
        outs = []
        for i in range(0, b, rows_per_chunk):
            xc = x[i : i + rows_per_chunk]
            bc = xc.shape[0]
            cond = self._cond_patches(xc).repeat_interleave(members, dim=0)
            z = torch.randn(
                bc * members,
                self.num_patches,
                self.var.shape[1],
                device=xc.device,
                dtype=xc.dtype,
            )
            z = z * self.var.sqrt()
            for block in reversed(self.blocks):
                z = cast(_MetaBlock, block).reverse(z, cond)
            img = self._unpatchify(z)  # (bc*M, T_out*C_out, H, W)
            rows = bc * members
            y = img.reshape(rows, self.n_steps_output, self.n_channels_out, h, h)
            y = y.permute(0, 1, 3, 4, 2)  # (bc*M, T_out, H, W, C_out)
            outs.append(y.reshape(bc, members, *y.shape[1:]))
        return torch.cat(outs, dim=0)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Exact negative log-likelihood (TarFlow objective) of x_{t+1} given x_t."""
        y_img = self._to_images(
            batch.encoded_output_fields, self.n_steps_output, self.n_channels_out
        )
        y_patches = self._patchify(y_img)
        cond = self._cond_patches(batch.encoded_inputs)
        z, logdets = self._encode(y_patches, cond)
        if self.training:
            self.var.lerp_((z**2).mean(dim=0).detach(), weight=self.var_lr)
        # 0.5 * ||z||^2 (per element) minus the mean log-det: the Gaussian NLL of
        # the latent under the change of variables, up to an additive constant.
        return 0.5 * z.pow(2).mean() - logdets.mean()
