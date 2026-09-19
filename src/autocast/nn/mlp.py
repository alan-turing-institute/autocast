"""Temporal MLP backbone for non-spatial / degenerate-grid forecasting.

The azula UNet/ViT backbones downsample spatial axes and so cannot run on a
1x1 (scalar SDE) or length-1 degenerate grid. This backbone flattens the
spatial-channel-time vector and applies a FiLM-modulated MLP, consuming the
same ``(B, mod_features)`` modulation bus as the convolutional backbones (so
``global_cond`` and, later, the lambda-conditioning input work unchanged).
Intended for the scalar / 1-D calibration testbeds, not production spatial data.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import torch
from torch import nn

from autocast.nn.base import TemporalBackboneBase

# The base backbone rearranges a fixed ``w h`` layout, so this backbone supports
# exactly two spatial axes (e.g. ``(1, 1)`` scalar grids, ``(40, 1)`` 1-D rings).
_N_SPATIAL_AXES = 2


class _FiLMBlock(nn.Module):
    """LayerNorm + Linear with a FiLM(mod) affine, SiLU, residual (AdaLN-Zero).

    The modulation projection is zero-initialised, so the block is the identity
    on its input at construction time and only starts to use ``mod`` once
    training drives the projection away from zero. This AdaLN-Zero behaviour
    stabilises early training and lets the lambda-conditioning learn its
    effect gradually rather than perturbing a randomly-initialised forecaster.
    """

    def __init__(self, dim: int, mod_features: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.to_scale_shift = nn.Linear(mod_features, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, h: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(mod).chunk(2, dim=-1)
        modulated = self.norm(self.lin(h)) * (1.0 + scale) + shift
        return h + torch.nn.functional.silu(modulated)


class _ModMLP(nn.Module):
    """Inner module with the azula ``forward(x, mod, cond)`` contract.

    Flattens the channels-first ``(B, T*C, *spatial)`` input (and conditioning),
    runs a stack of FiLM-modulated residual blocks at a single hidden width, and
    reshapes the output back to channels-first ``(B, T_out*out_C, *spatial)`` so
    the base backbone's ``b (t c) w h`` rearrange round-trips. A single width is
    used because the residual FiLM blocks preserve their dimension; ``len(hidden)``
    sets the depth and ``hidden[0]`` the width.
    """

    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        out_channels_first: int,
        spatial_shape: Sequence[int],
        mod_features: int,
        hidden: Sequence[int],
    ) -> None:
        super().__init__()
        self.out_channels_first = out_channels_first
        self.spatial_shape = tuple(spatial_shape)
        width = hidden[0]
        out_dim = out_channels_first * prod(self.spatial_shape)
        self.proj_in = nn.Linear(in_dim + cond_dim, width)
        self.blocks = nn.ModuleList(_FiLMBlock(width, mod_features) for _ in hidden)
        self.proj_out = nn.Linear(width, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> torch.Tensor:
        b = x.shape[0]
        h = x.reshape(b, -1)
        if cond is not None:
            h = torch.cat([h, cond.reshape(b, -1)], dim=-1)
        h = self.proj_in(h)
        for block in self.blocks:
            h = block(h, mod)
        out = self.proj_out(h)
        return out.reshape(b, self.out_channels_first, *self.spatial_shape)


class TemporalMLPBackbone(TemporalBackboneBase):
    """FiLM-modulated MLP backbone for scalar / degenerate-grid inputs.

    Mirrors ``TemporalUNetBackbone``: it inherits the whole modulation bus
    (time embedding, global-cond add, the ``t=None`` zeros path, the precomputed
    ``(B, mod_features)`` path) from ``TemporalBackboneBase`` and only supplies
    the inner module via ``_build_backbone``. The base rearranges to channels-
    first ``(B, T*C, *spatial)`` before calling the inner module and rearranges
    back afterwards, so this backbone supports exactly the two-spatial-axis
    layouts the base hard-codes (e.g. ``(1, 1)`` scalar grids and ``(40, 1)``
    1-D rings).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        n_steps_output: int,
        n_steps_input: int,
        global_cond_channels: int | None,
        include_global_cond: bool,
        spatial_shape: Sequence[int],
        mod_features: int = 256,
        hidden: Sequence[int] = (256, 256),
        temporal_method: str = "none",
        use_precomputed_modulation: bool = False,
        include_time_embedding: bool = True,
        include_loss_weight: bool = False,
    ) -> None:
        """Build a TemporalMLPBackbone.

        Args mirror ``TemporalBackboneBase`` (channel/step counts, the
        modulation-bus flags) plus two MLP-specific fields:

        Args:
            in_channels: Per-step input channel count.
            out_channels: Per-step output channel count.
            cond_channels: Per-step conditioning channel count (``0`` disables
                the concatenated conditioning input).
            n_steps_output: Number of output time steps.
            n_steps_input: Number of input (context) time steps.
            global_cond_channels: Channel count of the optional global
                conditioning vector (``None`` when unused).
            include_global_cond: Whether to add the global-conditioning signal
                to the modulation bus.
            mod_features: Width of the FiLM modulation feature vector.
            temporal_method: How the temporal axis is handled (``"none"``
                flattens the steps into channels).
            use_precomputed_modulation: Whether ``forward`` receives a
                precomputed ``(B, mod_features)`` modulation instead of a scalar
                time.
            include_time_embedding: Whether to embed the flow/diffusion time
                into the modulation bus.
            include_loss_weight: Whether to emit the per-sample loss weight on
                the modulation bus.
            spatial_shape: The two-axis spatial grid ``(W, H)`` the inner MLP
                flattens over. Must have length 2 (the base backbone rearranges
                a ``w h`` layout); use ``(1, 1)`` for scalar SDEs and ``(40, 1)``
                for a 1-D ring.
            hidden: Residual FiLM-block widths. The blocks preserve their
                dimension, so ``len(hidden)`` sets the MLP depth and ``hidden[0]``
                its width; any further entries are ignored (kept as a sequence
                only to mirror the convolutional backbones' config shape).
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            n_steps_output=n_steps_output,
            n_steps_input=n_steps_input,
            mod_features=mod_features,
            global_cond_channels=global_cond_channels,
            include_global_cond=include_global_cond,
            temporal_method=temporal_method,
            use_precomputed_modulation=use_precomputed_modulation,
            include_time_embedding=include_time_embedding,
            include_loss_weight=include_loss_weight,
        )
        if len(hidden) < 1:
            msg = f"TemporalMLPBackbone requires len(hidden) >= 1 (got {len(hidden)})."
            raise ValueError(msg)
        if len(spatial_shape) != _N_SPATIAL_AXES:
            msg = (
                "TemporalMLPBackbone supports exactly 2 spatial axes (the base "
                "backbone rearranges 'w h'); got spatial_shape="
                f"{tuple(spatial_shape)}."
            )
            raise ValueError(msg)
        self.spatial_shape = tuple(spatial_shape)
        n_spatial = prod(self.spatial_shape)
        self._mlp = self._build_backbone(
            in_dim=in_channels * self.n_steps_output * n_spatial,
            cond_dim=cond_channels * n_steps_input * n_spatial,
            out_channels_first=out_channels * n_steps_output,
            spatial_shape=self.spatial_shape,
            mod_features=mod_features,
            hidden=hidden,
        )

    def _build_backbone(self, **kwargs) -> nn.Module:
        """Build the inner FiLM-MLP (abstract on ``TemporalBackboneBase``)."""
        return _ModMLP(
            in_dim=kwargs["in_dim"],
            cond_dim=kwargs["cond_dim"],
            out_channels_first=kwargs["out_channels_first"],
            spatial_shape=kwargs["spatial_shape"],
            mod_features=kwargs["mod_features"],
            hidden=kwargs["hidden"],
        )

    @property
    def backbone(self) -> nn.Module:
        return self._mlp
