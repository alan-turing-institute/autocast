"""One-shot supervised ViT processor.

Takes encoded past observations (no noise, no diffusion ODE) and directly
regresses future states with MSE loss.  Intended for non-autoregressive
multi-step forecasting (e.g. 5 → 14 days).

Tensor layout follows the rest of the repo:
  encoded_inputs         : (B, T_in,  *spatial, C_in)   channels-last
  encoded_output_fields  : (B, T_out, *spatial, C_out)  channels-last
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from einops import rearrange, repeat
from torch import nn

from autocast.processors.base import Processor
from autocast.processors.vit import AViT
from autocast.types.batch import EncodedBatch
from autocast.types.types import Tensor


class SupervisedViTProcessor(Processor[EncodedBatch]):
    """Supervised one-shot ViT forecasting processor.

    Flattens T_in input timesteps (+ optional global conditioning broadcast
    to each spatial location) into (B, *spatial, T_in * C_in) channels,
    runs them through an AViT, then unfolds the output back to
    (B, T_out, *spatial, C_out).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_steps_input: int,
        n_steps_output: int,
        spatial_resolution: Sequence[int],
        global_cond_channels: int = 0,
        include_global_cond: bool = False,
        hidden_dim: int = 256,
        num_heads: int = 8,
        n_layers: int = 8,
        drop_path: float = 0.1,
        groups: int = 8,
        patch_size: int | None = None,
        loss_func: nn.Module | None = None,
        # unused by base Processor but accepted so _build_processor filter keeps them
        n_channels_out: int | None = None,
    ) -> None:
        super().__init__()

        self.n_spatial_dims = len(spatial_resolution)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.global_cond_channels = global_cond_channels
        self.include_global_cond = include_global_cond

        # How many channels feed into the ViT
        per_step = in_channels + (global_cond_channels if include_global_cond else 0)
        vit_in_channels = per_step * n_steps_input
        vit_out_channels = out_channels * n_steps_output

        self.model = AViT(
            dim_in=vit_in_channels,
            dim_out=vit_out_channels,
            n_spatial_dims=self.n_spatial_dims,
            spatial_resolution=list(spatial_resolution),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            processor_blocks=n_layers,
            drop_path=drop_path,
            groups=groups,
            n_noise_channels=None,
            patch_size=patch_size,
        )

        self.loss_func = loss_func if loss_func is not None else nn.MSELoss()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten_input(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Flatten time steps (+ optional global cond) into channel dim.

        Args:
            x:           (B, T_in, *spatial, C_in)
            global_cond: (B, D) or None

        Returns
        -------
            (B, *spatial, T_in * (C_in [+ D]))
        """
        if self.include_global_cond and global_cond is not None:
            _, t, *spatial, _ = x.shape
            if self.n_spatial_dims == 2:
                cond = repeat(
                    global_cond,
                    "b d -> b t w h d",
                    t=t,
                    w=spatial[0],
                    h=spatial[1],
                )
            elif self.n_spatial_dims == 3:
                cond = repeat(
                    global_cond,
                    "b d -> b t w h depth d",
                    t=t,
                    w=spatial[0],
                    h=spatial[1],
                    depth=spatial[2],
                )
            else:
                raise ValueError(f"Unsupported n_spatial_dims={self.n_spatial_dims}")
            x = torch.cat([x, cond], dim=-1)

        # (B, T, *spatial, C) -> (B, *spatial, T*C)
        return rearrange(x, "b t ... c -> b ... (t c)")

    def _unfold_output(self, y: Tensor) -> Tensor:
        """Unfold flattened ViT output channels back to time dimension.

        Args:
            y: (B, *spatial, T_out * C_out)

        Returns
        -------
            (B, T_out, *spatial, C_out)
        """
        return rearrange(
            y,
            "b ... (t c) -> b t ... c",
            t=self.n_steps_output,
            c=self.out_channels,
        )

    # ------------------------------------------------------------------
    # Processor interface
    # ------------------------------------------------------------------

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Forward inference: (B,T_in,*S,C_in) -> (B,T_out,*S,C_out)."""
        x_flat = self._flatten_input(x, global_cond)
        y_flat = self.model(x_flat, x_noise=None)
        return self._unfold_output(y_flat)

    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        return self.map(x, global_cond)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """MSE loss between predicted and target future states."""
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)
