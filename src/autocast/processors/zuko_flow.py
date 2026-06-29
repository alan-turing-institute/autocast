from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import zuko

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor

# flow_type -> zuko flow family. Coupling flows sample in a single network pass
# (one-forward); autoregressive flows sample in D sequential passes. All are
# trained on the EXACT log-likelihood (the logarithmic score).
_FLOW_CLASSES = {
    # coupling (one-forward sampling)
    "realnvp": zuko.flows.RealNVP,  # affine coupling
    "nice": zuko.flows.NICE,  # additive coupling
    "ncsf": zuko.flows.NCSF,  # neural coupling spline
    # autoregressive (expressive; D sequential sampling passes)
    "maf": zuko.flows.MAF,  # masked autoregressive (affine)
    "nsf": zuko.flows.NSF,  # masked autoregressive neural spline
}


class ZukoFlowProcessor(Processor):
    """Conditional normalizing-flow processor backed by the ``zuko`` library.

    Models the one-step predictive density ``p(x_{t+1} | x_t)`` exactly with a
    ``zuko`` flow over the *flattened* field, conditioned on the flattened
    current state. Training minimises the negative log-likelihood (the strictly
    proper logarithmic score, computed exactly via the change-of-variables
    rule). Each :meth:`map` call draws one ensemble member with a single
    ``distribution.sample()`` — one network pass for a coupling flow,
    ``features`` sequential passes for an autoregressive flow; ensembles are
    produced by calling :meth:`map` repeatedly.

    This is the generic, library-backed flow: ``flow_type`` selects a ``zuko``
    family — coupling (``realnvp`` / ``nice`` / ``ncsf``, one-pass sampling) or
    masked-autoregressive (``maf`` / ``nsf``, more expressive, including neural
    spline flows). Because it flattens the field it makes no use of spatial
    structure, so it suits scalar or small fields; the spatially structured
    flows (the convolutional-coupling and patch-autoregressive processors) are
    the bespoke alternatives for larger 2D fields.
    """

    def __init__(
        self,
        *,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        spatial_shape: Sequence[int] = (1, 1),
        global_cond_features: int = 0,
        flow_type: str = "nsf",
        transforms: int = 4,
        hidden_features: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if flow_type not in _FLOW_CLASSES:
            msg = (
                f"flow_type must be one of {sorted(_FLOW_CLASSES)}; got {flow_type!r}."
            )
            raise ValueError(msg)
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.n_channels_in = int(n_channels_in)
        self.n_channels_out = int(n_channels_out)
        self.spatial_shape = tuple(int(s) for s in spatial_shape)
        self.global_cond_features = int(global_cond_features)
        self.flow_type = flow_type

        spatial = int(math.prod(self.spatial_shape)) if self.spatial_shape else 1
        # target dimension D = T_out * spatial * C_out (the field, flattened)
        self.features = self.n_steps_output * spatial * self.n_channels_out
        # conditioning dimension = flattened current state (+ optional global cond)
        self.context = (
            self.n_steps_input * spatial * self.n_channels_in
            + self.global_cond_features
        )

        self.flow = _FLOW_CLASSES[flow_type](
            features=self.features,
            context=self.context,
            transforms=int(transforms),
            hidden_features=tuple(int(h) for h in hidden_features),
        )

    def _context(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Flatten the current state (and optional global cond) to (B, context)."""
        ctx = x.reshape(x.shape[0], -1)
        if global_cond is not None and self.global_cond_features > 0:
            ctx = torch.cat(
                [ctx, global_cond.reshape(global_cond.shape[0], -1)], dim=-1
            )
        return ctx

    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Alias to map for Lightning/PyTorch compatibility."""
        return self.map(x, global_cond)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Draw one stochastic next-state member, shape (B, T_out, *spatial, C_out)."""
        b = x.shape[0]
        ctx = self._context(x, global_cond)
        sample = self.flow(ctx).sample()  # (B, features)
        return sample.reshape(
            b, self.n_steps_output, *self.spatial_shape, self.n_channels_out
        )

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Exact negative log-likelihood of the next state given the current."""
        target = batch.encoded_output_fields
        if (
            target.shape[1] != self.n_steps_output
            or target.shape[-1] != self.n_channels_out
        ):
            msg = (
                f"Target shape {tuple(target.shape)} does not match configured "
                f"n_steps_output={self.n_steps_output}, "
                f"n_channels_out={self.n_channels_out}."
            )
            raise ValueError(msg)
        ctx = self._context(batch.encoded_inputs, batch.global_cond)
        y = target.reshape(target.shape[0], -1)  # (B, features)
        log_prob = self.flow(ctx).log_prob(y)  # (B,)
        return -log_prob.mean()
