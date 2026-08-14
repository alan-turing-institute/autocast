from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, cast

import torch
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class _FlowTeacher(Protocol):
    """The flow-matching teacher interface the distilled student consumes."""

    flow_ode_steps: int

    def flow_field(
        self, z: Tensor, t: Tensor, x: Tensor, global_cond: Tensor | None
    ) -> Tensor: ...


def _mlp(in_features: int, out_features: int, hidden: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


class DistilledProcessor(Processor):
    """One-pass student that *sample-matches* a flow-matching teacher.

    The student is a deterministic map ``g(z, x_t)`` from an input noise draw
    ``z`` (the same shape as the teacher's latent noise) and the current state
    ``x_t`` to a next-state sample. It is trained by **noise-paired**
    distillation: for each ``z`` the teacher's ODE is integrated from that exact
    ``z`` to its endpoint, and the student regresses to it. Because the teacher's
    flow is a deterministic, invertible transport, matching the noise->sample map
    pointwise reproduces the teacher's *whole* predictive distribution — the
    spread is inherited per toy, not set by a fixed noise hyperparameter. This is
    sample-matching, never endpoint/mean-matching (which would collapse to a
    deterministic forecaster with zero spread).

    Inference (:meth:`map`) is a single network pass per ensemble member, so the
    student is the cheap "can we keep the teacher's UQ at one-pass cost?" arm.
    The teacher is *injected* (:meth:`set_teacher`) only for training and is not
    a submodule, so it never enters the student checkpoint and inference needs no
    teacher. Operates on the flattened field, so it applies to every toy.
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
        hidden_features: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.n_channels_in = int(n_channels_in)
        self.n_channels_out = int(n_channels_out)
        self.spatial_shape = tuple(int(s) for s in spatial_shape)
        self.global_cond_features = int(global_cond_features)

        spatial = int(math.prod(self.spatial_shape)) if self.spatial_shape else 1
        self.spatial = spatial
        self.features = self.n_steps_output * spatial * self.n_channels_out
        self.context = (
            self.n_steps_input * spatial * self.n_channels_in
            + self.global_cond_features
        )
        self.student = _mlp(
            self.features + self.context,
            self.features,
            tuple(int(h) for h in hidden_features),
        )
        # Teacher is injected for training only (not a submodule -> excluded from
        # the checkpoint, so scoring loads the student alone).
        self._teacher: list[_FlowTeacher] = []

    def set_teacher(self, teacher: Processor) -> None:
        """Attach a frozen flow-matching teacher for distillation training."""
        if not callable(getattr(teacher, "flow_field", None)):
            msg = (
                "DistilledProcessor requires a flow-matching-capable teacher "
                "exposing a flow_field(z, t, x, global_cond) method; got "
                f"{type(teacher).__name__}."
            )
            raise TypeError(msg)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self._teacher = [cast(_FlowTeacher, teacher)]

    # -- layout helpers -------------------------------------------------------
    def _context(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        ctx = x.reshape(x.shape[0], -1)
        if global_cond is not None and self.global_cond_features > 0:
            ctx = torch.cat(
                [ctx, global_cond.reshape(global_cond.shape[0], -1)], dim=-1
            )
        return ctx

    def _z_shape(self, b: int) -> tuple[int, ...]:
        return (b, self.n_steps_output, *self.spatial_shape, self.n_channels_out)

    def _generate(self, z_flat: Tensor, ctx: Tensor) -> Tensor:
        """Student forward: (z_flat, ctx) -> next-state field (B, T, *spatial, C)."""
        out = self.student(torch.cat([z_flat, ctx], dim=-1))
        return out.reshape(self._z_shape(out.shape[0]))

    # -- teacher integration (noise-paired endpoint) --------------------------
    @torch.no_grad()
    def _teacher_endpoint(
        self, z: Tensor, x: Tensor, global_cond: Tensor | None
    ) -> Tensor:
        """Integrate the teacher's flow ODE from a *given* noise z to its sample."""
        teacher = self._teacher[0]
        steps = max(int(teacher.flow_ode_steps), 1)
        integrator = getattr(teacher, "integrator", "euler")
        dt = torch.tensor(1.0 / steps, device=z.device, dtype=z.dtype)
        t = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for _ in range(steps):
            if integrator == "heun":
                k1 = teacher.flow_field(z, t, x, global_cond)
                k2 = teacher.flow_field(z + dt * k1, t + dt, x, global_cond)
                z = z + dt * 0.5 * (k1 + k2)
            else:
                z = z + dt * teacher.flow_field(z, t, x, global_cond)
            t = t + dt
        return z

    # -- Processor API --------------------------------------------------------
    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        return self.map(x, global_cond)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Draw one member in a single pass: g(z, x_t), z ~ N(0, I)."""
        ctx = self._context(x, global_cond)
        z = torch.randn(self._z_shape(x.shape[0]), device=x.device, dtype=x.dtype)
        return self._generate(z.reshape(z.shape[0], -1), ctx)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Noise-paired sample-matching distillation loss against the teacher."""
        if not self._teacher:
            msg = (
                "DistilledProcessor.loss requires a teacher; call set_teacher(...) "
                "before training."
            )
            raise RuntimeError(msg)
        x = batch.encoded_inputs
        ctx = self._context(x, batch.global_cond)
        z = torch.randn(self._z_shape(x.shape[0]), device=x.device, dtype=x.dtype)
        target = self._teacher_endpoint(z, x, batch.global_cond)
        pred = self._generate(z.reshape(z.shape[0], -1), ctx)
        return torch.mean((pred - target) ** 2)
