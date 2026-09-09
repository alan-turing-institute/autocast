from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, cast

import torch
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class _FlowTeacher(Protocol):
    """The flow-matching teacher interface the consistency student consumes."""

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


class ConsistencyDistilledProcessor(Processor):
    r"""Small-step consistency-distilled student of a flow-matching teacher.

    The sibling of :class:`DistilledProcessor`. Where the one-pass distilled
    student regresses directly onto the teacher's *full* noise-to-data endpoint
    (``g(z, x) -> ODE_teacher(z | x)``), this student imposes the **self-consistency**
    property of consistency models [Song et al. 2023] over the teacher's
    probability-flow ODE: a single time-conditioned map ``f(z, t | x)`` is trained
    so that any two points on the *same* teacher trajectory map to the same
    data endpoint, learned one *small step* at a time rather than from the whole
    integrated trajectory.

    Concretely, with the conditional flow-matching path running from ``t = 0``
    (Gaussian noise) to ``t = 1`` (data) and the teacher velocity ``v(z, t | x)``:

      * a forward-marginal point ``z_t = (1 - t) z_0 + t y`` is sampled at a random
        grid time ``t_n`` (``y`` the true next state, ``z_0`` fresh noise);
      * the teacher takes ONE Euler step toward the data,
        ``ẑ_{t_{n+1}} = z_t + (t_{n+1} - t_n)\, v(z_t, t_n | x)``;
      * the student is trained to be consistent across that small step,
        ``f(z_t, t_n | x) \approx \mathrm{stopgrad}\, f(ẑ_{t_{n+1}}, t_{n+1} | x)``.

    The target is the SAME network evaluated at the adjacent, closer-to-data time
    with the gradient blocked (the modern no-EMA-target variant; the teacher, not an
    EMA copy, supplies the one solver step that makes the closer-to-data point more
    accurate). The map is parameterised as ``f(z, t | x) = z + (1 - t) F(z, t | x)``
    so the boundary condition ``f(z, 1 | x) = z`` (identity at the data end) holds by
    construction; ``F`` is a time-conditioned network.

    Inference (:meth:`map`) is a single network pass per member — draw ``z_0`` and
    return ``f(z_0, 0 | x)`` — so, like the one-pass distilled student, this is a
    cheap one-forward-pass generator, but trained by self-consistency over small
    steps rather than by whole-trajectory endpoint matching. The teacher is
    *injected* (:meth:`set_teacher`) for training only and is not a submodule, so
    it never enters the student checkpoint. Operates on the flattened field.
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
        n_consistency_steps: int = 18,
        n_time_frequencies: int = 6,
        boundary_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.n_channels_in = int(n_channels_in)
        self.n_channels_out = int(n_channels_out)
        self.spatial_shape = tuple(int(s) for s in spatial_shape)
        self.global_cond_features = int(global_cond_features)
        # Number of equal time sub-intervals on [0, 1]; the small step is 1/N. A
        # finer grid lowers the per-step discretisation error of the consistency
        # target at the cost of a smaller learning signal per step.
        self.n_consistency_steps = int(n_consistency_steps)
        self.n_time_frequencies = int(n_time_frequencies)
        # Sampling tilt toward the noise boundary (the deployed t=0 slice); 1.0 =
        # uniform over steps, larger concentrates supervision near t=0.
        self.boundary_bias = float(boundary_bias)

        spatial = int(math.prod(self.spatial_shape)) if self.spatial_shape else 1
        self.spatial = spatial
        self.features = self.n_steps_output * spatial * self.n_channels_out
        self.context = (
            self.n_steps_input * spatial * self.n_channels_in
            + self.global_cond_features
        )
        # Time conditioning: t itself plus sin/cos Fourier features at geometric
        # frequencies, concatenated to the flattened state + context.
        self.n_time_features = 1 + 2 * self.n_time_frequencies
        self.net = _mlp(
            self.features + self.context + self.n_time_features,
            self.features,
            tuple(int(h) for h in hidden_features),
        )
        # Teacher injected for training only (not a submodule -> excluded from the
        # checkpoint, so scoring loads the student alone).
        self._teacher: list[_FlowTeacher] = []

    def set_teacher(self, teacher: Processor) -> None:
        """Attach a frozen flow-matching teacher for consistency distillation."""
        if not callable(getattr(teacher, "flow_field", None)):
            msg = (
                "ConsistencyDistilledProcessor requires a flow-matching-capable "
                "teacher exposing a flow_field(z, t, x, global_cond) method; got "
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

    def _time_features(self, t: Tensor) -> Tensor:
        """Sin/cos Fourier features of t (shape (B,)) -> (B, n_time_features)."""
        feats = [t.unsqueeze(-1)]
        for k in range(self.n_time_frequencies):
            w = math.pi * (2.0**k)
            feats += [torch.sin(w * t).unsqueeze(-1), torch.cos(w * t).unsqueeze(-1)]
        return torch.cat(feats, dim=-1)

    @staticmethod
    def _bcast(t: Tensor, ref: Tensor) -> Tensor:
        """Reshape a per-sample time (B,) to broadcast against a field tensor ref."""
        return t.reshape(t.shape[0], *([1] * (ref.ndim - 1)))

    def _consistency_map(self, z: Tensor, t: Tensor, ctx: Tensor) -> Tensor:
        """Boundary-correct consistency map f(z, t | x) = z + (1 - t) F(z, t | x).

        z is a field (B, T_out, *spatial, C_out); t is (B,); ctx is the flattened
        conditioning. At t = 1 this returns z exactly (the data-end identity).
        """
        z_flat = z.reshape(z.shape[0], -1)
        inp = torch.cat([z_flat, ctx, self._time_features(t)], dim=-1)
        f = self.net(inp).reshape(self._z_shape(z.shape[0]))
        return z + (1.0 - self._bcast(t, z)) * f

    # -- Processor API --------------------------------------------------------
    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        return self.map(x, global_cond)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Draw one member in a single pass: f(z, 0 | x), z ~ N(0, I)."""
        ctx = self._context(x, global_cond)
        z = torch.randn(self._z_shape(x.shape[0]), device=x.device, dtype=x.dtype)
        t0 = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return self._consistency_map(z, t0, ctx)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Consistency-distillation loss over one small teacher ODE step."""
        if not self._teacher:
            msg = (
                "ConsistencyDistilledProcessor.loss requires a teacher; call "
                "set_teacher(...) before training."
            )
            raise RuntimeError(msg)
        teacher = self._teacher[0]
        x = batch.encoded_inputs
        y = batch.encoded_output_fields
        gc = batch.global_cond
        ctx = self._context(x, gc)
        b = x.shape[0]
        device, dtype = x.device, x.dtype

        # Grid step n in {0, ..., N-1}: t_n -> t_{n+1} = t_n + 1/N, one small step
        # toward the data end (t = 1). The noise-boundary slice (small n, t ~ 0) is
        # the one the one-pass inference draw f(z0, 0 | x) actually reads, yet under
        # uniform sampling it is the rarest and hardest to learn — which leaves the
        # deployed spread under-supervised and over-dispersed. `boundary_bias` >= 1
        # tilts the sampling toward small n (u**bias concentrates a uniform draw
        # near 0; bias = 1 recovers uniform), supervising the deployed slice more.
        n_steps = self.n_consistency_steps
        u = torch.rand(b, device=device)
        n = (u.pow(self.boundary_bias) * n_steps).long().clamp_(max=n_steps - 1)
        tn = (n.to(dtype)) / n_steps
        tnp1 = (n.to(dtype) + 1.0) / n_steps

        # Forward-marginal point at t_n on the conditional straight-line path.
        z0 = torch.randn_like(y)
        zt = (1.0 - self._bcast(tn, y)) * z0 + self._bcast(tn, y) * y

        # One teacher Euler step toward the data (t_n -> t_{n+1}).
        with torch.no_grad():
            v = teacher.flow_field(zt, tn, x, gc)
            zt_next = zt + self._bcast(tnp1 - tn, zt) * v

        # Self-consistency: the noisier point (with grad) matches the closer-to-data
        # point (stop-grad), which the teacher step made more accurate.
        pred = self._consistency_map(zt, tn, ctx)
        with torch.no_grad():
            target = self._consistency_map(zt_next, tnp1, ctx)
        return torch.mean((pred - target) ** 2)
