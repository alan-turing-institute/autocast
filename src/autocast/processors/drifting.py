"""Drifting processor algorithm core.

PyTorch port of the JAX reference implementation of the drift-field loss from
"Generative Modeling via Drifting" (Deng et al. 2026, arXiv:2602.04770).
Reference: https://github.com/lambertae/drifting (drift_loss.py, commit
c8b4fee).

Inherits the cohort-inflation forward path from ``OneStepCohortProcessor``;
this module owns the drift-field math (``_compute_drift_field`` and helpers)
and the ``_compute_cohort_loss`` override.
"""

# ruff: noqa: F722 — jaxtyping shape strings (Float[Tensor, "batch n_gen feature"])
# are nested inside the forward-annotation string under `from __future__ import
# annotations`; F722 flags the inner shape as if it were a Python expression.
from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
from jaxtyping import Float
from torch import nn
from torch.nn import functional as F

from autocast.processors.one_step_cohort import OneStepCohortProcessor
from autocast.types import Tensor


def _pairwise_distances(
    x: Float[Tensor, "batch n_x feature"],
    y: Float[Tensor, "batch n_y feature"],
    *,
    eps: float = 1e-8,
) -> Float[Tensor, "batch n_x n_y"]:
    """Pairwise Euclidean distances with a soft-floor on the squared form.

    Reimplemented rather than calling ``torch.cdist`` to match the JAX
    reference's ``sqrt(clamp(sq_dist, min=eps))`` form, which gives a
    well-defined backward at ``d == 0`` and avoids sqrt(0)-NaN gradients.

    Args:
        x: Query tensor of shape ``(B, N_x, D)``.
        y: Key tensor of shape ``(B, N_y, D)``.
        eps: Soft floor on the squared distance before the sqrt.

    Returns:
        Pairwise distance tensor of shape ``(B, N_x, N_y)``.
    """
    sq_dist = (
        (x * x).sum(dim=-1, keepdim=True)
        + (y * y).sum(dim=-1, keepdim=True).transpose(-1, -2)
        - 2.0 * torch.einsum("bnd,bmd->bnm", x, y)
    )
    return torch.sqrt(sq_dist.clamp_min(eps))


def _per_tau_force_field(
    *,
    dist_normed: Float[Tensor, "batch n_gen n_target"],
    targets_scaled: Float[Tensor, "batch n_target feature"],
    gen_scaled: Float[Tensor, "batch n_gen feature"],
    target_weights: Float[Tensor, "batch n_target"],
    n_neg_total: int,
    tau: float,
) -> tuple[Float[Tensor, "batch n_gen feature"], Float[Tensor, ""]]:
    """Compute the L2-normalised drift force at one kernel temperature.

    Args:
        dist_normed: Normalised pairwise distances, shape ``(B, N_gen, N_t)``.
        targets_scaled: Targets in scaled feature space, shape ``(B, N_t, D)``.
        gen_scaled: Trainable cohort in scaled feature space, shape
            ``(B, N_gen, D)``.
        target_weights: Per-target weights, shape ``(B, N_t)``.
        n_neg_total: Total negative-side slab width (gen + fixed_neg).
        tau: Kernel temperature for this scale.

    Returns:
        ``(force_l2, force_rms_sq)``: the per-tau force normalised to unit
        root-mean-square, and the pre-normalisation mean squared force
        magnitude (logged in ``info`` by the caller).
    """
    logits = -dist_normed / tau
    aff_row = torch.softmax(logits, dim=-1)
    aff_col = torch.softmax(logits, dim=-2)
    # 1e-6 floor: sqrt-stability guard on the geometric-mean affinity; matches
    # the JAX reference's clamp on the inner product before the sqrt.
    aff = torch.sqrt((aff_row * aff_col).clamp_min(1e-6))
    aff = aff * target_weights[:, None, :]

    aff_neg = aff[..., :n_neg_total]
    aff_pos = aff[..., n_neg_total:]
    coeff_neg = -aff_neg * aff_pos.sum(dim=-1, keepdim=True)
    coeff_pos = aff_pos * aff_neg.sum(dim=-1, keepdim=True)
    coeff = torch.cat([coeff_neg, coeff_pos], dim=-1)

    force_tau = torch.einsum("bny,byd->bnd", coeff, targets_scaled)
    force_tau = force_tau - coeff.sum(dim=-1, keepdim=True) * gen_scaled

    force_rms_sq = (force_tau**2).mean()
    # 1e-8 floor: RMS divide-by-zero guard for the L2 normaliser.
    return force_tau / force_rms_sq.clamp_min(1e-8).sqrt(), force_rms_sq


def _compute_drift_field(
    gen: Float[Tensor, "batch n_gen feature"],
    fixed_pos: Float[Tensor, "batch n_pos feature"],
    fixed_neg: Float[Tensor, "batch n_neg feature"] | None = None,
    *,
    tau_list: Sequence[float] = (0.02, 0.05, 0.2),
    weight_gen: Float[Tensor, "batch n_gen"] | None = None,
    weight_pos: Float[Tensor, "batch n_pos"] | None = None,
    weight_neg: Float[Tensor, "batch n_neg"] | None = None,
    diag_mask_value: float = 100.0,
    eps: float = 1e-3,
) -> tuple[
    Float[Tensor, "batch n_gen feature"],
    Float[Tensor, "batch n_gen feature"],
    dict[str, Tensor],
]:
    """Compute the kernelised contrastive drift field, JAX-faithful port.

    The drift moves each generated sample toward the positives and away from
    the negatives (the cohort detached + any extra fixed negatives), computed
    at multiple kernel temperatures and aggregated by per-scale L2
    normalisation. All arithmetic runs in fp32 for numerical stability of
    ``exp(-d / tau)`` with small ``tau``; the output stays in fp32 (the
    caller squares-and-reduces).

    The target buffer is structured as three slabs
    ``[gen_detached, fixed_neg, fixed_pos]``, with diagonal masking only on
    the leading gen-vs-gen block — matches the reference implementation
    exactly. Forward-compatible with the v2 memory-bank backlog (where
    ``fixed_neg`` becomes non-empty).

    Args:
        gen: Trainable cohort of generated samples of shape ``(B, N_gen, D)``.
        fixed_pos: Positives of shape ``(B, N_pos, D)``. In v1, ``N_pos = 1``
            per design — the unique ground truth for each cohort.
        fixed_neg: Extra fixed negatives (e.g. memory-bank samples) of shape
            ``(B, N_neg, D)``, or ``None`` (the v1 default).
        tau_list: Kernel temperatures. Default ``(0.02, 0.05, 0.2)`` matches
            the reference's library default.
        weight_gen: Per-sample weights for the cohort slab, shape
            ``(B, N_gen)``. Defaults to ones.
        weight_pos: Same for ``fixed_pos``, shape ``(B, N_pos)``.
        weight_neg: Same for ``fixed_neg``, shape ``(B, N_neg)``.
        diag_mask_value: Value added to the gen-vs-gen diagonal of the
            distance matrix to mask out self-affinity. Default 100 matches
            the reference.
        eps: Soft floor on the distance/feature scale denominators. Default
            ``1e-3`` matches the reference.

    Returns:
        A tuple ``(goal_scaled, gen_scaled, info)``:

        - ``goal_scaled`` of shape ``(B, N_gen, D)`` — the regression target
          in the scaled feature space, detached.
        - ``gen_scaled`` of shape ``(B, N_gen, D)`` — the trainable cohort
          rescaled into the same space; carries the live gradient through
          ``gen`` (the scale factor is detached, so the gradient is the
          clean ``1 / scale_inputs``).
        - ``info`` carrying per-scale diagnostics: ``"scale"`` (the
          distance-normalisation scalar) and ``"loss_<tau>"`` (the
          pre-normalisation mean squared force at each temperature).
    """
    # fp32 cast for numerical stability of exp(-d / tau) at small tau.
    gen_f32 = gen.to(torch.float32)
    fixed_pos_f32 = fixed_pos.to(torch.float32)
    batch_size, n_gen, feature_dim = gen_f32.shape
    fixed_neg_f32 = (
        gen_f32.new_zeros((batch_size, 0, feature_dim))
        if fixed_neg is None
        else fixed_neg.to(torch.float32)
    )
    weight_gen = (
        gen_f32.new_ones(gen_f32.shape[:-1])
        if weight_gen is None
        else weight_gen.to(torch.float32)
    )
    weight_pos = (
        gen_f32.new_ones(fixed_pos_f32.shape[:-1])
        if weight_pos is None
        else weight_pos.to(torch.float32)
    )
    weight_neg = (
        gen_f32.new_ones(fixed_neg_f32.shape[:-1])
        if weight_neg is None
        else weight_neg.to(torch.float32)
    )

    # Three-slab target buffer: [gen_detached, fixed_neg, fixed_pos].
    gen_detached = gen_f32.detach()
    targets = torch.cat([gen_detached, fixed_neg_f32, fixed_pos_f32], dim=1)
    target_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    dist = _pairwise_distances(gen_f32, targets)

    # Normalisation stats — detached so they act as constants downstream
    # (matches JAX's stop_gradient at the calculate-scaled-goal boundary).
    scale = (dist * target_weights[:, None, :]).mean() / target_weights.mean()
    scale = scale.detach()
    scale_inputs = (scale / (feature_dim**0.5)).clamp_min(eps)
    scale_clamped = scale.clamp_min(eps)

    gen_scaled = gen_f32 / scale_inputs
    targets_scaled = targets / scale_inputs
    dist_normed = dist / scale_clamped

    # Diagonal mask on the leading gen-vs-gen block ONLY. F.pad takes pads in
    # REVERSE axis order: (last_left, last_right, second_left, second_right).
    n_targets = dist_normed.shape[-1]
    eye_block = (
        torch.eye(n_gen, dtype=dist_normed.dtype, device=gen_f32.device)
        * diag_mask_value
    )
    eye_padded = F.pad(eye_block, (0, n_targets - n_gen, 0, 0))
    dist_normed = dist_normed + eye_padded[None, :, :]

    n_neg_total = n_gen + fixed_neg_f32.shape[1]
    info: dict[str, Tensor] = {"scale": scale}
    force_total = torch.zeros_like(gen_scaled)
    for tau in tau_list:
        force_l2, force_rms_sq = _per_tau_force_field(
            dist_normed=dist_normed,
            targets_scaled=targets_scaled,
            gen_scaled=gen_scaled,
            target_weights=target_weights,
            n_neg_total=n_neg_total,
            tau=tau,
        )
        info[f"loss_{tau}"] = force_rms_sq.detach()
        force_total = force_total + force_l2

    # Drift target is detached: the regression loss in the caller computes
    # ((gen_scaled - goal_scaled) ** 2).mean() and the gradient flows through
    # gen_scaled only.
    goal_scaled = (gen_scaled + force_total).detach()
    return goal_scaled, gen_scaled, info


class DriftingProcessor(OneStepCohortProcessor):
    """One-step generative processor trained via the drifting field loss.

    Implements the algorithm of Deng et al. 2026 (arXiv:2602.04770) as a
    drop-in autocast ``Processor``. Inherits the cohort-inflation forward
    path from ``OneStepCohortProcessor``; this class only specifies the
    extra hyperparameters (``tau_list``, ``diag_mask_value``) and the
    drift-field loss body.

    Note:
        With ``n_pos=1`` (the v1 default — the unique ground truth is the
        only positive per input), the positive-side dual softmax degenerates
        to a constant pull toward that single target; only the
        *negative-side* repulsion across the same-input cohort distinguishes
        drifting from plain MSE-toward-target. The ``MSECohortProcessor``
        ablation isolates that contribution.
    """

    _MIN_N_SAMPLES: ClassVar[int] = 2
    _MIN_N_SAMPLES_REASON: ClassVar[str] = (
        "cohort sizes below 2 collapse the negative-side dual softmax to a "
        "trivial self-affinity term"
    )

    def __init__(
        self,
        *,
        backbone: nn.Module,
        n_steps_output: int,
        n_channels_out: int,
        n_samples: int = 8,
        tau_list: Sequence[float] = (0.02, 0.05, 0.2),
        diag_mask_value: float = 100.0,
    ) -> None:
        """Build a DriftingProcessor.

        Args:
            backbone: One-step generator backbone. MUST be constructed with
                ``include_time_embedding=False`` (drifting has no
                integration time ``t``).
            n_steps_output: Number of output time steps per sample. Required;
                configs pass ``auto`` and let ``setup.py`` resolve from the
                datamodule's output shape.
            n_channels_out: Number of output channels per sample. Required;
                resolved via ``auto`` as for ``n_steps_output``.
            n_samples: Cohort size per input. Must be >= 2 (a cohort of one
                produces a trivially zero negative-side dual softmax;
                contrastive drift needs >= 2). Default 8 matches the
                reference's library default.
            tau_list: Kernel temperatures for the multi-scale aggregation.
                Default ``(0.02, 0.05, 0.2)`` matches the reference.
            diag_mask_value: Value added to the gen-vs-gen diagonal of the
                distance matrix to mask self-affinity.
        """
        if len(tau_list) < 1:
            msg = (
                f"DriftingProcessor requires len(tau_list) >= 1 (got {len(tau_list)})."
            )
            raise ValueError(msg)

        super().__init__(
            backbone=backbone,
            n_steps_output=n_steps_output,
            n_channels_out=n_channels_out,
            n_samples=n_samples,
        )

        self.tau_list = tuple(tau_list)
        self.diag_mask_value = diag_mask_value

    def _compute_cohort_loss(
        self,
        gen_b: Tensor,
        target_b: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the drift-field loss and per-scale force-RMS diagnostics.

        With ``n_pos = 1``, ``target_b[:, :1, :]`` recovers the unique
        positive per input (the K target copies along the middle axis are
        identical under repeat-interleave). The negative side comes
        implicitly from the gen-vs-gen block of the target buffer inside
        ``_compute_drift_field``.

        Args:
            gen_b: Cohort-flattened generator output.
            target_b: Cohort-flattened target.

        Returns:
            ``(loss, extra_diagnostics)`` per the
            ``OneStepCohortProcessor`` contract.
        """
        pos_b = target_b[:, :1, :]
        goal_scaled, gen_scaled, info = _compute_drift_field(
            gen_b,
            pos_b,
            tau_list=self.tau_list,
            diag_mask_value=self.diag_mask_value,
        )
        loss = ((gen_scaled - goal_scaled) ** 2).mean()
        extras: dict[str, Tensor] = {
            f"force_rms_tau{tau}": info[f"loss_{tau}"] for tau in self.tau_list
        }
        return loss, extras
