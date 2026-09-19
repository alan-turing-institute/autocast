"""Drifting processor algorithm core.

PyTorch port of the JAX reference implementation of the drift-field loss from
"Generative Modeling via Drifting" (Deng et al. 2026, arXiv:2602.04770).
Reference: https://github.com/lambertae/drifting (drift_loss.py, commit
c8b4fee).

Inherits the cohort-inflation forward path from
``OneStepCohortProcessor``; this module owns the drift-field math
(``_compute_drift_field`` and helpers) and the ``_compute_cohort_loss``
override.
"""

# ruff: noqa: F722 — jaxtyping shape strings (Float[Tensor, "batch n_gen feature"])
# are nested inside the forward-annotation string under `from __future__ import
# annotations`; F722 flags the inner shape as if it were a Python expression.
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

import torch
from jaxtyping import Float
from torch import nn
from torch.nn import functional as F

from autocast.metrics.ensemble import _energy_score_terms
from autocast.processors.one_step_cohort import BaseNoise, OneStepCohortProcessor
from autocast.types import EncodedBatch, Tensor


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
    -------
        Pairwise distance tensor of shape ``(B, N_x, N_y)``.
    """
    sq_dist = (
        (x * x).sum(dim=-1, keepdim=True)
        + (y * y).sum(dim=-1, keepdim=True).transpose(-1, -2)
        - 2.0 * torch.einsum("bnd,bmd->bnm", x, y)
    )
    return torch.sqrt(sq_dist.clamp_min(eps))


def _energy_score(
    gen_b: Float[Tensor, "batch n_gen feature"],
    pos_b: Float[Tensor, "batch 1 feature"],
    *,
    a_f: float = 0.95,
    sigma: float | Tensor | None = None,
    scaled: bool = False,
    eps: float = 1e-6,
) -> Float[Tensor, ""]:
    r"""Almost-fair joint-field energy score, averaged over the batch.

    The energy score is the multivariate generalisation of CRPS (Gneiting &
    Raftery 2007): the mean Euclidean distance of cohort members to the single
    truth, minus a spread reward over distinct member pairs. The distance is
    taken over the whole flattened field (``feature`` axis), so the score
    rewards getting the joint pattern right rather than per-point marginals.

    The spread coefficient uses the almost-fair estimator of Lang et al.
    (arXiv:2412.15832): ``c = (1 - eps) / (2 K (K - 1))`` with
    ``eps = (1 - a_f) / K``. This recovers the fair estimator at ``a_f = 1`` and
    the naive ``1 / (2 K^2)`` estimator at ``a_f = 0``. It is the same almost-fair
    convex blend as ``metrics.ensemble._alpha_fair_crps_terms``, but the
    coefficient carries the factor of 2 over distinct ordered pairs that the 1-D
    order-statistic CRPS reduction there omits (different objects, do not
    reconcile).

    Parameters
    ----------
    gen_b
        Generated cohort, shape ``(B, K, D)`` with ``K >= 2``; carries the live
        gradient through the backbone.
    pos_b
        The unique positive per input, shape ``(B, 1, D)``.
    a_f
        Almost-fair coefficient in ``(0, 1]``; ``0.95`` per AIFS-CRPS. Used only
        for the difference form (``scaled=False``); the SCRPS form keeps the
        pairwise term unbiased (shrinkage there is ``O(1/K^2)``).
    sigma
        Optional per-channel (broadcastable to ``D``) or scalar standard
        deviation; when given, both ``gen_b`` and ``pos_b`` are divided by it
        before the distances (the un-normalised-input fallback to the
        datamodule's z-score standardisation). Default ``None`` (no rescale).
    scaled
        If ``True`` return the *scaled* energy score (SCRPS; Bolin & Wallin
        2019), a locally scale-invariant proper score. The returned loss is
        ``-SCRPS = a / b + 1/2 log b`` with ``a`` the mean to-truth distance and
        ``b = pair_sum / (K (K - 1))`` the **unbiased** mean pairwise distance.
        ``b`` is *not* the difference form's spread term (that carries an extra
        ``1/2 (1 - eps_af)`` factor; reusing it would move the minimiser and
        break propriety). Default ``False`` (difference form, unchanged).

        SCRPS is unbiased only as ``K -> inf``: the ratio ``a / b`` has an
        ``O(1/K)`` finite-cohort bias, so model *selection* should use the
        unbiased almost-fair energy score (the ``afenergy`` val metric), not this
        training loss.
    eps
        Floor added to ``b`` in the SCRPS form, guarding the ``a / b`` barrier
        and ``log b`` against cohort collapse (``b -> 0``). Kept small relative
        to a standardised ``b ~ O(1)`` so it does not move the calibrated
        minimiser (it is a propriety floor, not a gradient controller). The
        ``a / b`` gradient is finite but grows as ``b`` shrinks; near-collapse
        conditioning relies on standardisation and gradient clipping, plus the
        barrier pushing away from collapse.

    Returns:
    -------
    Float[Tensor, ""]
        Scalar score, mean over the batch: the almost-fair energy score when
        ``scaled=False``, else the SCRPS loss ``-SCRPS``.
    """
    k = gen_b.shape[1]
    if k < 2:
        msg = f"_energy_score requires a cohort of at least 2 members, got {k}."
        raise ValueError(msg)

    if sigma is not None:
        gen_b = gen_b / sigma
        pos_b = pos_b / sigma

    # Both distance terms come from the shared kernel in ``metrics.ensemble`` so
    # that every energy score in the package computes them one way. The kernel
    # takes the vector axis LAST-but-one: (B, D, K) predictions against (B, D)
    # truth.
    dist_truth, dist_pair = _energy_score_terms(
        gen_b.transpose(-1, -2), pos_b.squeeze(-2)
    )  # (B, K), (B, K, K)
    term1 = dist_truth.mean(dim=-1)  # (B,)

    # Self-distances are EXACTLY zero under the kernel, so the full double sum
    # already equals the distinct-pair sum and needs no diagonal mask.
    pair_sum = dist_pair.sum(dim=(-2, -1))  # (B,)

    if scaled:
        # SCRPS loss: -SCRPS = a / b + 1/2 log b. b is the UNBIASED mean
        # pairwise distance pair_sum / (K(K-1)); eps floors the a/b barrier and
        # log b against cohort collapse. a_f / the spread coefficient are not
        # used here on purpose (the unbiased b is the SCRPS denominator).
        b = pair_sum / (k * (k - 1))  # (B,)
        b_eff = b + eps
        return (term1 / b_eff + 0.5 * b_eff.log()).mean()

    # Almost-fair energy score (difference form); eps_af is the almost-fair
    # shrinkage of Lang et al.
    eps_af = (1.0 - a_f) / k
    spread_coeff = (1.0 - eps_af) / (2.0 * k * (k - 1))
    return (term1 - spread_coeff * pair_sum).mean()


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
        targets_scaled: Targets in scaled feature space, shape
            ``(B, N_t, D)``.
        gen_scaled: Trainable cohort in scaled feature space, shape
            ``(B, N_gen, D)``.
        target_weights: Per-target weights, shape ``(B, N_t)``.
        n_neg_total: Total negative-side slab width (gen + fixed_neg).
        tau: Kernel temperature for this scale.

    Returns:
    -------
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
    ``exp(-d / tau)`` with small ``tau``; the output stays in fp32 (the caller
    squares-and-reduces).

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
    -------
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

    # The drift target is a fixed regression goal: the caller computes
    # ``((gen_scaled - goal_scaled) ** 2).mean()`` and back-props through
    # ``gen_scaled`` ONLY. Everything feeding ``goal_scaled`` (distances, the
    # normalisation scale, the per-tau forces) is therefore built under
    # ``torch.no_grad()`` so no throwaway autograd graph is recorded for work
    # that is discarded — this matches JAX's stop_gradient at the
    # calculate-scaled-goal boundary.
    with torch.no_grad():
        # Three-slab target buffer: [gen, fixed_neg, fixed_pos]. Under no_grad
        # the gen slab acts as a constant reference, so no explicit detach.
        targets = torch.cat([gen_f32, fixed_neg_f32, fixed_pos_f32], dim=1)
        target_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

        dist = _pairwise_distances(gen_f32, targets)

        # Per-batch normalisation scale (a constant downstream).
        scale = (dist * target_weights[:, None, :]).mean() / target_weights.mean()
        scale_inputs = (scale / (feature_dim**0.5)).clamp_min(eps)
        scale_clamped = scale.clamp_min(eps)

        targets_scaled = targets / scale_inputs
        dist_normed = dist / scale_clamped

        # Diagonal mask on the leading gen-vs-gen block ONLY. F.pad takes pads
        # in REVERSE axis order: (last_left, last_right, second_left, second_right).
        n_targets = dist_normed.shape[-1]
        eye_block = (
            torch.eye(n_gen, dtype=dist_normed.dtype, device=gen_f32.device)
            * diag_mask_value
        )
        eye_padded = F.pad(eye_block, (0, n_targets - n_gen, 0, 0))
        dist_normed = dist_normed + eye_padded[None, :, :]

        n_neg_total = n_gen + fixed_neg_f32.shape[1]

    # ``gen_scaled`` is the trainable path the caller's loss flows through, so
    # it is computed OUTSIDE the no_grad block. ``scale_inputs`` is a constant,
    # so the gradient through ``gen`` is the clean ``1 / scale_inputs``.
    gen_scaled = gen_f32 / scale_inputs

    with torch.no_grad():
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
            info[f"loss_{tau}"] = force_rms_sq
            force_total = force_total + force_l2

        # Drift target is a detached constant; gradient flows via gen_scaled only.
        goal_scaled = gen_scaled + force_total

    return goal_scaled, gen_scaled, info


def _accuracy_diagnostics(
    gen_b: Float[Tensor, "batch n_gen feature"],
    pos_b: Float[Tensor, "batch 1 feature"],
) -> dict[str, Tensor]:
    """Detached accuracy/calibration diagnostics for the generated cohort.

    None of these enter the loss; they exist so the training-time logs carry
    an accuracy signal. The drift loss alone is a kernel-redundancy proxy, not
    an error metric, so a run can descend on loss while the cohort collapses
    onto a poor mean. Computed under ``no_grad`` so they never touch the graph.

    Args:
        gen_b: Generated cohort in feature space, shape ``(B, K, D)``.
        pos_b: The unique positive per input, shape ``(B, 1, D)``.

    Returns:
    -------
        Three scalar tensors:
        - ``ens_mean_rmse``: RMSE of the cohort mean against the positive
          (ensemble-mean skill — the accuracy signal the loss lacks).
        - ``gen_pos_dist_p50``: median per-member Euclidean distance to the
          positive (re-added spread-vs-target monitor).
        - ``cohort_ssr``: spread/skill ratio proxy — cohort RMS spread over
          ``ens_mean_rmse``; ``<< 1`` flags cohort collapse.
    """
    with torch.no_grad():
        ens_mean = gen_b.mean(dim=1, keepdim=True)
        ens_mean_rmse = ((ens_mean - pos_b) ** 2).mean().sqrt()
        gen_pos_dist_p50 = torch.linalg.vector_norm(gen_b - pos_b, dim=-1).median()
        cohort_spread_rms = gen_b.var(dim=1, correction=0).mean().sqrt()
        cohort_ssr = cohort_spread_rms / ens_mean_rmse.clamp_min(1e-8)
    return {
        "ens_mean_rmse": ens_mean_rmse,
        "gen_pos_dist_p50": gen_pos_dist_p50,
        "cohort_ssr": cohort_ssr,
    }


class DriftingProcessor(OneStepCohortProcessor):
    """One-step generative processor trained via the drifting field loss.

    Implements the algorithm of Deng et al. 2026 (arXiv:2602.04770) as a
    drop-in autocast ``Processor``. Inherits the cohort-inflation forward
    path from ``OneStepCohortProcessor``; this class only specifies the
    extra hyperparameters (``tau_list``, ``diag_mask_value``) and the
    drift-field loss body.

    Notes:
    -----
    With ``n_pos=1`` (the v1 default — the unique ground truth is the only
    positive per input), the positive-side dual softmax degenerates to a
    constant pull toward that single target; only the *negative-side*
    repulsion across the same-input cohort distinguishes drifting from
    plain MSE-toward-target. The ``MSECohortProcessor`` ablation
    isolates that contribution.
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
        seed_sigma: Sequence[float] | float | None = None,
        tau_list: Sequence[float] = (0.02, 0.05, 0.2),
        diag_mask_value: float = 100.0,
        lambda_es: float = 0.0,
        energy_score_a_f: float = 0.95,
        energy_score_scaled: bool = True,
        condition_on_lambda: bool = False,
        lambda_log_min: float = 0.1,
        lambda_log_max: float = 30.0,
        lambda_zero_prob: float = 0.1,
        forcing_eta: float = 0.0,
        forcing_margin: float = 0.0,
        default_inference_lambda: float = 1.0,
    ) -> None:
        """Build a DriftingProcessor.

        Args:
            backbone: One-step generator backbone. MUST be constructed with
                ``include_time_embedding=False`` (drifting has no
                integration time ``t``).
            n_steps_output: Number of output time steps per sample.
                Required; configs pass ``auto`` and let ``setup.py``
                resolve from the datamodule's output shape.
            n_channels_out: Number of output channels per sample.
                Required; resolved via ``auto`` as for ``n_steps_output``.
            n_samples: Cohort size per input. Must be >= 2 (a cohort of one
                produces a trivially zero negative-side dual softmax;
                contrastive drift needs >= 2). Default 8 matches the
                reference's library default.
            seed_sigma: Optional per-channel (or scalar) standard deviation
                scaling the white-Gaussian cohort seed (the prescribed-noise
                injector of residual Drifting, Approach #2). ``None`` (default)
                keeps the unit seed — pure Drifting is unchanged. See
                ``OneStepCohortProcessor``.
            tau_list: Kernel temperatures for the multi-scale aggregation.
                Default ``(0.02, 0.05, 0.2)`` matches the reference.
            diag_mask_value: Value added to the gen-vs-gen diagonal of the
                distance matrix to mask self-affinity.
            lambda_es: Weight on the additive almost-fair energy-score term
                (``L = L_drift + lambda_es * ES``). Default 0.0 reproduces the
                pure drift loss exactly. Must be >= 0.
            energy_score_a_f: Almost-fair coefficient for the energy-score term
                in (0, 1]; 0.95 per AIFS-CRPS. Unused when ``lambda_es == 0`` or
                when ``energy_score_scaled`` is True (SCRPS keeps the pairwise
                term unbiased).
            energy_score_scaled: If True (default), the energy-score term uses
                the scale-invariant SCRPS form, so ``lambda_es > 0`` is a
                scale-free regulariser whose lambda transfers across datasets.
                If False, uses the difference-form almost-fair energy score.
                ``lambda_es == 0`` is byte-for-byte pure drift either way.
            condition_on_lambda: If True, train a single lambda-conditioned model:
                each step samples lambda, feeds the bounded weight
                ``w=1/(1+lambda)`` to the generator's loss-weight embedding, and
                uses the convex loss ``w*drift + (1-w)*ES``. The backbone must be
                built with ``include_loss_weight=True``. Default False keeps the
                fixed-``lambda_es`` additive path unchanged.
            lambda_log_min: Lower bound of the log-uniform lambda sampling range.
            lambda_log_max: Upper bound of the log-uniform lambda sampling range.
            lambda_zero_prob: Probability mass placed exactly on ``lambda=0`` (the
                pure-drift endpoint) when sampling.
            forcing_eta: Weight on the responsiveness forcing penalty (C3). 0.0
                (default) disables it; >0 penalises a lambda-insensitive cohort by
                comparing the spread at a low/high lambda pair.
            forcing_margin: Margin in the forcing penalty
                ``relu(spread(lambda_lo) - spread(lambda_hi) + margin)``.
            default_inference_lambda: The lambda used by ``map`` (and hence the
                Lightning validation/test metrics) when ``condition_on_lambda`` is
                on. The explicit inference sweep uses ``map_at_lambda`` instead.
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
            seed_sigma=seed_sigma,
        )

        self.tau_list = tuple(tau_list)
        self.diag_mask_value = diag_mask_value
        if lambda_es < 0.0:
            msg = f"DriftingProcessor requires lambda_es >= 0 (got {lambda_es})."
            raise ValueError(msg)
        if not (0.0 < energy_score_a_f <= 1.0):
            msg = (
                "DriftingProcessor requires energy_score_a_f in (0, 1] "
                f"(got {energy_score_a_f})."
            )
            raise ValueError(msg)
        self.lambda_es = lambda_es
        self.energy_score_a_f = energy_score_a_f
        self.energy_score_scaled = energy_score_scaled

        self.condition_on_lambda = condition_on_lambda
        if condition_on_lambda:
            if getattr(backbone, "include_loss_weight", False) is not True:
                msg = (
                    "DriftingProcessor(condition_on_lambda=True) requires a "
                    "backbone built with include_loss_weight=True."
                )
                raise ValueError(msg)
            if not (0.0 < lambda_log_min < lambda_log_max):
                msg = (
                    "condition_on_lambda requires 0 < lambda_log_min < "
                    f"lambda_log_max (got {lambda_log_min}, {lambda_log_max})."
                )
                raise ValueError(msg)
            if not (0.0 <= lambda_zero_prob < 1.0):
                msg = f"lambda_zero_prob must be in [0, 1) (got {lambda_zero_prob})."
                raise ValueError(msg)
            if forcing_eta < 0.0:
                msg = f"forcing_eta must be >= 0 (got {forcing_eta})."
                raise ValueError(msg)
        self.lambda_log_min = lambda_log_min
        self.lambda_log_max = lambda_log_max
        self.lambda_zero_prob = lambda_zero_prob
        self.forcing_eta = forcing_eta
        self.forcing_margin = forcing_margin
        self.default_inference_lambda = default_inference_lambda

    # ------------------------------------------------------------------
    # Lambda-conditioned path. Inactive unless condition_on_lambda.
    # ------------------------------------------------------------------

    def _sample_one_lambda(self) -> float:
        """Sample a single lambda: a point mass at 0, else log-uniform in range."""
        if torch.rand(()).item() < self.lambda_zero_prob:
            return 0.0
        u = torch.rand(()).item()
        log_lo, log_hi = math.log(self.lambda_log_min), math.log(self.lambda_log_max)
        return math.exp(log_lo + u * (log_hi - log_lo))

    def _lambda_term(
        self,
        cond: Tensor,
        global_cond: Tensor | None,
        pos_b: Tensor,
        lam: float,
        b: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One convex-loss term ``w*drift + (1-w)*ES`` at a fixed scalar lambda.

        Feeds ``w=1/(1+lambda)`` to the generator's loss-weight embedding so the
        sample is conditioned on lambda. Returns ``(term, spread, drift, es)``;
        all four tensors are live (not detached) so callers can backpropagate
        individual terms — diagnostics consumers detach at the call site.
        """
        w = 1.0 / (1.0 + lam)
        w_vec = cond.new_full((cond.shape[0],), w)
        gen = self._draw_one(cond, global_cond, loss_weight=w_vec)
        gen_b = gen.reshape(b, self.n_samples, -1)
        drift, es, spread = self._drift_energy_spread(gen_b, pos_b)
        term = w * drift + (1.0 - w) * es
        return term, spread, drift, es

    def _drift_energy_spread(
        self, gen_b: Tensor, pos_b: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the drift, energy-score, and cohort-spread terms for one draw.

        Live (not detached) so callers can backpropagate individual terms.
        """
        goal_scaled, gen_scaled, _ = _compute_drift_field(
            gen_b, pos_b, tau_list=self.tau_list, diag_mask_value=self.diag_mask_value
        )
        drift = ((gen_scaled - goal_scaled) ** 2).mean()
        es = _energy_score(
            gen_b, pos_b, a_f=self.energy_score_a_f, scaled=self.energy_score_scaled
        )
        spread = gen_b.std(dim=1, correction=0).mean()
        return drift, es, spread

    def _conditioned_loss(self, batch: EncodedBatch) -> Tensor:
        """Lambda-conditioned cohort loss: a low/high lambda pair per step.

        Two convex-loss terms at a sampled ``(lambda_lo, lambda_hi)`` pair give
        the network within-step lambda variation; an optional responsiveness
        penalty (forcing_eta>0) discourages a lambda-insensitive cohort.
        """
        self._latest_diagnostics = {}
        batch_n = batch.repeat(self.n_samples)
        cond = batch_n.encoded_inputs
        target = batch_n.encoded_output_fields
        b = cond.shape[0] // self.n_samples
        target_b = target.reshape(b, self.n_samples, -1)
        pos_b = target_b[:, :1, :]

        lam_a, lam_b = self._sample_one_lambda(), self._sample_one_lambda()
        lam_lo, lam_hi = (lam_a, lam_b) if lam_a <= lam_b else (lam_b, lam_a)

        term_lo, spread_lo, drift_lo, _ = self._lambda_term(
            cond, batch_n.global_cond, pos_b, lam_lo, b
        )
        term_hi, spread_hi, _, es_hi = self._lambda_term(
            cond, batch_n.global_cond, pos_b, lam_hi, b
        )
        loss = term_lo + term_hi

        diagnostics: dict[str, Tensor] = {
            "cohort_spread": 0.5 * (spread_lo.detach() + spread_hi.detach()),
            "lambda_lo": torch.as_tensor(lam_lo, device=cond.device),
            "lambda_hi": torch.as_tensor(lam_hi, device=cond.device),
            "spread_lo": spread_lo.detach(),
            "spread_hi": spread_hi.detach(),
            "drift_lo": drift_lo.detach(),
            "energy_hi": es_hi.detach(),
        }
        if self.forcing_eta > 0.0:
            p_resp = torch.relu(spread_lo - spread_hi + self.forcing_margin)
            loss = loss + self.forcing_eta * p_resp
            diagnostics["forcing_penalty"] = p_resp.detach()

        self._latest_diagnostics = diagnostics
        return loss

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Cohort training loss; routes to the lambda-conditioned path if enabled."""
        if self.condition_on_lambda:
            return self._conditioned_loss(batch)
        return super().loss(batch)

    def map(
        self,
        x: Tensor,
        global_cond: Tensor | None,
        base_noise: BaseNoise | None = None,
    ) -> Tensor:
        """Map inputs to one output. Conditioned models use the default lambda.

        The Lightning validation/test path calls ``map`` without a lambda, so a
        conditioned model evaluates at ``default_inference_lambda``; the explicit
        lambda sweep goes through ``map_at_lambda``. ``base_noise`` is forwarded
        on BOTH paths -- dropping it here would silently hand the sample-feed
        rollout a fresh white draw instead of the stateful AR(1) source it asked
        for, which is a wrong answer rather than an error.
        """
        if not self.condition_on_lambda:
            return super().map(x, global_cond, base_noise)
        return self.map_at_lambda(
            x, global_cond, self.default_inference_lambda, base_noise=base_noise
        )

    def map_at_lambda(
        self,
        x: Tensor,
        global_cond: Tensor | None,
        lam: float,
        base_noise: BaseNoise | None = None,
    ) -> Tensor:
        """Draw one cohort-of-one sample from the lambda-conditioned generator.

        Inference helper for the single-model lambda sweep: feeds
        ``w=1/(1+lambda)`` to the generator. Requires ``condition_on_lambda``.
        Respects the caller's autograd context (wrap in ``torch.no_grad`` for
        inference).
        """
        if not self.condition_on_lambda:
            msg = "map_at_lambda requires condition_on_lambda=True."
            raise ValueError(msg)
        w = 1.0 / (1.0 + lam)
        w_vec = x.new_full((x.shape[0],), w)
        return self._draw_one(x, global_cond, loss_weight=w_vec, base_noise=base_noise)

    def loss_terms(
        self, batch: EncodedBatch, lam: float | None = None
    ) -> dict[str, Tensor]:
        """Per-term losses of the hybrid objective for one cohort draw.

        Monitoring helper (see ``HybridLossMonitorCallback``): returns the raw
        ``drift`` and ``energy_score`` terms — NOT detached, so a caller can
        backpropagate each term separately — plus the cohort ``spread``. For a
        lambda-conditioned processor ``lam`` selects the probe lambda fed to
        the loss-weight embedding (required there); otherwise ``lam`` must be
        ``None`` and the terms mirror one ``_compute_cohort_loss`` evaluation.
        """
        batch_n = batch.repeat(self.n_samples)
        cond = batch_n.encoded_inputs
        target = batch_n.encoded_output_fields
        b = cond.shape[0] // self.n_samples
        target_b = target.reshape(b, self.n_samples, -1)
        pos_b = target_b[:, :1, :]
        if self.condition_on_lambda:
            if lam is None:
                msg = "loss_terms requires lam for a lambda-conditioned processor."
                raise ValueError(msg)
            _, spread, drift, es = self._lambda_term(
                cond, batch_n.global_cond, pos_b, lam, b
            )
        else:
            if lam is not None:
                msg = "lam is only valid for a lambda-conditioned processor."
                raise ValueError(msg)
            gen = self._draw_one(cond, batch_n.global_cond)
            gen_b = gen.reshape(b, self.n_samples, -1)
            drift, es, spread = self._drift_energy_spread(gen_b, pos_b)
        return {"drift": drift, "energy_score": es, "spread": spread}

    def _compute_cohort_loss(
        self,
        gen_b: Tensor,
        target_b: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute drift-field loss, optional energy-score term, and diagnostics.

        With ``n_pos = 1``, ``target_b[:, :1, :]`` recovers the unique
        positive per input (the K target copies along the middle axis are
        identical under repeat-interleave). The negative side comes
        implicitly from the gen-vs-gen block of the target buffer
        inside ``_compute_drift_field``.
        """
        pos_b = target_b[:, :1, :]
        goal_scaled, gen_scaled, info = _compute_drift_field(
            gen_b,
            pos_b,
            tau_list=self.tau_list,
            diag_mask_value=self.diag_mask_value,
        )
        drift_loss = ((gen_scaled - goal_scaled) ** 2).mean()
        extras: dict[str, Tensor] = {
            f"force_rms_tau{tau}": info[f"loss_{tau}"].sqrt() for tau in self.tau_list
        }
        extras.update(_accuracy_diagnostics(gen_b, pos_b))

        if self.lambda_es > 0.0:
            energy = _energy_score(
                gen_b, pos_b, a_f=self.energy_score_a_f, scaled=self.energy_score_scaled
            )
            weighted = self.lambda_es * energy
            loss = drift_loss + weighted
            extras["drift_loss"] = drift_loss.detach()
            extras["energy_score"] = energy.detach()
            extras["lambda_es"] = drift_loss.new_tensor(self.lambda_es)
            # Share of the loss magnitude carried by the energy term. Absolute
            # values keep the share in [0, 1] even when SCRPS goes negative
            # (its -0.5*log(spread) term has no sign guarantee).
            extras["energy_share"] = (
                weighted.abs() / (drift_loss.abs() + weighted.abs())
            ).detach()
        else:
            loss = drift_loss
        return loss, extras
