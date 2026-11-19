"""Sample-free scoring for low-rank Gaussian / Student-t predictive heads.

Every ensemble metric in :mod:`autocast.metrics.ensemble` (``LogScore``,
``CRPS``, ...) is fit from ``n_members`` sampled draws. That is the only option
for a processor with no closed-form predictive, but it is a poor instrument for
one that has one: at a heavy tail (``nu`` near its floor of 2), the deployed
covariance is ``nu/(nu-2) * Sigma`` and a handful of draws is dominated by which
side of the InvGamma mixing weight they happened to land on (see
``notes/design/rank_ladder/plans_and_designs/PLAN.md`` §3.9.1 -- an 8-member Student-t
fit reads
``log_score`` at +185.8 where the closed-form likelihood says the model is
BETTER than its Gaussian twin). §3.9.2 rule 5 states the fix generically: score
a closed-form head in closed form, and retire member-moment statistics for it.

This module is that closed form. It detects the capability by attribute, not by
class name (``LowRankGaussianNLLLoss._unpack_lowrank`` +
``_woodbury_quad_logdet`` -- "shared ... by the Gaussian loss, the variational
subclass, and any elliptical ... head", per that method's own docstring), reuses
the loss's own Woodbury solve and Student-t profile (never re-derived here), and
adds four columns without touching any existing one:

* ``log_score_closed_form`` -- the joint (not per-site) negative log-density of
  the truth under the deployed law, in the SAME dropped-additive-constant, per-
  element convention every loss in ``autocast.losses.gaussian`` already uses
  (see e.g. ``GaussianNLLLoss``'s docstring). It is therefore numerically equal
  to the loss's own training-time value for any head with no KL/prior term
  (:class:`LowRankGaussianNLLLoss`, :class:`LowRankVariationalARDStudentTNLLLoss`
  data_term="marginal", ...); for a head that adds a loading-KL prior
  (:class:`LowRankFactorAnalysisARDNLLLoss` / its Student-t twin) it is the DATA
  part alone, i.e. the same quantity ``val/fit`` reports.
* ``spread_closed_form`` -- ``sqrt(mean_i Sigma_eff_ii)`` pooled over every site,
  with ``Sigma_eff = nu/(nu-2) * Sigma`` for a Student-t head and ``Sigma`` for a
  Gaussian one.
* ``crps_gaussian_closed_form`` -- per-site closed-form CRPS treating each site's
  marginal as ``N(mu_i, Sigma_ii)``. Reuses
  :meth:`autocast.metrics.gaussian.GaussianCRPS._pointwise` rather than
  re-deriving it (searched first -- see the module import below).
* ``crps_studentt_closed_form`` -- per-site closed-form CRPS treating each site's
  marginal as the univariate Student-t with the SAME ``nu`` and scale
  ``sqrt(Sigma_ii)`` (the marginal of a multivariate ``t`` is univariate ``t``
  with unchanged ``nu``). Only populated when the loss exposes ``nu``; the
  formula is Jordan, Krueger & Lerch (2019), "Evaluating Probabilistic
  Forecasts with scoringRules", the ``crps_t`` case (verified against the
  package's own R source, ``scores_t.R``: ``y*(2*pt(y,df)-1) + 2/(df-1) *
  (dt(y,df)*(df+y^2) - sqrt(df)*bfrac)`` with ``bfrac = B(0.5, df-0.5) /
  B(0.5, df/2)^2``). ``crps_gaussian_closed_form`` is ALSO computed for a
  Student-t head (mis-specified: it uses the same raw ``Sigma`` without the
  ``nu/(nu-2)`` inflation), so the two columns together show exactly the
  Gaussian-vs-correct gap the member-sampled instrument could not see.

Neither ``eval_rank`` truncation nor log-variance bounding is re-implemented:
callers pass the processor's own ``_truncate_to_eval_rank`` so a truncated
deployment (``model.processor.eval_rank=<r>``) is scored under the SAME
moment-matched law :meth:`~autocast.processors.latent_lowrank.
LatentLowRankGaussianProcessor.sample_packed` draws from; the log-variance bound
is whatever ``loss_func._unpack_lowrank`` already applied (the loss's own
bound, not the sampler's).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torchmetrics import Metric

from autocast.metrics.gaussian import GaussianCRPS
from autocast.types import Tensor


def _no_truncation(var: Tensor, factor: Tensor) -> tuple[Tensor, Tensor]:
    """Identity truncation: the default when a caller has no ``eval_rank`` concept.

    So :func:`compute_closed_form_predictive_stats` is usable directly against
    a bare loss instance in tests, with no processor involved.
    """
    return var, factor


@dataclass
class ClosedFormPredictiveStats:
    """Per-frame closed-form predictive summary for one scored batch.

    ``n_frames = B * T`` (one row per whole latent/ambient frame, the loss's own
    convention); ``n_elements = prod(spatial) * C`` (the frame size ``N``).
    """

    #: Predictive mean, ``(n_frames, n_elements)``.
    mu: Tensor
    #: Truth, same shape and space as ``mu``.
    target: Tensor
    #: Marginal predictive variance ``Sigma_ii`` (the SCALE, not the deployed
    #: covariance -- callers wanting ``Sigma_eff`` multiply by :attr:`inflation`),
    #: ``(n_frames, n_elements)``.
    sigma_ii: Tensor
    #: Degrees of freedom, or ``None`` for a Gaussian (infinite-``nu``) head.
    nu: float | None
    #: Joint per-frame negative log-density, dropped-constant convention,
    #: ``(n_frames,)``.
    joint_nll_per_frame: Tensor
    #: Frame size ``N``, for the ``/N`` per-element normalisation.
    n_elements: int

    @property
    def inflation(self) -> float:
        """``nu / (nu - 2)`` for a Student-t head, ``1.0`` for a Gaussian one."""
        if self.nu is None:
            return 1.0
        return self.nu / (self.nu - 2.0)


def probe_closed_form_predictive(processor: Any) -> bool:
    """Cheap, side-effect-free check: does ``processor`` carry a scoreable loss?

    True when ``processor.loss_func`` exposes the shared elliptical core
    (``_unpack_lowrank`` + ``_woodbury_quad_logdet``, see the module docstring)
    AND ``processor`` exposes ``_truncate_to_eval_rank``. Checked by attribute,
    not ``isinstance``: a future heavy-tailed head (e.g. the per-eigen-direction
    design in ``PLAN.md`` §3.9.5) is picked up automatically as long as it keeps
    this shape, with no change here.

    This is necessary but not sufficient -- the plain (non-factor-analysis)
    :class:`~autocast.losses.gaussian.LowRankVariationalARDGaussianNLLLoss`
    inherits ``_unpack_lowrank`` from a parent expecting a narrower packing than
    it actually emits, so it passes this probe but its ``_unpack_lowrank`` call
    fails; :func:`compute_closed_form_predictive_stats` catches that and returns
    ``None`` per batch rather than raising.
    """
    loss_func = getattr(processor, "loss_func", None)
    return (
        loss_func is not None
        and callable(getattr(loss_func, "_unpack_lowrank", None))
        and callable(getattr(loss_func, "_woodbury_quad_logdet", None))
        and callable(getattr(processor, "_truncate_to_eval_rank", None))
    )


@torch.no_grad()
def compute_closed_form_predictive_stats(
    loss_func: Any,
    pred: Tensor,
    target: Tensor,
    *,
    truncate_to_eval_rank: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
    | None = None,
) -> ClosedFormPredictiveStats | None:
    """Build :class:`ClosedFormPredictiveStats` from one batch, or ``None``.

    ``None`` means "not applicable to this batch/loss" -- either ``loss_func``
    does not carry the shared elliptical core at all, or its ``_unpack_lowrank``
    raised (the packing-width mismatch described in
    :func:`probe_closed_form_predictive`'s docstring). Both are treated as "skip
    this column", never as an error: a caller iterating a whole eval set must
    not abort the run over one processor family's known gap.

    Args:
        loss_func: The processor's loss module (``processor.loss_func``).
        pred: Packed processor output, ``(B, T, ..., width)``.
        target: Truth in the SAME space/packing ``pred`` was produced in
            (e.g. ``encoded_batch.encoded_output_fields``), ``(B, T, ..., C)``.
        truncate_to_eval_rank: The processor's own
            ``_truncate_to_eval_rank(var, factor) -> (var, factor)``, so a
            truncated deployment is scored under the same moment-matched law
            the sampler draws from. Defaults to the identity (no truncation) --
            correct whenever ``eval_rank`` is unset, and what makes this
            function testable against a bare loss with no processor.
    """
    if not (
        callable(getattr(loss_func, "_unpack_lowrank", None))
        and callable(getattr(loss_func, "_woodbury_quad_logdet", None))
    ):
        return None
    truncate = truncate_to_eval_rank or _no_truncation

    try:
        mu, log_var, u_factor = loss_func._unpack_lowrank(pred, target)
    except ValueError:
        # The packing width doesn't match this loss's simple (mu, log_var, U)
        # unpacker -- e.g. the plain variational marginal head, whose actual
        # (2 + 2*rank)*C packing is wider than the inherited unpacker expects.
        # Not an error: this processor family is out of scope for now (see the
        # module docstring), and the caller should just omit these columns.
        return None

    n_frames = int(pred.shape[0] * pred.shape[1])
    n_elements = math.prod(target.shape[2:])
    mu_flat = mu.reshape(n_frames, n_elements)
    target_flat = target.reshape(n_frames, n_elements)
    log_var_flat = log_var.reshape(n_frames, n_elements)
    rank = u_factor.shape[-1]
    u_flat = u_factor.reshape(n_frames, n_elements, rank)

    var_eff, u_eff = truncate(torch.exp(log_var_flat), u_flat)
    residual = target_flat - mu_flat
    quad, logdet = loss_func._woodbury_quad_logdet(residual, torch.log(var_eff), u_eff)

    nu_attr = getattr(loss_func, "nu", None)
    is_student_t = callable(getattr(loss_func, "_student_t_frame_nll", None))
    if is_student_t and nu_attr is not None:
        nu = float(nu_attr.detach()) if isinstance(nu_attr, Tensor) else float(nu_attr)
        joint_nll_per_frame = loss_func._student_t_frame_nll(quad, logdet, n_elements)
    else:
        nu = None
        # The Gaussian data term is exactly `_woodbury_frame_nll`'s formula
        # (0.5 * (quad + logdet)); `quad`/`logdet` are already the reused
        # Woodbury solve, so this is not a re-derivation of it.
        joint_nll_per_frame = 0.5 * (quad + logdet)

    sigma_ii = var_eff + u_eff.square().sum(dim=-1)
    return ClosedFormPredictiveStats(
        mu=mu_flat,
        target=target_flat,
        sigma_ii=sigma_ii,
        nu=nu,
        joint_nll_per_frame=joint_nll_per_frame,
        n_elements=n_elements,
    )


class ClosedFormLogScore(Metric):
    r"""Per-element joint negative log-density of the truth, closed form.

    ``mean_frames(joint_nll_per_frame) / N`` -- the same dropped-additive-
    constant, per-element convention as every loss in
    :mod:`autocast.losses.gaussian`, so this is numerically the "held-out
    likelihood" ``val/fit`` reports at training time, evaluated fresh on
    whatever batches :meth:`update` is fed.
    """

    name: str = "log_score_closed_form"

    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_frames", default=torch.tensor(0), dist_reduce_fx="sum")
        self._n_elements: int | None = None

    def update(self, stats: ClosedFormPredictiveStats) -> None:
        """Accumulate one batch's per-frame joint NLL."""
        if self._n_elements is None:
            self._n_elements = stats.n_elements
        self.sum_nll = self.sum_nll + stats.joint_nll_per_frame.sum()
        self.n_frames = self.n_frames + stats.joint_nll_per_frame.numel()

    def compute(self) -> Tensor:
        if self.n_frames == 0 or self._n_elements is None:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)
        return self.sum_nll / self.n_frames / self._n_elements


class ClosedFormSpread(Metric):
    r"""Pooled closed-form spread ``sqrt(mean_i Sigma_eff_ii)``.

    ``Sigma_eff = nu/(nu-2) * Sigma`` for a Student-t head (the deployed
    covariance; ``Sigma`` itself is only the t SCALE matrix) and ``Sigma`` for a
    Gaussian one -- see :attr:`ClosedFormPredictiveStats.inflation`.
    """

    name: str = "spread_closed_form"

    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_sigma_eff", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_elements_total", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, stats: ClosedFormPredictiveStats) -> None:
        sigma_eff_sum = (stats.sigma_ii * stats.inflation).sum()
        self.sum_sigma_eff = self.sum_sigma_eff + sigma_eff_sum
        self.n_elements_total = self.n_elements_total + stats.sigma_ii.numel()

    def compute(self) -> Tensor:
        if self.n_elements_total == 0:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)
        return torch.sqrt(self.sum_sigma_eff / self.n_elements_total)


class ClosedFormGaussianCRPS(Metric):
    """Per-site closed-form CRPS under a Gaussian marginal ``N(mu_i, Sigma_ii)``.

    Reuses :meth:`autocast.metrics.gaussian.GaussianCRPS._pointwise` (searched
    for and found first -- see the module docstring) rather than re-deriving the
    closed-form Gaussian CRPS. ``Sigma_ii`` is the RAW scale, never inflated --
    correct for a Gaussian head, and deliberately mis-specified for a Student-t
    one (see the module docstring).
    """

    name: str = "crps_gaussian_closed_form"

    def __init__(self, eps: float = 1e-6, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._core = GaussianCRPS(eps=eps, dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_sites", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stats: ClosedFormPredictiveStats) -> None:
        log_var = torch.log(stats.sigma_ii.clamp_min(self._core.eps**2))
        pointwise = self._core._pointwise(stats.mu, log_var, stats.target)
        self.sum_crps = self.sum_crps + pointwise.sum()
        self.n_sites = self.n_sites + pointwise.numel()

    def compute(self) -> Tensor:
        if self.n_sites == 0:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)
        return self.sum_crps / self.n_sites


#: Degrees-of-freedom floor below which the Student-t CRPS closed form (which
#: divides by ``nu - 1``) is not defined -- see Jordan/Krueger/Lerch's own
#: ``df[df <= 1] <- NaN`` guard in ``scores_t.R``. The repo's own ``nu_floor``
#: (`LowRankFactorAnalysisStudentTNLLLoss`) defaults to and is ratified at 2.0
#: (PLAN.md §3.9.2 rule 1), safely above this.
_MIN_CRPS_STUDENT_T_DF = 1.0

#: Fixed continued-fraction iterations for :func:`_regularized_incomplete_beta`.
#: No early-exit check: an early-exit would need a `.item()`/`.all()` host sync
#: every iteration, which is the exact per-step CPU round-trip this function
#: exists to remove. Measured against ``scipy.special.betainc`` on a fine grid
#: spanning every ``a = nu/2`` this repo's ``nu_floor``-to-Gaussian-limit range
#: touches (``nu`` in ``{2.5, 4, 30, 1000}``, i.e. ``a`` in ``{1.25, 2, 15,
#: 500}``, at ``x`` spanning ``(0, 1)``, the hardest slower-than-``x = nu/(nu +
#: z^2)`` case): max abs error 1.0e-12 at 30 iterations, already far past what
#: a CRPS reported to a CSV needs; more iterations buys precision this metric
#: has no use for while directly costing wall time (see
#: :func:`_cpu_single_threaded_scope` below).
_BETAINC_ITERATIONS = 30

#: Floor added to a continued-fraction denominator that would otherwise pass
#: through exactly 0 -- Numerical Recipes' own ``FPMIN`` guard for ``betacf``.
_BETAINC_FPMIN = 1e-30


@contextlib.contextmanager
def _cpu_single_threaded_scope() -> Iterator[None]:
    """Temporarily set torch's CPU intra-op thread count to 1, then restore it.

    ``_betacf`` is a long sequential chain of small elementwise ops on a
    (batch, time, site, channel)-shaped tensor -- data-parallel across that
    whole block (never a loop over it), but each of the ~20 fixed algorithm
    iterations still pays PyTorch's per-op CPU dispatch/thread-fan-out
    overhead once. Measured directly on this machine (20 CPU threads): a
    single elementwise ``torch.where`` on a 65536-element float64 tensor costs
    0.66 ms at the default thread count and 0.13 ms at ``num_threads=1`` --
    the tensor is far too small for intra-op parallelism to pay for itself,
    and a ~20-iteration chain of such ops multiplies that overhead into the
    dominant cost (measured: 300+ ms end to end, vs 20-25 ms with this scope
    applied -- see the benchmark in the E3 report). Restored via ``finally``
    so a later, larger batched op elsewhere in the SAME process (e.g. the
    loss's own ``O(N * rank^2)`` Woodbury Cholesky solve, which genuinely
    benefits from more threads) is unaffected. Scoped narrowly around the
    incomplete-beta evaluation only, not the whole eval loop, for exactly that
    reason. A GPU tensor is unaffected either way: ``torch.set_num_threads``
    governs CPU intra-op parallelism only.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _betacf(x: Tensor, a: float, b: float) -> Tensor:
    """Lentz's continued fraction for the incomplete beta function's tail.

    Numerical Recipes (3rd ed.) §6.4, ``betacf``: evaluates the continued
    fraction ``a`` and ``b`` are constant scalars (the repo's Student-t heads
    carry a single global ``nu``, never a per-site one), so only ``x`` is
    tensor-valued -- every iteration below is one batched elementwise op over
    the WHOLE input tensor, run a fixed :data:`_BETAINC_ITERATIONS` times; there
    is no branch, host sync, or loop over sites/leads/members/batch anywhere in
    this function.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = torch.ones_like(x)
    d = 1.0 - qab * x / qap
    d = torch.where(d.abs() < _BETAINC_FPMIN, torch.full_like(d, _BETAINC_FPMIN), d)
    d = 1.0 / d
    h = d.clone()
    for m in range(1, _BETAINC_ITERATIONS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = torch.where(d.abs() < _BETAINC_FPMIN, torch.full_like(d, _BETAINC_FPMIN), d)
        c = 1.0 + aa / c
        c = torch.where(c.abs() < _BETAINC_FPMIN, torch.full_like(c, _BETAINC_FPMIN), c)
        d = 1.0 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = torch.where(d.abs() < _BETAINC_FPMIN, torch.full_like(d, _BETAINC_FPMIN), d)
        c = 1.0 + aa / c
        c = torch.where(c.abs() < _BETAINC_FPMIN, torch.full_like(c, _BETAINC_FPMIN), c)
        d = 1.0 / d
        h = h * (d * c)
    return h


def _regularized_incomplete_beta(x: Tensor, a: float, b: float) -> Tensor:
    r"""``I_x(a, b)``, evaluated batched and entirely in torch (no scipy).

    Numerical Recipes (3rd ed.) §6.4, ``betai``: the log-space prefactor
    ``x^a (1-x)^b / (a B(a,b))`` times :func:`_betacf`, using whichever of the
    two continued-fraction expansions converges fast at this ``x``
    (``I_x(a,b) = 1 - I_{1-x}(b,a)``). ``math.lgamma`` (Python stdlib) gives the
    log-Beta normaliser; nothing here reaches for ``scipy``.

    **Why this exists instead of ``scipy.special.betainc``** (scipy is an
    installed, working dependency -- rung 5 of the minimum-viable-code ladder
    normally means "use it", and this is the one place in the module that
    deliberately does not): ``scipy``/``numpy`` are CPU-only, so scoring a CUDA
    tensor through them forces a device round-trip (``.cpu().numpy()`` out,
    ``torch.as_tensor(..., device=...)`` back) on every batch. This module's
    caller (`scripts/eval/encoder_processor_decoder.py`'s
    ``_compute_closed_form_predictive_row``) is the standard ``autocast eval``
    CLI's test pass, which for a real (non-toy) PDE model runs via
    ``lightning.Fabric`` on whatever accelerator the run is configured for --
    GPU for every gs64/ad64-scale run this repo actually publishes. This
    module's own tests and benchmark (see the E3 report) were run CPU-only
    per that task's own instruction, so the CUDA-round-trip cost itself is NOT
    directly measured here; it is the motivating reason for removing scipy
    from this hot path, stated as such rather than left implicit.

    Verified against ``scipy.special.betainc`` in
    ``tests/metrics/test_closed_form_predictive.py`` (that comparison is
    test-only, per the instruction to keep any MC/oracle reference out of
    ``src/``) -- including at ``a = nu/2`` for the LIVE gs64 Student-t
    checkpoint's own deployed ``nu`` (``GS64_DEPLOYED_NU`` in that file,
    ``2.00688``, closer to the ``a=1`` boundary than any other value tested):
    max abs error ``2.0e-15`` there, no worse than at any larger ``a``.
    """
    eps = torch.finfo(x.dtype).eps
    x_clamped = x.clamp(eps, 1.0 - eps)
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * torch.log(x_clamped)
        + b * torch.log1p(-x_clamped)
    )
    bt = torch.exp(log_bt)
    swap = x_clamped < (a + 1.0) / (a + b + 2.0)
    with _cpu_single_threaded_scope():
        result = torch.where(
            swap,
            bt * _betacf(x_clamped, a, b) / a,
            1.0 - bt * _betacf(1.0 - x_clamped, b, a) / b,
        )
    result = torch.where(x <= 0, torch.zeros_like(result), result)
    return torch.where(x >= 1, torch.ones_like(result), result)


def _student_t_cdf(z: Tensor, nu: float) -> Tensor:
    r"""Standard Student-t CDF ``F_nu(z)``, batched, torch-native.

    ``F_nu(z) = 1 - 0.5 I_x(nu/2, 1/2)`` for ``z > 0`` and ``0.5 I_x(nu/2, 1/2)``
    otherwise, with ``x = nu / (nu + z^2)`` (Abramowitz & Stegun 26.7.1).
    Replaces the earlier ``scipy.stats.t.cdf`` CPU/numpy round-trip: verified to
    agree with it to float64 machine precision (max abs error ~1e-14 across
    ``nu`` in ``{2.5, 4, 6, 30, 1000}`` and ``z`` in ``[-8, 8]``) in
    ``tests/metrics/test_closed_form_predictive.py``.
    """
    x = nu / (nu + z * z)
    ib = _regularized_incomplete_beta(x, 0.5 * nu, 0.5)
    return torch.where(z > 0, 1.0 - 0.5 * ib, 0.5 * ib)


def student_t_crps_pointwise(
    mu: Tensor, sigma_ii: Tensor, nu: float, target: Tensor
) -> Tensor:
    r"""Closed-form CRPS of a univariate ``t_nu(mu, sigma)`` at ``target``.

    ``sigma = sqrt(sigma_ii)``. Jordan, Krueger & Lerch (2019), "Evaluating
    Probabilistic Forecasts with scoringRules", the ``crps_t`` case -- verified
    against the package's own R source (``scores_t.R``, fetched 2026-09-12):

    .. code-block:: r

        bfrac <- beta(0.5, df - 0.5) / beta(0.5, 0.5 * df)^2
        y * (2 * pt(y, df) - 1) + 2 / (df - 1) *
          (dt(y, df) * (df + y^2) - sqrt(df) * bfrac)

    with the location-scale case scaling as ``sigma * crps_std(y / sigma, df)``.
    Every term is a batched tensor op over the whole ``(mu, sigma_ii, target)``
    block at once -- no Python loop over sites, leads, members, or batch
    elements, and no scipy/numpy round-trip:

    * ``bfrac`` is a SCALAR (``nu`` is the single global tail index the loss
      learns, never per-site), computed once via ``math.lgamma`` in log-space
      so it stays finite at the large-``nu`` (near-Gaussian) end where
      ``Beta(0.5, nu-0.5)`` itself underflows;
    * the CDF term is :func:`_student_t_cdf` (torch-native, see its docstring);
    * the PDF term reuses ``torch.distributions.StudentT.log_prob`` (torch's
      own implementation, not re-derived) -- a single vectorised call over the
      whole tensor, never a per-element scalar one.

    Both are evaluated in float64 regardless of the input dtype (mirrors
    ``LowRankGaussianNLLLoss._woodbury_quad_logdet``'s own float64 upcast for a
    numerically-sensitive sub-solve): ``log_prob`` computes
    ``lgamma((nu+1)/2) - lgamma(nu/2)``, a difference of two
    ``O(nu * log(nu))``-magnitude terms whose ``O(1)`` result loses essentially
    all precision in float32 once ``nu`` reaches the low hundreds -- measured
    directly: at ``nu=1e6`` the float32 path read a PDF ~8% off the Gaussian
    limit it should recover, while float64 matched to six decimal places.
    """
    if nu <= _MIN_CRPS_STUDENT_T_DF:
        msg = (
            f"student_t_crps_pointwise requires nu > {_MIN_CRPS_STUDENT_T_DF} "
            f"(the crps_t closed form divides by nu - 1); got nu={nu}. The "
            f"repo's nu_floor is ratified at 2.0, so a caller reaching this is "
            f"scoring an out-of-family nu, not a healthy model."
        )
        raise ValueError(msg)
    dtype, device = target.dtype, target.device
    sigma = torch.sqrt(sigma_ii.double().clamp_min(1e-24))
    z = (target.double() - mu.double()) / sigma

    # `StudentT(df=<python float>)` stores `df` as a float32 tensor regardless
    # of `z`'s dtype (measured: `StudentT(df=1e6).df.dtype` is float32), so an
    # explicit float64 `df` tensor is required for `log_prob` to actually run
    # the lgamma cancellation in float64 -- passing a bare float silently
    # reintroduces the precision loss the docstring above describes.
    nu_tensor = torch.tensor(nu, dtype=torch.float64, device=z.device)
    pdf = torch.distributions.StudentT(df=nu_tensor).log_prob(z).exp()
    cdf = _student_t_cdf(z, nu)

    log_bfrac = (
        math.lgamma(nu - 0.5)
        + 2.0 * math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5)
        - math.lgamma(nu)
        - 2.0 * math.lgamma(0.5 * nu)
    )
    sqrt_nu_bfrac = math.exp(0.5 * math.log(nu) + log_bfrac)

    crps_std = z * (2.0 * cdf - 1.0) + (2.0 / (nu - 1.0)) * (
        pdf * (nu + z.square()) - sqrt_nu_bfrac
    )
    return (sigma * crps_std).to(dtype=dtype, device=device)


class ClosedFormStudentTCRPS(Metric):
    """Per-site closed-form CRPS under the correct univariate Student-t marginal.

    Only accumulates on batches carrying a ``nu`` (a genuine Student-t head);
    :meth:`compute` raises the usual "no samples" ``RuntimeError`` for a batch
    stream that never had one, exactly like every other metric here on an empty
    accumulator -- the caller (see ``scripts/eval/encoder_processor_decoder.py``)
    already catches and skips that case for the existing metric registry.
    """

    name: str = "crps_studentt_closed_form"

    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_sites", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stats: ClosedFormPredictiveStats) -> None:
        if stats.nu is None:
            return
        pointwise = student_t_crps_pointwise(
            stats.mu, stats.sigma_ii, stats.nu, stats.target
        )
        self.sum_crps = self.sum_crps + pointwise.sum()
        self.n_sites = self.n_sites + pointwise.numel()

    def compute(self) -> Tensor:
        if self.n_sites == 0:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)
        return self.sum_crps / self.n_sites
