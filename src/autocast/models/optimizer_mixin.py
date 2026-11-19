"""Optimizer configuration mixin for Lightning modules."""

import math
import os
import statistics
import warnings
from collections import deque
from enum import StrEnum
from functools import partial
from typing import Any

import heavyball
import torch
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig, OmegaConf
from torch import nn

#: How many recent monitored values the plateau rule keeps to estimate the
#: metric's own epoch-to-epoch noise. Long enough for a stable median absolute
#: successive difference, short enough to reflect the CURRENT regime rather than
#: the steep early descent.
_NOISE_WINDOW_EPOCHS = 12


#: A parameter with fewer dimensions (a bias, a gain, a scalar) is not a matrix.
_MATRIX_NDIM = 2

#: ``plateau_monitor`` when not named: the rank ladder's ``val/fit``, else ``val_loss``.
PLATEAU_MONITOR_AUTO = "auto"


class WeightDecayScope(StrEnum):
    """Which parameters ``optimizer.weight_decay`` applies to. A closed set.

    ``ALL`` (the default) decays every parameter outside the shape/auxiliary
    groups, exactly as before this option existed. ``MATRICES`` decays only
    parameters with two or more dimensions and none a module declares through
    ``no_weight_decay_parameters()``: biases and gains are never decayed, and a
    distribution head is never decayed (decaying the log-variance bias would
    pull the predicted variance towards 1, a change unrelated to the weights it
    is meant to regularise).
    """

    ALL = "all"
    MATRICES = "matrices"


#: The optimizer-config key a warm start sets to start Prodigy's step-size estimate at
#: the source's (a name shared with ``analysis.warm_start_inherit``, not a setting).
PRODIGY_D0_KEY = "prodigy_d0"
#: The optimizer-config key that scales Prodigy's step-size estimate (its ``d_coef``).
PRODIGY_D_COEF_KEY = "prodigy_d_coef"


class OptimizerName(StrEnum):
    """The optimizers ``_create_optimizer`` can build. A closed set."""

    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    PSGD = "psgd"
    #: Adam whose step size is its own running estimate times ``learning_rate``, a
    #: multiplier left at 1 (Mishchenko and Defazio, ICML 2024; ``prodigyopt``).
    #: Added 2026-11-15 after D208's 0.08 x band top diverged in the first epoch on
    #: 10 toy runs: a fitted fraction carried to systems it was not fitted on.
    PRODIGY = "prodigy"

    @property
    def sets_own_step_size(self) -> bool:
        """Whether ``learning_rate`` is a multiplier on the optimizer's own step."""
        return self is OptimizerName.PRODIGY


def optimizer_name(cfg: DictConfig | dict[str, Any]) -> OptimizerName:
    """The configured optimizer, refused by name when it is not one we build."""
    raw = str(cfg.get("optimizer")).lower()
    try:
        return OptimizerName(raw)
    except ValueError:
        msg = f"Unsupported optimizer: {raw}"
        raise ValueError(msg) from None


class SchedulerName(StrEnum):
    """The schedules ``_create_scheduler`` can build. A closed set."""

    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    STEP = "step"
    PLATEAU = "plateau"


#: Which config keys each schedule actually READS. The union is the set of keys
#: that mean something to some schedule; anything outside it (``learning_rate``,
#: ``betas``, ``weight_decay``, ...) is not a scheduler key and is never judged
#: here.
#:
#: This map exists to make a silently-ignored key impossible. Setting
#: ``plateau_patience`` on a cosine run, or ``cosine_epochs`` on a plateau one,
#: did exactly nothing and said exactly nothing. That is not hypothetical: the
#: campaign's schedule was ruled as "constant LR after warmup", ``warmup`` was
#: read only by the cosine branch, and every cell in a 105-cell campaign trained
#: with no warmup at all. Nothing surfaced it, because an ignored key raises
#: nothing. Keep a schedule's entry in step with what its branch reads.
SCHEDULER_KEYS: dict[SchedulerName, frozenset[str]] = {
    SchedulerName.COSINE: frozenset(
        {
            "scheduler_interval",
            "min_lr_ratio",
            "warmup",
            "cosine_epochs",
            "cosine_steps",
            "cosine_t_max",
        }
    ),
    SchedulerName.COSINE_WITH_RESTARTS: frozenset(
        {
            "scheduler_interval",
            "min_lr_ratio",
            "warmup",
            "cosine_epochs",
            "cosine_steps",
            "cosine_t_max",
        }
    ),
    SchedulerName.STEP: frozenset({"scheduler_interval", "step_size", "gamma"}),
    SchedulerName.PLATEAU: frozenset(
        {
            "scheduler_interval",
            "min_lr_ratio",
            "warmup",
            "plateau_patience",
            "plateau_factor",
            "plateau_cooldown",
            "plateau_threshold",
            "plateau_threshold_mode",
            "plateau_monitor",
        }
    ),
}

#: Every key that means something to at least one schedule.
ALL_SCHEDULER_KEYS: frozenset[str] = frozenset().union(*SCHEDULER_KEYS.values())


def min_epochs_to_lr_floor(
    *, patience: int, cooldown: int, factor: float, min_lr_ratio: float
) -> int:
    """Fewest epochs in which a plateau schedule COULD reach its rate floor.

    The convergence gate asks whether the schedule bottomed out, so a run whose
    budget cannot physically hold this many epochs can never pass it, however
    well it trains. The count is fixed by config alone -- it assumes the metric
    never improves once, which is the fastest the rate can possibly fall -- so it
    is a floor on the budget, not a prediction. A real run needs this PLUS
    however many epochs it spends still improving.

    The arithmetic, MEASURED against the scheduler itself rather than read off
    torch's source (2026-08-15, four configurations, all exact):

    * the first ``step`` records ``best`` instead of counting a bad epoch, so the
      first cut lands at ``patience + 2``, not ``patience + 1``;
    * after a cut, ``cooldown`` epochs pass with the bad-epoch counter held at
      zero, so later cuts are ``patience + cooldown + 1`` apart;
    * ``n_cuts`` is how many multiplications by ``factor`` reach the floor.

    Checked: ``patience=15, cooldown=5, factor=0.3, min_lr_ratio=0.01`` gives 4
    cuts at epochs 17/38/59/**80** -- the shipped ``campaign2_plateau`` config,
    and the "80 epochs of NO improvement" in `DECISIONS.md`.
    """
    if not 0.0 < factor < 1.0:
        msg = f"factor must be in (0, 1). Got: {factor}"
        raise ValueError(msg)
    if not 0.0 < min_lr_ratio < 1.0:
        msg = (
            "min_lr_ratio must be in (0, 1) to have a reachable floor. "
            f"Got: {min_lr_ratio}"
        )
        raise ValueError(msg)
    n_cuts = math.ceil(math.log(min_lr_ratio) / math.log(factor))
    return (patience + 2) + (n_cuts - 1) * (patience + cooldown + 1)


#: Set to "1" to resume a cosine run under a DIFFERENT horizon than it was
#: trained with, accepting that its early epochs annealed toward a horizon that
#: no longer exists.
ALLOW_COSINE_HORIZON_CHANGE_ENV_VAR = "AUTOCAST_ALLOW_COSINE_HORIZON_CHANGE"


class HorizonAwareLambdaLR(torch.optim.lr_scheduler.LambdaLR):
    """``LambdaLR`` that REMEMBERS its cosine horizon and refuses a silent change.

    The horizon is what makes a cosine schedule a schedule: it decides how fast
    the rate anneals and therefore what every epoch before now was aiming at.
    But it lives in a Python closure, and ``LambdaLR.state_dict`` deliberately
    drops plain-function lambdas, so nothing about the horizon reached the
    checkpoint. Resuming with a different ``trainer.max_epochs`` silently
    re-plans the anneal: measured 2026-08-15, a run resumed with a larger budget
    climbed from its 5% floor back to 75.7% of base and then followed the longer
    run's path exactly.

    That is mechanically clean and scientifically incoherent -- the run's early
    epochs annealed toward a horizon that no longer exists, so the result is
    neither the short schedule nor the long one, and nothing on disk records
    that it happened. Every other attribute rides in ``state_dict`` for free
    (torch keeps all of ``__dict__`` except the optimizer and the lambdas), so
    the horizon does too once it is an attribute; this class adds the comparison
    that turns a silent re-plan into a refusal.

    The escape hatch is deliberate and explicit: extending a cosine run IS
    sometimes what you want, and the ruling on record is that it counts as
    PASS-mechanical / FAIL-coherent. Set
    ``AUTOCAST_ALLOW_COSINE_HORIZON_CHANGE=1`` to say so out loud.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_lambda: Any,
        *,
        cosine_horizon: int,
        horizon_units: str,
    ) -> None:
        self.cosine_horizon = int(cosine_horizon)
        self.horizon_units = horizon_units
        super().__init__(optimizer, lr_lambda=lr_lambda)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, refusing a horizon that disagrees with the checkpoint."""
        raw = state_dict.get("cosine_horizon")
        saved_units = state_dict.get("horizon_units", self.horizon_units)
        saved = None if raw is None else int(raw)
        current = self.cosine_horizon
        if saved is None or saved == current:
            super().load_state_dict(state_dict)
            return

        if os.environ.get(ALLOW_COSINE_HORIZON_CHANGE_ENV_VAR) != "1":
            msg = (
                f"REFUSING TO RESUME: this checkpoint was trained under a cosine "
                f"horizon of {saved} {saved_units}, but this run resolves to "
                f"{current} {self.horizon_units}. The rate would jump back up and "
                f"re-anneal toward a horizon the completed epochs never targeted, "
                f"so the result is neither schedule and nothing records the "
                f"change. Resume with the original horizon (e.g. set "
                f"optimizer.cosine_epochs={saved}), or set "
                f"{ALLOW_COSINE_HORIZON_CHANGE_ENV_VAR}=1 to extend deliberately "
                f"-- an extended cosine run is mechanically sound but cannot be "
                f"read for convergence."
            )
            raise ValueError(msg)

        super().load_state_dict(state_dict)
        # The operator opted in; keep the NEW horizon (that is the point of
        # extending) but leave a record that the schedule was re-planned.
        # `super().load_state_dict` has just overwritten it with the saved
        # value, so it must be restored here, after the update.
        self.cosine_horizon = current
        warnings.warn(
            f"Cosine horizon changed on resume: {saved} -> {current} "
            f"{self.horizon_units}. The epochs already trained annealed "
            f"toward the old horizon, so this run's tail cannot be read for "
            f"convergence.",
            UserWarning,
            stacklevel=2,
        )

    @property
    def lr(self) -> list[float]:
        """Current learning rates for each parameter group.

        Exposed for compatibility with PyTorch Lightning's ``Tuner.lr_find``,
        which reads ``scheduler.lr[0]`` in its batch start callback.
        """
        return [float(group["lr"]) for group in self.optimizer.param_groups]


class WarmupReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """``ReduceLROnPlateau`` with a linear warmup and a guard on ``rel`` mode.

    Closes two defects found on 2026-08-08 against a finished 105-cell campaign.

    THE WARMUP. The campaign's schedule ruling (``CAMPAIGN_PROTOCOL_2026-08-05``
    D1) reads "constant LR **after warmup**, decayed by ``ReduceLROnPlateau``",
    but warmup was implemented only on the cosine branch — the plateau branch
    never read the key, so ``warmup`` was silently inert and every cell trained
    with none. Nothing surfaced it because an ignored key raises nothing.

    THE SIGN GUARD. torch computes the ``rel`` improvement bar as
    ``best * (1 - threshold)`` (``ReduceLROnPlateau.is_better``). For a NEGATIVE
    objective that bar sits **above** the running best — with ``best = -4.58``
    and ``threshold = 0.0072`` it is ``-4.547`` — so a value equal to, or even
    slightly worse than, the best still counts as an improvement, the bad-epoch
    counter resets every epoch and the rate never decays. Measured consequence:
    all five ``l96c``/``lowrank`` cells finished at their base rate having never
    cut once, which then failed the convergence gate's learning-rate leg for a
    configuration reason rather than a scientific one. A relative bar is not
    meaningful for a sign-indefinite objective, so this raises rather than
    quietly re-deriving one: ``plateau_threshold_mode: abs`` is the fix, and the
    run should be told at its first validation rather than after its budget.

    Parameters
    ----------
    optimizer
        The optimizer whose learning rate is scheduled.
    warmup_epochs
        Number of epochs to ramp the rate linearly from ``base_lr /
        (warmup_epochs + 1)`` to ``base_lr``. Zero disables the ramp, leaving
        plain ``ReduceLROnPlateau`` behaviour.
    **kwargs
        Passed through to :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_epochs: int = 0,
        **kwargs: Any,
    ) -> None:
        # torch validates threshold_mode in its own __init__ and knows only
        # 'rel'/'abs'. Hand it a valid one, then restore ours -- the base class
        # reads threshold_mode nowhere except is_better, which we override.
        sign_safe_rel = kwargs.get("threshold_mode") == "rel_abs"
        if sign_safe_rel:
            kwargs = {**kwargs, "threshold_mode": "abs"}
        super().__init__(optimizer, **kwargs)
        if sign_safe_rel:
            self.threshold_mode = "rel_abs"
        if warmup_epochs < 0:
            msg = f"warmup_epochs must be >= 0. Got: {warmup_epochs}"
            raise ValueError(msg)
        self.warmup_epochs = warmup_epochs
        self.warmup_seen = 0
        # Recent monitored values, for the threshold-vs-noise check in `step`.
        self._recent: deque[float] = deque(maxlen=_NOISE_WINDOW_EPOCHS)
        self._noise_warned = False
        # Captured before the first ramp so the peak is the configured rate,
        # not whatever the ramp last wrote.
        self._warmup_base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if self.warmup_epochs > 0:
            self._apply_warmup_lr()

    @property
    def in_warmup(self) -> bool:
        """Whether the ramp is still running."""
        return self.warmup_seen < self.warmup_epochs

    @property
    def lr(self) -> list[float]:
        """Current learning rates for each parameter group.

        Exposed for compatibility with PyTorch Lightning's ``Tuner.lr_find``,
        which reads ``scheduler.lr[0]`` in its batch start callback.
        """
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _apply_warmup_lr(self) -> None:
        """Set every group's rate to the current point on the linear ramp."""
        scale = float(self.warmup_seen + 1) / float(self.warmup_epochs + 1)
        for group, base in zip(
            self.optimizer.param_groups, self._warmup_base_lrs, strict=True
        ):
            group["lr"] = base * scale
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    #: Diagnostic-only attributes, kept OUT of the checkpoint. See `state_dict`.
    _EPHEMERAL_STATE = ("_recent", "_noise_warned")

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state, WITHOUT the noise-check's working memory.

        torch's ``ReduceLROnPlateau.state_dict`` returns every attribute except
        the optimizer, so the ``deque`` backing the threshold-vs-noise check
        would ride into every checkpoint. MEASURED on torch 2.8: a checkpoint
        containing it **cannot be loaded with ``weights_only=True``**, which has
        been ``torch.load``'s default since 2.6 --
        ``UnpicklingError: Weights only load failed``. That would have made
        every new checkpoint unloadable on the standard path, the third defect
        in this class on this class today, and the two before it both reached a
        finished campaign before anyone noticed.

        Dropping it costs nothing: the window is a within-run diagnostic that
        refills in ``_NOISE_WINDOW_EPOCHS`` epochs after a resume, and
        ``load_state_dict`` tolerates its absence (a legacy checkpoint has none
        either).
        """
        state = super().state_dict()
        for key in self._EPHEMERAL_STATE:
            state.pop(key, None)
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state AND repair the rates ``__init__`` mutated on the way in.

        ``__init__`` captures ``_warmup_base_lrs`` and then, when
        ``warmup_epochs > 0``, applies the first ramp point immediately -- both
        with a ``warmup_seen`` of 0, because no state has been restored yet. The
        inherited ``load_state_dict`` puts the counter back but never touches the
        optimizer, so without this override a resumed run continues at
        ``base / (warmup_epochs + 1)`` for the rest of training: measured before
        this fix, a run resumed after a 2-epoch warmup came back at 0.00333
        instead of 0.01, and once ``warmup_seen`` is past ``warmup_epochs`` the
        ramp never runs again to correct it.

        ``_last_lr`` is the authority for the post-warmup case, NOT
        ``_warmup_base_lrs`` -- by then the plateau may legitimately have decayed
        the rate, and restoring the base would silently undo that decay.

        THE ``rel_abs`` DANCE, and why it must be repeated here. ``__init__``
        hides our custom threshold mode from torch's validator by passing
        ``"abs"`` and restoring ``"rel_abs"`` afterwards. torch's
        ``load_state_dict`` does ``self.__dict__.update(state_dict)`` and THEN
        calls ``_init_is_better(threshold_mode=self.threshold_mode)``, which
        validates against ``{"rel", "abs"}`` -- so a state dict carrying
        ``"rel_abs"`` raises ``ValueError: threshold mode rel_abs is unknown!``
        and **the run cannot be resumed at all**. Measured 2026-08-15: every
        checkpoint written by ``optimizer/campaign2_plateau.yaml`` since
        ``8a542354`` was unresumable, and it went unnoticed because
        ``test_warmup_plateau_resume.py`` never sets ``threshold_mode`` and so
        exercises only torch's default.
        """
        sign_safe_rel = state_dict.get("threshold_mode") == "rel_abs"
        if sign_safe_rel:
            state_dict = {**state_dict, "threshold_mode": "abs"}
        super().load_state_dict(state_dict)
        if sign_safe_rel:
            self.threshold_mode = "rel_abs"
        if self.warmup_epochs <= 0:
            return  # __init__ mutated nothing; there is nothing to repair
        if self.in_warmup:
            self._apply_warmup_lr()
            return
        last_lr = getattr(self, "_last_lr", None)
        if last_lr:
            for group, lr in zip(self.optimizer.param_groups, last_lr, strict=True):
                group["lr"] = lr

    def _fallback_is_better(self, a: Any, best: Any) -> bool:
        """Delegate to torch's native comparison, whichever name it goes by.

        The public ``is_better`` was renamed to a private ``_is_better`` in
        PyTorch 2.10, so neither name can be assumed present. The literal
        formulas are the last resort for a version carrying neither.
        """
        base = super()
        native = getattr(base, "_is_better", None) or getattr(base, "is_better", None)
        if native is not None:
            return bool(native(a, best))
        if self.mode == "min" and self.threshold_mode == "rel":
            return bool(a < best * (1.0 - self.threshold))
        if self.mode == "min" and self.threshold_mode == "abs":
            return bool(a < best - self.threshold)
        if self.mode == "max" and self.threshold_mode == "rel":
            return bool(a > best * (self.threshold + 1.0))
        return bool(a > best + self.threshold)

    def is_better(self, a: Any, best: Any) -> bool:
        """Sign-safe relative comparison for ``threshold_mode='rel_abs'``.

        torch offers only ``rel`` (``best * (1 - threshold)``) and ``abs``
        (``best - threshold``). The first is what "an improvement smaller than
        one epoch's noise, expressed relative to the current best" was meant to
        say, but its formula is only valid for a POSITIVE metric: scaling a
        negative best by ``(1 - threshold)`` moves the bar UP, above the best,
        so anything at all counts as an improvement.

        ``rel_abs`` states the intended semantics directly -- the bar is
        ``best - threshold * |best|`` -- which is well defined for either sign
        and **identical to torch's ``rel`` whenever ``best > 0``**, so it is a
        strict generalisation rather than a behaviour change for the arms that
        were already fine.

        Why this and not simply ``abs``: an absolute bar needs the noise scale
        of each arm's own objective, and the pilot measured that for the anchor
        arm alone (the campaign's per-system thresholds are ``sigma_epoch /
        best_value`` from that one arm). A relative bar is what makes one
        measured number transfer across arms whose objectives differ in scale;
        it just has to be written sign-safely.
        """
        if self.threshold_mode != "rel_abs":
            return self._fallback_is_better(a, best)
        if best == self.mode_worse:
            # `best` is torch's `mode_worse` sentinel (+/-inf) until the first
            # epoch is scored, and the relative bar is UNDEFINED there:
            # `inf - threshold*abs(inf)` is nan, and every comparison against
            # nan is False. Without this branch `best` never leaves infinity,
            # nothing ever counts as an improvement, and the bad-epoch counter
            # runs free -- so the rate decays on a fixed patience+cooldown
            # timer, blind to the metric. Measured on a 132-epoch run: cuts at
            # epochs 16/37/58/79, exactly 21 apart, with `best` still inf in
            # the final checkpoint. torch's own `rel` and `abs` bars are
            # unaffected (`inf*(1-t)` and `inf-t` are both inf, which any
            # finite value beats), which is why only this branch needs it.
            # Any finite value improves on "nothing recorded yet"; a nan does
            # not. Tested against the SENTINEL specifically, not against any
            # non-finite `best`: a run whose metric genuinely reached -inf in
            # `min` mode has a real best that nothing can improve on, and it
            # must fall through to the bar below (where the margin is inf and
            # the comparison correctly returns False) rather than be treated as
            # unset.
            return bool(math.isfinite(a))
        margin = self.threshold * abs(best)
        if self.mode == "min":
            return bool(a < best - margin)
        return bool(a > best + margin)

    # PyTorch 2.10 renamed the comparison hook `is_better` -> `_is_better`, and
    # `step` calls the private name. Overriding only the public one left the
    # `rel_abs` logic above UNREACHABLE on >=2.10: torch's own `_is_better` has
    # no `rel_abs` branch, so it fell through to its `mode='max'` arm,
    # `a > best + threshold`, which against the `inf` sentinel is always False.
    # `best` then never left infinity and the rate cut on a blind
    # patience+cooldown timer -- measured on job 6029506 (Isambard, torch
    # 2.10.0+cu126): 136 epochs, `best: inf` in the final checkpoint, cuts at
    # epochs 16/37/58/79, exactly 21 apart. Both names must be defined; keep
    # them delegating to one implementation so they cannot drift. A delegating
    # method rather than an alias (`_is_better = is_better`), so that a subclass
    # overriding `is_better` is still the one torch ends up calling.
    def _is_better(self, a: Any, best: Any) -> bool:
        """Torch >=2.10 calls this name; :meth:`is_better` is the implementation."""
        return self.is_better(a, best)

    def _improvement_bar(self) -> float | None:
        """The absolute improvement this rule currently demands, or None.

        None while ``best`` is still the ``mode_worse`` sentinel, where no
        relative bar is defined.
        """
        best = self.best
        if best is None or not math.isfinite(float(best)):
            return None
        best = float(best)
        if self.threshold_mode == "rel_abs":
            return self.threshold * abs(best)
        if self.threshold_mode == "rel":
            return abs(best) * self.threshold
        return self.threshold

    def _check_threshold_against_noise(self) -> None:
        """Warn when the improvement bar sits UNDER the metric's own noise.

        A plateau rule only works if "did it improve?" is answerable. When the
        bar the rule demands is well below the metric's epoch-to-epoch scatter,
        noise alone clears it: the bad-epoch counter resets on luck, and the rate
        decays late or never. The run looks healthy throughout -- it simply never
        settles, and the convergence gate then reports a rate-floor failure whose
        real cause is the threshold.

        This is not hypothetical for the arm this was written on. Measured
        2026-08-15 over 47 epochs with the rate pinned: ``val_loss`` scatters
        with sd 0.001047 while ``val/fit`` scatters with sd 0.000028, a factor of
        37, because ``val_loss`` scores a randomly drawn rank per batch item plus
        a regulariser. Against a best near 0.036 the shipped relative threshold
        of 1e-4 asks for an improvement of ~3.6e-6 -- comfortably UNDER even the
        quieter metric's noise.

        Warns rather than raises: the numbers only exist mid-run, and killing a
        job at epoch 12 over a diagnostic would destroy work that may still be
        usable. The warning carries every number needed to act.
        """
        if self._noise_warned or len(self._recent) < _NOISE_WINDOW_EPOCHS:
            return
        values = list(self._recent)
        # SCATTER, not step size, so the curve is DETRENDED first. Two wrong
        # ways, both tried and both caught by this module's own tests:
        #   * the median absolute successive difference reads a steadily
        #     improving run as pure noise -- a run marching down by 0.001 an
        #     epoch has |diff| = 0.001 every epoch and is the healthiest case
        #     there is;
        #   * taking those differences' deviation about their own median fixes
        #     that but collapses on a period-2 curve, where an odd number of
        #     alternating +n/-n differences has median +n (not 0), leaving every
        #     deviation at 0 or 2n and the median at 0 -- i.e. "no noise" for
        #     the noisiest possible curve.
        # A least-squares line removes any monotone trend without either
        # failure; the residual spread about it is the scatter we mean.
        index = list(range(len(values)))
        slope, intercept = statistics.linear_regression(index, values)
        residuals = [
            v - (slope * i + intercept) for i, v in zip(index, values, strict=True)
        ]
        level = statistics.median(residuals)
        noise = statistics.median(abs(r - level) for r in residuals)
        bar = self._improvement_bar()
        if bar is None or noise <= 0.0 or bar >= noise:
            return
        self._noise_warned = True
        warnings.warn(
            f"The plateau rule's improvement bar ({bar:.6g}) is BELOW the "
            f"monitored metric's own epoch-to-epoch noise ({noise:.6g}, median "
            f"absolute successive difference over {_NOISE_WINDOW_EPOCHS} epochs). "
            f"Noise alone will clear the bar and reset the bad-epoch counter, so "
            f"the rate will decay late or never, and the run will fail the "
            f"convergence gate's rate-floor leg for a threshold reason rather "
            f"than a scientific one. Raise optimizer.plateau_threshold above "
            f"{noise / max(abs(float(self.best)), 1e-30):.3g} (relative), or "
            f"monitor a quieter metric -- an all-rungs `val/fit` scatters 37x "
            f"less than `val_loss` on the ladder arms.",
            UserWarning,
            stacklevel=2,
        )

    def step(self, metrics: Any, epoch: Any = None) -> None:
        """Advance the schedule, ramping first and only then decaying."""
        current = float(metrics)
        if self.threshold_mode == "rel" and current < 0.0:
            bar = current * (1.0 - self.threshold)
            msg = (
                "plateau_threshold_mode='rel' cannot schedule a NEGATIVE "
                f"monitored value. Got: {current}. torch's relative bar is "
                f"best*(1-threshold) = {bar}, which is ABOVE the value itself, "
                "so every epoch counts as an improvement and the rate will "
                "never decay. Set optimizer.plateau_threshold_mode=abs (and an "
                "absolute optimizer.plateau_threshold) for an objective that "
                "can go negative."
            )
            raise ValueError(msg)
        if self.in_warmup:
            self.warmup_seen += 1
            self.last_epoch += 1
            self._apply_warmup_lr()
            return
        super().step(metrics, epoch)
        # After the step, so `best` reflects this epoch and the bar is the one
        # the NEXT epoch must clear.
        self._recent.append(current)
        self._check_threshold_against_noise()


class OptimizerMixin(nn.Module):
    """Mixin class providing optimizer configuration for Lightning modules.

    Inherits from nn.Module to ensure parameters() method is available.
    Requires the class to have:
        - self.optimizer_config: DictConfig | dict[str, Any] | None
        - self.trainer: Lightning Trainer instance (optional, for scheduler)
    """

    # Type hints for attributes expected from the concrete class
    optimizer_config: DictConfig | dict[str, Any] | None

    @staticmethod
    def _precond_prob_schedule(
        n: int,
        *,
        max_prob: float = 1.0,
        min_prob: float = 0.01,
        decay: float = 0.999,
        flat_start: int = 0,
    ) -> float:
        """Preconditioner update probability schedule.

        Implemented to align with LOLA:
        https://github.com/francois-rozet/lola/blob/21a4354b327e6e5ee06da5075ba3bd1dd88c61f1/lola/optim.py
        """
        return max(min_prob, max_prob * decay ** max(n - flat_start, 0))

    @staticmethod
    def _reject_ignored_scheduler_keys(
        cfg: dict[str, Any], scheduler_name: SchedulerName | None
    ) -> None:
        """Refuse a config whose scheduler keys the chosen schedule never reads.

        An ignored key is the quietest defect this codebase has produced. Twice
        now a knob was set, recorded in the resolved config, believed to be in
        force, and read by nobody: ``warmup`` on the plateau branch (2026-08-08,
        a whole 105-cell campaign), and every ``plateau_*`` key whenever
        ``scheduler`` is null -- which is how the constant-rate arm was launched
        on 2026-08-15, carrying five plateau knobs that could not act. Both were
        found by reading the source, not by anything the run said.

        Only keys that belong to SOME schedule are judged, so this cannot fire
        on optimizer-level settings or on a config carrying extra fields for
        other consumers. A key explicitly set to null is "unset", not "ignored".

        Raises:
        ValueError
            If any key set here is read by a different schedule than the one
            selected, naming the key, its owner, and the fix.
        """
        read_here = (
            frozenset() if scheduler_name is None else SCHEDULER_KEYS[scheduler_name]
        )
        offending: dict[str, list[str]] = {}
        for key in sorted(set(cfg) & ALL_SCHEDULER_KEYS):
            if cfg.get(key) is None or key in read_here:
                continue  # unset, or genuinely read by the selected schedule
            offending[key] = sorted(
                name.value for name, keys in SCHEDULER_KEYS.items() if key in keys
            )
        if not offending:
            return

        selected = "null (no scheduler)" if scheduler_name is None else scheduler_name
        detail = "; ".join(
            f"{key!r}={cfg[key]!r} is read only by scheduler={'/'.join(owners)}"
            for key, owners in offending.items()
        )
        fix = (
            "use optimizer=constant, which declares a constant rate as the "
            "schedule rather than as a plateau config with its scheduler removed"
            if scheduler_name is None
            else "remove it, or select the scheduler that reads it"
        )
        msg = (
            f"REFUSING TO START: optimizer.scheduler is {selected}, so these "
            f"keys are set but will never be read -- {detail}. An ignored knob "
            f"is recorded in the resolved config and looks in force, which is "
            f"how a 105-cell campaign trained with no warmup. To fix: {fix}."
        )
        raise ValueError(msg)

    def _get_scheduler_interval(self, cfg: dict[str, Any]) -> str:
        """Return scheduler interval ('epoch', 'step', or 'time')."""
        interval = str(cfg.get("scheduler_interval", "epoch")).lower()
        if interval not in {"epoch", "step", "time"}:
            msg = (
                "scheduler_interval must be 'epoch', 'step', or 'time'. "
                f"Got: {interval}"
            )
            raise ValueError(msg)
        return interval

    def _as_positive_int(self, value: Any, field_name: str) -> int:
        """Convert value to a positive integer."""
        int_value = int(value)
        if int_value <= 0:
            msg = f"{field_name} must be a positive integer. Got: {value}"
            raise ValueError(msg)
        return int_value

    def _resolve_cosine_horizon(self, cfg: dict[str, Any], interval: str) -> int:
        """Resolve the cosine horizon used in the scheduler lambda."""
        explicit_key = "cosine_steps" if interval == "step" else "cosine_epochs"
        explicit_value = cfg.get(explicit_key)
        if explicit_value is not None:
            return self._as_positive_int(explicit_value, explicit_key)

        trainer = getattr(self, "trainer", None)
        if interval == "step" and trainer is not None:
            estimated_steps = getattr(trainer, "estimated_stepping_batches", None)
            if estimated_steps is not None:
                try:
                    estimated_int = int(estimated_steps)
                except (TypeError, ValueError):
                    estimated_int = 0
                if estimated_int > 0:
                    return estimated_int

        if trainer is not None and trainer.max_epochs is not None:
            return self._as_positive_int(trainer.max_epochs, "trainer.max_epochs")

        fallback = cfg.get("cosine_t_max", 1)
        return self._as_positive_int(fallback, "cosine_t_max")

    def _shape_parameter_ids(self) -> set[int]:
        """Identity set of parameters submodules declare as *shape* parameters.

        A shape parameter (e.g. the Student-t tail index ``raw_nu``) is a single
        global scalar whose only signal is the strictly-proper log-score. A
        module opts in by exposing ``shape_parameters() -> list[nn.Parameter]``;
        :meth:`_build_param_groups` then gives those parameters a dedicated LR
        and excludes them from weight decay.
        """
        ids: set[int] = set()
        for module in self.modules():
            get_shape = getattr(module, "shape_parameters", None)
            if callable(get_shape):
                shape_params: Any = get_shape()
                ids.update(id(p) for p in shape_params)
        return ids

    def _auxiliary_parameter_ids(self) -> set[int]:
        """Identity set of parameters submodules declare as *auxiliary*.

        An auxiliary parameter belongs to a training-time-only network that
        tracks a moving target and must therefore be stepped at its own rate --
        Mean Flow Distillation's velocity field is the case this exists for,
        where the authors run the auxiliary at 10x the student's rate. A module
        opts in by exposing ``auxiliary_parameters() -> list[nn.Parameter]``.
        """
        ids: set[int] = set()
        for module in self.modules():
            get_aux = getattr(module, "auxiliary_parameters", None)
            if callable(get_aux):
                aux_params: Any = get_aux()
                ids.update(id(p) for p in aux_params)
        return ids

    def _no_weight_decay_parameter_ids(
        self, cfg: DictConfig | dict[str, Any], weight_decay: float
    ) -> set[int]:
        """Identity set of parameters ``optimizer.weight_decay_scope`` exempts.

        Empty for :attr:`WeightDecayScope.ALL` or when there is no decay, so the
        optimizer is built exactly as before the option existed.
        """
        scope = WeightDecayScope(
            str(cfg.get("weight_decay_scope", WeightDecayScope.ALL))
        )
        if scope is WeightDecayScope.ALL or weight_decay == 0.0:
            return set()
        ids = {id(p) for p in self.parameters() if p.ndim < _MATRIX_NDIM}
        for module in self.modules():
            get_exempt = getattr(module, "no_weight_decay_parameters", None)
            if callable(get_exempt):
                exempt: Any = get_exempt()
                ids.update(id(p) for p in exempt)
        return ids

    def _build_param_groups(
        self, cfg: DictConfig | dict[str, Any], *, lr: float, weight_decay: float
    ) -> Any:
        """Split parameters so shape parameters get their own LR and no decay.

        Returns ``self.parameters()`` unchanged — a single implicit group,
        byte-identical to the previous construction — when no submodule declares
        ``shape_parameters()``. Otherwise returns two explicit groups: the bulk
        of the network at ``lr`` / ``weight_decay`` and the shape parameters at
        ``lr * nu_lr_mult`` with ``weight_decay=0``.

        The scheduler installs a single lambda, which torch applies to every
        group, so the shape group keeps the same cosine shape, scaled by
        ``nu_lr_mult``. The multiplier defaults to 1.0 (LR unchanged); the only
        divergence from the single-group path is then the decay exclusion, and
        only when ``weight_decay > 0`` — decay on a tail index is an arbitrary
        prior, so shape parameters never decay.
        """
        shape_ids = self._shape_parameter_ids()
        aux_ids = self._auxiliary_parameter_ids()
        no_decay_ids = self._no_weight_decay_parameter_ids(cfg, weight_decay)
        if not shape_ids and not aux_ids and not no_decay_ids:
            return self.parameters()
        if no_decay_ids and aux_ids:
            msg = (
                "optimizer.weight_decay_scope=matrices is not supported with "
                "auxiliary parameters (their group would decay its biases)."
            )
            raise ValueError(msg)

        nu_lr_mult = float(cfg.get("nu_lr_mult", 1.0))
        if nu_lr_mult <= 0.0:
            msg = f"nu_lr_mult must be positive. Got: {nu_lr_mult}"
            raise ValueError(msg)
        aux_lr_mult = float(cfg.get("aux_lr_mult", 1.0))
        if aux_lr_mult <= 0.0:
            msg = f"aux_lr_mult must be positive. Got: {aux_lr_mult}"
            raise ValueError(msg)

        shape_params: list[nn.Parameter] = []
        aux_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        rest_params: list[nn.Parameter] = []
        for param in self.parameters():
            # Shape first: the two sets are disjoint in every built model, and
            # stating the precedence beats leaving an overlap to iteration order.
            if id(param) in shape_ids:
                shape_params.append(param)
            elif id(param) in aux_ids:
                aux_params.append(param)
            elif id(param) in no_decay_ids:
                no_decay_params.append(param)
            else:
                rest_params.append(param)

        groups: list[dict[str, Any]] = [
            {"params": rest_params, "lr": lr, "weight_decay": weight_decay}
        ]
        if no_decay_params:
            groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
        if shape_params:
            groups.append(
                {"params": shape_params, "lr": lr * nu_lr_mult, "weight_decay": 0.0}
            )
        if aux_params:
            # Weight decay is NOT excluded here, unlike the shape group: the
            # auxiliary is an ordinary network, not a lone scalar whose decay
            # would be an arbitrary prior.
            groups.append(
                {
                    "params": aux_params,
                    "lr": lr * aux_lr_mult,
                    "weight_decay": weight_decay,
                }
            )
        return groups

    def _create_optimizer(
        self, cfg: DictConfig | dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        if not cfg.get("optimizer"):
            msg = "Optimizer name is required in optimizer_config."
            raise ValueError(msg)
        if cfg.get("learning_rate") is None:
            msg = "learning_rate is required in optimizer_config."
            raise ValueError(msg)
        if cfg.get("grad_clip") is not None:
            msg = (
                "optimizer.grad_clip is no longer read; gradient clipping is "
                "wired through trainer.gradient_clip_val. Move the value to "
                "trainer.gradient_clip_val (or set it to null to disable)."
            )
            raise ValueError(msg)
        name = optimizer_name(cfg)
        lr = cfg.get("learning_rate")
        if not isinstance(lr, (float, int)):
            msg = "learning_rate must be a number in optimizer_config."
            raise TypeError(msg)
        lr = float(lr)
        weight_decay = cfg.get("weight_decay", 0.0)
        # Route any declared shape parameters (e.g. the Student-t tail index) to
        # a dedicated LR / no-decay group; falls back to self.parameters() (a
        # single implicit group, byte-identical) when none are declared.
        params = self._build_param_groups(cfg, lr=lr, weight_decay=float(weight_decay))

        if name is OptimizerName.ADAMW:
            betas = cfg.get("betas", [0.9, 0.999])
            return torch.optim.AdamW(
                params, lr=lr, betas=betas, weight_decay=weight_decay
            )
        if name is OptimizerName.ADAM:
            betas = cfg.get("betas", [0.9, 0.999])
            return torch.optim.Adam(
                params, lr=lr, betas=betas, weight_decay=weight_decay
            )
        if name is OptimizerName.SGD:
            momentum = cfg.get("momentum", 0.9)
            return torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        if name is OptimizerName.PRODIGY:
            # The authors' defaults otherwise (d0, d_coef, no growth cap, decoupled
            # decay): nothing here is tuned to our runs. It refuses parameter groups
            # whose rates differ unless they are 0, so a shape group at another
            # multiplier fails loudly at the first step rather than silently.
            from prodigyopt import Prodigy  # noqa: PLC0415

            betas = cfg.get("betas", [0.9, 0.999])
            # ``prodigy_d0`` is set only by a warm start that carries the source's
            # step-size estimate over (``inherit_prodigy_d0``); absent, Prodigy's own
            # default d0 stands. ``prodigy_d_coef`` scales the step estimate, the
            # authors' knob for a smaller step (their README: 0.5 or 0.1); absent,
            # their 1.0.
            keys = (("d0", PRODIGY_D0_KEY), ("d_coef", PRODIGY_D_COEF_KEY))
            # Any: Prodigy also takes bool and int keywords, so pyright refuses a
            # dict of floats unpacked into its signature.
            extra: dict[str, Any] = {
                name: float(cfg[key]) for name, key in keys if key in cfg
            }
            return Prodigy(
                params, lr=lr, betas=betas, weight_decay=weight_decay, **extra
            )
        return self._create_psgd(
            cfg, params=params, lr=lr, weight_decay=float(weight_decay)
        )

    def _create_psgd(
        self,
        cfg: DictConfig | dict[str, Any],
        *,
        params: Any,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        betas = cfg.get("betas", [0.9])
        beta = float(next(iter(betas)))

        precondition_frequency = float(cfg.get("precondition_frequency", 16))
        precondition_frequency_decay = float(
            cfg.get("precondition_frequency_decay", 0.999)
        )
        precondition_size = int(cfg.get("precondition_size", 4096))
        merge_dims = bool(cfg.get("merge_dims", False))

        preconditioner_update_probability = cfg.get("preconditioner_update_probability")
        if preconditioner_update_probability is None:
            preconditioner_update_probability = partial(
                self._precond_prob_schedule,
                min_prob=1.0 / max(precondition_frequency, 1.0),
                decay=precondition_frequency_decay,
            )

        return heavyball.ForeachCachedDelayedPSGDKron(
            params,
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=precondition_size,
            merge_dims=merge_dims,
        )

    def _resolve_warmup(self, cfg: dict[str, Any], horizon: int) -> int:
        """Resolve warmup steps/epochs from config.

        A float in (0, 1) is treated as a fraction of the cosine horizon
        (e.g. 0.05 -> 5% of horizon). Anything else is interpreted as an
        absolute count. Negative values clamp to 0. NaN / inf are rejected.
        """
        warmup_raw = cfg.get("warmup", 0)
        if isinstance(warmup_raw, float) and not math.isfinite(warmup_raw):
            msg = f"warmup must be finite. Got: {warmup_raw}"
            raise ValueError(msg)
        if isinstance(warmup_raw, float) and 0.0 < warmup_raw < 1.0:
            warmup = int(warmup_raw * horizon)
        else:
            warmup = int(warmup_raw)
        return max(warmup, 0)

    def _resolve_time_warmup(self, cfg: dict[str, Any]) -> float:
        """Resolve warmup as a fraction of the wall-clock budget.

        With ``scheduler_interval='time'`` warmup is a fraction of
        ``trainer.max_time`` (a float in ``[0, 1)``); absolute step/epoch
        counts are meaningless against a wall-clock budget and are rejected.
        """
        warmup_cfg = cfg.get("warmup", 0)
        warmup_raw = 0.0 if warmup_cfg is None else float(warmup_cfg)
        if not math.isfinite(warmup_raw):
            msg = f"warmup must be finite. Got: {warmup_raw}"
            raise ValueError(msg)
        if not 0.0 <= warmup_raw < 1.0:
            msg = (
                "With scheduler_interval='time', warmup must be a fraction in "
                f"[0, 1) of the wall-clock budget. Got: {warmup_raw}"
            )
            raise ValueError(msg)
        return warmup_raw

    def _find_training_timer(self) -> Timer | None:
        """Return Lightning's Timer callback if one is installed."""
        trainer = getattr(self, "trainer", None)
        callbacks = getattr(trainer, "callbacks", []) if trainer is not None else []
        return next((cb for cb in callbacks if isinstance(cb, Timer)), None)

    def _require_time_budget(self) -> None:
        """Ensure a wall-clock budget exists for time-based scheduling.

        ``trainer.max_time`` installs Lightning's ``Timer`` with a duration
        (``max_time`` itself is not exposed as a trainer attribute). A Timer
        without a duration, or no Timer at all, means there is no budget to
        anneal over, so fail loud rather than silently never decaying.
        """
        timer = self._find_training_timer()
        if timer is None or timer.time_remaining() is None:
            msg = (
                "scheduler_interval='time' requires a wall-clock budget; set "
                "trainer.max_time (which installs Lightning's Timer)."
            )
            raise ValueError(msg)

    def _training_time_progress(self) -> float:
        """Fraction in [0, 1] of the wall-clock budget elapsed.

        Reads Lightning's ``Timer`` callback, whose elapsed offset is restored
        from checkpoints (and cleared by ``reset_resume_time_budget``).
        Rank zero's progress is broadcast so all model replicas use the same
        LR even when their local clocks or callback timings differ. The
        schedule therefore tracks the same clock that enforces ``max_time``: a
        single-job run anneals over the job; a full-state-resumed run anneals
        over the cumulative training time. Returns 0.0 (schedule start) if no
        Timer / budget is available, which only happens before training begins.
        """
        timer = self._find_training_timer()
        if timer is None:
            return 0.0
        remaining = timer.time_remaining()
        if remaining is None:
            return 0.0
        elapsed = timer.time_elapsed()
        budget = elapsed + remaining
        if budget <= 0.0:
            return 0.0
        progress = min(max(elapsed / budget, 0.0), 1.0)
        trainer = getattr(self, "trainer", None)
        strategy = getattr(trainer, "strategy", None)
        if strategy is not None:
            progress = strategy.broadcast(progress, src=0)
        return float(progress)

    def _create_time_cosine(
        self,
        optimizer: torch.optim.Optimizer,
        cfg: dict[str, Any],
        min_lr_ratio: float,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Cosine LR annealed over wall-clock time (``trainer.max_time``).

        The lambda ignores the step counter and instead reads elapsed training
        time from Lightning's ``Timer`` each step, so the LR reaches
        ``min_lr_ratio`` exactly as the budget is exhausted regardless of
        throughput. Stepped per optimizer step (see ``configure_optimizers``).
        """
        warmup_frac = self._resolve_time_warmup(cfg)

        def lr_lambda(_step: int) -> float:
            progress = self._training_time_progress()
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scaled = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
            if warmup_frac > 0.0 and progress < warmup_frac:
                return (progress / warmup_frac) * scaled
            return scaled

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _create_plateau(
        self,
        optimizer: torch.optim.Optimizer,
        cfg: dict[str, Any],
        min_lr_ratio: float,
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Build the plateau schedule with every knob read from config.

        They were hardcoded (``patience=10``, torch's ``factor=0.1``, no floor)
        until 2026-08-05, which made the decay schedule unreachable from Hydra
        and therefore unrecorded in any run's config. The campaign protocol
        needs it declared per run, because the decay shape is what the
        convergence gate later reads the learning-rate floor from.

        ``warmup`` is honoured here as of 2026-08-08. It had been read only by
        the cosine branch, so the schedule ruled as "constant LR after warmup"
        ran with none — see :class:`WarmupReduceLROnPlateau` for the measured
        consequence. Unlike the cosine branch it must be an INTEGER number of
        epochs: a fraction there means "of the annealing horizon", and a
        constant-rate schedule deliberately has no horizon to take a fraction
        of. That is the whole reason cosine was retired for the campaign, so a
        fractional warmup is rejected rather than silently reinterpreted.
        """
        patience = self._as_positive_int(
            cfg.get("plateau_patience", 10), "plateau_patience"
        )
        factor = float(cfg.get("plateau_factor", 0.1))
        if not 0.0 < factor < 1.0:
            msg = f"plateau_factor must be in (0, 1). Got: {factor}"
            raise ValueError(msg)
        cooldown = int(cfg.get("plateau_cooldown", 0))
        if cooldown < 0:
            msg = f"plateau_cooldown must be >= 0. Got: {cooldown}"
            raise ValueError(msg)
        threshold = float(cfg.get("plateau_threshold", 1e-4))
        if threshold < 0.0:
            msg = f"plateau_threshold must be >= 0. Got: {threshold}"
            raise ValueError(msg)
        threshold_mode = str(cfg.get("plateau_threshold_mode", "rel")).lower()
        if threshold_mode not in {"rel", "abs", "rel_abs"}:
            msg = (
                "plateau_threshold_mode must be 'rel', 'abs' or 'rel_abs'. "
                f"Got: {threshold_mode}"
            )
            raise ValueError(msg)
        warmup_epochs = self._resolve_plateau_warmup(cfg)
        # min_lr as a RATIO of the base LR, mirroring the cosine branch's
        # min_lr_ratio, so the two schedules are configured in the same units.
        min_lr = min_lr_ratio * float(cfg["learning_rate"])
        return WarmupReduceLROnPlateau(
            optimizer,
            warmup_epochs=warmup_epochs,
            mode="min",
            patience=patience,
            factor=factor,
            cooldown=cooldown,
            threshold=threshold,
            threshold_mode=threshold_mode,
            min_lr=min_lr,
        )

    @staticmethod
    def _resolve_plateau_warmup(cfg: dict[str, Any]) -> int:
        """Resolve the plateau branch's warmup, in whole epochs.

        Rejects the fractional form the cosine branch accepts: see
        :meth:`_create_plateau` for why a constant-rate schedule has no horizon
        for a fraction to refer to.
        """
        raw = cfg.get("warmup", 0)
        if raw is None:
            return 0
        value = float(raw)
        if not math.isfinite(value):
            msg = f"warmup must be finite. Got: {raw}"
            raise ValueError(msg)
        if value < 0.0:
            msg = f"warmup must be >= 0. Got: {raw}"
            raise ValueError(msg)
        if value != int(value):
            msg = (
                "With scheduler='plateau', warmup must be a whole number of "
                f"epochs, not a fraction. Got: {raw}. A fractional warmup is a "
                "fraction of the annealing horizon, and a constant-rate "
                "schedule has none."
            )
            raise ValueError(msg)
        return int(value)

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer, cfg: dict[str, Any]
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler from config."""
        scheduler_name = str(cfg.get("scheduler", "")).lower()
        scheduler_interval = self._get_scheduler_interval(cfg)

        min_lr_ratio = float(cfg.get("min_lr_ratio", 0.0))
        if not 0.0 <= min_lr_ratio <= 1.0:
            msg = f"min_lr_ratio must be in [0, 1]. Got: {min_lr_ratio}"
            raise ValueError(msg)

        if scheduler_name in {
            SchedulerName.COSINE,
            SchedulerName.COSINE_WITH_RESTARTS,
        }:
            if min_lr_ratio == 0.0:
                # The default anneals the LR to EXACTLY zero, so the loss tail
                # goes flat whether or not the model converged -- a run's own
                # logs then cannot answer "did this converge?". That silence
                # (not the value) is the defect: it has invalidated verdicts in
                # three separate workstreams. The default stays 0.0 because
                # changing it would break reproduction of every earlier run
                # (tests/scripts/test_training.py); it is made loud instead.
                # See
                # notes/design/training_infrastructure/model_engineering_checklist/checklist.md.  # noqa: E501
                warnings.warn(
                    "min_lr_ratio=0.0: the cosine schedule anneals the learning "
                    "rate to exactly zero, so this run's loss tail will be flat "
                    "by construction and CANNOT be used to judge convergence. "
                    "Set optimizer.min_lr_ratio>0 (e.g. optimizer="
                    "updated_defaults/adamw, which uses 0.05) for any run whose "
                    "convergence you intend to assess.",
                    UserWarning,
                    stacklevel=2,
                )
            if scheduler_interval == "time":
                if scheduler_name == SchedulerName.COSINE_WITH_RESTARTS:
                    msg = (
                        "cosine_with_restarts is not supported with "
                        "scheduler_interval='time'."
                    )
                    raise ValueError(msg)
                self._require_time_budget()
                return self._create_time_cosine(optimizer, cfg, min_lr_ratio)

            horizon = self._resolve_cosine_horizon(cfg, scheduler_interval)
            warmup = self._resolve_warmup(cfg, horizon)
            use_restarts = scheduler_name == SchedulerName.COSINE_WITH_RESTARTS

            def cosine_lambda(t: int) -> float:
                phase_t = t % horizon if use_restarts else t
                cosine = 0.5 * (
                    1.0 + math.cos(math.pi * float(phase_t) / float(horizon))
                )
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            if warmup > 0:

                def lr_lambda(t: int) -> float:
                    warm = min(1.0, float(t + 1) / float(warmup + 1))
                    return warm * cosine_lambda(t)

            else:
                lr_lambda = cosine_lambda

            return HorizonAwareLambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
                cosine_horizon=horizon,
                horizon_units="steps" if scheduler_interval == "step" else "epochs",
            )

        if scheduler_name == SchedulerName.STEP:
            step_size = cfg.get("step_size", 30)
            gamma = cfg.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        if scheduler_name == SchedulerName.PLATEAU:
            return self._create_plateau(optimizer, cfg, min_lr_ratio)
        msg = f"Unsupported scheduler: {scheduler_name}"
        raise ValueError(msg)

    def _processor_target_hook(self, name: str) -> Any:
        """A processor's training hook by name, or ``None``.

        Duck-typed on purpose, mirroring how ``_log_processor_diagnostics``
        reaches ``processor._latest_diagnostics``: only some processors carry a
        hook (the distillation students' target network, the rank ladder's
        ``after_optimiser_step``), and every other model must stay untouched by
        its presence.
        """
        return getattr(getattr(self, "processor", None), name, None)

    def on_train_start(self) -> None:
        """Tell a processor carrying an EMA target that it will be driven.

        Without this the processor cannot distinguish "no driver is wired" from
        "the driver has not fired yet", and an unwired EMA target would train
        against its own initial weights for a whole run with no failing signal.
        """
        super().on_train_start()  # type: ignore[misc]
        attach = self._processor_target_hook("attach_target_driver")
        if attach is not None:
            attach()
        attach_cycle = self._processor_target_hook("attach_cycle_driver")
        if attach_cycle is not None:
            attach_cycle()

    def on_train_batch_start(self, *args: Any, **kwargs: Any) -> None:
        """Tell a two-timescale processor which OPTIMISER step is coming.

        ⭐ ``global_step`` counts optimiser steps, not micro-batches, so a
        processor alternating between two objectives stays in phase with the
        updates it is scheduling however many micro-batches feed each one.
        Letting the processor count its own ``loss()`` calls instead would shear
        the cycle by a factor of ``accumulate_grad_batches`` -- silently, which
        is precisely the defect recorded against the EMA target in D100.
        """
        super().on_train_batch_start(*args, **kwargs)  # type: ignore[misc]
        set_step = self._processor_target_hook("set_optimiser_step")
        if set_step is not None:
            set_step(int(self.global_step))  # type: ignore[arg-type]

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        """Step the optimiser, then fold the new weights into any EMA target.

        ⭐ **This hook is the reason the update is accumulation-correct.**
        Lightning calls ``optimizer_step`` only on batches that actually step
        (``_AutomaticOptimization.run`` takes the closure-only branch while
        ``_should_accumulate()`` holds), so the target advances once per
        OPTIMISER step however many micro-batches feed it. Updating from
        ``loss()`` instead would fold the same unchanged weights in once per
        micro-batch and give an effective decay of ``mu ** N``, silently.

        It runs AFTER ``super()``, so the weights folded in are the ones the
        step just produced -- the post-step update Song et al. specify, rather
        than a copy that trails it by one step.
        """
        super().optimizer_step(*args, **kwargs)  # type: ignore[misc]
        # After every real step for the same reason: a processor that constrains
        # its weights (the rank ladder's patterns_off_term) re-applies it here.
        for name in ("update_target_network", "after_optimiser_step"):
            hook = self._processor_target_hook(name)
            if hook is not None:
                hook()

    def build_optimizer(self) -> torch.optim.Optimizer:
        """The optimizer a training run builds, without its schedule.

        The schedule is left out because a time-based one reads the budget off
        the attached Trainer; this is what a caller timing steps outside a
        Trainer needs.
        """
        return self._create_optimizer(self._optimizer_config_dict())

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers for training."""
        cfg = self._optimizer_config_dict()
        optimizer = self._create_optimizer(cfg)
        scheduler_name = cfg.get("scheduler", None)

        # Before anything is built: refuse a config carrying knobs the selected
        # schedule cannot read. Checked here rather than in `_create_scheduler`
        # so the scheduler=null case is covered too -- that path returns early
        # and is exactly where a full set of plateau knobs goes unread.
        selected: SchedulerName | None = None
        if scheduler_name is not None:
            try:
                selected = SchedulerName(str(scheduler_name).lower())
            except ValueError:
                msg = (
                    f"Unsupported scheduler: {scheduler_name}. Expected one of: "
                    f"{', '.join(s.value for s in SchedulerName)}."
                )
                raise ValueError(msg) from None
        self._reject_ignored_scheduler_keys(cfg, selected)

        # Return optimizer only if no scheduler
        if scheduler_name is None:
            return optimizer

        return self._build_scheduler_config(optimizer, cfg)

    def _optimizer_config_dict(self) -> dict[str, Any]:
        """``optimizer_config`` as a validated plain dict."""
        if self.optimizer_config is None:
            msg = "optimizer_config is required for training."
            raise ValueError(msg)

        # Accept both plain dict and Hydra DictConfig
        cfg_any: Any = self.optimizer_config
        if not isinstance(cfg_any, dict):
            cfg_any = OmegaConf.to_container(cfg_any, resolve=True)
        if not isinstance(cfg_any, dict):
            msg = (
                "optimizer_config must be a mapping (dict-like). "
                f"Got: {type(cfg_any).__name__}"
            )
            raise TypeError(msg)
        cfg = cfg_any
        if not cfg:
            msg = "optimizer_config cannot be empty."
            raise ValueError(msg)
        return cfg

    def _plateau_monitor(self, cfg: dict[str, Any]) -> str:
        """The metric a plateau schedule decays on.

        ``auto`` (also when the key is absent) is the processor's own
        ``plateau_monitor`` when it declares one, else ``val_loss``. The rank-ladder
        (ARFC) processor declares ``val/fit``. On the ladder,
        ``val_loss`` scores one randomly drawn rung per batch item plus the
        loading regulariser, while ``val/fit`` averages the fit over every rung
        (``RungFitSummaryCallback``), the quantity checkpoints and early stopping
        already read (D203). Measured 2026-10-25 on ten ARFC runs (shallow water
        32x32 and 64x64): at the starting rate the two scatter alike, so the
        first cut comes at about the same epoch on either; as the rate falls
        the model stops moving between epochs, ``val/fit`` goes quiet and
        ``val_loss`` keeps its rung-draw noise (2-5x ``val/fit``'s after the
        first cut, 10-37x near the floor), which hurried every run to the rate
        floor by epoch 150-265 while ``val/fit`` was still improving.
        """
        monitor = str(cfg.get("plateau_monitor", PLATEAU_MONITOR_AUTO))
        if monitor != PLATEAU_MONITOR_AUTO:
            return monitor
        declared = getattr(getattr(self, "processor", None), "plateau_monitor", None)
        return str(declared) if declared is not None else "val_loss"

    def _build_scheduler_config(
        self, optimizer: torch.optim.Optimizer, cfg: dict[str, Any]
    ) -> OptimizerLRScheduler:
        """Build the Lightning ``lr_scheduler`` dict for a configured schedule.

        Split out of :meth:`configure_optimizers` so the branch below can be
        unit-tested against a config alone. It decides which quantity a plateau
        schedule steers by, and until 2026-09-05 that decision was a literal
        with a config key pointing at it that reached nothing -- the kind of
        thing only a test that asserts the VALUE can catch, and there was no
        seam to write one against.
        """
        scheduler = self._create_scheduler(optimizer, cfg)
        scheduler_interval = self._get_scheduler_interval(cfg)

        # ReduceLROnPlateau needs special handling
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ⛔ `plateau_monitor` has been in this module's allowed-keys list
            # since the plateau schedule was added, and it was wired to
            # NOTHING: the quantity was hard-coded to `val_loss` here. A config
            # setting it passed every validation the repo has -- the key is
            # spelled correctly and is accepted -- and then trained against a
            # different signal from the one it named. That is the "declared but
            # not in force" shape this repo has now paid for three times
            # (a guard registered in YAML that Hydra dropped; an eval remedy
            # reachable from no config; this).
            #
            # ⭐ It matters for the distillation restarts specifically: those
            # runs select on `val_rmse`, so a plateau schedule reading
            # `val_loss` would decay the rate against a quantity nobody is
            # steering by.
            # ⛔ This branch hard-coded `epoch` until 2026-09-05. Honouring
            # `scheduler_interval` made BOTH "time" and "step" reachable, and
            # the argument against "time" applies verbatim to "step": a plateau
            # schedule decays on a MONITORED METRIC, and the metrics anything
            # here monitors (`val_loss`, `val_rmse`) exist once an epoch.
            # Stepping it per batch polls a value that has not moved, so
            # `plateau_patience` counts batches while meaning epochs -- a
            # schedule nobody configured, off by the number of batches in an
            # epoch. Refusing both is the honest reading of one sentence.
            # ⭐ Nothing scored moves: `campaign2_plateau.yaml` is the only
            # config selecting `scheduler: "plateau"` and it sets no
            # `scheduler_interval`; "step" was reachable only from a CLI
            # override, and only since the commit that stopped hard-coding.
            # ⭐ Reuses `scheduler_interval` from above rather than re-deriving
            # it -- the branch called `_get_scheduler_interval(cfg)` twice.
            if scheduler_interval != "epoch":
                msg = (
                    f"scheduler_interval={scheduler_interval!r} is meaningless "
                    "for ReduceLROnPlateau: it decays on a monitored METRIC, "
                    "not on a wall-clock or per-batch lambda, and the metrics "
                    "it monitors are logged once an epoch. Use 'epoch' (the "
                    "default) with plateau_patience, or a cosine schedule for "
                    "a time- or step-based decay."
                )
                raise ValueError(msg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # Always "epoch" past the guard above; named rather than
                    # literal so the two cannot drift apart.
                    "interval": scheduler_interval,
                    "monitor": self._plateau_monitor(cfg),
                },
            }

        # Time-based cosine steps per optimizer step (the lambda reads the
        # wall-clock; Lightning's lr_scheduler only accepts "step"/"epoch").
        lightning_interval = (
            "step" if scheduler_interval == "time" else scheduler_interval
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": lightning_interval,
                "frequency": 1,
            },
        }
