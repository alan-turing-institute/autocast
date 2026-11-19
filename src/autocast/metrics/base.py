import abc
import dataclasses
from abc import ABC
from collections.abc import Callable, Mapping
from typing import Generic, TypeVar

import torch
from torchmetrics import Metric

from autocast.types import TensorBSC, TensorBTC, TensorBTSC
from autocast.types.types import ArrayLike, Tensor

TPred = TypeVar("TPred", bound=Tensor)
TTrue = TypeVar("TTrue", bound=Tensor)

# Default cap on the all-pairs ``(rows, ..., D, D, ...)`` block that the
# variogram-family metrics build, in tensor elements. 2**27 elements is 1 GiB of
# float64 (512 MiB of float32), chosen to leave a 30 GB workstation room for the
# rest of an assessment suite while staying large enough that ordinary training
# and eval batches take the single-block path.
DEFAULT_MAX_PAIR_ELEMENTS = 2**27


def rows_per_pair_block(
    elements_per_row: int, max_elements: int = DEFAULT_MAX_PAIR_ELEMENTS
) -> int:
    """Leading-axis rows whose all-pairs block fits within ``max_elements``.

    All-pairs metrics cost ``O(D^2)`` per row and are independent across the
    leading axis, so an oversized input is processed in row blocks rather than
    refused: scoring a whole test split in one call costs extra passes instead of
    extra memory, and the result is unchanged.

    Returns at least ``1`` -- a single row is the indivisible unit, so a cap
    below one row's cost degrades to row-by-row rather than raising.
    """
    return max(1, max_elements // max(1, elements_per_row))


def block_pairwise_frames(
    y_pred_vector: Tensor,
    y_true_vector: Tensor,
    elements_per_row: int,
    max_elements: int,
    compute: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Apply an all-pairs ``compute`` frame-by-frame, in memory-capped blocks.

    ``y_pred_vector`` is ``(*leading, D, M)`` and ``y_true_vector`` is
    ``(*leading, D)``, the output of a metric's ``vector_dims`` selection
    (:func:`autocast.metrics.ensemble._vectorize_selected_dims`). Depending on
    ``vector_dims``, ``leading`` can carry batch alone, or batch together with
    lead-time and/or channel -- but every entry along it is an independent
    scoring frame: the all-pairs work for one frame never touches another.
    Blocking is therefore exact, not an approximation -- flattening every
    ``leading`` axis into one blockable row axis (rather than only the batch
    axis) lets the row cap from :func:`rows_per_pair_block` reach below a
    single original batch row, e.g. also splitting across lead-time and
    channel for a spatial vector.

    Args:
        y_pred_vector: Ensemble predictions, ``(*leading, D, M)``.
        y_true_vector: Ground truth, ``(*leading, D)``.
        elements_per_row: Element count of ``compute``'s dominant intermediate
            for a *single* frame (e.g. ``D * D * M`` for an all-pairs
            ``(D, D, M)`` tensor, ``D * D`` for a ``(D, D)`` covariance).
        max_elements: Cap on one block's total element count.
        compute: Maps one row-block of ``(pred, true)`` -- shapes
            ``(rows, D, M)`` and ``(rows, D)`` -- to its per-frame result,
            ``(rows, ...)``.

    Returns:
        ``compute``'s per-frame results, concatenated and reshaped back onto
        the original ``leading`` axes: ``(*leading, ...)``.
    """
    leading_shape = y_true_vector.shape[:-1]
    d = y_true_vector.shape[-1]
    m = y_pred_vector.shape[-1]
    pred_flat = y_pred_vector.reshape(-1, d, m)
    true_flat = y_true_vector.reshape(-1, d)
    n_rows = pred_flat.shape[0]

    rows = rows_per_pair_block(elements_per_row, max_elements)
    blocks = [
        compute(pred_flat[start : start + rows], true_flat[start : start + rows])
        for start in range(0, n_rows, rows)
    ]
    result = torch.cat(blocks, dim=0)
    return result.reshape(*leading_shape, *result.shape[1:])


@dataclasses.dataclass(frozen=True, slots=True)
class LeadStateSpec:
    """The shape and dtype one rank's :class:`BaseMetric` accumulator holds.

    Exchanged between ranks so every one of them enters ``compute()`` with a
    stack-compatible state -- see :meth:`BaseMetric.align_lead_state` for why a
    disagreement is fatal rather than cosmetic.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype


def _grown_to_lead_width(state: Tensor, n_leads: int) -> Tensor:
    """``state`` with its leading (lead-step) axis zero-extended to ``n_leads``.

    Returns ``state`` untouched when it is already at least that wide, so the
    common path costs a shape comparison and no allocation.
    """
    if state.shape[0] >= n_leads:
        return state
    grown = state.new_zeros((n_leads, *state.shape[1:]))
    grown[: state.shape[0]] = state
    return grown


class BaseMetric(Metric, Generic[TPred, TTrue], ABC):
    """Shared template for spatial metrics that reduce over spatial axes.

    ⛔ **The accumulator is RAGGED along the lead axis, and it has to be.**
    ``InvalidFrameAction.TRUNCATE`` feeds each trajectory its valid prefix
    (:class:`~autocast.analysis.validity.InvalidFrameAction`), so consecutive
    ``update`` calls carry different numbers of lead steps. A short call
    accumulates into the LEADING SLICE of the buffer and a long one GROWS it;
    ``lead_samples`` records, per lead step, how many trajectories actually
    reached that far, and :meth:`compute` divides per lead rather than by one
    global count.

    ⛔ **The per-lead count is not bookkeeping -- it is the survivorship guard**
    (D101). Once trajectories truncate at different frames a long-lead average
    runs over a shrinking, non-random subset, so a model looks BETTER at long
    leads precisely because its worst cases dropped out. The count is a metric
    state, reduced across ranks with the score, and it is what the evaluator
    publishes beside every windowed number.

    ⚠ Before this existed, ``self.sum_score += score_summed`` failed TWO ways on
    a ragged prefix, measured 2026-08-26 on jobs `6140907`/`6140908`: a
    length-3 score into a length-4 buffer RAISED, and -- worse -- a length-1
    score into a length-4 buffer BROADCAST silently, writing one lead's value
    into all four.
    """

    # Per-class flag for callers that gate expensive metrics (e.g. the eval
    # script's memory guard). Subclasses whose covariance/vector state scales
    # as D-by-D in the field size override this to True so the flag travels
    # with the class -- any registry alias of that class inherits it too,
    # instead of a separately maintained name allowlist.
    memory_intensive: bool = False

    # Per-class minimum number of lead timesteps a call needs to be valid.
    # Subclasses that reduce over the lead axis (e.g. a lag/pair-based
    # temporal-dependence diagnostic) override this so the requirement
    # travels with the class -- any registry alias inherits it too, instead
    # of a separately maintained name-to-minimum table.
    min_window_steps: int = 1

    # Names of the states whose LEADING axis is the lead step and which hold a
    # per-lead sum. Subclasses accumulating something other than `sum_score`
    # (e.g. a pooled numerator and denominator that must not be divided until
    # the end) name theirs here, and the ragged-prefix handling, the per-lead
    # count and the cross-rank shape agreement all follow -- rather than each
    # such subclass re-deriving them and getting the ragged case wrong again.
    lead_axis_states: tuple[str, ...] = ("sum_score",)

    # Declared so the states read as tensors rather than as the
    # `Tensor | Module` union `nn.Module.__getattr__` returns. Annotation only:
    # `add_state` below is still what creates and registers them.
    sum_score: Tensor
    total_samples: Tensor
    lead_samples: Tensor

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.reduce_all = reduce_all

        # States shared by all derived metrics
        self.add_state("sum_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        # Per-lead-step contributing trajectory count -- the divisor `compute`
        # uses, and the survivorship population the evaluator publishes. Summed
        # across ranks like the score, so the two can never disagree.
        self.add_state(
            "lead_samples",
            default=torch.zeros(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

        # Internal flag to set shape of sum_score
        self._initialized = False

    @abc.abstractmethod
    def _check_input(
        self, y_pred: ArrayLike, y_true: ArrayLike
    ) -> tuple[TPred, TTrue]: ...

    @abc.abstractmethod
    def _score(
        self, y_pred: TPred, y_true: TTrue
    ) -> TensorBTC | TensorBSC | TensorBTSC: ...

    def score(
        self, y_pred: ArrayLike, y_true: ArrayLike
    ) -> TensorBTC | TensorBSC | TensorBTSC:
        y_pred_tensor, y_true_tensor = self._check_input(y_pred, y_true)
        return self._score(y_pred_tensor, y_true_tensor)

    def update(self, y_pred: ArrayLike, y_true: ArrayLike) -> None:
        """Update metric state with a batch of predictions and targets.

        A call may carry FEWER lead steps than an earlier one (a truncated
        prefix) or MORE (a full batch after a truncated one). Both are handled:
        the score lands on the leading ``score.shape[1]`` lead steps of the
        buffer, and the buffer grows when a call is wider than it is.
        """
        # Compute score for the batch. Shape of score depends on
        # reduction strategy
        score = self.score(y_pred, y_true)

        batch_size = score.shape[0]

        # Sum over batch dimension: (B, T, C) -> (T, C)
        score_summed = torch.sum(score, dim=0)
        self._accumulate_over_leads({"sum_score": score_summed}, batch_size)

    def _accumulate_over_leads(
        self, updates: Mapping[str, Tensor], n_samples: int
    ) -> None:
        """Add per-lead sums into the leading slice of the named states.

        The one place ragged prefixes are handled, so every accumulator that
        carries a lead axis gets the same treatment: a short update lands on
        the leading leads, a long one widens the buffer, and ``lead_samples``
        records how many trajectories reached each lead.

        Args:
            updates: This call's per-lead sums, keyed by state name. Every
                entry must share the same leading (lead-step) length.
            n_samples: Trajectories this call summed over -- what each lead it
                touched gains in population.

        Raises:
            ValueError: If a state and its update disagree anywhere but the
                lead axis.
        """
        n_leads = next(iter(updates.values())).shape[0]

        # Lazily set correct shape for the accumulators on the first batch
        if not self._initialized:
            for name, value in updates.items():
                setattr(self, name, torch.zeros_like(value))
            self.lead_samples = torch.zeros(
                n_leads, dtype=torch.long, device=next(iter(updates.values())).device
            )
            self._initialized = True
        else:
            # ⛔ Only the LEAD axis may differ between calls. A mismatch
            # anywhere else means two different score shapes reached one
            # metric, which `+=` would either broadcast silently or reject with
            # a message naming an axis rather than the cause.
            for name, value in updates.items():
                state = getattr(self, name)
                if state.shape[1:] != value.shape[1:]:
                    msg = (
                        f"metric state '{name}' and its update disagree away "
                        f"from the lead axis: accumulated {tuple(state.shape)} "
                        f"vs this call's {tuple(value.shape)}. Only the leading "
                        "(lead-step) axis may vary between updates -- everything "
                        "after it is fixed by the field, so a difference there is "
                        "two different quantities being summed into one "
                        "accumulator."
                    )
                    raise ValueError(msg)
                setattr(self, name, _grown_to_lead_width(state, n_leads))
            self.lead_samples = _grown_to_lead_width(self.lead_samples, n_leads)

        for name, value in updates.items():
            getattr(self, name)[:n_leads] += value
        self.lead_samples[:n_leads] += n_samples
        self.total_samples += n_samples

    def compute(self) -> Tensor:
        """Compute final metric value.

        ⭐ **Per-lead division, and it reduces to the old one exactly.** Each
        lead step is divided by the number of trajectories that actually
        reached it, so the leads a truncated run never got to do not dilute the
        ones it did. When nothing truncates every entry of ``lead_samples``
        equals ``total_samples`` and this is bit-for-bit the previous
        ``sum_score / total_samples``.

        ⭐ **``reduce_all`` averages the LEADS EQUALLY, not by population**, and
        the choice is deliberate. A population-weighted headline would shift
        weight onto the early leads exactly as a model truncates more, so
        diverging would IMPROVE its headline -- a second flattering mechanism
        stacked on top of the thinned-subset one. Equal lead weight keeps two
        models' headlines composed of the same leads however differently they
        truncate. It does NOT make the scalar survivorship-safe: a long lead
        averaged over three survivors still counts for as much as a short one
        averaged over three hundred, which is why ``lead_samples`` is published
        beside the number and why a headline is quoted only over the
        full-population horizon (D101).
        """
        if self.total_samples == 0:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)

        # Every lead index below the buffer's width was covered by the update
        # that set that width, so no count here is zero; a defensive 0/0 would
        # give NaN, which is the right answer for a lead nothing reached.
        counts = self.lead_samples.to(self.sum_score.dtype)
        score = self.sum_score / counts.reshape(-1, *(1,) * (self.sum_score.ndim - 1))

        if self.reduce_all:
            # Average over time and channels
            return score.mean()

        return score

    def lead_state_spec(self) -> LeadStateSpec | None:
        """This metric's accumulator shape, or None if it never updated.

        Read off the first of `lead_axis_states`; a subclass declaring several
        accumulates them from one score, so they share a shape.
        """
        if not self._initialized:
            return None
        state = getattr(self, self.lead_axis_states[0])
        return LeadStateSpec(tuple(state.shape), state.dtype)

    def align_lead_state(self, spec: LeadStateSpec) -> None:
        """Grow (or materialise) the accumulator to ``spec``, keeping its value.

        ⛔ **A DDP requirement, not a tidy-up.** ``dist_reduce_fx="sum"`` routes
        every state through torchmetrics' ``_sync_dist``, which gathers the
        ranks' tensors and then calls ``torch.stack`` on them
        (``torchmetrics/metric.py:533``). ``gather_all_tensors`` pads to the
        largest size and trims each entry back to its OWN size
        (``torchmetrics/utilities/distributed.py:138-150``), so the stack sees
        differently-shaped tensors and RAISES. Under TRUNCATE the widths are a
        per-rank fact -- one rank's trajectories may all die at lead 3 while
        another's run to 4 -- so the widths have to be agreed before
        ``compute()``. A rank that never updated is the same hazard in its
        sharpest form: its default state is 0-dimensional, which sends it down
        ``gather_all_tensors``' scalar branch while its peers all-gather sizes,
        and mismatched collectives hang rather than fail.

        Args:
            spec: The world's agreed accumulator shape and dtype.

        Raises:
            ValueError: If ``spec`` is narrower than this rank's own state, or
                disagrees with it away from the lead axis -- either means the
                agreement was computed wrongly, and silently reshaping would
                sum two different quantities together.
        """
        device = getattr(self, self.lead_axis_states[0]).device
        if self._initialized:
            for name in self.lead_axis_states:
                state = getattr(self, name)
                if state.shape[1:] != spec.shape[1:]:
                    msg = (
                        f"cannot align metric state '{name}' "
                        f"{tuple(state.shape)} to {spec.shape}: the axes after "
                        "the lead axis differ, so these are not the same quantity."
                    )
                    raise ValueError(msg)
                if spec.shape[0] < state.shape[0]:
                    msg = (
                        f"cannot align metric state '{name}' "
                        f"{tuple(state.shape)} to the narrower {spec.shape}: "
                        "lead steps this rank scored would be dropped."
                    )
                    raise ValueError(msg)
                setattr(self, name, _grown_to_lead_width(state, spec.shape[0]))
            self.lead_samples = _grown_to_lead_width(self.lead_samples, spec.shape[0])
            return

        for name in self.lead_axis_states:
            setattr(
                self, name, torch.zeros(spec.shape, dtype=spec.dtype, device=device)
            )
        self.lead_samples = torch.zeros(spec.shape[0], dtype=torch.long, device=device)
        self._initialized = True

    def reset(self) -> None:
        """Reset metric state and initialization flag."""
        super().reset()
        self._initialized = False

    def _infer_n_spatial_dims(self, tensor: Tensor) -> int:
        """Infer number of spatial dimensions from tensor shape."""
        return tensor.ndim - 3  # Subtract B, T, C
