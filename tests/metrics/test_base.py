"""Ragged lead-step accumulation in `BaseMetric`, and its survivorship count.

⛔ **The fixture these tests exist to BE.** 294 synthetic tests passed while
`InvalidFrameAction.TRUNCATE` crashed on its first real run (jobs `6140907`,
`6140908`, 2026-08-26): the truncation tests counted frames with a scalar-state
metric and the metric tests never truncated, so no fixture in the repository
could produce a ragged prefix. Every test here therefore drives a REAL metric
whose state is per lead step, with update lengths that differ.
"""

import pytest
import torch
from torchmetrics import Metric

from autocast.metrics.base import BaseMetric, LeadStateSpec
from autocast.metrics.deterministic import MSE
from autocast.metrics.ensemble import TemporalAutocorrelationMatch
from autocast.scripts.eval.encoder_processor_decoder import (
    AVAILABLE_METRICS,
    AVAILABLE_METRICS_ENSEMBLE,
)


def _mse(**kwargs) -> MSE:
    return MSE(reduce_all=False, **kwargs)


def _constant_batch(n_traj: int, n_leads: int, error: float):
    """A ``(B, T, S, C)`` pair whose per-lead MSE is exactly ``error ** 2``."""
    true = torch.zeros(n_traj, n_leads, 8, 1)
    return true + error, true


class TestARaggedPrefixAccumulates:
    """The reported crash, and the silent corruption hiding beside it."""

    def test_a_shorter_call_lands_on_the_leading_leads(self):
        """⛔ The measured failure: `sum_score` (4) += `score_summed` (3).

        Mutation: restore `self.sum_score += score_summed` -- RuntimeError,
        "The size of tensor a (4) must match the size of tensor b (3)".
        """
        metric = _mse()
        metric.update(*_constant_batch(2, 4, 1.0))
        metric.update(*_constant_batch(1, 3, 1.0))

        assert metric.lead_samples.tolist() == [3, 3, 3, 2]
        assert metric.compute().flatten().tolist() == pytest.approx([1.0] * 4)

    def test_a_one_lead_call_is_not_broadcast_across_every_lead(self):
        """⛔ The WORSE half, and nothing raised on it.

        A length-1 score broadcasts into a length-4 buffer under `+=`, writing
        one lead's value into all four. Measured before the fix: leads 1-3 read
        1.0 as 3.667.

        Mutation: restore `self.sum_score += score_summed` -- every lead reads
        3.667 instead of leads 1-3 reading 1.0.
        """
        metric = _mse()
        metric.update(*_constant_batch(2, 4, 1.0))
        metric.update(*_constant_batch(1, 1, 3.0))

        assert metric.compute().flatten().tolist() == pytest.approx(
            [(2.0 + 9.0) / 3, 1.0, 1.0, 1.0]
        )

    def test_a_wider_call_after_a_narrow_one_grows_the_buffer(self):
        """⛔ The MIRROR of the reported failure, reachable across batches.

        Lazy init sizes the buffer from the first batch, so a batch whose
        longest valid prefix is 3 leaves the accumulator too NARROW for the
        next, untruncated one.

        Mutation: drop the `_grown_to_lead_width` calls in `update` --
        RuntimeError, "The size of tensor a (3) must match the size of tensor
        b (4)".
        """
        metric = _mse()
        metric.update(*_constant_batch(1, 3, 1.0))
        metric.update(*_constant_batch(2, 4, 1.0))

        assert metric.lead_samples.tolist() == [3, 3, 3, 2]
        assert metric.compute().shape[0] == 4

    def test_the_lead_axis_is_the_only_one_allowed_to_differ(self):
        """⛔ A channel-count change would BROADCAST or fail naming an axis.

        Mutation: delete the `shape[1:]` guard in `update` -- the call raises
        a bare shape mismatch from `+=` (or, for a singleton, silently
        broadcasts) instead of naming the cause.
        """
        metric = _mse()
        metric.update(torch.zeros(2, 4, 8, 3), torch.ones(2, 4, 8, 3))
        with pytest.raises(ValueError, match="away from the lead axis"):
            metric.update(torch.zeros(2, 4, 8, 1), torch.ones(2, 4, 8, 1))


class TestEachLeadIsDividedByItsOwnPopulation:
    """⛔ One global divisor is the survivorship corruption in arithmetic form."""

    def test_a_lead_only_some_trajectories_reached_uses_only_their_count(self):
        """Mutation: divide by `total_samples` -- lead 3 reads 0.667, not 1.0."""
        metric = _mse()
        metric.update(*_constant_batch(2, 4, 1.0))
        metric.update(*_constant_batch(1, 3, 1.0))

        assert int(metric.total_samples) == 3
        assert metric.lead_samples.tolist() == [3, 3, 3, 2]
        assert metric.compute()[3].item() == pytest.approx(1.0)

    def test_the_count_is_a_metric_state_so_it_reduces_with_the_score(self):
        """⛔ A count kept outside the state cannot be right under DDP.

        Mutation: make `lead_samples` a plain attribute rather than an
        `add_state` -- it disappears from `_reductions` and a two-rank sum
        divides the world's score by one rank's count.
        """
        assert "lead_samples" in _mse()._reductions

    def test_reset_clears_the_per_lead_count_with_everything_else(self):
        """Mutation: drop `lead_samples` from the state -- it survives reset."""
        metric = _mse()
        metric.update(*_constant_batch(2, 4, 1.0))
        metric.reset()

        assert metric.lead_samples.numel() == 0
        with pytest.raises(RuntimeError, match="No samples"):
            metric.compute()


class TestTheDefaultPathIsUnchanged:
    """⛔ Every number produced so far was computed by the old expression."""

    def test_an_untruncated_run_is_bit_for_bit_the_old_division(self):
        """`sum_score / total_samples`, pinned as a literal expression.

        Mutation: divide by anything but the per-lead count (e.g. by
        `total_samples + 1`, or by a clamped count) -- `torch.equal` fails.
        """
        metric = _mse()
        metric.update(*_constant_batch(3, 5, 0.5))
        metric.update(*_constant_batch(2, 5, 1.5))

        assert torch.equal(metric.compute(), metric.sum_score / metric.total_samples)

    def test_reduce_all_is_bit_for_bit_the_old_scalar(self):
        """Mutation: population-weight `reduce_all` -- `torch.equal` fails."""
        metric = MSE(reduce_all=True)
        metric.update(*_constant_batch(3, 5, 0.5))
        metric.update(*_constant_batch(2, 5, 1.5))

        expected = (metric.sum_score / metric.total_samples).mean()
        assert torch.equal(metric.compute(), expected)

    def test_a_time_reducing_metric_keeps_its_singleton_lead_axis(self):
        """⚠ Not every metric's axis 0 is a real lead count.

        A metric that reduces over the lead axis restores it as a singleton, so
        every call contributes to lead 0 whatever window length it saw. The
        count must therefore stay a TRAJECTORY count, accumulating across calls
        of different window lengths, and the buffer must stay one lead wide.

        ⭐ This is also why `_accumulate_over_leads` reads its width off the
        SCORE and never off the input: the input's lead count is invisible to
        it, so a lead-reducing metric cannot accidentally widen its state.

        Mutation: count updates instead of trajectories (`lead_samples[:n] += 1`)
        -- the counts read [1] and [2].
        """
        metric = TemporalAutocorrelationMatch(reduce_all=False)
        metric.update(torch.rand(2, 5, 4, 1, 3), torch.rand(2, 5, 4, 1))
        assert metric.lead_samples.tolist() == [2]
        metric.update(torch.rand(3, 3, 4, 1, 3), torch.rand(3, 3, 4, 1))
        assert metric.lead_samples.tolist() == [5]


class TestReduceAllWeightsEveryLeadEqually:
    """⭐ The choice made when leads carry unequal populations, and its reason.

    A population-weighted headline shifts weight onto the early leads exactly
    as a model truncates more, so diverging would IMPROVE its own headline --
    a second flattering mechanism on top of the thinned-subset one. Equal lead
    weight keeps two models' headlines composed of the same leads.
    """

    def test_a_thinly_populated_lead_is_not_discounted(self):
        """Mutation: weight by `lead_samples` -- the scalar reads 1.75, not 2.5.

        Leads 0-2 are measured at 1.0 over three trajectories; lead 3 at 4.0
        over one. Equal lead weight gives (1+1+1+4)/4 = 1.75... under a
        population weighting it would be (3+3+3+4)/10 = 1.3.
        """
        metric = MSE(reduce_all=True)
        metric.update(*_constant_batch(1, 4, 2.0))  # per-lead MSE 4.0
        metric.update(*_constant_batch(2, 3, 1.0))  # per-lead MSE 1.0

        per_lead = [(4.0 + 1.0 + 1.0) / 3] * 3 + [4.0]
        assert metric.compute().item() == pytest.approx(sum(per_lead) / 4)

    def test_the_headline_is_not_the_population_weighted_mean(self):
        """The contrast, measured rather than asserted."""
        metric = MSE(reduce_all=True)
        metric.update(*_constant_batch(1, 4, 2.0))
        metric.update(*_constant_batch(2, 3, 1.0))

        weighted = float(metric.sum_score.sum()) / float(metric.lead_samples.sum())
        assert metric.compute().item() != pytest.approx(weighted)


def _fake_gather(peer: Metric, local: Metric):
    """A `gather_all_tensors` stand-in for a two-rank world holding ``peer``.

    ⭐ Faithful to the real one in the way that matters: `gather_all_tensors`
    pads to the largest size and then trims each entry back to its OWN size
    (`torchmetrics/utilities/distributed.py:138-150`), so the list torchmetrics
    stacks holds the ranks' original shapes. Keyed by object identity because
    `_sync_dist` hands the gather function the state tensors themselves, with
    no attribute name.
    """
    peer_states = {name: getattr(peer, name) for name in ("sum_score", "total_samples")}
    peer_states["lead_samples"] = peer.lead_samples

    def gather(result, group=None):  # noqa: ARG001 -- signature parity
        for name, tensor in peer_states.items():
            if result is getattr(local, name):
                return [result, tensor]
        msg = "the fake gather was handed a tensor it does not know"
        raise AssertionError(msg)

    return gather


class TestRanksMustAgreeOnTheAccumulatorShape:
    """⛔ `dist_reduce_fx="sum"` cannot reduce states of different widths.

    `_sync_dist` calls `torch.stack` on the gathered list
    (`torchmetrics/metric.py:533`), which raises the moment two ranks report
    different shapes -- and TRUNCATE makes the width a per-rank data fact.
    """

    @staticmethod
    def _rank(n_traj: int, n_leads: int, error: float) -> MSE:
        metric = _mse()
        metric.update(*_constant_batch(n_traj, n_leads, error))
        return metric

    def test_unaligned_widths_cannot_be_reduced(self):
        """The hazard itself, measured -- the reason alignment exists.

        Mutation: none needed; this pins the torchmetrics behaviour the fix is
        built on. If a future version stopped raising here, `align_lead_state`
        would be dead weight rather than load-bearing.
        """
        mine, peer = self._rank(2, 4, 1.0), self._rank(1, 3, 1.0)
        gather = _fake_gather(peer, mine)
        with pytest.raises(RuntimeError, match="stack expects each tensor"):
            mine.sync(dist_sync_fn=gather, distributed_available=lambda: True)

    def test_aligned_widths_reduce_to_the_world_answer(self):
        """Mutation: skip `align_lead_state` -- the sync raises, as above.

        Read off the synced STATE rather than `compute()`: torchmetrics refuses
        a second sync, and the state is the thing the reduction produces.
        """
        mine, peer = self._rank(2, 4, 1.0), self._rank(1, 3, 1.0)
        spec = LeadStateSpec((4, 1), torch.float32)
        mine.align_lead_state(spec)
        peer.align_lead_state(spec)

        gather = _fake_gather(peer, mine)
        mine.sync(dist_sync_fn=gather, distributed_available=lambda: True)

        assert mine.lead_samples.tolist() == [3, 3, 3, 2]
        assert int(mine.total_samples) == 3
        per_lead = mine.sum_score / mine.lead_samples.unsqueeze(-1)
        assert per_lead.flatten().tolist() == pytest.approx([1.0] * 4)

    def test_a_rank_that_never_updated_materialises_a_stackable_state(self):
        """⛔ Its default state is 0-DIMENSIONAL, which HANGS rather than fails.

        A 0-dim tensor takes `gather_all_tensors`' scalar branch while its peers
        all-gather sizes, so the two ranks enter different collectives. The
        filled-in metrics `_agree_on_window_metrics` creates are exactly this
        case.

        Mutation: make `align_lead_state` return early when uninitialised --
        `sum_score` stays a scalar and the shapes no longer match.
        """
        idle = _mse()
        assert idle.sum_score.ndim == 0

        idle.align_lead_state(LeadStateSpec((4, 1), torch.float32))

        assert idle.sum_score.shape == (4, 1)
        assert idle.lead_samples.tolist() == [0, 0, 0, 0]

    def test_alignment_never_narrows_a_state(self):
        """⛔ Narrowing would silently drop leads this rank actually scored.

        Mutation: replace the width check with an unconditional resize -- the
        call succeeds and lead 3's contribution vanishes.
        """
        metric = self._rank(2, 4, 1.0)
        with pytest.raises(ValueError, match="narrower"):
            metric.align_lead_state(LeadStateSpec((3, 1), torch.float32))

    def test_alignment_refuses_a_shape_that_differs_off_the_lead_axis(self):
        """Mutation: drop the `shape[1:]` check -- two quantities get summed."""
        metric = self._rank(2, 4, 1.0)
        with pytest.raises(ValueError, match="not the same quantity"):
            metric.align_lead_state(LeadStateSpec((4, 3), torch.float32))

    def test_a_metric_that_never_updated_reports_no_spec(self):
        """So the agreement takes its shape from a rank that has one."""
        assert _mse().lead_state_spec() is None
        assert self._rank(2, 4, 1.0).lead_state_spec() == LeadStateSpec(
            (4, 1), torch.float32
        )


def _selectable_base_metrics() -> list[tuple[str, object]]:
    """Every registry entry a run can select that derives from `BaseMetric`.

    ⛔ Built from the EVALUATOR's own tables, not a hand-kept list. A metric
    added to the registry and not to a list here would be exactly the untested
    one that reaches a cluster job.
    """
    selectable = []
    for name, factory in sorted(
        {**AVAILABLE_METRICS, **AVAILABLE_METRICS_ENSEMBLE}.items()
    ):
        try:
            metric = factory(reduce_all=False)
        except TypeError:
            continue  # not a `BaseMetric` signature; excluded below anyway
        if isinstance(metric, BaseMetric):
            selectable.append((name, factory))
    return selectable


_SELECTABLE = _selectable_base_metrics()


def test_the_registry_sweep_is_not_a_degenerate_fixture():
    """⛔ D107, third occurrence: a sweep over nothing passes silently."""
    assert len(_SELECTABLE) >= 30, [name for name, _ in _SELECTABLE]


@pytest.mark.parametrize(
    "factory", [f for _, f in _SELECTABLE], ids=[n for n, _ in _SELECTABLE]
)
def test_every_selectable_metric_survives_a_ragged_prefix(factory):
    """⛔ THE FIXTURE THAT DID NOT EXIST -- truncation meets per-lead state.

    A full batch followed by a shorter one is precisely what
    `InvalidFrameAction.TRUNCATE` hands a metric, and it is what jobs
    `6140907`/`6140908` died on. Sweeping the registry rather than one class is
    what catches a metric that keeps its own accumulators (`ssr_pooled` does,
    and was broken in exactly the same way).

    Mutation: restore `self.sum_score += score_summed` in
    `BaseMetric._accumulate_over_leads` -- every metric whose score keeps a
    real lead axis raises "The size of tensor a (4) must match the size of
    tensor b (3)".
    """
    torch.manual_seed(0)
    metric = factory(reduce_all=False)
    metric.update(torch.rand(2, 4, 8, 1, 4), torch.rand(2, 4, 8, 1))
    metric.update(torch.rand(1, 3, 8, 1, 4), torch.rand(1, 3, 8, 1))

    n_leads = metric.lead_samples.numel()
    # A metric that reduces over the lead axis keeps a singleton; one that does
    # not shows the thinning. Either way the count must be per lead and the
    # trailing leads must be the thin ones.
    assert metric.lead_samples.tolist() == ([3] if n_leads == 1 else [3, 3, 3, 2])
    assert torch.isfinite(metric.compute()).all()
