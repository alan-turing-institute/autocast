import lightning as L
import pytest
import torch
from conftest import get_optimizer_config
from torch.utils.data import DataLoader, Dataset

from autocast.models.processor import ProcessorModel
from autocast.nn.mlp import TemporalMLPBackbone
from autocast.nn.unet import TemporalUNetBackbone
from autocast.processors.drifting import (
    DriftingProcessor,
    _accuracy_diagnostics,
    _compute_drift_field,
    _energy_score,
)
from autocast.types import EncodedBatch


def _make_backbone(
    *,
    n_steps_in: int = 2,
    n_steps_out: int = 2,
    ch_in: int = 1,
    ch_out: int = 1,
    include_time_embedding: bool = False,
) -> TemporalUNetBackbone:
    return TemporalUNetBackbone(
        in_channels=ch_out,
        out_channels=ch_out,
        cond_channels=ch_in,
        n_steps_output=n_steps_out,
        n_steps_input=n_steps_in,
        include_global_cond=False,
        global_cond_channels=None,
        temporal_method="none",
        mod_features=256,
        hid_channels=(32, 64, 128),
        hid_blocks=(2, 2, 2),
        spatial=2,
        periodic=False,
        include_time_embedding=include_time_embedding,
    )


def _make_batch(
    *,
    b: int = 2,
    t_in: int = 2,
    t_out: int = 2,
    ch_in: int = 1,
    ch_out: int = 1,
    spatial: int = 16,
) -> EncodedBatch:
    return EncodedBatch(
        encoded_inputs=torch.randn(b, t_in, spatial, spatial, ch_in),
        encoded_output_fields=torch.randn(b, t_out, spatial, spatial, ch_out),
        global_cond=None,
        encoded_info={},
    )


def test_constructor_rejects_backbone_with_time_embedding():
    backbone = _make_backbone(include_time_embedding=True)
    with pytest.raises(ValueError, match="include_time_embedding=False"):
        DriftingProcessor(backbone=backbone, n_steps_output=2, n_channels_out=1)


def test_constructor_rejects_n_samples_lt_2():
    backbone = _make_backbone()
    with pytest.raises(ValueError, match="n_samples >= 2"):
        DriftingProcessor(
            backbone=backbone, n_steps_output=2, n_channels_out=1, n_samples=1
        )


def test_constructor_rejects_empty_tau_list():
    backbone = _make_backbone()
    with pytest.raises(ValueError, match="tau_list"):
        DriftingProcessor(
            backbone=backbone, n_steps_output=2, n_channels_out=1, tau_list=()
        )


def test_constructor_rejects_negative_lambda_es():
    backbone = _make_backbone()
    with pytest.raises(ValueError, match="lambda_es >= 0"):
        DriftingProcessor(
            backbone=backbone, n_steps_output=2, n_channels_out=1, lambda_es=-1.0
        )


@pytest.mark.parametrize("a_f", [0.0, -0.1, 1.5])
def test_constructor_rejects_energy_score_a_f_out_of_range(a_f):
    backbone = _make_backbone()
    with pytest.raises(ValueError, match=r"energy_score_a_f in \(0, 1\]"):
        DriftingProcessor(
            backbone=backbone,
            n_steps_output=2,
            n_channels_out=1,
            energy_score_a_f=a_f,
        )


def test_constructor_accepts_valid_backbone_defaults():
    backbone = _make_backbone()
    processor = DriftingProcessor(backbone=backbone, n_steps_output=2, n_channels_out=1)
    assert processor.n_samples == 8
    assert processor.tau_list == (0.02, 0.05, 0.2)
    assert processor.diag_mask_value == 100.0


def test_forward_returns_expected_shape():
    b, t_in, t_out, ch_in, ch_out = 2, 2, 2, 1, 1
    backbone = _make_backbone(
        n_steps_in=t_in, n_steps_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    processor = DriftingProcessor(
        backbone=backbone, n_steps_output=t_out, n_channels_out=ch_out
    )
    x = torch.randn(b, t_in, 16, 16, ch_in)
    out = processor.forward(x, None)
    assert out.shape == (b, t_out, 16, 16, ch_out)


@pytest.mark.parametrize("b", [1, 2])
def test_loss_returns_finite_scalar(b):
    t_in, t_out, ch_in, ch_out, n_samples = 2, 2, 1, 1, 4
    backbone = _make_backbone(
        n_steps_in=t_in, n_steps_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    processor = DriftingProcessor(
        backbone=backbone,
        n_steps_output=t_out,
        n_channels_out=ch_out,
        n_samples=n_samples,
    )
    batch = _make_batch(b=b, t_in=t_in, t_out=t_out, ch_in=ch_in, ch_out=ch_out)
    loss = processor.loss(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss).all()


def test_loss_populates_latest_diagnostics():
    """Pin the diagnostic side-channel that the LightningModule pulls from.

    With n_+ = 1 the positive-side dual softmax collapses to a constant
    pull at the unique ground truth, so only the negative-side cohort
    contrast does extra work over plain MSE. ``cohort_spread`` (the std
    across the K cohort members) and the per-scale ``force_rms_tau*`` are
    the signals that catch the contrastive-collapse failure mode mid-run.
    """
    b, t_in, t_out, ch_in, ch_out, n_samples = 2, 2, 2, 1, 1, 4
    tau_list = (0.02, 0.05, 0.2)
    backbone = _make_backbone(
        n_steps_in=t_in, n_steps_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    processor = DriftingProcessor(
        backbone=backbone,
        n_steps_output=t_out,
        n_channels_out=ch_out,
        n_samples=n_samples,
        tau_list=tau_list,
    )
    assert processor._latest_diagnostics == {}

    batch = _make_batch(b=b, t_in=t_in, t_out=t_out, ch_in=ch_in, ch_out=ch_out)
    processor.loss(batch)

    diagnostics = processor._latest_diagnostics
    expected_keys = {
        "cohort_spread",
        "ens_mean_rmse",
        "gen_pos_dist_p50",
        "cohort_ssr",
        *(f"force_rms_tau{tau}" for tau in tau_list),
    }
    assert set(diagnostics.keys()) == expected_keys
    for name, value in diagnostics.items():
        assert isinstance(value, torch.Tensor), name
        assert value.shape == (), name
        assert torch.isfinite(value).all(), name
    assert diagnostics["cohort_spread"].item() > 0.0


def test_accuracy_diagnostics_value_sanity():
    """Pin the semantics of the accuracy/calibration diagnostics.

    A perfect cohort (every member exactly on the positive) has zero
    ensemble-mean error, zero member distance, and — with no spread — zero
    SSR. A dispersed cohort has strictly positive error, distance, and SSR.
    """
    b, k, d = 2, 8, 3
    pos = torch.randn(b, 1, d)

    perfect = pos.expand(b, k, d).contiguous()
    perfect_diag = _accuracy_diagnostics(perfect, pos)
    assert perfect_diag["ens_mean_rmse"].item() == pytest.approx(0.0, abs=1e-6)
    assert perfect_diag["gen_pos_dist_p50"].item() == pytest.approx(0.0, abs=1e-6)
    # zero spread over a clamped-positive denominator -> exactly 0.
    assert perfect_diag["cohort_ssr"].item() == pytest.approx(0.0, abs=1e-6)

    # A cohort dispersed around the positive: mean is still off (finite K),
    # members sit at a distance, and spread is real.
    dispersed = pos + torch.randn(b, k, d)
    dispersed_diag = _accuracy_diagnostics(dispersed, pos)
    assert dispersed_diag["ens_mean_rmse"].item() > 0.0
    assert dispersed_diag["gen_pos_dist_p50"].item() > 0.0
    assert dispersed_diag["cohort_ssr"].item() > 0.0

    # Diagnostics are detached even when the cohort requires grad.
    grad_cohort = (pos + torch.randn(b, k, d)).requires_grad_(True)
    for value in _accuracy_diagnostics(grad_cohort, pos).values():
        assert not value.requires_grad


def test_loss_matches_manual_cohort_slicing():
    """Pin the n_pos=1 cohort slicing in ``DriftingProcessor.loss``.

    With ``EncodedBatch.repeat(K)`` using repeat-interleave semantics, the
    target layout is ``[t_0]*K + [t_1]*K + ...``. The slicing
    ``target.reshape(b, K, -1)[:, :1, :]`` recovers one copy per input. If
    the slicing axis or index were wrong, the per-input pull would mix
    targets across batch elements and the loss would diverge from the
    manual recomputation below.
    """
    b, t_in, t_out, ch_in, ch_out, n_samples = 2, 2, 2, 1, 1, 4
    backbone = _make_backbone(
        n_steps_in=t_in, n_steps_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    processor = DriftingProcessor(
        backbone=backbone,
        n_steps_output=t_out,
        n_channels_out=ch_out,
        n_samples=n_samples,
    )
    batch = _make_batch(b=b, t_in=t_in, t_out=t_out, ch_in=ch_in, ch_out=ch_out)
    batch.encoded_output_fields[1].fill_(5.0)

    torch.manual_seed(42)
    loss_via_processor = processor.loss(batch)

    torch.manual_seed(42)
    batch_n = batch.repeat(n_samples)
    cond = batch_n.encoded_inputs
    target = batch_n.encoded_output_fields
    gen = processor._draw_one(cond, batch_n.global_cond)
    gen_b = gen.reshape(b, n_samples, -1)
    pos_b = target.reshape(b, n_samples, -1)[:, :1, :]
    goal_scaled, gen_scaled, _ = _compute_drift_field(
        gen_b,
        pos_b,
        tau_list=processor.tau_list,
        diag_mask_value=processor.diag_mask_value,
    )
    expected_loss = ((gen_scaled - goal_scaled) ** 2).mean()
    torch.testing.assert_close(loss_via_processor, expected_loss)


def test_loss_backward_populates_backbone_gradients():
    b, t_in, t_out, ch_in, ch_out, n_samples = 2, 2, 2, 1, 1, 4
    backbone = _make_backbone(
        n_steps_in=t_in, n_steps_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    processor = DriftingProcessor(
        backbone=backbone,
        n_steps_output=t_out,
        n_channels_out=ch_out,
        n_samples=n_samples,
    )
    batch = _make_batch(b=b, t_in=t_in, t_out=t_out, ch_in=ch_in, ch_out=ch_out)
    loss = processor.loss(batch)
    loss.backward()
    grad_norms = [
        p.grad.norm().item() for p in processor.parameters() if p.grad is not None
    ]
    assert len(grad_norms) > 0
    assert any(g > 0.0 for g in grad_norms)


def test_compute_drift_field_keeps_gen_gradient_and_detaches_goal():
    """Pin the no_grad-wrapped goal boundary in ``_compute_drift_field``.

    The goal target is built under ``torch.no_grad()`` for efficiency, but
    ``gen_scaled`` is the trainable path the caller's regression loss flows
    through. Assert the gradient still reaches ``gen`` via ``gen_scaled``
    while ``goal_scaled`` carries none, so the target acts as a constant.
    """
    torch.manual_seed(0)
    b, n_gen, d = 2, 4, 6
    gen = torch.randn(b, n_gen, d, requires_grad=True)
    pos = torch.randn(b, 1, d)
    goal_scaled, gen_scaled, _ = _compute_drift_field(
        gen, pos, tau_list=[0.05, 0.2], diag_mask_value=100.0
    )
    assert gen_scaled.requires_grad
    assert not goal_scaled.requires_grad
    ((gen_scaled - goal_scaled) ** 2).mean().backward()
    assert gen.grad is not None
    assert torch.isfinite(gen.grad).all()
    assert gen.grad.abs().sum() > 0.0


class _DriftingSyntheticDataset(Dataset):
    """Minimal synthetic `EncodedBatch` source for integration smoke tests."""

    def __init__(
        self,
        *,
        t_in: int,
        t_out: int,
        ch_in: int,
        ch_out: int,
        spatial: int = 16,
        length: int = 2,
    ) -> None:
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.spatial = spatial
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _: int) -> EncodedBatch:
        return EncodedBatch(
            encoded_inputs=torch.randn(
                1, self.t_in, self.spatial, self.spatial, self.ch_in
            ),
            encoded_output_fields=torch.randn(
                1, self.t_out, self.spatial, self.spatial, self.ch_out
            ),
            global_cond=None,
            encoded_info={},
        )


def _single_item_collate(items):
    return items[0]


def test_drifting_processor_lightning_fit_smoke():
    """End-to-end Lightning train+val cycle on a synthetic EncodedBatch
    dataset. Confirms training_step, validation_step, and the
    cohort-inflated drift loss all run cleanly through a real Trainer."""
    t_in, t_out, ch_in, ch_out, n_samples = 2, 2, 1, 1, 4
    dataset = _DriftingSyntheticDataset(
        t_in=t_in, t_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    loader = DataLoader(
        dataset, batch_size=1, collate_fn=_single_item_collate, num_workers=0
    )

    backbone = _make_backbone(
        n_steps_in=t_in, n_steps_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    processor = DriftingProcessor(
        backbone=backbone,
        n_steps_output=t_out,
        n_channels_out=ch_out,
        n_samples=n_samples,
    )
    model = ProcessorModel(processor, optimizer_config=get_optimizer_config())

    train_loss = model.training_step(next(iter(loader)), 0)
    assert train_loss.shape == ()
    assert torch.isfinite(train_loss).all()

    L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        enable_model_summary=False,
        accelerator="cpu",
    ).fit(model, train_dataloaders=loader, val_dataloaders=loader)


def test_drifting_lambda_zero_is_pure_drift():
    """At lambda_es=0 (default) the loss equals the pure drift recompute.

    Mirrors ``test_loss_matches_manual_cohort_slicing`` exactly, but also
    asserts that neither 'drift_loss' nor 'energy_score' appear in
    ``_latest_diagnostics`` (the else-branch adds neither at lambda_es=0).
    """
    b, t_in, t_out, ch_in, ch_out, n_samples = 2, 2, 2, 1, 1, 4
    backbone = _make_backbone(
        n_steps_in=t_in, n_steps_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    processor = DriftingProcessor(
        backbone=backbone,
        n_steps_output=t_out,
        n_channels_out=ch_out,
        n_samples=n_samples,
        lambda_es=0.0,
    )
    batch = _make_batch(b=b, t_in=t_in, t_out=t_out, ch_in=ch_in, ch_out=ch_out)
    batch.encoded_output_fields[1].fill_(5.0)

    torch.manual_seed(42)
    loss_via_processor = processor.loss(batch)

    torch.manual_seed(42)
    batch_n = batch.repeat(n_samples)
    cond = batch_n.encoded_inputs
    target = batch_n.encoded_output_fields
    gen = processor._draw_one(cond, batch_n.global_cond)
    gen_b = gen.reshape(b, n_samples, -1)
    pos_b = target.reshape(b, n_samples, -1)[:, :1, :]
    goal_scaled, gen_scaled, _ = _compute_drift_field(
        gen_b,
        pos_b,
        tau_list=processor.tau_list,
        diag_mask_value=processor.diag_mask_value,
    )
    expected_loss = ((gen_scaled - goal_scaled) ** 2).mean()

    torch.testing.assert_close(loss_via_processor, expected_loss)
    assert "drift_loss" not in processor._latest_diagnostics
    assert "energy_score" not in processor._latest_diagnostics


@pytest.mark.parametrize("scaled", [False, True])
def test_drifting_lambda_adds_energy_score(scaled):
    """With lambda_es=3.0, loss == drift_loss + 3.0 * energy_score, for both the
    difference-form (scaled=False) and the SCRPS (scaled=True) energy-score term.

    Reconstructs gen_b / pos_b under the same seed as processor.loss, then checks
    the combined value to atol=1e-5 — confirming the ``energy_score_scaled`` flag
    propagates to the additive path.
    """
    b, t_in, t_out, ch_in, ch_out, n_samples = 2, 2, 2, 1, 1, 4
    lambda_val = 3.0
    backbone = _make_backbone(
        n_steps_in=t_in, n_steps_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    processor = DriftingProcessor(
        backbone=backbone,
        n_steps_output=t_out,
        n_channels_out=ch_out,
        n_samples=n_samples,
        lambda_es=lambda_val,
        energy_score_a_f=0.95,
        energy_score_scaled=scaled,
    )
    batch = _make_batch(b=b, t_in=t_in, t_out=t_out, ch_in=ch_in, ch_out=ch_out)
    batch.encoded_output_fields[1].fill_(5.0)

    torch.manual_seed(42)
    loss_via_processor = processor.loss(batch)

    torch.manual_seed(42)
    batch_n = batch.repeat(n_samples)
    cond = batch_n.encoded_inputs
    target = batch_n.encoded_output_fields
    gen = processor._draw_one(cond, batch_n.global_cond)
    gen_b = gen.reshape(b, n_samples, -1)
    pos_b = target.reshape(b, n_samples, -1)[:, :1, :]
    goal_scaled, gen_scaled, _ = _compute_drift_field(
        gen_b,
        pos_b,
        tau_list=processor.tau_list,
        diag_mask_value=processor.diag_mask_value,
    )
    drift_loss_manual = ((gen_scaled - goal_scaled) ** 2).mean()
    es_manual = _energy_score(gen_b, pos_b, a_f=0.95, scaled=scaled)
    expected = drift_loss_manual + lambda_val * es_manual

    torch.testing.assert_close(loss_via_processor, expected, atol=1e-5, rtol=0.0)


def test_drifting_energy_score_scaled_defaults_on():
    """The SCRPS form is the default regulariser (lambda_es>0 yields SCRPS)."""
    backbone = _make_backbone(n_steps_in=2, n_steps_out=2, ch_in=1, ch_out=1)
    processor = DriftingProcessor(backbone=backbone, n_steps_output=2, n_channels_out=1)
    assert processor.energy_score_scaled is True


def test_drifting_lambda_records_term_diagnostics():
    """With lambda_es>0, diagnostics contain finite 'drift_loss' and 'energy_score'."""
    b, t_in, t_out, ch_in, ch_out, n_samples = 2, 2, 2, 1, 1, 4
    backbone = _make_backbone(
        n_steps_in=t_in, n_steps_out=t_out, ch_in=ch_in, ch_out=ch_out
    )
    processor = DriftingProcessor(
        backbone=backbone,
        n_steps_output=t_out,
        n_channels_out=ch_out,
        n_samples=n_samples,
        lambda_es=1.5,
    )
    batch = _make_batch(b=b, t_in=t_in, t_out=t_out, ch_in=ch_in, ch_out=ch_out)
    processor.loss(batch)

    diagnostics = processor._latest_diagnostics
    assert "drift_loss" in diagnostics
    assert "energy_score" in diagnostics
    assert torch.isfinite(diagnostics["drift_loss"])
    assert torch.isfinite(diagnostics["energy_score"])


# ---------------------------------------------------------------------------
# Lambda-conditioning: a single model conditioned on lambda.
# ---------------------------------------------------------------------------


def _mlp_lw_backbone(*, include_loss_weight=True, n_steps_in=2, n_steps_out=2):
    return TemporalMLPBackbone(
        in_channels=1,
        out_channels=1,
        cond_channels=1,
        n_steps_output=n_steps_out,
        n_steps_input=n_steps_in,
        global_cond_channels=None,
        include_global_cond=False,
        spatial_shape=(1, 1),
        mod_features=32,
        hidden=(32, 32),
        include_time_embedding=False,
        include_loss_weight=include_loss_weight,
    )


def _scalar_batch(*, b=4, t_in=2, t_out=2):
    return EncodedBatch(
        encoded_inputs=torch.randn(b, t_in, 1, 1, 1),
        encoded_output_fields=torch.randn(b, t_out, 1, 1, 1),
        global_cond=None,
        encoded_info={},
    )


def _conditioned_processor(**kw):
    return DriftingProcessor(
        backbone=_mlp_lw_backbone(),
        n_steps_output=2,
        n_channels_out=1,
        n_samples=4,
        condition_on_lambda=True,
        **kw,
    )


def test_condition_on_lambda_requires_loss_weight_backbone():
    bb = _mlp_lw_backbone(include_loss_weight=False)
    with pytest.raises(ValueError, match="include_loss_weight=True"):
        DriftingProcessor(
            backbone=bb,
            n_steps_output=2,
            n_channels_out=1,
            condition_on_lambda=True,
        )


def test_conditioned_loss_runs_scalar_and_differentiable():
    torch.manual_seed(0)
    proc = _conditioned_processor()
    loss = proc.loss(_scalar_batch())
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in proc.generator.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_conditioned_loss_logs_lambda_diagnostics():
    torch.manual_seed(0)
    proc = _conditioned_processor()
    proc.loss(_scalar_batch())
    diag = proc._latest_diagnostics
    for key in ("cohort_spread", "lambda_lo", "lambda_hi", "spread_lo", "spread_hi"):
        assert key in diag, f"missing diagnostic {key}"
    assert float(diag["lambda_lo"]) <= float(diag["lambda_hi"])


def test_forcing_penalty_logged_when_eta_positive():
    torch.manual_seed(0)
    proc = _conditioned_processor(forcing_eta=1.0, forcing_margin=0.01)
    proc.loss(_scalar_batch())
    assert "forcing_penalty" in proc._latest_diagnostics
    assert torch.isfinite(proc._latest_diagnostics["forcing_penalty"])


def test_unconditioned_path_unchanged_and_has_no_lambda_diagnostics():
    """condition_on_lambda=False routes to the fixed-lambda path: standard drift extras,
    no lambda diagnostics. The non-conditioned path uses a plain backbone."""
    torch.manual_seed(0)
    bb = _mlp_lw_backbone(include_loss_weight=False)
    proc = DriftingProcessor(
        backbone=bb,
        n_steps_output=2,
        n_channels_out=1,
        n_samples=4,
        condition_on_lambda=False,
        lambda_es=1.0,
    )
    proc.loss(_scalar_batch())
    diag = proc._latest_diagnostics
    assert "drift_loss" in diag
    assert "energy_score" in diag
    assert "lambda_lo" not in diag


def test_map_at_lambda_requires_conditioning():
    bb = _mlp_lw_backbone()
    proc = DriftingProcessor(
        backbone=bb, n_steps_output=2, n_channels_out=1, condition_on_lambda=False
    )
    x = torch.randn(3, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="condition_on_lambda=True"):
        proc.map_at_lambda(x, None, lam=1.0)


def test_map_at_lambda_produces_output_shape():
    torch.manual_seed(0)
    proc = _conditioned_processor()
    x = torch.randn(3, 2, 1, 1, 1)
    out = proc.map_at_lambda(x, None, lam=2.0)
    assert out.shape == (3, 2, 1, 1, 1)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# loss_terms — the monitoring surface consumed by HybridLossMonitorCallback
# ---------------------------------------------------------------------------


def _fixed_lambda_processor(**kw):
    return DriftingProcessor(
        backbone=_make_backbone(),
        n_steps_output=2,
        n_channels_out=1,
        n_samples=4,
        lambda_es=1.5,
        **kw,
    )


def test_loss_terms_matches_loss_diagnostics_under_same_seed():
    """Same RNG state => loss_terms mirrors the loss() term diagnostics.

    Both paths draw one cohort through the same generator, so resetting the
    seed must reproduce the exact drift and energy values that loss()
    records — the guarantee that the monitoring numbers describe the same
    objective the optimizer sees.
    """
    proc = _fixed_lambda_processor()
    batch = _make_batch()
    torch.manual_seed(123)
    proc.loss(batch)
    diag = proc._latest_diagnostics
    torch.manual_seed(123)
    terms = proc.loss_terms(batch)
    torch.testing.assert_close(terms["drift"].detach(), diag["drift_loss"])
    torch.testing.assert_close(terms["energy_score"].detach(), diag["energy_score"])


def test_loss_terms_returns_live_scalars():
    torch.manual_seed(0)
    proc = _fixed_lambda_processor()
    terms = proc.loss_terms(_make_batch())
    for key in ("drift", "energy_score", "spread"):
        assert terms[key].shape == ()
        assert torch.isfinite(terms[key]).all()
        assert terms[key].requires_grad


def test_loss_terms_terms_backpropagate_separately():
    """Each term must reach the backbone so per-term grad norms exist."""
    torch.manual_seed(0)
    proc = _fixed_lambda_processor()
    terms = proc.loss_terms(_make_batch())
    params = [p for p in proc.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        terms["drift"], params, retain_graph=True, allow_unused=True
    )
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    grads = torch.autograd.grad(terms["energy_score"], params, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_loss_terms_rejects_lam_on_fixed_lambda_processor():
    proc = _fixed_lambda_processor()
    with pytest.raises(ValueError, match="lambda-conditioned"):
        proc.loss_terms(_make_batch(), lam=3.0)


def test_loss_terms_conditioned_requires_lam():
    proc = _conditioned_processor()
    with pytest.raises(ValueError, match="requires lam"):
        proc.loss_terms(_scalar_batch())


def test_loss_terms_conditioned_evaluates_at_probe_lambda():
    torch.manual_seed(0)
    proc = _conditioned_processor()
    terms = proc.loss_terms(_scalar_batch(), lam=3.0)
    for key in ("drift", "energy_score", "spread"):
        assert terms[key].shape == ()
        assert torch.isfinite(terms[key]).all()
        assert terms[key].requires_grad


def test_drifting_lambda_records_share_diagnostics():
    """lambda_es>0 adds lambda_es and an energy_share in [0,1]."""
    torch.manual_seed(0)
    proc = _fixed_lambda_processor()
    proc.loss(_make_batch())
    diag = proc._latest_diagnostics
    torch.testing.assert_close(diag["lambda_es"], torch.tensor(1.5))
    assert 0.0 <= float(diag["energy_share"]) <= 1.0
