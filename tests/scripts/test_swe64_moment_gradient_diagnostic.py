"""Test radial/output-gradient diagnostics without data or GPU jobs."""

import json
import math
import runpy
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autocast.losses.moments import SpectralMomentLoss
from autocast.types import Sample

ROOT = Path(__file__).resolve().parents[2]
HELPER = runpy.run_path(str(ROOT / "scripts/swe64s/diagnose_moment_gradients.py"))


def test_common_differential_split_is_orthogonal_and_retains_member_measure():
    g = torch.randn(2, 1, 8, 8, 3, 8, generator=torch.Generator().manual_seed(42))
    parts = HELPER["ensemble_parts"](g)
    torch.testing.assert_close(parts["common"] + parts["differential"], g)
    torch.testing.assert_close(
        parts["differential"].mean(-1), torch.zeros_like(g[..., 0]), atol=2e-7, rtol=0
    )
    assert (parts["common"] * parts["differential"]).mean().item() == pytest.approx(
        0, abs=1e-7
    )
    assert g.square().mean().item() == pytest.approx(
        (parts["common"].square().mean() + parts["differential"].square().mean()).item()
    )


def test_shell_energy_is_parseval_and_weights_scale_gradient_not_alignment():
    g = torch.randn(2, 1, 8, 8, 3, 8, generator=torch.Generator().manual_seed(42))
    value = HELPER["gradient_statistics"](g, 2 * g, g, coefficient=0.002)
    total = value["parts"]["total"]
    assert total["gradient_rms"] == pytest.approx(g.square().mean().sqrt().item())
    assert total["weighted_rms_ratio_to_baseline"] == pytest.approx(0.001)
    assert total["cosine_to_baseline"] == pytest.approx(1)
    shell_energy = torch.tensor(total["gradient_energy_channel_shell"])
    torch.testing.assert_close(shell_energy.sum(-1), g.square().mean((0, 1, 2, 3, 5)))
    torch.testing.assert_close(
        torch.tensor(total["weighted_gradient_energy_channel_shell"]),
        shell_energy * 0.002**2,
    )
    assert value["parts"]["differential"]["gradient_rms"] > 0


def test_single_mode_is_in_expected_shell_and_descent_sign_reduces_anomaly():
    x = torch.arange(16, dtype=torch.float64)
    wave = torch.cos(2 * math.pi * 3 * x / 16)
    anomaly = wave[None, None, :, None, None, None].expand(1, 1, 16, 16, 1, 8).clone()
    anomaly[..., 4:] *= -1
    result = HELPER["gradient_statistics"](anomaly, anomaly, anomaly, coefficient=0.5)
    part = result["parts"]["differential"]
    energy = torch.tensor(part["gradient_energy_channel_shell"])[0]
    assert energy.argmax() == 3
    assert energy[3] / energy.sum() == pytest.approx(1)
    derivative = torch.tensor(
        part["anomaly_power_descent_derivative_channel_shell"]
    ).sum()
    assert derivative == pytest.approx(-0.5)
    assert result["parts"]["common"]["gradient_rms"] < 1e-15
    assert result["parts"]["common"]["cosine_to_baseline"] is None


def test_all_components_match_direct_joint_gradient_and_leave_inputs_untouched():
    cfg = HELPER["compose_config"](HELPER["BASELINE"])
    physical, fourier = (instantiate(item) for item in cfg.model.loss_func.losses)
    moments = SpectralMomentLoss()
    rng = torch.Generator().manual_seed(42)
    pred = torch.randn(2, 1, 4, 4, 3, 8, generator=rng)
    target = torch.randn(2, 1, 4, 4, 3, generator=rng)
    original = pred.clone()
    result = HELPER["diagnose_batch"](physical, fourier, moments, pred, target)
    leaf = pred.clone().requires_grad_()
    base = physical(leaf, target) + 0.1 * fourier(leaf, target)
    base_g = torch.autograd.grad(base, leaf)[0]
    leaf = pred.clone().requires_grad_()
    joint_g = torch.autograd.grad(moments(leaf, target), leaf)[0]
    ratio = 0.0005 * joint_g.square().mean().sqrt() / base_g.square().mean().sqrt()
    assert result["weighted_joint_moments"]["0.0005"]["parts"]["total"][
        "weighted_rms_ratio_to_baseline"
    ] == pytest.approx(ratio.item())
    assert result["weighted_joint_moments"]["0.002"]["parts"]["total"][
        "weighted_gradient_rms"
    ] == pytest.approx(
        4
        * result["weighted_joint_moments"]["0.0005"]["parts"]["total"][
            "weighted_gradient_rms"
        ]
    )
    assert (
        result["weighted_joint_moments"]["0.0"]["parts"]["total"][
            "weighted_gradient_rms"
        ]
        == 0
    )
    torch.testing.assert_close(pred, original)
    assert pred.grad is None
    assert target.grad is None


@pytest.mark.parametrize("coefficient", [-1, math.nan, math.inf])
def test_invalid_coefficient_fails_closed(coefficient):
    value = torch.ones(1, 1, 4, 4, 1, 8)
    with pytest.raises(ValueError, match="coefficient"):
        HELPER["gradient_statistics"](value, value, value, coefficient=coefficient)


def test_checkpoint_validation_checks_hash_and_counter_and_keeps_raw_weights(tmp_path):
    checkpoint = tmp_path / "test.ckpt"
    torch.save(
        {
            "global_step": 34000,
            "state_dict": {"a": torch.tensor(1.0)},
            "ema_state_dict": {"a": torch.tensor(2.0)},
            "optimizer_states": ["not restored"],
        },
        checkpoint,
    )
    sha = HELPER["_sha256"](checkpoint)
    raw, record = HELPER["validate_checkpoint"](
        checkpoint, expected_sha256=sha, expected_step=34000
    )
    assert raw["a"] == 1
    assert record["sha256"] == sha
    with pytest.raises(ValueError, match="SHA256"):
        HELPER["validate_checkpoint"](
            checkpoint, expected_sha256="wrong", expected_step=34000
        )
    with pytest.raises(ValueError, match="step"):
        HELPER["validate_checkpoint"](
            checkpoint, expected_sha256=sha, expected_step=33000
        )


def test_nonfinite_gradient_is_rejected():
    value = torch.ones(1, 1, 4, 4, 1, 8)
    with pytest.raises(ValueError, match="finite predictions"):
        HELPER["gradient_statistics"](value * math.nan, value, value, coefficient=1)


def test_env_override_allows_outside_repo_script(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCAST_ROOT", str(ROOT))
    copied = tmp_path / "diagnostic.py"
    copied.write_text(
        (ROOT / "scripts/swe64s/diagnose_moment_gradients.py").read_text()
    )
    assert runpy.run_path(str(copied))["ROOT"] == ROOT


def test_preparation_is_train_only_and_pairs_windows_and_noise(monkeypatch, tmp_path):
    scope = HELPER["prepare_diagnostic"].__globals__
    train = tmp_path / "train/data.pt"
    train.parent.mkdir()
    train.write_bytes(b"training fixture")
    (tmp_path / "stats.yml").write_text("normalization fixture")
    manifest = json.loads(
        (ROOT / "local_hydra/local_experiment/swe64s/dataset.json").read_text()
    )
    sha = HELPER["_sha256"]
    models, opened = [], []

    class TrainingDataset:
        def __init__(self, **kwargs):
            opened.append(kwargs["data_path"])
            assert kwargs["data_path"] == str(train)

        def __len__(self):
            return 64000

        def __getitem__(self, index):
            field = torch.full((1, 4, 4, 3), index / 64000)
            return Sample(field, field + 0.1, None, None, None)

    class ToyEnsemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = torch.nn.Parameter(torch.tensor(0.0))
            self.noise = []

        def forward(self, batch):
            assert not self.training
            assert not torch.is_grad_enabled()
            noise = torch.randn(*batch.output_fields.shape, 8)
            self.noise.append(noise)
            return batch.input_fields[..., None] + self.amplitude * noise

    def setup(*_args):
        model = ToyEnsemble()
        models.append(model)
        return model

    def data_hash(path):
        return manifest["sha256"]["train/data.pt"] if path == train else sha(path)

    monkeypatch.setitem(scope, "SpatioTemporalDataset", TrainingDataset)
    monkeypatch.setitem(scope, "setup_epd_model", setup)
    monkeypatch.setitem(scope, "_sha256", data_hash)
    args = {}
    for label, preset, step, amplitude in (
        ("baseline", HELPER["BASELINE"], 36000, 0.2),
        ("moments", HELPER["PRESET"], 34000, 0.3),
    ):
        cfg = HELPER["compose_config"](preset, [f"datamodule.data_path={tmp_path}"])
        cfg.model.decoder.output_channels = 3
        config = tmp_path / f"{label}.yaml"
        OmegaConf.save(cfg, config)
        checkpoint = tmp_path / f"{label}.ckpt"
        torch.save(
            {
                "global_step": step,
                "state_dict": {"amplitude": torch.tensor(amplitude)},
                "ema_state_dict": {"amplitude": torch.tensor(0.9)},
            },
            checkpoint,
        )
        args.update(
            {
                f"{label}_config": config,
                f"{label}_checkpoint": checkpoint,
                f"{label}_sha256": sha(checkpoint),
            }
        )
    result = HELPER["prepare_diagnostic"](
        **args, data_path=tmp_path, device="cpu", batch_count=1, draws_per_batch=2
    )
    assert opened == [str(train)]
    assert result["split_used"] == "train"
    assert not result["training_executed"]
    assert not result["optimizer_state_restored"]
    for baseline_row, moment_row in zip(
        result["stages"]["baseline"]["rows"],
        result["stages"]["moments"]["rows"],
        strict=True,
    ):
        assert (
            baseline_row["training_window_indices"]
            == moment_row["training_window_indices"]
        )
        assert baseline_row["prediction_seed"] == moment_row["prediction_seed"]
    for first, second in zip(models[0].noise, models[1].noise, strict=True):
        torch.testing.assert_close(first, second)
    assert not torch.equal(models[0].noise[0], models[0].noise[1])
    for model in models:
        assert model.amplitude.grad is None
    assert float(models[0].amplitude.detach()) == pytest.approx(0.2)
    assert float(models[1].amplitude.detach()) == pytest.approx(0.3)


def test_resolved_source_accepts_only_known_decoder_shape_and_preserves_candidate():
    candidate = HELPER["compose_config"](HELPER["BASELINE"])
    assert candidate.model.decoder.output_channels is None
    source = deepcopy(candidate)
    source.model.decoder.output_channels = 3
    HELPER["validate_resolved_source"](source, candidate)
    assert candidate.model.decoder.output_channels is None
    source.model.decoder.output_channels = 4
    with pytest.raises(ValueError, match="baseline model"):
        HELPER["validate_resolved_source"](source, candidate)
    source.model.decoder.output_channels = 3
    source.model.n_members = 4
    with pytest.raises(ValueError, match="baseline model"):
        HELPER["validate_resolved_source"](source, candidate)
    candidate.model.decoder.output_channels = 4
    with pytest.raises(ValueError, match="three decoder"):
        HELPER["validate_resolved_source"](source, candidate)
