"""Check joint moment calibration without real data, training, or checkpoints."""

import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autocast.losses.moments import SpectralMomentLoss
from autocast.types import Sample

ROOT = Path(__file__).resolve().parents[2]
HELPER = runpy.run_path(str(ROOT / "scripts/swe64s/calibrate_moment_weight.py"))


class LinearBase(torch.nn.Module):
    def forward(self, prediction, target):
        return 2 * (prediction - target.unsqueeze(-1)).mean()


class LinearMoments(torch.nn.Module):
    def __init__(self, mean_factor=1.0, power_factor=-0.5):
        super().__init__()
        self.mean_factor = mean_factor
        self.power_factor = power_factor

    def components(self, prediction, target):
        value = (prediction - target.unsqueeze(-1)).mean()
        return self.mean_factor * value - 10, self.power_factor * value - 10


def test_cap_uses_the_joint_gradient_and_all_batches():
    base = instantiate(HELPER["compose_config"](HELPER["BASELINE"]).model.loss_func)
    auxiliary = SpectralMomentLoss()
    generator = torch.Generator().manual_seed(42)
    batches = [
        (
            scale * torch.randn(2, 1, 4, 4, 3, 8, generator=generator),
            torch.randn(2, 1, 4, 4, 3, generator=generator),
        )
        for scale in (0.1, 0.5, 2.0)
    ]
    originals = [(prediction.clone(), target.clone()) for prediction, target in batches]
    result = HELPER["calibrate_batches"](base, auxiliary, iter(batches))
    assert result["weight"] == min(row["candidate_weight"] for row in result["rows"])
    assert result["weight"] > 0
    for row, (prediction, target), (saved_prediction, saved_target) in zip(
        result["rows"], batches, originals, strict=True
    ):
        reference_prediction = prediction.detach().requires_grad_(True)
        (joint_gradient,) = torch.autograd.grad(
            auxiliary(reference_prediction, target), reference_prediction
        )
        assert row["joint_gradient_rms"] == pytest.approx(
            float(joint_gradient.square().mean().sqrt()), rel=1e-6
        )
        assert row["joint_loss"] == pytest.approx(row["mean_loss"] + row["power_loss"])
        assert row["candidate_weight"] == pytest.approx(
            0.1 * row["base_gradient_rms"] / row["joint_gradient_rms"]
        )
        assert row["weighted_gradient_ratio"] <= 0.1 + 1e-12
        torch.testing.assert_close(prediction, saved_prediction)
        torch.testing.assert_close(target, saved_target)
        assert prediction.grad is None
        assert target.grad is None
    cfg = HELPER["compose_config"](HELPER["PRESET"], [result["weight_override"]])
    assert list(instantiate(cfg.model.loss_func).weights) == [
        1.0,
        0.1,
        result["weight"],
    ]


@pytest.mark.parametrize("mean_factor", [0.0, 1.0])
def test_negative_components_and_zero_component_gradients_are_valid(mean_factor):
    prediction = torch.zeros(2, 1, 4, 1, 8)
    target = torch.zeros(2, 1, 4, 1)
    result = HELPER["calibrate_batches"](
        LinearBase(), LinearMoments(mean_factor=mean_factor), [(prediction, target)]
    )
    row = result["rows"][0]
    assert row["mean_loss"] == row["power_loss"] == -10
    assert result["weight"] == pytest.approx(0.4)
    assert row["weighted_gradient_ratio"] == pytest.approx(0.1)
    if mean_factor == 0:
        assert row["mean_gradient_rms"] == 0
    else:
        assert row["joint_gradient_rms"] < row["mean_gradient_rms"]


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, math.nan, math.inf])
def test_rejects_invalid_gradient_fraction(fraction):
    with pytest.raises(ValueError, match="gradient_fraction"):
        HELPER["calibrate_batches"](
            LinearBase(), LinearMoments(), [], gradient_fraction=fraction
        )


def test_empty_batches_and_zero_joint_gradient_cannot_select_a_weight():
    with pytest.raises(ValueError, match="At least one training batch"):
        HELPER["calibrate_batches"](LinearBase(), LinearMoments(), [])
    with pytest.raises(ValueError, match="nonzero base/joint gradients"):
        HELPER["calibrate_batches"](
            LinearBase(),
            LinearMoments(power_factor=-1),
            [(torch.ones(2, 1, 4, 1, 8), torch.zeros(2, 1, 4, 1))],
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_losses_fail_closed(value):
    with pytest.raises(ValueError, match="finite scalar losses"):
        HELPER["calibrate_batches"](
            LinearBase(),
            LinearMoments(),
            [(torch.full((2, 1, 4, 1, 8), value), torch.zeros(2, 1, 4, 1))],
        )


def test_nonfinite_gradient_with_finite_loss_fails_closed():
    class NonfiniteGradient(torch.autograd.Function):
        @staticmethod
        def forward(_ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, *grad_outputs):
            del ctx
            return torch.full_like(grad_outputs[0], math.nan)

    class InvalidBase(torch.nn.Module):
        def forward(self, prediction, target):
            return NonfiniteGradient.apply((prediction - target.unsqueeze(-1)).mean())

    with pytest.raises(ValueError, match="finite gradients"):
        HELPER["calibrate_batches"](
            InvalidBase(),
            LinearMoments(),
            [(torch.ones(2, 1, 4, 1, 8), torch.zeros(2, 1, 4, 1))],
        )


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    scope = HELPER["prepare_calibration"].__globals__
    source_config = tmp_path / "source.yaml"
    checkpoint = tmp_path / "control.ckpt"
    train_path = tmp_path / "train/data.pt"
    train_path.parent.mkdir()
    train_path.write_bytes(b"training fixture")
    (tmp_path / "stats.yml").write_text("normalization fixture")
    cfg = HELPER["compose_config"](
        HELPER["BASELINE"], [f"datamodule.data_path={tmp_path}"]
    )
    OmegaConf.save(cfg, source_config)
    torch.save(
        {
            "global_step": 36000,
            "state_dict": {"amplitude": torch.tensor(0.2)},
            "ema_state_dict": {"amplitude": torch.tensor(0.9)},
            "optimizer_states": ["Must not be restored"],
        },
        checkpoint,
    )
    original_hash = scope["_sha256"]
    manifest = json.loads(
        (ROOT / "local_hydra/local_experiment/swe64s/dataset.json").read_text()
    )
    opened_paths, models = [], []

    class TrainWindows:
        def __init__(self, **kwargs):
            opened_paths.append(kwargs["data_path"])
            assert kwargs["data_path"] == str(train_path)
            assert kwargs["normalization_path"] == str(tmp_path / "stats.yml")
            assert kwargs["use_normalization"]

        def __len__(self):
            return 64000

        def __getitem__(self, index):
            field = torch.full((1, 2, 2, 3), index / 64000)
            return Sample(field, field + 0.1, None, None, None)

    class ToyEnsemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = torch.nn.Parameter(torch.rand(()))
            self.initial_amplitude = self.amplitude.detach().clone()
            self.noise = []

        def forward(self, batch):
            assert not self.training
            assert not torch.is_grad_enabled()
            noise = torch.randn(*batch.output_fields.shape, 8)
            self.noise.append(noise)
            return batch.input_fields[..., None] + self.amplitude * noise

    def build_model(config, stats, datamodule):
        assert config.seed == 42
        assert stats["channel_count"] == 3
        assert isinstance(datamodule.train_dataset, TrainWindows)
        model = ToyEnsemble()
        models.append(model)
        return model

    def data_hash(path):
        return (
            manifest["sha256"]["train/data.pt"]
            if path == train_path
            else original_hash(path)
        )

    def unexpected_wandb(*args, **kwargs):
        pytest.fail("Calibration must not initialize W&B")

    monkeypatch.setitem(scope, "_sha256", data_hash)
    monkeypatch.setitem(scope, "SpatioTemporalDataset", TrainWindows)
    monkeypatch.setitem(scope, "setup_epd_model", build_model)
    monkeypatch.setattr("wandb.init", unexpected_wandb)
    return SimpleNamespace(
        kwargs={
            "source_config": source_config,
            "checkpoint": checkpoint,
            "data_path": tmp_path,
            "device": "cpu",
        },
        source_config=source_config,
        checkpoint=checkpoint,
        checkpoint_hash=original_hash(checkpoint),
        original_hash=original_hash,
        train_path=train_path,
        opened_paths=opened_paths,
        models=models,
        scope=scope,
    )


def test_preparation_uses_fixed_train_batches_cold_and_raw_warm_without_training(
    preparation,
):
    fixture = preparation
    result = HELPER["prepare_calibration"](**fixture.kwargs)
    assert fixture.opened_paths == [str(fixture.train_path)]
    assert result["batch_count_per_stage"] == 8
    assert result["batch_size"] == 32
    assert result["members"] == 8
    assert result["model_seed"] == 42
    assert result["checkpoint_step"] == 36000
    assert result["checkpoint_sha256"] == fixture.checkpoint_hash
    assert result["split_used"] == "train"
    assert not result["training_executed"]
    assert not result["optimizer_state_restored"]
    assert not result["jobs_submitted"]
    assert fixture.original_hash(fixture.checkpoint) == fixture.checkpoint_hash
    cold, warm = fixture.models
    expected = torch.rand((), generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(cold.initial_amplitude, expected)
    torch.testing.assert_close(warm.initial_amplitude, expected)
    torch.testing.assert_close(cold.amplitude.detach(), expected)
    torch.testing.assert_close(warm.amplitude.detach(), torch.tensor(0.2))
    assert cold.amplitude.grad is warm.amplitude.grad is None
    for cold_noise, warm_noise in zip(cold.noise, warm.noise, strict=True):
        torch.testing.assert_close(cold_noise, warm_noise)
        assert not torch.equal(cold_noise[..., 0], cold_noise[..., 1])
    cold_rows, warm_rows = result["rows"][:8], result["rows"][8:]
    assert len(cold_rows) == len(warm_rows) == 8
    for cold_row, warm_row in zip(cold_rows, warm_rows, strict=True):
        assert cold_row["stage"] == "cold"
        assert warm_row["stage"] == "warm"
        assert (
            cold_row["training_window_indices"] == warm_row["training_window_indices"]
        )
        assert cold_row["prediction_seed"] == warm_row["prediction_seed"]
    sampled = [index for row in cold_rows for index in row["training_window_indices"]]
    assert len(sampled) == len(set(sampled)) == 256
    assert result["weight"] == min(row["candidate_weight"] for row in result["rows"])
    assert all(row["weighted_gradient_ratio"] <= 0.1 + 1e-12 for row in result["rows"])
    for key in (
        "source_config_sha256",
        "checkpoint_sha256",
        "train_sha256",
        "normalization_sha256",
        "dataset_manifest_sha256",
    ):
        assert len(result[key]) == 64
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("invalid", ["train_hash", "stats", "step"])
def test_preparation_checks_source_before_inference(preparation, monkeypatch, invalid):
    fixture = preparation
    if invalid == "train_hash":
        monkeypatch.setitem(fixture.scope, "_sha256", fixture.original_hash)
        message = "Training split"
    elif invalid == "stats":
        source_stats = fixture.source_config.parent / "different-stats.yml"
        source_stats.write_text("Different normalization")
        source = OmegaConf.load(fixture.source_config)
        source.datamodule.normalization_path = str(source_stats)
        OmegaConf.save(source, fixture.source_config)
        message = "normalization differs"
    else:
        torch.save({"global_step": 72000, "state_dict": {}}, fixture.checkpoint)
        message = "36,000-step"
    with pytest.raises(ValueError, match=message):
        HELPER["prepare_calibration"](**fixture.kwargs)
    assert not fixture.models
    assert not fixture.opened_paths


def test_preparation_refuses_a_cap_above_ten_percent():
    with pytest.raises(ValueError, match="gradient_fraction"):
        HELPER["prepare_calibration"](
            source_config=Path("absent"),
            checkpoint=Path("absent"),
            data_path=Path("absent"),
            device="cpu",
            gradient_fraction=0.2,
        )


def test_cli_refuses_to_overwrite_before_loading_anything(tmp_path, monkeypatch):
    output = tmp_path / "calibration.json"
    output.write_text("Existing result")
    monkeypatch.setattr(
        "sys.argv",
        [
            "calibrate",
            "--source-config",
            "absent",
            "--checkpoint",
            "absent",
            "--data-path",
            "absent",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError):
        HELPER["main"]()
    assert output.read_text() == "Existing result"
