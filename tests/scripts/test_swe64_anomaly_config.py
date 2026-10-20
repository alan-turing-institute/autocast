"""Prepare the matched online-power control without datasets or submissions."""

import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from the_well.data.normalization import ZScoreNormalization

from autocast.losses import AlphaFairCRPSLoss, SpectralSpreadErrorLoss, WeightedSumLoss
from autocast.scripts.setup import setup_epd_model
from autocast.types import Batch, Sample

ROOT = Path(__file__).resolve().parents[2]
HELPER = runpy.run_path(str(ROOT / "scripts/swe64s/calibrate_anomaly_weight.py"))
EDGES = [6 / 64, 14 / 64, 22 / 64, math.sqrt(2) / 2]


def _config(weight=None):
    overrides = [] if weight is None else [f"model.loss_func.weights.2={weight}"]
    return HELPER["compose_config"](HELPER["PRESET"], overrides)


def test_control_changes_only_the_loss_and_name():
    baseline = HELPER["compose_config"](HELPER["BASELINE"])
    cfg = _config(0.003)
    baseline.experiment_name = cfg.experiment_name
    expected = OmegaConf.to_container(baseline, resolve=True)
    actual = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(expected, dict)
    assert isinstance(actual, dict)
    expected["experiment_name"] = actual["experiment_name"]
    expected["model"]["loss_func"] = actual["model"]["loss_func"]
    assert actual == expected
    assert cfg.model.loss_func.losses[:2] == baseline.model.loss_func.losses
    assert cfg.model.loss_func.weights[:2] == baseline.model.loss_func.weights
    assert cfg.model.processor.noise_mode == "spatial"
    assert cfg.model.processor.n_noise_input_channels == 1
    assert cfg.model.processor.hidden_dim == 224
    assert cfg.model.processor.n_layers == 12
    assert cfg.model.processor.num_heads == 8
    assert cfg.model.processor.patch_size == 4
    assert cfg.model.n_members == 8
    assert cfg.datamodule.batch_size == 32
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 256
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.min_lr_ratio == cfg.optimizer.warmup == 0
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert not cfg.model.processor.include_global_cond
    assert not cfg.model.encoder.with_constants
    assert cfg.model.loss_func.weights == [1.0, 0.1, 0.003]
    loss = instantiate(cfg.model.loss_func)
    assert isinstance(loss, WeightedSumLoss)
    assert loss.requires_ambient_predictions


def test_auxiliary_has_physical_scale_bands_and_complete_diagonal_coverage():
    settings = _config(0.003).model.loss_func.losses[2]
    assert settings.edges == EDGES
    assert settings.power_scale == 1.0
    assert settings.include_dc
    auxiliary = instantiate(settings)
    assert isinstance(auxiliary, SpectralSpreadErrorLoss)
    weights = auxiliary._band_weights(torch.zeros(1, 1, 64, 64, 3), (2, 3))
    assert weights.shape == (5, 4096)
    torch.testing.assert_close(weights.sum(1), torch.ones(5))
    assert torch.all((weights > 0).sum(0) == 1)
    assert weights[0, 0] == 1
    assert weights[-1, 32 * 64 + 32] > 0  # Diagonal Nyquist.
    assert weights[1, 3] > 0  # Forcing ring.
    assert weights[2, 6] > 0
    assert weights[3, 14] > 0
    assert weights[3, 16] > 0  # First token lattice harmonic.
    assert weights[4, 22] > 0


def test_uncalibrated_weight_cannot_be_instantiated():
    cfg = _config()
    assert OmegaConf.is_missing(cfg.model.loss_func.weights, 2)
    with pytest.raises(MissingMandatoryValue):
        instantiate(cfg.model.loss_func)


def test_four_gpu_wandb_preflight_preserves_matched_budget():
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        cfg = compose(
            config_name="encoder_processor_decoder",
            return_hydra_config=True,
            overrides=[
                f"local_experiment=swe64s/{HELPER['PRESET']}",
                f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
                "model.loss_func.weights.2=0.003",
                "+distributed=ddp_4gpu_slurm",
                "trainer.max_time=00:07:45:00",
                "logging.wandb.enabled=true",
                "logging.wandb.mode=online",
            ],
        )
    assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
    assert cfg.hydra.launcher.tasks_per_node == 4
    assert cfg.hydra.launcher.timeout_min == 480
    assert cfg.trainer.max_time == "00:07:45:00"
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "online"
    assert 64000 // (cfg.datamodule.batch_size * 4) * cfg.trainer.max_epochs == 128000


def test_prepared_preset_backpropagates_through_decoded_residual_states():
    cfg = _config(0.003)  # Test value only; not a calibrated experiment weight.
    cfg.model.processor.hidden_dim = 16
    cfg.model.processor.n_layers = 1
    cfg.model.processor.num_heads = 2
    fields = ["h", "u", "v"]
    norm = ZScoreNormalization(
        stats={
            "mean": dict.fromkeys(fields, 0.0),
            "std": dict.fromkeys(fields, 1.0),
            "mean_delta": dict.fromkeys(fields, 0.0),
            "std_delta": dict.fromkeys(fields, 0.2),
        },
        core_field_names=fields,
        core_constant_field_names=[],
    )
    batch = Batch(
        input_fields=torch.randn(2, 1, 64, 64, 3),
        output_fields=torch.randn(2, 1, 64, 64, 3),
        constant_fields=None,
        constant_scalars=None,
    )
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": batch,
    }
    datamodule = cast(Any, SimpleNamespace(train_dataset=SimpleNamespace(norm=norm)))
    model = setup_epd_model(cfg, stats, datamodule)
    value, prediction = model.loss(batch)
    assert prediction is not None
    assert prediction.shape == (2, 1, 64, 64, 3, 8)
    expected = instantiate(cfg.model.loss_func)(prediction, batch.output_fields)
    torch.testing.assert_close(value, expected)
    value.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.processor.parameters()
    )


def test_source_guard_accepts_resolved_shape_placeholders_only():
    baseline = HELPER["compose_config"](HELPER["BASELINE"])
    source = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    source.model.processor.in_channels = 3
    source.model.processor.out_channels = 3
    HELPER["validate_source"](source, baseline)
    source.model.processor.n_noise_input_channels = 4
    with pytest.raises(ValueError, match="baseline model"):
        HELPER["validate_source"](source, baseline)


@pytest.mark.parametrize("component", ["model", "datamodule"])
def test_source_guard_rejects_altered_loss_or_task(component):
    baseline = HELPER["compose_config"](HELPER["BASELINE"])
    source = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    if component == "model":
        source.model.loss_func.weights[1] = 0.2
    else:
        source.datamodule.n_steps_output = 4
    with pytest.raises(ValueError, match=r"(baseline model|datamodule)"):
        HELPER["validate_source"](source, baseline)


def test_train_batch_gradient_calibration_caps_every_recorded_batch():
    cfg = _config(1)
    loss = instantiate(cfg.model.loss_func)
    base = WeightedSumLoss(list(loss.losses[:2]), [1.0, 0.1])
    auxiliary = loss.losses[2]
    rng = torch.Generator().manual_seed(42)
    batches = [
        (
            scale * torch.randn(4, 1, 8, 8, 3, 8, generator=rng),
            torch.randn(4, 1, 8, 8, 3, generator=rng),
        )
        for scale in (0.3, 0.8, 1.5)
    ]
    preserved = [(prediction.clone(), target.clone()) for prediction, target in batches]
    result = HELPER["calibrate_batches"](base, auxiliary, iter(batches))
    assert result["weight"] == min(row["candidate_weight"] for row in result["rows"])
    assert result["weight"] > 0
    assert len(result["rows"]) == 3
    for row, (prediction, target), (saved_prediction, saved_target) in zip(
        result["rows"], batches, preserved, strict=True
    ):
        assert row["weighted_gradient_ratio"] <= 0.1 + 1e-12
        torch.testing.assert_close(prediction, saved_prediction)
        torch.testing.assert_close(target, saved_target)
        assert prediction.grad is None
        assert target.grad is None
        assert "spread_power_t_band_c" in row
        assert "residual_power_t_band_c" in row
    overrides = [result["weight_override"]]
    calibrated = HELPER["compose_config"](HELPER["PRESET"], overrides)
    assert instantiate(calibrated.model.loss_func).weights[2] == result["weight"]


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, math.nan, math.inf])
def test_rejects_invalid_gradient_fraction(fraction):
    with pytest.raises(ValueError, match="gradient_fraction"):
        HELPER["calibrate_batches"](
            AlphaFairCRPSLoss(),
            SpectralSpreadErrorLoss([0.5]),
            [],
            gradient_fraction=fraction,
        )


def test_empty_batches_or_collapsed_ensemble_cannot_select_a_weight():
    base, auxiliary = AlphaFairCRPSLoss(), SpectralSpreadErrorLoss([0.5])
    with pytest.raises(ValueError, match="At least one training batch"):
        HELPER["calibrate_batches"](base, auxiliary, [])
    with pytest.raises(ValueError, match="nonzero finite gradients"):
        HELPER["calibrate_batches"](
            base, auxiliary, [(torch.zeros(2, 1, 4, 1, 8), torch.ones(2, 1, 4, 1))]
        )


def test_calibration_preparation_uses_only_training_and_never_updates_model(
    tmp_path, monkeypatch
):
    scope = HELPER["prepare_calibration"].__globals__
    source_config = tmp_path / "source.yaml"
    checkpoint = tmp_path / "control.ckpt"
    train_path = tmp_path / "train/data.pt"
    train_path.parent.mkdir()
    train_path.write_bytes(b"training fixture")
    (tmp_path / "stats.yml").write_text("normalization fixture")
    OmegaConf.save(
        HELPER["compose_config"](
            HELPER["BASELINE"], [f"datamodule.data_path={tmp_path}"]
        ),
        source_config,
    )
    torch.save(
        {"global_step": 36000, "state_dict": {"amplitude": torch.tensor(0.2)}},
        checkpoint,
    )
    original_hash = scope["_sha256"]
    checkpoint_hash = original_hash(checkpoint)
    manifest = json.loads(
        (ROOT / "local_hydra/local_experiment/swe64s/dataset.json").read_text()
    )
    opened_paths, models = [], []

    class TrainWindows:
        def __init__(self, **kwargs):
            opened_paths.append(kwargs["data_path"])
            assert kwargs["normalization_path"] == str(tmp_path / "stats.yml")
            assert kwargs["use_normalization"]

        def __len__(self):
            return 64000

        def __getitem__(self, index):
            field = torch.full((1, 8, 8, 3), index / 64000)
            return Sample(field, field + 0.1, None, None, None)

    class ToyEnsemble(torch.nn.Module):
        def __init__(self, loss):
            super().__init__()
            self.amplitude = torch.nn.Parameter(torch.tensor(0.1))
            self.loss_func = loss

        def forward(self, batch):
            shape = (*batch.output_fields.shape, 8)
            return batch.input_fields[..., None] + self.amplitude * torch.randn(shape)

    def build_model(cfg, stats, datamodule):
        assert stats["channel_count"] == 3
        assert isinstance(datamodule.train_dataset, TrainWindows)
        model = ToyEnsemble(instantiate(cfg.model.loss_func))
        models.append(model)
        return model

    def data_hash(path):
        return (
            manifest["sha256"]["train/data.pt"]
            if path == train_path
            else original_hash(path)
        )

    monkeypatch.setitem(scope, "_sha256", data_hash)
    monkeypatch.setitem(scope, "SpatioTemporalDataset", TrainWindows)
    monkeypatch.setitem(scope, "setup_epd_model", build_model)
    result = HELPER["prepare_calibration"](
        source_config=source_config,
        checkpoint=checkpoint,
        data_path=tmp_path,
        device="cpu",
        batch_count=2,
    )

    assert opened_paths == [str(train_path)]
    assert result["checkpoint_step"] == 36000
    assert result["checkpoint_sha256"] == checkpoint_hash
    assert result["split_used"] == "train"
    assert not result["training_executed"]
    assert not result["optimizer_state_restored"]
    assert not result["jobs_submitted"]
    assert original_hash(checkpoint) == checkpoint_hash
    assert models[0].amplitude.detach() == 0.2
    assert models[0].amplitude.grad is None
    sampled = [
        index for row in result["rows"] for index in row["training_window_indices"]
    ]
    assert len(sampled) == len(set(sampled)) == 64
    assert (
        len(result["source_config_sha256"]) == len(result["normalization_sha256"]) == 64
    )


def test_calibration_cli_refuses_to_overwrite_before_loading_anything(
    tmp_path, monkeypatch
):
    output = tmp_path / "calibration.json"
    output.write_text("existing result")
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
    assert output.read_text() == "existing result"
