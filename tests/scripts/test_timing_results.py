"""Timing runs persist small results without model or optimizer checkpoints."""

import json
from unittest.mock import Mock

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from autocast.scripts import training
from autocast.scripts.workflow import commands
from autocast.scripts.workflow.cli import build_parser
from autocast.scripts.workflow.overrides import extract_override_value


class _TimingModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def training_step(self, batch, batch_idx):
        del batch, batch_idx
        return self.weight.square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _TimingData(L.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.arange(4.0)), batch_size=1)


@pytest.mark.parametrize("kind", ["epd", "ae"])
def test_timing_only_training_writes_json_without_checkpoint(
    kind, tmp_path, monkeypatch
):
    config = OmegaConf.create(
        {
            "logging": {"wandb": {"enabled": False}},
            "output": {"timing_only": True, "save_config": False},
            "trainer": {
                "_target_": "lightning.pytorch.Trainer",
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 2,
                "limit_train_batches": 2,
                "limit_val_batches": 0,
                "num_sanity_val_steps": 0,
                "enable_checkpointing": False,
                "enable_model_summary": False,
                "enable_progress_bar": False,
                "callbacks": [],
            },
        }
    )
    model, data = _TimingModel(), _TimingData()
    save = Mock(side_effect=AssertionError("Timing must not save checkpoints"))
    monkeypatch.setattr(L.Trainer, "save_checkpoint", save)
    if kind == "epd":
        training.run_training(config, model, data, tmp_path, skip_test=True)
    else:
        monkeypatch.setattr(training, "setup_datamodule", lambda cfg: (data, cfg, {}))
        monkeypatch.setattr(
            training, "setup_autoencoder_model", Mock(return_value=model)
        )
        assert training.train_autoencoder(config, tmp_path) == tmp_path / "timing.json"
    path = tmp_path / "timing.json"
    result = json.loads(path.read_text())
    assert result["global_step"] == 4
    assert result["world_size"] == 1
    assert len(result["epoch_times_s"]) == 2
    assert all(value > 0 for value in result["epoch_times_s"])
    assert model.weight.item() < 0.5
    assert path.stat().st_size < 4096
    assert not list(tmp_path.rglob("*.ckpt"))
    save.assert_not_called()


def test_time_epochs_uses_json_and_disables_checkpoint_callbacks(tmp_path, monkeypatch):
    captured = []

    def run(module, overrides, **kwargs):
        del module, kwargs
        captured.extend(overrides)
        (tmp_path / "timing.json").write_text(json.dumps({"epoch_times_s": [2.0, 4.0]}))

    monkeypatch.setattr(commands, "run_module", run)
    result = commands.time_epochs_command(
        mode="local",
        dataset="zarr",
        output_base=str(tmp_path),
        overrides=[],
        num_epochs=2,
        work_dir=str(tmp_path),
    )
    assert result is not None
    assert result["seconds_per_epoch"] == 3.0
    normalized = [value.lstrip("+") for value in captured]
    assert extract_override_value(normalized, "output.timing_only") == "true"
    assert extract_override_value(normalized, "trainer.enable_checkpointing") == "false"
    assert extract_override_value(normalized, "trainer.callbacks") == "[]"
    assert not any("timing.ckpt" in value for value in captured)


@pytest.mark.parametrize("flag", ["--from-timing", "--from-checkpoint"])
def test_timing_result_cli_accepts_both_spellings(flag):
    args = build_parser().parse_args(["time-epochs", flag, "timing.json"])
    assert args.from_checkpoint == "timing.json"


def test_legacy_timing_checkpoint_remains_readable(tmp_path):
    path = tmp_path / "timing.ckpt"
    torch.save(
        {"callbacks": {"TrainingTimerCallback": {"epoch_times_s": [1.0, 2.0]}}}, path
    )
    assert commands._extract_epoch_times(path) == [1.0, 2.0]


@pytest.mark.parametrize("content", ["invalid JSON", "{}", '{"epoch_times_s": []}'])
def test_missing_or_invalid_timing_result_returns_none(tmp_path, content):
    path = tmp_path / "timing.json"
    assert commands._extract_epoch_times(path) is None
    path.write_text(content)
    assert commands._extract_epoch_times(path) is None
