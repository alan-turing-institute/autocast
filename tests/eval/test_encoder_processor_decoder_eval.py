import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from autocast.decoders.base import Decoder
from autocast.encoders.base import Encoder
from autocast.eval.encoder_processor_decoder import (
    _build_metrics,
    _evaluate_metrics,
    _infer_spatial_dims,
    _resolve_csv_path,
    _resolve_device,
    _write_csv,
    main,
)
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.processors.base import Processor
from autocast.types import Batch, EncodedBatch


class DummyMetric(nn.Module):
    def forward(self, preds: torch.Tensor, trues: torch.Tensor) -> torch.Tensor:
        return (preds - trues) ** 2


class _IdentityEncoder(Encoder):
    def __init__(self, time_steps: int = 1, channels: int = 1) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.channels = channels
        self.encoder_model = nn.Identity()
        self.latent_dim = time_steps * channels

    def forward(self, batch: Batch) -> torch.Tensor:
        b, t, w, h, c = batch.input_fields.shape
        return batch.input_fields.view(b, t * c, w, h)

    def encode(self, batch: Batch) -> torch.Tensor:
        return self.forward(batch)

    def encode_batch(
        self,
        batch: Batch,
        encoded_info: dict | None = None,
    ) -> EncodedBatch:
        encoded_inputs = self.encode(batch)
        encoded_outputs = self.encode(batch)
        return EncodedBatch(
            encoded_inputs=encoded_inputs,
            encoded_output_fields=encoded_outputs,
            encoded_info=encoded_info or {},
        )


class _IdentityDecoder(Decoder):
    def __init__(self, time_steps: int = 1, channels: int = 1) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.channels = channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, _tc, w, h = z.shape
        return z.view(b, self.time_steps, w, h, self.channels)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)


class _IdentityProcessor(Processor[torch.Tensor]):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def map(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def loss(self, batch: torch.Tensor) -> torch.Tensor:  # pragma: no cover - unused
        return batch.mean()


def _make_epd_model(time_steps: int = 1, channels: int = 1) -> EncoderProcessorDecoder:
    encoder = _IdentityEncoder(time_steps=time_steps, channels=channels)
    decoder = _IdentityDecoder(time_steps=time_steps, channels=channels)
    processor = _IdentityProcessor()
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)
    return EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=nn.MSELoss(),
    )


def test_resolve_csv_path_defaults(tmp_path):
    args = argparse.Namespace(csv_path=None, work_dir=tmp_path)
    resolved = _resolve_csv_path(args)
    assert resolved == (tmp_path / "evaluation_metrics.csv").resolve()

    custom = tmp_path / "custom.csv"
    args = argparse.Namespace(csv_path=custom, work_dir=tmp_path)
    assert _resolve_csv_path(args) == custom.resolve()


def test_resolve_device_auto_prefers_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", None, raising=False)

    auto_device = _resolve_device("auto")
    explicit_device = _resolve_device("cpu")

    assert auto_device.type == "cpu"
    assert explicit_device.type == "cpu"


def test_build_metrics_respects_requested_keys():
    metrics = _build_metrics(["mae", "rmse"])
    assert set(metrics) == {"mae", "rmse"}
    assert all(isinstance(metric, nn.Module) for metric in metrics.values())


def test_infer_spatial_dims_and_overrides():
    args = argparse.Namespace(n_spatial_dims=None)
    output_shape = (2, 3, 4, 5, 6)
    assert _infer_spatial_dims(args, output_shape) == 2

    args = argparse.Namespace(n_spatial_dims=1)
    assert _infer_spatial_dims(args, output_shape) == 1

    with pytest.raises(ValueError, match="Unable to infer spatial dimensions"):
        _infer_spatial_dims(argparse.Namespace(n_spatial_dims=None), (2, 3))


def _make_batch(batch_size: int = 2, true_value: float = 1.0) -> Batch:
    shape = (batch_size, 1, 1, 1, 1)
    return Batch(
        input_fields=torch.zeros(shape),
        output_fields=torch.full(shape, true_value),
        constant_scalars=None,
        constant_fields=None,
    )


def test_evaluate_metrics_generates_rows():
    model = _make_epd_model()
    dataloader = [_make_batch(), _make_batch()]
    metrics: dict[str, nn.Module] = {"dummy": DummyMetric()}

    rows = _evaluate_metrics(model, dataloader, metrics, torch.device("cpu"))

    assert len(rows) == 3
    batch_rows = [row for row in rows if row["batch_index"] != "all"]
    assert all(row["num_samples"] == 2 for row in batch_rows)
    assert all(row["dummy"] == pytest.approx(1.0) for row in batch_rows)

    aggregate = next(row for row in rows if row["batch_index"] == "all")
    assert aggregate["num_samples"] == 4
    assert aggregate["dummy"] == pytest.approx(1.0)


def test_write_csv_creates_file(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    rows = [
        {
            "dataset_split": "test",
            "batch_index": 0,
            "num_samples": 1,
            "dummy": 0.5,
        }
    ]

    _write_csv(rows, csv_path, ["dummy"])

    content = csv_path.read_text().splitlines()
    assert "dataset_split,batch_index,num_samples,dummy" in content[0]
    assert "test,0,1,0.5" in content[1]


def test_main_runs_without_checkpoint(tmp_path: Path):
    # Minimal args; checkpoint is unused because _load_model is patched.
    checkpoint_path = tmp_path / "dummy.ckpt"
    work_dir = tmp_path / "work"
    args = argparse.Namespace(
        config_dir=tmp_path,
        config_name="encoder_processor_decoder",
        overrides=[],
        autoencoder_checkpoint=None,
        freeze_autoencoder=None,
        n_steps_input=None,
        n_steps_output=None,
        stride=None,
        checkpoint=checkpoint_path,
        work_dir=work_dir,
        csv_path=None,
        metrics=None,
        n_spatial_dims=None,
        batch_indices=[],
        video_dir=None,
        video_format="mp4",
        video_sample_index=0,
        fps=5,
        device="cpu",
        free_running_only=True,
    )

    dataloader = [_make_batch(batch_size=1)]
    csv_calls: list[tuple[Any, Any, Any]] = []
    log_calls: list[tuple[Any, Any]] = []

    class _Datamodule:
        def test_dataloader(self):
            return dataloader

        def rollout_test_dataloader(self):  # pragma: no cover - batch_indices empty
            return dataloader

    with (
        patch("autocast.eval.encoder_processor_decoder.parse_args", lambda: args),
        patch(
            "autocast.eval.encoder_processor_decoder.compose_training_config",
            lambda _: OmegaConf.create(
                {
                    "seed": 0,
                    "logging": {},
                    "experiment_name": "epd",
                    "training": {},
                    "model": {},
                }
            ),
        ),
        patch(
            "autocast.eval.encoder_processor_decoder.create_wandb_logger",
            lambda *a, **k: (MagicMock(), MagicMock()),
        ),
        patch(
            "autocast.eval.encoder_processor_decoder.resolve_training_params",
            lambda _cfg, _a: SimpleNamespace(n_steps_input=1, n_steps_output=1),
        ),
        patch(
            "autocast.eval.encoder_processor_decoder.update_data_cfg",
            lambda *a, **k: None,
        ),
        patch(
            "autocast.eval.encoder_processor_decoder.configure_module_dimensions",
            lambda *a, **k: None,
        ),
        patch(
            "autocast.eval.encoder_processor_decoder.normalize_processor_cfg",
            lambda *a, **k: None,
        ),
        patch(
            "autocast.eval.encoder_processor_decoder.prepare_datamodule",
            lambda _cfg: (_Datamodule(), 1, 1, 1, None, None),
        ),
        patch(
            "autocast.eval.encoder_processor_decoder._build_metrics",
            lambda _names: {"dummy": DummyMetric()},
        ),
        patch(
            "autocast.eval.encoder_processor_decoder._load_model",
            lambda _cfg, _ckpt: _make_epd_model(),
        ),
        patch(
            "autocast.eval.encoder_processor_decoder._resolve_device",
            lambda _arg: torch.device("cpu"),
        ),
        patch(
            "autocast.eval.encoder_processor_decoder._evaluate_metrics",
            lambda _model, _dl, _metrics, _device: [
                {
                    "dataset_split": "test",
                    "batch_index": "all",
                    "num_samples": 1,
                    "dummy": 0.0,
                }
            ],
        ),
        patch(
            "autocast.eval.encoder_processor_decoder._write_csv",
            lambda rows, path, metric_names: csv_calls.append(
                (rows, path, metric_names)
            ),
        ),
        patch(
            "autocast.eval.encoder_processor_decoder.log_metrics",
            lambda logger, payload: log_calls.append((logger, payload)),
        ),
        patch(
            "autocast.eval.encoder_processor_decoder.L.seed_everything",
            lambda *a, **k: None,
        ),
    ):
        main()

    assert csv_calls, "CSV writer was not called"
    written_rows, written_path, metric_names = csv_calls[0]
    assert written_path == (work_dir / "evaluation_metrics.csv").resolve()
    assert metric_names == ["dummy"]
    assert written_rows[0]["dummy"] == 0.0
    assert log_calls
    assert log_calls[0][1] == {"test/dummy": 0.0}
