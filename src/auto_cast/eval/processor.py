"""Evaluation CLI for encoder-processor-decoder checkpoints."""

from __future__ import annotations

import argparse
import csv
import logging
from argparse import BooleanOptionalAction
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from auto_cast.metrics.spatiotemporal import (
    MAE,
    MSE,
    NMAE,
    NMSE,
    NRMSE,
    RMSE,
    VMSE,
    VRMSE,
    LInfinity,
)
from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.models.encoder_processor_decoder import EncoderProcessorDecoder
from auto_cast.train.configuration import (
    compose_training_config,
    configure_module_dimensions,
    normalize_processor_cfg,
    prepare_datamodule,
    resolve_training_params,
    update_data_cfg,
)
from auto_cast.types import Batch
from auto_cast.utils import plot_spatiotemporal_video

log = logging.getLogger(__name__)

AVAILABLE_METRICS = {
    "mae": MAE,
    "mse": MSE,
    "nmse": NMSE,
    "nmae": NMAE,
    "rmse": RMSE,
    "nrmse": NRMSE,
    "vmse": VMSE,
    "vrmse": VRMSE,
    "linf": LInfinity,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation utility."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained encoder-processor-decoder checkpoint."
    )
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--config-dir",
        "--config-path",
        dest="config_dir",
        type=Path,
        default=repo_root / "configs",
        help="Path to the Hydra config directory (defaults to <repo>/configs).",
    )
    parser.add_argument(
        "--config-name",
        default="processor",
        help="Hydra config name to compose (defaults to 'processor').",
    )
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=[],
        help="Optional Hydra override, e.g. --override trainer.max_epochs=5",
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=None,
        help="Retained for parity with training; ignored for evaluation.",
    )
    parser.add_argument(
        "--freeze-autoencoder",
        action=BooleanOptionalAction,
        default=None,
        help="Retained for parity with training; ignored for evaluation.",
    )
    parser.add_argument(
        "--n-steps-input",
        type=int,
        default=None,
        help="Override training.n_steps_input (number of input time steps).",
    )
    parser.add_argument(
        "--n-steps-output",
        type=int,
        default=None,
        help="Override training.n_steps_output (number of target time steps).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the encoder-processor-decoder checkpoint to evaluate.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where evaluation artifacts (csv/videos) are saved.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional explicit path for the metrics CSV (defaults to work-dir).",
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        choices=sorted(AVAILABLE_METRICS.keys()),
        default=None,
        help="Metrics to compute (defaults to mse and rmse).",
    )
    parser.add_argument(
        "--n-spatial-dims",
        type=int,
        default=None,
        help="Override the number of spatial dims (inferred when omitted).",
    )
    parser.add_argument(
        "--batch-index",
        dest="batch_indices",
        type=int,
        action="append",
        default=[],
        help="Batch indices from rollout_test_dataloader() to visualize.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory to save rollout videos (defaults to <work-dir>/videos).",
    )
    parser.add_argument(
        "--video-format",
        choices=("gif", "mp4"),
        default="mp4",
        help="File extension used for saved rollout animations.",
    )
    parser.add_argument(
        "--video-sample-index",
        type=int,
        default=0,
        help="Sample index within the batch to visualize (default: 0).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for the generated videos.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device to run evaluation on (default tries CUDA/MPS before CPU).",
    )
    parser.add_argument(
        "--free-running-only",
        action=BooleanOptionalAction,
        default=True,
        help="Whether to disable teacher forcing during rollouts (default: True).",
    )
    return parser.parse_args()


def _resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv_path is not None:
        return args.csv_path.expanduser().resolve()
    return (args.work_dir / "evaluation_metrics.csv").resolve()


def _resolve_video_dir(args: argparse.Namespace) -> Path:
    if args.video_dir is not None:
        return args.video_dir.expanduser().resolve()
    return (args.work_dir / "videos").resolve()


def _resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_available = getattr(torch.backends, "mps", None)
    if mps_available is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        input_fields=batch.input_fields.to(device),
        output_fields=batch.output_fields.to(device),
        constant_scalars=(
            batch.constant_scalars.to(device)
            if batch.constant_scalars is not None
            else None
        ),
        constant_fields=(
            batch.constant_fields.to(device)
            if batch.constant_fields is not None
            else None
        ),
    )


def _build_metrics(metric_names: Sequence[str]):
    names = metric_names or ("mse", "rmse")
    metrics = {}
    for name in names:
        metric_cls = AVAILABLE_METRICS[name]
        metrics[name] = metric_cls()
    return metrics


def _infer_spatial_dims(args: argparse.Namespace, output_shape: Sequence[int]) -> int:
    if args.n_spatial_dims is not None:
        return args.n_spatial_dims
    spatial_dims = len(output_shape) - 3
    if spatial_dims < 1:
        msg = "Unable to infer spatial dimensions from output shape %s"
        raise ValueError(msg % (output_shape,))
    return spatial_dims


def _evaluate_metrics(
    model: EncoderProcessorDecoder,
    dataloader,
    metrics: dict[str, nn.Module],
    device: torch.device,
    n_spatial_dims: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    totals = dict.fromkeys(metrics, 0.0)
    total_weight = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_on_device = _batch_to_device(batch, device)
            preds = model(batch_on_device)
            trues = batch_on_device.output_fields
            batch_size = preds.shape[0]
            total_weight += batch_size

            row: dict[str, object] = {
                "dataset_split": "test",
                "batch_index": batch_idx,
                "num_samples": batch_size,
            }
            for name, metric in metrics.items():
                value = metric(preds, trues, n_spatial_dims)
                scalar = float(value.mean().item())
                row[name] = scalar
                totals[name] += scalar * batch_size
            rows.append(row)

    if total_weight == 0:
        return rows

    aggregate = {
        "dataset_split": "test",
        "batch_index": "all",
        "num_samples": total_weight,
    }
    for name in metrics:
        aggregate[name] = totals[name] / total_weight
    rows.append(aggregate)
    return rows


def _write_csv(
    rows: list[dict[str, object]],
    csv_path: Path,
    metric_names: Sequence[str],
):
    if not rows:
        log.warning("No evaluation rows to write; skipping CSV generation.")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["dataset_split", "batch_index", "num_samples"]
    fieldnames = base_fields + list(metric_names)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("Wrote metrics CSV to %s", csv_path)


def _load_state_dict(checkpoint_path: Path) -> OrderedDict[str, torch.Tensor]:
    checkpoint_real = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_real,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(checkpoint, Mapping):
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:  # pragma: no cover - defensive fallback for unexpected formats
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        msg = f"Checkpoint {checkpoint_real} does not contain a valid state_dict."
        raise TypeError(msg)
    if isinstance(state_dict, OrderedDict):
        state_dict = state_dict.copy()
    else:
        state_dict = OrderedDict(state_dict)
    state_dict.pop("_metadata", None)
    return state_dict


def _load_model(cfg: DictConfig, checkpoint_path: Path) -> EncoderProcessorDecoder:
    encoder = instantiate(cfg.encoder)
    decoder = instantiate(cfg.decoder)
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)
    processor = instantiate(cfg.processor)
    epd_cfg = cfg.get("encoder_processor_decoder") or {}
    learning_rate = epd_cfg.get("learning_rate", 1e-3)
    training_cfg = cfg.get("training")
    stride = 1
    if isinstance(training_cfg, DictConfig):
        stride = training_cfg.get("stride", 1)
    teacher_forcing_ratio = epd_cfg.get("teacher_forcing_ratio", 0.5)
    max_rollout_steps = epd_cfg.get("max_rollout_steps", 10)
    loss_cfg = epd_cfg.get("loss_func")
    loss_func = instantiate(loss_cfg) if loss_cfg is not None else nn.MSELoss()

    checkpoint_real = checkpoint_path.expanduser().resolve()
    if not checkpoint_real.exists():
        msg = f"Checkpoint not found: {checkpoint_real}"
        raise FileNotFoundError(msg)
    log.info("Loading checkpoint from %s", checkpoint_real)

    state_dict = _load_state_dict(checkpoint_real)

    model = EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        learning_rate=learning_rate,
        stride=stride,
        teacher_forcing_ratio=teacher_forcing_ratio,
        max_rollout_steps=max_rollout_steps,
        loss_func=loss_func,
    )
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        msg = (
            "Checkpoint parameters do not match the instantiated model. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}."
        )
        raise RuntimeError(msg)
    return model


def _ensure_rollout_stride(batch: Batch, stride: int) -> Batch:
    current_steps = batch.input_fields.shape[1]
    if current_steps >= stride:
        return batch
    deficit = stride - current_steps
    if batch.output_fields.shape[1] < deficit:
        msg = (
            "Rollout requested stride longer than available ground-truth window: "
            f"needed {stride}, have {current_steps} inputs and "
            f"{batch.output_fields.shape[1]} outputs."
        )
        raise ValueError(msg)
    padding = batch.output_fields[:, :deficit, ...]
    new_inputs = torch.cat([batch.input_fields, padding], dim=1)
    new_outputs = batch.output_fields[:, deficit:, ...]
    return Batch(
        input_fields=new_inputs,
        output_fields=new_outputs,
        constant_scalars=batch.constant_scalars,
        constant_fields=batch.constant_fields,
    )


def _render_rollouts(
    model: EncoderProcessorDecoder,
    dataloader,
    batch_indices: Sequence[int],
    video_dir: Path,
    sample_index: int,
    fmt: str,
    fps: int,
    device: torch.device,
    free_running_only: bool,
) -> list[Path]:
    if not batch_indices:
        return []
    targets = set(batch_indices)
    saved_paths: list[Path] = []
    rendered_batches: set[int] = set()
    video_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx not in targets:
                continue
            try:
                stride_ready_batch = _ensure_rollout_stride(batch, model.stride)
            except ValueError as exc:
                log.warning("Skipping batch %s: %s", batch_idx, exc)
                continue
            batch_on_device = _batch_to_device(stride_ready_batch, device)
            preds, trues = model.rollout(
                batch_on_device,
                free_running_only=free_running_only,
            )
            if trues is None:
                log.warning(
                    "Rollout for batch %s did not return ground truth; skipping video.",
                    batch_idx,
                )
                continue
            if sample_index >= preds.shape[0]:
                log.warning(
                    "Requested sample %s but batch %s has only %s samples.",
                    sample_index,
                    batch_idx,
                    preds.shape[0],
                )
                continue
            filename = video_dir / f"batch_{batch_idx}_sample_{sample_index}.{fmt}"
            plot_spatiotemporal_video(
                true=trues.cpu(),
                pred=preds.cpu(),
                batch_idx=sample_index,
                fps=fps,
                save_path=str(filename),
            )
            saved_paths.append(filename)
            rendered_batches.add(batch_idx)
            log.info("Saved rollout visualization to %s", filename)
    missing = targets - rendered_batches
    for batch_idx in sorted(missing):
        log.warning("Requested batch %s was not found in the dataloader.", batch_idx)
    return saved_paths


def main() -> None:
    """Entry point for CLI-based evaluation."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _resolve_csv_path(args)
    video_dir = _resolve_video_dir(args)
    cfg = compose_training_config(args)
    training_params = resolve_training_params(cfg, args)
    update_data_cfg(cfg, training_params.n_steps_input, training_params.n_steps_output)

    L.seed_everything(cfg.get("seed", 42), workers=True)

    (
        datamodule,
        channel_count,
        inferred_n_steps_input,
        inferred_n_steps_output,
        _input_shape,
        output_shape,
    ) = prepare_datamodule(cfg)

    configure_module_dimensions(
        cfg,
        channel_count=channel_count,
        n_steps_input=inferred_n_steps_input,
        n_steps_output=inferred_n_steps_output,
    )
    normalize_processor_cfg(cfg)

    n_spatial_dims = _infer_spatial_dims(args, output_shape)
    metrics = _build_metrics(args.metrics or ("mse", "rmse"))

    model = _load_model(cfg, args.checkpoint)
    device = _resolve_device(args.device)
    model.to(device)

    test_loader = datamodule.test_dataloader()
    rows = _evaluate_metrics(model, test_loader, metrics, device, n_spatial_dims)
    _write_csv(rows, csv_path, list(metrics.keys()))

    if args.batch_indices:
        rollout_loader = datamodule.rollout_test_dataloader()
        _render_rollouts(
            model,
            rollout_loader,
            args.batch_indices,
            video_dir,
            args.video_sample_index,
            args.video_format,
            args.fps,
            device,
            args.free_running_only,
        )


if __name__ == "__main__":
    main()
