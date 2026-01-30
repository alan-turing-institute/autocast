"""Evaluation CLI for encoder-processor-decoder checkpoints."""

import argparse
import csv
import logging
from argparse import BooleanOptionalAction
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from autocast.logging import create_wandb_logger, log_metrics
from autocast.metrics import (
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
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.scripts.cli import add_common_config_args, add_work_dir_arg
from autocast.scripts.config import load_config
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.types import Batch
from autocast.utils import plot_spatiotemporal_video

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
    add_common_config_args(parser, "encoder_processor_decoder")
    # Required for evaluation
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the encoder-processor-decoder checkpoint to evaluate.",
    )
    add_work_dir_arg(parser)

    # Optional overrides typically handled by Hydra, but kept for convenience
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override training stride used for rollouts.",
    )

    # Evaluation specific
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


def _resolve_csv_path(args: argparse.Namespace, work_dir: Path) -> Path:
    if args.csv_path is not None:
        return args.csv_path.expanduser().resolve()
    return (work_dir / "evaluation_metrics.csv").resolve()


def _resolve_video_dir(args: argparse.Namespace, work_dir: Path) -> Path:
    if args.video_dir is not None:
        return args.video_dir.expanduser().resolve()
    return (work_dir / "videos").resolve()


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
    names = metric_names or ("mse", "rmse", "vrmse")
    metrics = {}
    for name in names:
        metric_cls = AVAILABLE_METRICS[name]
        metrics[name] = metric_cls()
    return metrics


def _evaluate_metrics(
    model: EncoderProcessorDecoder,
    dataloader,
    metrics: dict[str, nn.Module],
    device: torch.device,
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
                value = metric(preds, trues)
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


def _render_rollouts(
    model: EncoderProcessorDecoder,
    dataloader,
    batch_indices: Sequence[int],
    video_dir: Path,
    sample_index: int,
    fmt: str,
    fps: int,
    device: torch.device,
    stride: int,
    max_rollout_steps: int,
    free_running_only: bool,
    n_members: int | None = None,
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
            batch_on_device = _batch_to_device(batch, device)
            preds, trues = model.rollout(
                batch_on_device,
                stride=stride,
                max_rollout_steps=max_rollout_steps,
                free_running_only=free_running_only,
                n_members=n_members if n_members and n_members > 1 else None,
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
            # Limit the rollout to the available ground truth rollout length
            if trues.shape[1] < preds.shape[1]:
                preds = preds[:, : trues.shape[1]]
            # Reduce ensemble dimension for plotting if present.
            # When n_members > 1, the rollout output has shape (B, T, ..., C, M).
            if n_members is not None and n_members > 1:
                preds_mean = preds.mean(dim=-1)
                trues_mean = trues.mean(dim=-1)
                preds_uq = preds.std(dim=-1)
            else:
                preds_mean = preds
                trues_mean = trues
                preds_uq = None
            plot_spatiotemporal_video(
                true=trues_mean.cpu(),
                pred=preds_mean.cpu(),
                pred_uq=preds_uq.cpu() if preds_uq is not None else None,
                batch_idx=sample_index,
                fps=fps,
                save_path=str(filename),
                colorbar_mode="column",
                pred_uq_label="Ensemble Std Dev",
            )
            saved_paths.append(filename)
            rendered_batches.add(batch_idx)
            log.info("Saved rollout visualization to %s", filename)
    missing = targets - rendered_batches
    for batch_idx in sorted(missing):
        log.warning("Requested batch %s was not found in the dataloader.", batch_idx)
    return saved_paths


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
    else:
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


def main() -> None:
    """Entry point for CLI-based evaluation."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _resolve_csv_path(args, work_dir)
    video_dir = _resolve_video_dir(args, work_dir)

    cfg = load_config(args)

    # Setup datamodule and resolve config
    datamodule, cfg, stats = setup_datamodule(cfg)

    # Setup Model
    model = setup_epd_model(cfg, stats)

    # Load checkpoint
    log.info("Loading checkpoint from %s", args.checkpoint)
    state_dict = _load_state_dict(args.checkpoint)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        msg = (
            "Checkpoint parameters do not match the instantiated model. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}."
        )
        raise RuntimeError(msg)

    # Setup WandB (if enabled by config)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logging_cfg = cfg.get("logging")
    logging_cfg = (
        OmegaConf.to_container(logging_cfg, resolve=True)
        if logging_cfg is not None
        else {}
    )
    wandb_logger, _ = create_wandb_logger(
        logging_cfg,  # type: ignore  # noqa: PGH003
        experiment_name=cfg.get("experiment_name"),
        job_type="evaluate-encoder-processor-decoder",
        work_dir=work_dir,
        config={
            "hydra": resolved_cfg,
            "evaluation": {
                "checkpoint": str(args.checkpoint),
                "metrics": args.metrics or ("mse", "rmse"),
                "batch_indices": args.batch_indices,
            },
        },
    )

    metrics = _build_metrics(args.metrics or ("mse", "rmse"))

    model_cfg = cfg.get("model", {})
    n_members = model_cfg.get("n_members", 1)

    device = _resolve_device(args.device)
    model.to(device)

    # Evaluation
    test_loader = datamodule.test_dataloader()
    rows = _evaluate_metrics(model, test_loader, metrics, device)
    _write_csv(rows, csv_path, list(metrics.keys()))

    aggregate_row = next((row for row in rows if row.get("batch_index") == "all"), None)
    if aggregate_row is not None:
        payload = {
            f"test/{name}": float(aggregate_row[name])  # type: ignore[arg-type]
            for name in metrics
            if name in aggregate_row
        }
        log_metrics(wandb_logger, payload)

    # Rollouts
    if args.batch_indices:
        rollout_loader = datamodule.rollout_test_dataloader(batch_size=1)
        # Check explicit eval config or assume defaults
        eval_cfg = cfg.get("eval", {})
        max_rollout_steps = eval_cfg.get("max_rollout_steps", 10)

        # Use stride from args, or config, or fallback to n_steps_output (from stats)
        data_config = cfg.get("datamodule", {})
        rollout_stride = (
            args.stride
            if args.stride is not None
            else (data_config.get("stride") or stats["n_steps_output"])
        )

        _render_rollouts(
            model,
            rollout_loader,
            args.batch_indices,
            video_dir,
            args.video_sample_index,
            args.video_format,
            args.fps,
            device,
            stride=rollout_stride,
            max_rollout_steps=max_rollout_steps,
            free_running_only=args.free_running_only,
            n_members=n_members,
        )


if __name__ == "__main__":
    main()
