"""Evaluation CLI for encoder-processor-decoder checkpoints."""

import csv
import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
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
from autocast.metrics.coverage import MultiCoverage
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.scripts.config import save_resolved_config
from autocast.scripts.data import batch_to_device
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.utils import get_default_config_path
from autocast.utils import plot_spatiotemporal_video
from autocast.utils.plots import compute_coverage_scores_from_dataloader

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


def _resolve_csv_path(eval_cfg: DictConfig, work_dir: Path) -> Path:
    csv_path = eval_cfg.get("csv_path")
    if csv_path is not None:
        return Path(csv_path).expanduser().resolve()
    return (work_dir / "evaluation_metrics.csv").resolve()


def _resolve_video_dir(eval_cfg: DictConfig, work_dir: Path) -> Path:
    video_dir = eval_cfg.get("video_dir")
    if video_dir is not None:
        return Path(video_dir).expanduser().resolve()
    return (work_dir / "videos").resolve()


def _resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_available = getattr(torch.backends, "mps", None)
    if mps_available is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


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
            batch_on_device = batch_to_device(batch, device)
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


def _evaluate_rollout_coverage(
    model: EncoderProcessorDecoder,
    dataloader,
    device: torch.device,
    stride: int,
    max_rollout_steps: int,
    free_running_only: bool,
    n_members: int | None,
    windows: list[tuple[int, int] | None] | None = None,
    coverage_levels: list[float] | None = None,
) -> dict[None | tuple[int, int], MultiCoverage]:
    """Evaluate rollout coverage using the dataloader helper."""

    def rollout_predict(batch):
        """Predict function for rollout evaluation."""
        batch_on_device = batch_to_device(batch, device)
        preds, trues = model.rollout(
            batch_on_device,
            stride=stride,
            max_rollout_steps=max_rollout_steps,
            free_running_only=free_running_only,
            n_members=n_members if n_members and n_members > 1 else None,
        )
        if trues is None:
            return None, None

        # Match dimensions
        min_len = min(preds.shape[1], trues.shape[1])
        return preds[:, :min_len], trues[:, :min_len]

    # Use the helper function with predict_fn
    metrics_per_window, _ = compute_coverage_scores_from_dataloader(
        dataloader=dataloader,
        predict_fn=rollout_predict,
        coverage_levels=coverage_levels,
        windows=windows,
        return_tensors=False,
    )

    return metrics_per_window


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
            batch_on_device = batch_to_device(batch, device)
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


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    """Entry point for CLI-based evaluation."""
    logging.basicConfig(level=logging.INFO)

    # Work directory is managed by Hydra
    work_dir = Path.cwd()

    # Get eval config
    eval_cfg = cfg.get("eval", {})
    eval_batch_size: int = eval_cfg.get("batch_size", 1)

    # Validate that checkpoint is provided
    checkpoint_path = eval_cfg.get("checkpoint")
    if checkpoint_path is None:
        msg = (
            "No checkpoint specified. Please provide a checkpoint path via:\n"
            "  eval.checkpoint=/path/to/checkpoint.ckpt\n"
            "Or add it to your config file."
        )
        raise ValueError(msg)
    checkpoint_path = Path(checkpoint_path)

    if cfg.get("output", {}).get("save_config"):
        save_resolved_config(cfg, work_dir, filename="resolved_eval_config.yaml")

    csv_path = _resolve_csv_path(eval_cfg, work_dir)
    video_dir = _resolve_video_dir(eval_cfg, work_dir)

    # Setup datamodule and resolve config
    datamodule, cfg, stats = setup_datamodule(cfg)

    # Setup Model
    model = setup_epd_model(cfg, stats)

    # Load checkpoint
    log.info("Loading checkpoint from %s", checkpoint_path)
    state_dict = _load_state_dict(checkpoint_path)
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

    # Get eval parameters from config
    metrics_list = eval_cfg.get("metrics", ["mse", "rmse"])
    batch_indices = eval_cfg.get("batch_indices", [])

    wandb_logger, _ = create_wandb_logger(
        logging_cfg,  # type: ignore  # noqa: PGH003
        experiment_name=cfg.get("experiment_name"),
        job_type="evaluate-encoder-processor-decoder",
        work_dir=work_dir,
        config={
            "hydra": resolved_cfg,
            "evaluation": {
                "checkpoint": str(checkpoint_path),
                "metrics": metrics_list,
                "batch_indices": batch_indices,
            },
        },
    )

    metrics = _build_metrics(metrics_list)

    model_cfg = cfg.get("model", {})
    n_members = model_cfg.get("n_members", 1)

    device = _resolve_device(eval_cfg.get("device", "auto"))
    model.to(device)

    # Evaluation
    test_loader = datamodule.test_dataloader()

    # Compute coverage using helper function
    test_coverage, _ = compute_coverage_scores_from_dataloader(
        dataloader=test_loader,
        model=model,
        coverage_levels=None,
        return_tensors=False,
    )
    for window, coverage_metric in test_coverage.items():
        log.info("Test coverage for window %s: %s", window, coverage_metric)
        window_str = f"{window[0]}-{window[1]}" if window is not None else "all"
        coverage_metric.plot(
            save_path=csv_path / f"test_coverage_window_{window}.png",
            title=f"Test Coverage Window {window}",
        )

    # Compute other metrics
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
    compute_rollout_coverage = eval_cfg.get("compute_rollout_coverage", False)
    if batch_indices or compute_rollout_coverage:
        max_rollout_steps = eval_cfg.get("max_rollout_steps", 10)

        # Use rollout_stride config or fallback to n_steps_output (from stats)
        data_config = cfg.get("datamodule", {})
        rollout_stride = data_config.get("rollout_stride") or stats["n_steps_output"]

        if batch_indices:
            _render_rollouts(
                model,
                datamodule.rollout_test_dataloader(batch_size=eval_batch_size),
                batch_indices,
                video_dir,
                eval_cfg.get("video_sample_index", 0),
                eval_cfg.get("video_format", "mp4"),
                eval_cfg.get("fps", 5),
                device,
                stride=rollout_stride,
                max_rollout_steps=max_rollout_steps,
                free_running_only=eval_cfg.get("free_running_only", True),
                n_members=n_members,
            )

        if compute_rollout_coverage:
            log.info("Computing rollout coverage metrics...")
            windows = eval_cfg.get("coverage_windows", [(6, 12), (13, 30)])
            rollout_coverage_per_window = _evaluate_rollout_coverage(
                model,
                datamodule.rollout_test_dataloader(batch_size=eval_batch_size),
                device,
                stride=rollout_stride,
                max_rollout_steps=max_rollout_steps,
                free_running_only=eval_cfg.get("free_running_only", True),
                windows=windows,
                n_members=n_members,
            )
            for window, coverage_metric in rollout_coverage_per_window.items():
                log.info(
                    "Rollout coverage for window %s: %s",
                    window,
                    coverage_metric,
                )
                window_str = f"{window[0]}-{window[1]}" if window is not None else "all"
                coverage_metric.plot(
                    save_path=csv_path / f"rollout_coverage_window_{window_str}.png",
                    title=f"Rollout Coverage Window {window_str}",
                )


if __name__ == "__main__":
    main()
