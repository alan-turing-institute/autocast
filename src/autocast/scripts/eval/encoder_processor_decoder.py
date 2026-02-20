"""Evaluation CLI for encoder-processor-decoder checkpoints."""

import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
from torchmetrics import Metric

from autocast.metrics import MAE, MSE, NMAE, NMSE, NRMSE, RMSE, VMSE, VRMSE, LInfinity
from autocast.metrics.base import BaseMetric
from autocast.metrics.coverage import MultiCoverage
from autocast.metrics.ensemble import CRPS, AlphaFairCRPS, FairCRPS
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.scripts.config import save_resolved_config
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.utils import get_default_config_path
from autocast.utils import plot_spatiotemporal_video
from autocast.utils.plots import compute_metrics_from_dataloader

# Set matmul precision for A100/H100
torch.set_float32_matmul_precision("high")

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

AVAILABLE_METRICS_ENSEMBLE = {
    "crps": CRPS,
    "fcrps": FairCRPS,
    "afcrps": AlphaFairCRPS,
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


def _limit_batches(dataloader, max_batches: int | None):
    if max_batches is None or max_batches <= 0:
        return dataloader

    def _generator():
        for index, batch in enumerate(dataloader):
            if index >= max_batches:
                break
            yield batch

    return _generator()


def _build_metrics(metric_names: Sequence[str]) -> dict[str, BaseMetric]:
    names = metric_names or ("mse", "rmse", "vrmse")
    metrics = {}
    for name in names:
        metric_cls = AVAILABLE_METRICS[name]
        metrics[name] = metric_cls()
    return metrics


def _process_metrics_results(
    results: dict[None | tuple[int, int], dict[str, Metric]],
    per_batch_rows: list[dict[str, float | str]] | None = None,
    log_prefix: str = "Test",
    plot_dir: Path | None = None,
) -> list[dict[str, float | str]]:
    """Process metric results into CSV rows and plots."""
    rows = []
    plot_dir = plot_dir or Path.cwd()

    for window, window_metrics in results.items():
        window_str = f"{window[0]}-{window[1]}" if window is not None else "all"
        row: dict[str, float | str] = {"window": window_str, "batch_idx": "all"}

        for name, metric in window_metrics.items():
            log.info(
                "%s metric '%s' for window %s: %s",
                log_prefix,
                name,
                window,
                metric.compute(),
            )

            # If this is coverage, also plot it
            if name == "coverage" and isinstance(metric, MultiCoverage):
                metric.plot(
                    save_path=plot_dir
                    / f"{log_prefix.lower()}_coverage_window_{window_str}.png",
                    title=f"{log_prefix} Coverage Window {window}",
                )

            # Try to get a scalar value for csv
            try:
                val = metric.compute()
                if val.numel() == 1:
                    row[name] = float(val.item())
                elif hasattr(val, "mean"):
                    row[name] = float(val.mean().item())
            except Exception as e:
                msg = f"Could not extract scalar for metric {name}: {e}"
                log.warning(msg)

        rows.append(row)

    if per_batch_rows:
        rows.extend(per_batch_rows)

    return rows


def _map_windows(
    windows: Sequence[Sequence[int] | None] | None,
) -> list[tuple[int, int] | None] | None:
    if windows is None:
        return None

    # Convert to tuple pairs
    tuple_windows: list[tuple[int, int] | None] = []
    for w in list(windows):
        if w is not None and len(w) != 2:
            msg = f"Coverage window must be (start, end) indices or None. Got {w}"
            raise ValueError(msg)
        tuple_windows.append((w[0], w[1]) if w is not None else None)
    return tuple_windows


def _evaluate_rollout_metrics(
    model: EncoderProcessorDecoderEnsemble | EncoderProcessorDecoder,
    dataloader,
    stride: int,
    max_rollout_steps: int,
    free_running_only: bool,
    n_members: int | None,
    metric_fns: dict[str, Callable[[], Metric]],
    windows: list[tuple[int, int] | None] | None = None,
) -> tuple[
    dict[None | tuple[int, int], dict[str, Metric]],
    list[dict[str, float | str]] | None,
]:
    """Evaluate rollout metrics using the dataloader helper."""

    def rollout_predict(batch):
        """Predict function for rollout evaluation."""
        preds, trues = model.rollout(
            batch,
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
    metrics_per_window, _, per_batch_rows = compute_metrics_from_dataloader(
        dataloader=dataloader,
        metric_fns=metric_fns,
        predict_fn=rollout_predict,
        windows=windows,
        return_per_batch=True,
    )

    return metrics_per_window, per_batch_rows


def _render_rollouts(
    model: EncoderProcessorDecoder | EncoderProcessorDecoderEnsemble,
    dataloader,
    batch_indices: Sequence[int],
    video_dir: Path,
    sample_index: int,
    fmt: str,
    fps: int,
    stride: int,
    max_rollout_steps: int,
    free_running_only: bool,
    n_members: int | None = None,
) -> list[Path]:
    # Return early if no batches are requested
    if not batch_indices:
        return []

    # Create sets to enable logging warnings for any missing batches
    targets = set(batch_indices)
    saved_paths: list[Path] = []
    rendered_batches: set[int] = set()
    video_dir.mkdir(parents=True, exist_ok=True)

    # Perform rollouts and save videos for requested batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx not in targets:
                continue
            preds, trues = model.rollout(
                batch,
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
                preds_uq = preds.std(dim=-1)
                trues_mean = trues
            else:
                preds_mean = preds
                trues_mean = trues
                preds_uq = None

            # Plot video
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

    # Check for any missing batches that were requested but not rendered
    missing = targets - rendered_batches
    for batch_idx in sorted(missing):
        log.warning("Requested batch %s was not found in the dataloader.", batch_idx)

    return saved_paths


def _write_csv(rows: list[dict[str, float | str]], csv_path: Path):
    if not rows:
        log.warning("No evaluation rows to write; skipping CSV generation.")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _load_checkpoint_payload(checkpoint_path: Path) -> Mapping[str, Any]:
    checkpoint_real = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_real,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        msg = f"Checkpoint {checkpoint_real} does not contain a valid payload."
        raise TypeError(msg)
    return checkpoint


def _extract_state_dict(
    checkpoint: Mapping[str, Any],
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        msg = "Checkpoint payload does not contain a valid state_dict."
        raise TypeError(msg)
    if isinstance(state_dict, OrderedDict):
        state_dict = state_dict.copy()
    else:
        state_dict = OrderedDict(state_dict)
    state_dict.pop("_metadata", None)
    return state_dict


def _make_metadata_row(
    *,
    category: str,
    metric: str,
    value: float | int,
    loader: str | None = None,
    batch_idx: int | str = "all",
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "window": "meta",
        "batch_idx": batch_idx,
        "category": category,
        "metric": metric,
        "value": float(value),
    }
    if loader is not None:
        row["loader"] = loader
    return row


def _parameter_count_rows(
    model: EncoderProcessorDecoderEnsemble | EncoderProcessorDecoder,
) -> list[dict[str, float | str]]:
    def _count(module: torch.nn.Module | None, *, trainable: bool = False) -> int:
        if module is None:
            return 0
        params = module.parameters()
        if trainable:
            params = (param for param in params if param.requires_grad)
        return sum(param.numel() for param in params)

    encoder_module = getattr(getattr(model, "encoder_decoder", None), "encoder", None)
    decoder_module = getattr(getattr(model, "encoder_decoder", None), "decoder", None)
    processor_module = getattr(model, "processor", None)

    return [
        _make_metadata_row(
            category="params",
            metric="encoder_total",
            value=_count(encoder_module),
        ),
        _make_metadata_row(
            category="params",
            metric="decoder_total",
            value=_count(decoder_module),
        ),
        _make_metadata_row(
            category="params",
            metric="processor_total",
            value=_count(processor_module),
        ),
        _make_metadata_row(
            category="params",
            metric="model_total",
            value=_count(model),
        ),
        _make_metadata_row(
            category="params",
            metric="model_trainable",
            value=_count(model, trainable=True),
        ),
    ]


def _extract_training_runtime_total_s(
    checkpoint_payload: Mapping[str, Any],
) -> float | None:
    for key in ("training_runtime_total_s", "train_runtime_total_s", "runtime_total_s"):
        value = checkpoint_payload.get(key)
        if isinstance(value, int | float):
            return float(value)

    callbacks = checkpoint_payload.get("callbacks")
    if isinstance(callbacks, Mapping):
        for callback_state in callbacks.values():
            if not isinstance(callback_state, Mapping):
                continue
            for key in ("time_elapsed", "time_elapsed_s", "total_time", "total_time_s"):
                value = callback_state.get(key)
                if isinstance(value, int | float):
                    return float(value)

    return None


def _training_runtime_rows(
    checkpoint_payload: Mapping[str, Any],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    epoch_value = checkpoint_payload.get("epoch")
    global_step_value = checkpoint_payload.get("global_step")

    epochs_completed = (
        int(epoch_value) + 1
        if isinstance(epoch_value, int) and epoch_value >= 0
        else None
    )
    global_steps = (
        int(global_step_value) if isinstance(global_step_value, int) else None
    )

    if epochs_completed is not None:
        rows.append(
            _make_metadata_row(
                category="runtime_train",
                metric="epochs_completed",
                value=epochs_completed,
            )
        )
    if global_steps is not None:
        rows.append(
            _make_metadata_row(
                category="runtime_train",
                metric="steps_completed",
                value=global_steps,
            )
        )

    total_runtime_s = _extract_training_runtime_total_s(checkpoint_payload)
    if total_runtime_s is None or total_runtime_s <= 0:
        return rows

    rows.append(
        _make_metadata_row(
            category="runtime_train",
            metric="total_s",
            value=total_runtime_s,
        )
    )

    if epochs_completed is not None and epochs_completed > 0:
        rows.append(
            _make_metadata_row(
                category="runtime_train",
                metric="per_epoch_s",
                value=total_runtime_s / epochs_completed,
            )
        )

    if global_steps is not None and global_steps > 0:
        rows.append(
            _make_metadata_row(
                category="runtime_train",
                metric="per_step_s",
                value=total_runtime_s / global_steps,
            )
        )

    return rows


def _eval_runtime_rows(
    loader_name: str,
    total_runtime_s: float,
    per_batch_runtimes_s: Sequence[float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    rows.append(
        _make_metadata_row(
            category="runtime_eval",
            metric="total_s",
            value=total_runtime_s,
            loader=loader_name,
        )
    )

    batch_count = len(per_batch_runtimes_s)
    rows.append(
        _make_metadata_row(
            category="runtime_eval",
            metric="batch_count",
            value=batch_count,
            loader=loader_name,
        )
    )

    if batch_count > 0:
        rows.append(
            _make_metadata_row(
                category="runtime_eval",
                metric="per_batch_mean_s",
                value=total_runtime_s / batch_count,
                loader=loader_name,
            )
        )
        for batch_idx, batch_runtime_s in enumerate(per_batch_runtimes_s):
            rows.append(
                _make_metadata_row(
                    category="runtime_eval",
                    metric="per_batch_s",
                    value=batch_runtime_s,
                    loader=loader_name,
                    batch_idx=batch_idx,
                )
            )

    return rows


def _with_batch_timing(
    predict_fn: Callable,
    per_batch_runtimes_s: list[float],
) -> Callable:
    def wrapped(batch):
        start_s = perf_counter()
        result = predict_fn(batch)
        per_batch_runtimes_s.append(perf_counter() - start_s)
        return result

    return wrapped


def _evaluation_metadata_rows(
    checkpoint_payload: Mapping[str, Any],
    model: EncoderProcessorDecoderEnsemble | EncoderProcessorDecoder,
    test_eval_total_s: float,
    test_batch_runtimes_s: Sequence[float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    rows.extend(_training_runtime_rows(checkpoint_payload))
    rows.extend(
        _eval_runtime_rows(
            "test_dataloader",
            test_eval_total_s,
            test_batch_runtimes_s,
        )
    )
    rows.extend(_parameter_count_rows(model))
    return rows


def _rollout_metadata_rows(
    rollout_eval_total_s: float,
    rollout_batch_runtimes_s: Sequence[float],
) -> list[dict[str, float | str]]:
    return _eval_runtime_rows(
        "rollout_dataloader",
        rollout_eval_total_s,
        rollout_batch_runtimes_s,
    )


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:
    """Entry point for CLI-based evaluation."""
    run_evaluation(cfg)


def run_evaluation(cfg: DictConfig, work_dir: Path | None = None) -> None:  # noqa: PLR0912, PLR0915
    """Run evaluation using an already-composed config."""
    logging.basicConfig(level=logging.INFO)

    umask_value = cfg.get("umask")
    if umask_value is not None:
        os.umask(int(str(umask_value), 8))
        log.info("Applied process umask %s", umask_value)

    work_dir = work_dir or Path.cwd()

    # Get eval config
    eval_cfg = cfg.get("eval", {})
    eval_batch_size: int = eval_cfg.get("batch_size", 1)
    max_test_batches = eval_cfg.get("max_test_batches")
    max_rollout_batches = eval_cfg.get("max_rollout_batches", max_test_batches)

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

    # Override model n_members from eval config if specified
    if "n_members" in eval_cfg:
        with open_dict(cfg.model):
            cfg.model.n_members = eval_cfg.n_members
        log.info(
            "Overriding model.n_members with %s from eval config", eval_cfg.n_members
        )

    # Setup Model
    model = setup_epd_model(cfg, stats, datamodule=datamodule)

    # Load checkpoint
    log.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
    state_dict = _extract_state_dict(checkpoint_payload)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        msg = (
            "Checkpoint parameters do not match the instantiated model. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}."
        )
        raise RuntimeError(msg)

    # Get eval parameters from config
    metrics_list = eval_cfg.get("metrics", ["mse", "rmse"])
    batch_indices = eval_cfg.get("batch_indices", [])

    # Construct metrics (deprecated usage in _evaluate_metrics)
    # metrics = _build_metrics(metrics_list)

    # Get number of ensemble members from config if available
    n_members = cfg.get("model", {}).get("n_members", 1)

    # Setup Fabric for device management
    accelerator = eval_cfg.get("device", "auto")
    fabric = L.Fabric(accelerator=accelerator, devices=1)
    fabric.launch()

    # Setup model and loader with Fabric
    log.info("Model configuration n_members: %s", n_members)
    log.info("Model class: %s", type(model))

    model.to(fabric.device)
    model.eval()
    test_loader = _limit_batches(
        fabric.setup_dataloaders(datamodule.test_dataloader()),
        max_test_batches,
    )

    # Evaluation

    # Prepare metric functions for test pass
    test_metric_fns: dict[str, Callable[[], Metric]] = {}

    # Add standard metrics from config
    for name in metrics_list:
        if name in AVAILABLE_METRICS:
            test_metric_fns[name] = AVAILABLE_METRICS[name]
        else:
            msg = f"Metric {name} not found in AVAILABLE_METRICS"
            log.warning(msg)

    # Add coverage if we have an ensemble
    compute_coverage = eval_cfg.get("compute_coverage", False)
    if (n_members > 1) or compute_coverage:

        def coverage_factory() -> Metric:
            return MultiCoverage(coverage_levels=eval_cfg.get("coverage_levels", None))

        test_metric_fns["coverage"] = coverage_factory

    log.info("Computing test metrics: %s", list(test_metric_fns.keys()))

    # Use metric_windows from config (apply to all metrics)
    test_windows = _map_windows(eval_cfg.get("metric_windows", None))

    test_batch_runtimes_s: list[float] = []
    timed_test_predict = _with_batch_timing(model, test_batch_runtimes_s)

    test_eval_start_s = perf_counter()

    test_metrics_results, _, test_per_batch_rows = compute_metrics_from_dataloader(
        dataloader=test_loader,
        metric_fns=test_metric_fns,
        predict_fn=timed_test_predict,
        windows=test_windows,
        return_per_batch=True,
    )
    test_eval_total_s = perf_counter() - test_eval_start_s

    # Process and save test metrics
    test_rows = _process_metrics_results(
        test_metrics_results,
        per_batch_rows=test_per_batch_rows,
        log_prefix="Test",
        plot_dir=work_dir,
    )

    evaluation_rows: list[dict[str, float | str]] = []
    evaluation_rows.extend(test_rows)
    evaluation_rows.extend(
        _evaluation_metadata_rows(
            checkpoint_payload=checkpoint_payload,
            model=model,
            test_eval_total_s=test_eval_total_s,
            test_batch_runtimes_s=test_batch_runtimes_s,
        )
    )

    # Rollouts
    compute_rollout_coverage = eval_cfg.get("compute_rollout_coverage", False)
    compute_rollout_metrics = eval_cfg.get("compute_rollout_metrics", False)

    if batch_indices or compute_rollout_coverage or compute_rollout_metrics:
        max_rollout_steps = eval_cfg.get("max_rollout_steps", 10)

        # Use rollout_stride config or fallback to n_steps_output (from stats)
        data_config = cfg.get("datamodule", {})
        rollout_stride = data_config.get("rollout_stride") or stats["n_steps_output"]

        if batch_indices:
            rollout_loader = _limit_batches(
                fabric.setup_dataloaders(
                    datamodule.rollout_test_dataloader(batch_size=eval_batch_size)
                ),
                max_rollout_batches,
            )
            _render_rollouts(
                model,
                rollout_loader,
                batch_indices,
                video_dir,
                eval_cfg.get("video_sample_index", 0),
                eval_cfg.get("video_format", "mp4"),
                eval_cfg.get("fps", 5),
                stride=rollout_stride,
                max_rollout_steps=max_rollout_steps,
                free_running_only=eval_cfg.get("free_running_only", True),
                n_members=n_members,
            )

        # Prepare metric functions for rollouts
        rollout_metric_fns: dict[str, Callable[[], Metric]] = {}

        if compute_rollout_metrics:
            for name in metrics_list:
                if name in AVAILABLE_METRICS:
                    rollout_metric_fns[name] = AVAILABLE_METRICS[name]
                else:
                    msg = f"Metric {name} not found in AVAILABLE_METRICS"
                    log.warning(msg)

        if compute_rollout_coverage and n_members and n_members > 1:
            log.info("Adding rollout coverage to metrics...")
            assert isinstance(model, EncoderProcessorDecoderEnsemble)

            def coverage_factory() -> Metric:
                return MultiCoverage(
                    coverage_levels=eval_cfg.get("coverage_levels", None)
                )

            rollout_metric_fns["coverage"] = coverage_factory

        if rollout_metric_fns:
            log.info("Computing rollout metrics: %s", list(rollout_metric_fns.keys()))
            windows = _map_windows(
                eval_cfg.get("metric_windows_rollout", [(0, 1), (6, 12), (13, 30)])
            )

            rollout_batch_runtimes_s: list[float] = []

            def rollout_predict(batch):
                preds, trues = model.rollout(
                    batch,
                    stride=rollout_stride,
                    max_rollout_steps=max_rollout_steps,
                    free_running_only=eval_cfg.get("free_running_only", True),
                    n_members=n_members if n_members and n_members > 1 else None,
                )
                if trues is None:
                    return None, None

                min_len = min(preds.shape[1], trues.shape[1])
                return preds[:, :min_len], trues[:, :min_len]

            timed_rollout_predict = _with_batch_timing(
                rollout_predict,
                rollout_batch_runtimes_s,
            )

            rollout_eval_start_s = perf_counter()

            rollout_metrics_loader = _limit_batches(
                fabric.setup_dataloaders(
                    datamodule.rollout_test_dataloader(batch_size=eval_batch_size)
                ),
                max_rollout_batches,
            )

            rollout_metrics_per_window, _, rollout_per_batch_rows = (
                compute_metrics_from_dataloader(
                    dataloader=rollout_metrics_loader,
                    metric_fns=rollout_metric_fns,
                    predict_fn=timed_rollout_predict,
                    windows=windows,
                    return_per_batch=True,
                )
            )
            rollout_eval_total_s = perf_counter() - rollout_eval_start_s

            rollout_runtime_rows = _rollout_metadata_rows(
                rollout_eval_total_s,
                rollout_batch_runtimes_s,
            )
            evaluation_rows.extend(rollout_runtime_rows)

            # Process and log results
            rollout_csv_rows = _process_metrics_results(
                rollout_metrics_per_window,
                per_batch_rows=rollout_per_batch_rows,
                log_prefix="Rollout",
                plot_dir=csv_path.parent,
            )

            # Save rollout metrics to CSV
            rollout_csv_path = csv_path.parent / "rollout_metrics.csv"
            rollout_csv_rows.extend(rollout_runtime_rows)
            if rollout_csv_rows:
                _write_csv(rollout_csv_rows, rollout_csv_path)
                log.info("Wrote rollout metrics to %s", rollout_csv_path)

    _write_csv(evaluation_rows, csv_path)
    log.info("Wrote metrics CSV to %s", csv_path)


if __name__ == "__main__":
    main()
