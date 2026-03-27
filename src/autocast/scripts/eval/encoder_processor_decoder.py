"""Evaluation CLI for encoder-processor-decoder checkpoints."""

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
from torchmetrics import Metric

from autocast.benchmarking import benchmark_model, benchmark_rollout
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
    PowerSpectrumCCRMSE,
    PowerSpectrumCCRMSEHigh,
    PowerSpectrumCCRMSELow,
    PowerSpectrumCCRMSEMid,
    PowerSpectrumCCRMSETail,
    PowerSpectrumRMSE,
    PowerSpectrumRMSEHigh,
    PowerSpectrumRMSELow,
    PowerSpectrumRMSEMid,
    PowerSpectrumRMSETail,
)
from autocast.metrics.coverage import MultiCoverage
from autocast.metrics.ensemble import (
    CRPS,
    AlphaFairCRPS,
    EnergyScore,
    FairCRPS,
    SpreadSkillRatio,
    VariogramScore,
)
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.scripts.config import save_resolved_config
from autocast.scripts.execution import (
    benchmark_metric_rows,
    extract_state_dict,
    load_checkpoint_payload,
    resolve_benchmark_csv_path,
    resolve_checkpoint_path,
    resolve_hydra_work_dir,
)
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.utils import get_default_config_path
from autocast.types.batch import Batch
from autocast.utils import plot_spatiotemporal_video
from autocast.utils.plots import (
    compute_metrics_from_dataloader,
    compute_metrics_per_timestep_from_dataloader,
)

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
    "psrmse": PowerSpectrumRMSE,
    "psrmse_low": PowerSpectrumRMSELow,
    "psrmse_mid": PowerSpectrumRMSEMid,
    "psrmse_high": PowerSpectrumRMSEHigh,
    "psrmse_tail": PowerSpectrumRMSETail,
    "pscc": PowerSpectrumCCRMSE,
    "pscc_low": PowerSpectrumCCRMSELow,
    "pscc_mid": PowerSpectrumCCRMSEMid,
    "pscc_high": PowerSpectrumCCRMSEHigh,
    "pscc_tail": PowerSpectrumCCRMSETail,
}

AVAILABLE_METRICS_ENSEMBLE = {
    "crps": CRPS,
    "fcrps": FairCRPS,
    "afcrps": AlphaFairCRPS,
    "energy": EnergyScore,
    "variogram": VariogramScore,
    "ssr": SpreadSkillRatio,
}

DEFAULT_EVAL_METRICS = [
    "mse",
    "mae",
    "rmse",
    "vrmse",
    "psrmse",
    "psrmse_low",
    "psrmse_mid",
    "psrmse_high",
    "psrmse_tail",
    "pscc",
    "pscc_low",
    "pscc_mid",
    "pscc_high",
    "pscc_tail",
    "crps",
    "fcrps",
    "afcrps",
    "energy",
    "ssr",
]

MEMORY_INTENSIVE_METRICS = {"variogram"}


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


def _unwrap_module(module: Any, max_depth: int = 10) -> Any:
    """Return the underlying model when wrapped by Fabric/DDP-style wrappers."""
    unwrapped = module
    for _ in range(max_depth):
        next_module = getattr(unwrapped, "module", None)
        if next_module is None or next_module is unwrapped:
            break
        unwrapped = next_module
    return unwrapped


def _mark_forward_methods_if_available(model: Any, methods: Sequence[str]) -> None:
    """Mark custom methods as valid forwards when using Fabric wrappers."""
    mark_forward_method = getattr(model, "mark_forward_method", None)
    if not callable(mark_forward_method):
        return
    for method in methods:
        mark_forward_method(method)


def _limit_batches(dataloader, max_batches: int | None):
    if max_batches is None or max_batches <= 0:
        return dataloader

    def _generator():
        for index, batch in enumerate(dataloader):
            if index >= max_batches:
                break
            yield batch

    return _generator()


def _resolve_rollout_batch_limit(eval_cfg: DictConfig) -> int | None:
    max_test_batches = eval_cfg.get("max_test_batches")
    max_rollout_batches = eval_cfg.get("max_rollout_batches")
    if max_rollout_batches is None:
        return max_test_batches
    return max_rollout_batches


def _resolve_rollout_timestep_limit(
    *,
    max_rollout_steps: int | None,
    rollout_stride: int,
) -> int | None:
    """Resolve the cap for flattened rollout timesteps.

    Rollout metrics are computed on flattened predictions where each rollout window
    contributes ``rollout_stride`` timesteps. To cap by simulated lead time,
    convert max rollout windows to max flattened timesteps.
    """
    if max_rollout_steps is None:
        return None
    if max_rollout_steps <= 0:
        return None
    if rollout_stride <= 0:
        return None
    return int(max_rollout_steps) * int(rollout_stride)


def _resolve_rollout_channel_names(dataset: Any) -> list[str] | None:
    if dataset is None:
        return None

    norm = getattr(dataset, "norm", None)
    raw_names = getattr(norm, "core_field_names", None)

    if not isinstance(raw_names, Sequence) or isinstance(raw_names, str):
        normalization_stats = getattr(dataset, "normalization_stats", None)
        if isinstance(normalization_stats, Mapping):
            raw_names = normalization_stats.get("core_field_names")

    if not isinstance(raw_names, Sequence) or isinstance(raw_names, str):
        return None

    channel_names = [str(name) for name in raw_names]
    if not channel_names:
        return None

    output_channel_idxs = getattr(dataset, "output_channel_idxs", None)
    if output_channel_idxs is not None:
        try:
            channel_names = [channel_names[idx] for idx in output_channel_idxs]
        except (TypeError, IndexError):
            log.warning(
                "Could not apply output_channel_idxs=%s to channel names %s.",
                output_channel_idxs,
                channel_names,
            )
            return None

    return channel_names


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


def _collect_rollout_sample_targets_for_batch(
    *,
    targets: set[int],
    rendered_targets: set[int],
    batch_idx: int,
    batch_size: int,
    sample_index: int,
    global_sample_offset: int,
) -> dict[int, int]:
    sample_targets: dict[int, int] = {}

    # Legacy support: target refers to dataloader batch index.
    if batch_idx in targets and batch_idx not in rendered_targets:
        if sample_index < batch_size:
            sample_targets[batch_idx] = int(sample_index)
        else:
            log.warning(
                "Requested sample %s for dataloader batch %s, "
                "but batch has only %s samples.",
                sample_index,
                batch_idx,
                batch_size,
            )

    # Preferred behavior: target refers to global sample index.
    for local_idx in range(batch_size):
        global_idx = global_sample_offset + local_idx
        if global_idx in targets and global_idx not in rendered_targets:
            sample_targets[global_idx] = local_idx

    return sample_targets


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
    channel_names: list[str] | None = None,
    preserve_aspect: bool = False,
) -> list[Path]:
    # Return early if no rollout indices are requested
    if not batch_indices:
        return []

    # Targets are interpreted as rollout sample indices across the full dataloader
    # stream. We also preserve legacy behavior where an index can refer to a
    # dataloader batch index together with `sample_index`.
    targets = {int(idx) for idx in batch_indices}
    saved_paths: list[Path] = []
    rendered_targets: set[int] = set()
    global_sample_offset = 0
    video_dir.mkdir(parents=True, exist_ok=True)

    # Perform rollouts and save videos for requested target indices.
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if rendered_targets == targets:
                break

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
                global_sample_offset += int(preds.shape[0])
                continue

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

            names_for_plot = channel_names
            n_channels = int(trues_mean.shape[-1])
            if names_for_plot is not None and len(names_for_plot) != n_channels:
                log.warning(
                    "Ignoring channel names for video plotting due to length mismatch "
                    "(names=%s, channels=%s).",
                    len(names_for_plot),
                    n_channels,
                )
                names_for_plot = None

            batch_size = int(preds.shape[0])
            sample_targets = _collect_rollout_sample_targets_for_batch(
                targets=targets,
                rendered_targets=rendered_targets,
                batch_idx=batch_idx,
                batch_size=batch_size,
                sample_index=sample_index,
                global_sample_offset=global_sample_offset,
            )

            for target_idx, local_idx in sample_targets.items():
                if target_idx in rendered_targets:
                    continue

                filename = video_dir / f"batch_{target_idx}_sample_{local_idx}.{fmt}"

                # Plot one sample at a time for deterministic target-to-file mapping.
                plot_spatiotemporal_video(
                    true=trues_mean[local_idx : local_idx + 1].cpu(),
                    pred=preds_mean[local_idx : local_idx + 1].cpu(),
                    pred_uq=(
                        preds_uq[local_idx : local_idx + 1].cpu()
                        if preds_uq is not None
                        else None
                    ),
                    batch_idx=0,
                    fps=fps,
                    save_path=str(filename),
                    colorbar_mode="column",
                    pred_uq_label="Ensemble Std Dev",
                    channel_names=names_for_plot,
                    preserve_aspect=preserve_aspect,
                )
                saved_paths.append(filename)
                rendered_targets.add(target_idx)
                log.info("Saved rollout visualization to %s", filename)

            global_sample_offset += batch_size

    # Check for any missing rollout sample indices that were requested
    missing = targets - rendered_targets
    for target_idx in sorted(missing):
        log.warning(
            "Requested rollout index %s was not found in rendered samples.",
            target_idx,
        )

    return saved_paths


def _write_csv(rows: list[dict[str, float | str]], csv_path: Path):
    if not rows:
        log.warning("No evaluation rows to write; skipping CSV generation.")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_rows = [_normalize_row_values_for_csv(row) for row in rows]
    pd.DataFrame(serializable_rows).to_csv(csv_path, index=False)


def _to_csv_scalar(value: Any) -> Any:
    """Convert common tensor/numpy-like scalar values to plain Python scalars."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass

    return value


def _normalize_row_values_for_csv(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize all row values so pandas writes clean scalar values to CSV."""
    return {str(key): _to_csv_scalar(value) for key, value in row.items()}


def _build_per_timestep_metric_factory(
    metric_cls: type[Metric],
) -> Callable[[], Metric]:
    """Build a metric factory configured for per-timestep outputs.

    Some metrics expose ``reduce_all`` in ``__init__``, while others hardcode
    constructor args. We first try ``reduce_all=False`` and then fall back to
    setting ``metric.reduce_all`` directly when available.
    """

    def _factory(cls: type[Metric] = metric_cls) -> Metric:
        try:
            return cls(reduce_all=False)
        except TypeError:
            metric = cls()
            if hasattr(metric, "reduce_all"):
                metric.reduce_all = False
            return metric

    return _factory


def _should_skip_metric(name: str) -> bool:
    """Return True when a metric should be excluded due to memory cost."""
    return name in MEMORY_INTENSIVE_METRICS


def _resolve_metric_fns(
    metrics_list: Sequence[str],
    *,
    has_ensemble: bool,
) -> dict[str, Callable[[], Metric]]:
    """Resolve a list of metric names into a dict of metric factory callables."""
    resolved: dict[str, Callable[[], Metric]] = {}
    for name in metrics_list:
        if _should_skip_metric(name):
            log.info("Skipping metric '%s' due to memory cost.", name)
            continue
        if name in AVAILABLE_METRICS:
            resolved[name] = AVAILABLE_METRICS[name]
        elif name in AVAILABLE_METRICS_ENSEMBLE:
            if has_ensemble:
                resolved[name] = AVAILABLE_METRICS_ENSEMBLE[name]
            else:
                log.info(
                    "Skipping ensemble metric '%s' because n_members <= 1.",
                    name,
                )
        else:
            log.warning("Metric '%s' not found in available metrics.", name)
    return resolved


def _normalize_per_batch_rows(rows: Any) -> list[dict[str, float | str]]:
    """Flatten gathered per-batch rows and keep only mapping-like row records."""
    normalized: list[dict[str, float | str]] = []

    def _visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, Mapping):
            normalized.append(_normalize_row_values_for_csv(item))
            return
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                _visit(sub_item)
            return

        log.warning(
            "Skipping unexpected per-batch row type %s while normalizing rows.",
            type(item),
        )

    _visit(rows)
    return normalized


def _extract_int_batch_idx(value: Any) -> int | None:
    scalar = _to_csv_scalar(value)
    if isinstance(scalar, bool):
        return int(scalar)
    if isinstance(scalar, int):
        return scalar
    if isinstance(scalar, float) and scalar.is_integer():
        return int(scalar)
    return None


def _reindex_per_batch_rows_by_rank(
    rows_by_rank: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, float | str]]:
    """Convert per-rank local batch_idx values to a global, deterministic index.

    The mapping uses ``global_batch_idx = local_batch_idx * world_size + rank``,
    which matches the common distributed-eval ordering where dataloader shards are
    rank-interleaved and ``shuffle=False``.
    """
    if not rows_by_rank:
        return []

    world_size = len(rows_by_rank)
    grouped_rows: list[dict[int, list[dict[str, Any]]]] = []
    passthrough_rows: list[list[dict[str, Any]]] = []
    max_local_batch_idx = -1

    for rank_rows in rows_by_rank:
        grouped: dict[int, list[dict[str, Any]]] = {}
        passthrough: list[dict[str, Any]] = []
        for row in rank_rows:
            row_dict = dict(row)
            local_batch_idx = _extract_int_batch_idx(row_dict.get("batch_idx"))
            if local_batch_idx is None:
                passthrough.append(row_dict)
                continue
            max_local_batch_idx = max(max_local_batch_idx, local_batch_idx)
            grouped.setdefault(local_batch_idx, []).append(row_dict)
        grouped_rows.append(grouped)
        passthrough_rows.append(passthrough)

    reindexed_rows: list[dict[str, float | str]] = []
    for local_batch_idx in range(max_local_batch_idx + 1):
        for rank in range(world_size):
            rank_group = grouped_rows[rank]
            for row in rank_group.get(local_batch_idx, []):
                updated = dict(row)
                updated["batch_idx"] = local_batch_idx * world_size + rank
                reindexed_rows.append(_normalize_row_values_for_csv(updated))

    for passthrough in passthrough_rows:
        for row in passthrough:
            reindexed_rows.append(_normalize_row_values_for_csv(row))

    return reindexed_rows


def _gather_per_batch_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fabric: L.Fabric,
) -> list[dict[str, float | str]]:
    """Gather per-rank row objects and normalize global batch indices."""
    local_rows = _normalize_per_batch_rows(rows)
    world_size = int(getattr(fabric, "world_size", 1))
    if world_size <= 1:
        return local_rows

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        log.warning(
            "Distributed row gather requested but torch.distributed is unavailable; "
            "falling back to local per-batch rows."
        )
        return local_rows

    gathered_rows: list[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_rows, local_rows)
    normalized_rows_by_rank = [
        _normalize_per_batch_rows(item) for item in gathered_rows
    ]
    return _reindex_per_batch_rows_by_rank(normalized_rows_by_rank)


def _is_metadata_row(row: Mapping[str, float | str]) -> bool:
    return row.get("window") == "meta" or "category" in row


def _split_metric_and_metadata_rows(
    rows: list[dict[str, float | str]],
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    metric_rows: list[dict[str, float | str]] = []
    metadata_rows: list[dict[str, float | str]] = []

    for row in rows:
        if not isinstance(row, Mapping):
            log.warning("Skipping malformed evaluation row with type %s", type(row))
            continue
        if _is_metadata_row(row):
            metadata_rows.append(dict(row))
        else:
            metric_rows.append(dict(row))

    return metric_rows, metadata_rows


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


def _extract_training_timer_callback_state(
    checkpoint_payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the TrainingTimerCallback state dict from checkpoint callbacks."""
    callbacks = checkpoint_payload.get("callbacks")
    if not isinstance(callbacks, Mapping):
        return None
    for callback_state in callbacks.values():
        if not isinstance(callback_state, Mapping):
            continue
        if any(
            key in callback_state
            for key in (
                "training_runtime_total_s",
                "training_runtime_elapsed_s",
                "mean_epoch_s",
                "min_epoch_s",
                "max_epoch_s",
                "epoch_times_s",
            )
        ):
            return callback_state
    return None


def _training_runtime_rows(
    checkpoint_payload: Mapping[str, Any],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    callback_state = _extract_training_timer_callback_state(checkpoint_payload)
    if callback_state is None:
        return rows

    # Prefer the final runtime if available. Fall back to an "elapsed so far"
    # snapshot saved in epoch-end checkpoints (see TrainingTimerCallback docs).
    total_value = callback_state.get("training_runtime_total_s")
    elapsed_value = callback_state.get("training_runtime_elapsed_s")
    runtime_total_s: float | None = None
    if isinstance(total_value, int | float) and float(total_value) > 0:
        runtime_total_s = float(total_value)
    elif isinstance(elapsed_value, int | float) and float(elapsed_value) > 0:
        runtime_total_s = float(elapsed_value)

    if runtime_total_s is not None:
        rows.append(
            _make_metadata_row(
                category="runtime_train",
                metric="total_s",
                value=runtime_total_s,
            )
        )

    for callback_key, metric_name in (
        ("mean_epoch_s", "mean_epoch_s"),
        ("min_epoch_s", "min_epoch_s"),
        ("max_epoch_s", "max_epoch_s"),
    ):
        value = callback_state.get(callback_key)
        if isinstance(value, int | float) and float(value) > 0:
            rows.append(
                _make_metadata_row(
                    category="runtime_train",
                    metric=metric_name,
                    value=float(value),
                )
            )

    return rows


def _evaluation_metadata_rows(
    checkpoint_payload: Mapping[str, Any],
    model: EncoderProcessorDecoderEnsemble | EncoderProcessorDecoder,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    rows.extend(_training_runtime_rows(checkpoint_payload))
    rows.extend(_parameter_count_rows(model))
    return rows


def _collect_benchmark_rows(
    *,
    eval_cfg: DictConfig,
    cfg: DictConfig,
    stats: Mapping[str, Any],
    model: EncoderProcessorDecoderEnsemble | EncoderProcessorDecoder,
    checkpoint_path: Path,
    device: str,
    eval_batch_size: int,
) -> list[dict[str, float | str | int | None]]:
    """Collect benchmark rows for model and optional rollout benchmark runs."""
    rows: list[dict[str, float | str | int | None]] = []

    benchmark_cfg = eval_cfg.get("benchmark", {})
    if benchmark_cfg.get("enabled", False):
        benchmark_batch_size = int(benchmark_cfg.get("batch_size", eval_batch_size))
        benchmark_n_warmup = int(benchmark_cfg.get("n_warmup", 5))
        benchmark_n_benchmark = int(benchmark_cfg.get("n_benchmark", 50))
        example_batch = stats.get("example_batch")
        if isinstance(example_batch, Batch):
            log.info(
                (
                    "Running inference benchmark "
                    "(batch_size=%s, n_warmup=%s, n_benchmark=%s)"
                ),
                benchmark_batch_size,
                benchmark_n_warmup,
                benchmark_n_benchmark,
            )
            benchmark_metrics = benchmark_model(
                model,
                example_batch,
                n_warmup=benchmark_n_warmup,
                n_benchmark=benchmark_n_benchmark,
                batch_size=benchmark_batch_size,
            )
            rows.extend(
                benchmark_metric_rows(
                    benchmark_type="model",
                    checkpoint_path=checkpoint_path,
                    device=device,
                    batch_size=benchmark_batch_size,
                    n_warmup=benchmark_n_warmup,
                    n_benchmark=benchmark_n_benchmark,
                    metrics=benchmark_metrics,
                )
            )
        else:
            log.warning(
                "Skipping inference benchmark: expected Batch example, got %s",
                type(example_batch),
            )

    rollout_benchmark_cfg = eval_cfg.get("benchmark_rollout", {})
    if rollout_benchmark_cfg.get("enabled", False):
        rb_batch_size = int(rollout_benchmark_cfg.get("batch_size", eval_batch_size))
        rb_n_warmup = int(rollout_benchmark_cfg.get("n_warmup", 5))
        rb_n_benchmark = int(rollout_benchmark_cfg.get("n_benchmark", 20))
        rb_max_rollout_steps = int(
            rollout_benchmark_cfg.get("max_rollout_steps")
            or eval_cfg.get("max_rollout_steps", 10)
        )
        rb_free_running_only = bool(
            rollout_benchmark_cfg.get(
                "free_running_only", eval_cfg.get("free_running_only", True)
            )
        )
        data_config = cfg.get("datamodule", {})
        rb_stride = int(
            rollout_benchmark_cfg.get("stride")
            or data_config.get("rollout_stride")
            or stats["n_steps_output"]
        )
        rb_example_batch = stats.get("example_batch")
        if isinstance(rb_example_batch, Batch):
            log.info(
                (
                    "Running rollout benchmark "
                    "(batch_size=%s, n_warmup=%s, n_benchmark=%s, "
                    "stride=%s, max_rollout_steps=%s)"
                ),
                rb_batch_size,
                rb_n_warmup,
                rb_n_benchmark,
                rb_stride,
                rb_max_rollout_steps,
            )
            rollout_benchmark_metrics = benchmark_rollout(
                model,
                rb_example_batch,
                stride=rb_stride,
                max_rollout_steps=rb_max_rollout_steps,
                n_warmup=rb_n_warmup,
                n_benchmark=rb_n_benchmark,
                batch_size=rb_batch_size,
                free_running_only=rb_free_running_only,
            )
            rows.extend(
                benchmark_metric_rows(
                    benchmark_type="rollout",
                    checkpoint_path=checkpoint_path,
                    device=device,
                    batch_size=rb_batch_size,
                    n_warmup=rb_n_warmup,
                    n_benchmark=rb_n_benchmark,
                    metrics=rollout_benchmark_metrics,
                    stride=rb_stride,
                    max_rollout_steps=rb_max_rollout_steps,
                )
            )
        else:
            log.warning(
                "Skipping rollout benchmark: expected Batch example, got %s",
                type(rb_example_batch),
            )

    return rows


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:
    """Entry point for CLI-based evaluation."""
    run_evaluation(cfg)


def _run_test_metrics(
    *,
    model: Any,
    test_loader: Any,
    eval_cfg: DictConfig,
    metrics_list: Sequence[str],
    has_ensemble: bool,
    n_members: int,
    fabric: L.Fabric,
    work_dir: Path,
) -> list[dict[str, float | str]]:
    """Run the test-set metric evaluation phase."""
    test_metric_fns = _resolve_metric_fns(metrics_list, has_ensemble=has_ensemble)

    compute_coverage = eval_cfg.get("compute_coverage", False)
    if (n_members > 1) or compute_coverage:

        def coverage_factory() -> Metric:
            return MultiCoverage(coverage_levels=eval_cfg.get("coverage_levels", None))

        test_metric_fns["coverage"] = coverage_factory

    log.info("Computing test metrics: %s", list(test_metric_fns.keys()))

    test_windows = _map_windows(eval_cfg.get("metric_windows", None))

    test_metrics_results, _, test_per_batch_rows = compute_metrics_from_dataloader(
        dataloader=test_loader,
        metric_fns=test_metric_fns,
        predict_fn=model,
        windows=test_windows,
        return_per_batch=True,
        device=fabric.device,
    )

    if test_per_batch_rows:
        test_per_batch_rows = _gather_per_batch_rows(test_per_batch_rows, fabric=fabric)

    return _process_metrics_results(
        test_metrics_results,
        per_batch_rows=test_per_batch_rows,  # pyright: ignore[reportArgumentType]
        log_prefix="Test",
        plot_dir=work_dir,
    )


def _run_rollout_videos(
    *,
    model: Any,
    datamodule: Any,
    eval_cfg: DictConfig,
    fabric: L.Fabric,
    eval_batch_size: int,
    max_rollout_batches: int | None,
    batch_indices: Sequence[int],
    video_dir: Path,
    rollout_stride: int,
    max_rollout_steps: int,
    n_members: int,
) -> None:
    """Render rollout videos for requested sample indices."""
    rollout_test_loader = fabric.setup_dataloaders(
        datamodule.rollout_test_dataloader(batch_size=eval_batch_size)
    )
    rollout_channel_names = _resolve_rollout_channel_names(
        getattr(rollout_test_loader, "dataset", None)
    )
    if rollout_channel_names is not None:
        log.info(
            "Using rollout video channel labels from core_field_names: %s",
            rollout_channel_names,
        )
    else:
        log.info(
            "No rollout video channel labels found in core_field_names; "
            "falling back to generic channel indices."
        )

    rollout_loader = _limit_batches(rollout_test_loader, max_rollout_batches)
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
        channel_names=rollout_channel_names,
        preserve_aspect=eval_cfg.get("preserve_aspect", False),
    )


def _run_rollout_metrics(
    *,
    model: Any,
    datamodule: Any,
    eval_cfg: DictConfig,
    fabric: L.Fabric,
    metrics_list: Sequence[str],
    has_ensemble: bool,
    n_members: int,
    eval_batch_size: int,
    max_rollout_batches: int | None,
    rollout_stride: int,
    max_rollout_steps: int,
    csv_path: Path,
) -> None:
    """Compute rollout metrics (windowed, per-batch, and per-timestep)."""
    rollout_metric_fns = _resolve_metric_fns(metrics_list, has_ensemble=has_ensemble)

    compute_rollout_coverage = eval_cfg.get("compute_rollout_coverage", False)
    if compute_rollout_coverage and n_members and n_members > 1:
        log.info("Adding rollout coverage to metrics...")
        unwrapped_model = _unwrap_module(model)
        if not isinstance(unwrapped_model, EncoderProcessorDecoderEnsemble):
            msg = (
                "Rollout coverage requires an ensemble model, but got "
                f"{type(unwrapped_model)!r}."
            )
            raise TypeError(msg)

        def coverage_factory() -> Metric:
            return MultiCoverage(coverage_levels=eval_cfg.get("coverage_levels", None))

        rollout_metric_fns["coverage"] = coverage_factory

    if not rollout_metric_fns:
        return

    log.info("Computing rollout metrics: %s", list(rollout_metric_fns.keys()))
    windows = _map_windows(
        eval_cfg.get("metric_windows_rollout", [(0, 1), (6, 12), (13, 30)])
    )

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
            predict_fn=rollout_predict,
            windows=windows,
            return_per_batch=True,
            device=fabric.device,
        )
    )
    if rollout_per_batch_rows:
        rollout_per_batch_rows = _gather_per_batch_rows(
            rollout_per_batch_rows,
            fabric=fabric,
        )

    rollout_csv_rows = _process_metrics_results(
        rollout_metrics_per_window,
        per_batch_rows=rollout_per_batch_rows,  # pyright: ignore[reportArgumentType]
        log_prefix="Rollout",
        plot_dir=csv_path.parent,
    )

    rollout_metric_rows, rollout_metadata_rows = _split_metric_and_metadata_rows(
        rollout_csv_rows
    )

    if fabric.global_rank == 0 and rollout_metric_rows:
        rollout_csv_path = csv_path.parent / "rollout_metrics.csv"
        _write_csv(rollout_metric_rows, rollout_csv_path)
        log.info("Wrote rollout metrics to %s", rollout_csv_path)

    if fabric.global_rank == 0 and rollout_metadata_rows:
        rollout_metadata_csv_path = csv_path.parent / "rollout_metadata.csv"
        _write_csv(rollout_metadata_rows, rollout_metadata_csv_path)
        log.info("Wrote rollout metadata to %s", rollout_metadata_csv_path)

    # Per-timestep, per-channel rollout metrics
    _run_rollout_per_timestep_metrics(
        rollout_metric_fns=rollout_metric_fns,
        rollout_predict=rollout_predict,
        datamodule=datamodule,
        fabric=fabric,
        eval_batch_size=eval_batch_size,
        max_rollout_batches=max_rollout_batches,
        max_rollout_steps=max_rollout_steps,
        rollout_stride=rollout_stride,
        csv_path=csv_path,
    )


def _run_rollout_per_timestep_metrics(
    *,
    rollout_metric_fns: dict[str, Callable[[], Metric]],
    rollout_predict: Callable,
    datamodule: Any,
    fabric: L.Fabric,
    eval_batch_size: int,
    max_rollout_batches: int | None,
    max_rollout_steps: int,
    rollout_stride: int,
    csv_path: Path,
) -> None:
    """Compute per-timestep, per-channel rollout metrics."""
    per_timestep_metric_fns: dict[str, Callable[[], Metric]] = {}
    for name, metric_factory in rollout_metric_fns.items():
        if _should_skip_metric(name):
            log.info(
                "Skipping rollout per-timestep metric '%s' due to memory cost.",
                name,
            )
            continue
        if name == "coverage":
            per_timestep_metric_fns[name] = metric_factory
        else:
            metric_cls = AVAILABLE_METRICS.get(name)
            if metric_cls is None:
                metric_cls = AVAILABLE_METRICS_ENSEMBLE.get(name)
            if metric_cls is not None:
                per_timestep_metric_fns[name] = _build_per_timestep_metric_factory(
                    metric_cls
                )

    if not per_timestep_metric_fns:
        return

    max_rollout_timesteps = _resolve_rollout_timestep_limit(
        max_rollout_steps=max_rollout_steps,
        rollout_stride=int(rollout_stride),
    )
    rollout_loader_per_timestep = _limit_batches(
        fabric.setup_dataloaders(
            datamodule.rollout_test_dataloader(batch_size=eval_batch_size)
        ),
        max_rollout_batches,
    )
    per_timestep_results = compute_metrics_per_timestep_from_dataloader(
        dataloader=rollout_loader_per_timestep,
        metric_fns=per_timestep_metric_fns,
        predict_fn=rollout_predict,
        max_timesteps=max_rollout_timesteps,
        device=fabric.device,
    )
    if not per_timestep_results or fabric.global_rank != 0:
        return

    T, C = next(iter(per_timestep_results.values())).shape
    timestep_cols = [str(t) for t in range(T)]
    timestep_index = pd.Index(timestep_cols)
    for c in range(C):
        df = pd.DataFrame.from_dict(
            {
                metric: per_timestep_results[metric][:, c].tolist()
                for metric in per_timestep_results
            },
            orient="index",
            columns=timestep_index,
        )
        out_path = csv_path.parent / f"rollout_metrics_per_timestep_channel_{c}.csv"
        df.to_csv(out_path)
        log.info(
            "Wrote rollout metrics per timestep (channel %s) to %s",
            c,
            out_path,
        )
    df_all = pd.DataFrame.from_dict(
        {
            metric: per_timestep_results[metric].mean(axis=1).tolist()
            for metric in per_timestep_results
        },
        orient="index",
        columns=timestep_index,
    )
    out_path_all = csv_path.parent / "rollout_metrics_per_timestep_channel_all.csv"
    df_all.to_csv(out_path_all)
    log.info(
        "Wrote rollout metrics per timestep (channel all) to %s",
        out_path_all,
    )


def _write_final_csvs(
    *,
    evaluation_rows: list[dict[str, float | str]],
    benchmark_rows: list[dict[str, float | str | int | None]],
    fabric: L.Fabric,
    csv_path: Path,
    eval_cfg: DictConfig,
    work_dir: Path,
) -> None:
    """Write final evaluation, metadata, and benchmark CSVs (rank 0 only)."""
    metric_rows, metadata_rows = _split_metric_and_metadata_rows(evaluation_rows)

    if fabric.global_rank == 0 and metric_rows:
        _write_csv(metric_rows, csv_path)
        log.info("Wrote metrics CSV to %s", csv_path)

    metadata_csv_path = csv_path.parent / "evaluation_metadata.csv"
    if fabric.global_rank == 0 and metadata_rows:
        _write_csv(metadata_rows, metadata_csv_path)
        log.info("Wrote evaluation metadata to %s", metadata_csv_path)

    if benchmark_rows and fabric.global_rank == 0:
        benchmark_csv_path = resolve_benchmark_csv_path(eval_cfg, work_dir)
        benchmark_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(benchmark_rows).to_csv(benchmark_csv_path, index=False)
        log.info("Wrote benchmark CSV to %s", benchmark_csv_path)


def _setup_fabric(eval_cfg: DictConfig) -> L.Fabric:
    """Create and launch a Fabric instance from eval config."""
    # Prefer eval.accelerator to mirror Lightning API, while keeping
    # eval.device as a backwards-compatible fallback.
    accelerator = eval_cfg.get("accelerator", None)
    if accelerator is None:
        accelerator = eval_cfg.get("device", "auto")
    elif "device" in eval_cfg:
        log.warning(
            "Both eval.accelerator and deprecated eval.device were provided; "
            "using eval.accelerator=%s.",
            accelerator,
        )
    devices = eval_cfg.get("devices", 1)
    fabric = L.Fabric(accelerator=accelerator, devices=devices)
    fabric.launch()
    return fabric


def _load_model_checkpoint(
    model: EncoderProcessorDecoder | EncoderProcessorDecoderEnsemble,
    checkpoint_path: Path,
) -> Mapping[str, Any]:
    """Load checkpoint into model, returning the full checkpoint payload."""
    log.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    state_dict = extract_state_dict(checkpoint_payload)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        msg = (
            "Checkpoint parameters do not match the instantiated model. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}."
        )
        raise RuntimeError(msg)
    return checkpoint_payload


def _resolve_eval_paths(
    cfg: DictConfig,
    eval_cfg: DictConfig,
    work_dir: Path,
) -> tuple[Path, Path, Path]:
    """Resolve checkpoint, CSV, and video directory paths."""
    checkpoint_path = resolve_checkpoint_path(
        eval_cfg,
        work_dir,
        missing_message=(
            "No checkpoint specified. Please provide a checkpoint path via:\n"
            "  eval.checkpoint=/path/to/checkpoint.ckpt\n"
            "Or add it to your config file."
        ),
    )
    if cfg.get("output", {}).get("save_config"):
        save_resolved_config(cfg, work_dir, filename="resolved_eval_config.yaml")

    csv_path = _resolve_csv_path(eval_cfg, work_dir)
    video_dir = _resolve_video_dir(eval_cfg, work_dir)
    return checkpoint_path, csv_path, video_dir


def run_evaluation(cfg: DictConfig, work_dir: Path | None = None) -> None:
    """Run evaluation using an already-composed config."""
    logging.basicConfig(level=logging.INFO)

    umask_value = cfg.get("umask")
    if umask_value is not None:
        os.umask(int(str(umask_value), 8))
        log.info("Applied process umask %s", umask_value)

    work_dir = resolve_hydra_work_dir(work_dir)

    eval_cfg = cfg.get("eval", {})
    eval_batch_size: int = eval_cfg.get("batch_size", 1)
    max_test_batches = eval_cfg.get("max_test_batches")
    max_rollout_batches = _resolve_rollout_batch_limit(eval_cfg)
    log.info(
        "Batch limits: max_test_batches=%s, max_rollout_batches=%s",
        max_test_batches,
        max_rollout_batches,
    )

    checkpoint_path, csv_path, video_dir = _resolve_eval_paths(cfg, eval_cfg, work_dir)

    # Setup datamodule, model, and checkpoint
    datamodule, cfg, stats = setup_datamodule(cfg)

    if "n_members" in eval_cfg:
        with open_dict(cfg.model):
            cfg.model.n_members = eval_cfg.n_members
        log.info(
            "Overriding model.n_members with %s from eval config", eval_cfg.n_members
        )

    model = setup_epd_model(cfg, stats, datamodule=datamodule)
    checkpoint_payload = _load_model_checkpoint(model, checkpoint_path)

    metrics_list = eval_cfg.get("metrics", DEFAULT_EVAL_METRICS)
    batch_indices = eval_cfg.get("batch_indices", [])
    n_members = cfg.get("model", {}).get("n_members", 1)
    has_ensemble = bool(n_members and n_members > 1)

    # Setup Fabric and wrap model
    fabric = _setup_fabric(eval_cfg)
    log.info("Model configuration n_members: %s", n_members)
    log.info("Model class: %s", type(model))

    model = fabric.setup_module(model)
    _mark_forward_methods_if_available(model, methods=("rollout",))
    model.eval()

    # Unwrap once for functions that need direct access to model internals
    # (parameter counting, benchmark shaping, etc.). The Fabric-wrapped `model`
    # is still used for all forward/rollout calls.
    unwrapped_model = _unwrap_module(model)

    test_loader = _limit_batches(
        fabric.setup_dataloaders(datamodule.test_dataloader()),
        max_test_batches,
    )

    # Phase 1: Test metrics
    test_rows = _run_test_metrics(
        model=model,
        test_loader=test_loader,
        eval_cfg=eval_cfg,
        metrics_list=metrics_list,
        has_ensemble=has_ensemble,
        n_members=n_members,
        fabric=fabric,
        work_dir=work_dir,
    )

    evaluation_rows: list[dict[str, float | str]] = []
    evaluation_rows.extend(test_rows)
    evaluation_rows.extend(
        _evaluation_metadata_rows(
            checkpoint_payload=checkpoint_payload,
            model=unwrapped_model,
        )
    )
    benchmark_rows = _collect_benchmark_rows(
        eval_cfg=eval_cfg,
        cfg=cfg,
        stats=stats,
        model=unwrapped_model,
        checkpoint_path=checkpoint_path,
        device=str(fabric.device),
        eval_batch_size=eval_batch_size,
    )

    # Phase 2: Rollout videos and metrics
    compute_rollout_coverage = eval_cfg.get("compute_rollout_coverage", False)
    compute_rollout_metrics = eval_cfg.get("compute_rollout_metrics", False)

    if batch_indices or compute_rollout_coverage or compute_rollout_metrics:
        max_rollout_steps = eval_cfg.get("max_rollout_steps", 10)

        data_config = cfg.get("datamodule", {})
        rollout_stride = data_config.get("rollout_stride") or stats["n_steps_output"]

        if batch_indices:
            _run_rollout_videos(
                model=unwrapped_model,
                datamodule=datamodule,
                eval_cfg=eval_cfg,
                fabric=fabric,
                eval_batch_size=eval_batch_size,
                max_rollout_batches=max_rollout_batches,
                batch_indices=batch_indices,
                video_dir=video_dir,
                rollout_stride=rollout_stride,
                max_rollout_steps=max_rollout_steps,
                n_members=n_members,
            )

        if compute_rollout_metrics or compute_rollout_coverage:
            _run_rollout_metrics(
                model=model,
                datamodule=datamodule,
                eval_cfg=eval_cfg,
                fabric=fabric,
                metrics_list=metrics_list,
                has_ensemble=has_ensemble,
                n_members=n_members,
                eval_batch_size=eval_batch_size,
                max_rollout_batches=max_rollout_batches,
                rollout_stride=rollout_stride,
                max_rollout_steps=max_rollout_steps,
                csv_path=csv_path,
            )

    # Phase 3: Write final CSVs
    _write_final_csvs(
        evaluation_rows=evaluation_rows,
        benchmark_rows=benchmark_rows,
        fabric=fabric,
        csv_path=csv_path,
        eval_cfg=eval_cfg,
        work_dir=work_dir,
    )


if __name__ == "__main__":
    main()
