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
import yaml
from einops import rearrange
from omegaconf import DictConfig, OmegaConf, open_dict
from the_well.data.normalization import ZScoreNormalization
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
    WinklerScore,
)
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.models.processor import ProcessorModel
from autocast.models.processor_ensemble import ProcessorModelEnsemble
from autocast.scripts.config import save_resolved_config
from autocast.scripts.execution import (
    benchmark_metric_rows,
    extract_state_dict,
    load_checkpoint_payload,
    resolve_benchmark_csv_path,
    resolve_checkpoint_path,
    resolve_hydra_work_dir,
)
from autocast.scripts.setup import (
    setup_autoencoder_components,
    setup_datamodule,
    setup_epd_model,
    setup_processor_model,
)
from autocast.scripts.training import apply_float32_matmul_precision
from autocast.scripts.utils import get_default_config_path
from autocast.types.batch import Batch, EncodedBatch
from autocast.utils import plot_spatiotemporal_video
from autocast.utils.plots import (
    compute_metrics_from_dataloader,
    compute_metrics_per_timestep_from_dataloader,
)

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
    "winkler": WinklerScore,
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
    "winkler",
]

MEMORY_INTENSIVE_METRICS = {"variogram"}


def _decode_tensor(
    x: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_members: int | None = None,
    decode_chunk_size: int = 4,
) -> torch.Tensor:
    """Decode a tensor while preserving an optional trailing ensemble axis.

    When ``n_members`` is set, the ensemble dimension is flattened into the
    batch dimension before decoding.  To avoid OOM when the resulting batch is
    large (e.g. rollout with many members), decoding is done in chunks of
    ``decode_chunk_size`` along the batch dimension.
    """
    if n_members is not None and n_members > 1 and x.shape[-1] == n_members:
        batch_size = x.shape[0]
        flattened = x.movedim(-1, 1).flatten(0, 1)
        # Chunk along batch dim to avoid OOM for large rollouts
        chunks = []
        for i in range(0, flattened.shape[0], decode_chunk_size):
            chunks.append(decode_fn(flattened[i : i + decode_chunk_size]))
        decoded = torch.cat(chunks, dim=0)
        return decoded.unflatten(0, (batch_size, n_members)).movedim(1, -1)
    return decode_fn(x)


def _build_eval_predict_fn(
    model: Any,
    *,
    is_processor_model: bool,
    decode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    n_members: int | None = None,
) -> Callable[[Any], Any]:
    """Build the prediction callable used by evaluation metrics."""
    if is_processor_model:

        def predict_fn(batch):
            # For ensemble models, expand batch and rearrange to add member dim
            if n_members is not None and n_members > 1:
                b = batch.encoded_inputs.shape[0]
                expanded_batch = batch.repeat(n_members)
                latent_pred = model._predict(expanded_batch)
                latent_pred = rearrange(
                    latent_pred, "(b m) ... -> b ... m", b=b, m=n_members
                )
            else:
                latent_pred = model._predict(batch)
            if decode_fn is not None:
                return (
                    _decode_tensor(latent_pred, decode_fn, n_members=n_members),
                    _decode_tensor(
                        batch.encoded_output_fields, decode_fn, n_members=None
                    ),
                )
            return latent_pred, batch.encoded_output_fields

        return predict_fn

    return model


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


def _unwrap_module(module: Any) -> Any:
    """Return the underlying model when wrapped by Fabric/DDP-style wrappers."""
    unwrapped = module
    while hasattr(unwrapped, "module"):
        next_module = unwrapped.module
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
    names_already_subset = raw_names is not None

    if not isinstance(raw_names, Sequence) or isinstance(raw_names, str):
        normalization_stats = getattr(dataset, "normalization_stats", None)
        if isinstance(normalization_stats, Mapping):
            raw_names = normalization_stats.get("core_field_names")
            names_already_subset = False

    if not isinstance(raw_names, Sequence) or isinstance(raw_names, str):
        return None

    channel_names = [str(name) for name in raw_names]
    if not channel_names:
        return None

    channel_idxs = getattr(dataset, "channel_idxs", None)
    if channel_idxs is not None and not names_already_subset:
        try:
            channel_names = [channel_names[idx] for idx in channel_idxs]
        except (TypeError, IndexError):
            log.warning(
                "Could not apply channel_idxs=%s to channel names %s.",
                channel_idxs,
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
            try:
                val = metric.compute()
            except RuntimeError:
                log.warning(
                    "%s metric '%s' for window %s: skipped (no samples)",
                    log_prefix,
                    name,
                    window,
                )
                continue

            log.info(
                "%s metric '%s' for window %s: %s",
                log_prefix,
                name,
                window,
                val,
            )

            # If this is coverage, also plot it
            if name == "coverage" and isinstance(metric, MultiCoverage):
                metric.plot(
                    save_path=plot_dir
                    / f"{log_prefix.lower()}_coverage_window_{window_str}.png",
                    title=f"{log_prefix} Coverage Window {window}",
                )

            # Try to get a scalar value for csv
            if val.numel() == 1:
                row[name] = float(val.item())
            elif hasattr(val, "mean"):
                row[name] = float(val.mean().item())

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


def _render_rollouts(  # noqa: PLR0912
    model: (
        EncoderProcessorDecoder
        | EncoderProcessorDecoderEnsemble
        | ProcessorModel
        | ProcessorModelEnsemble
    ),
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
    decode_fn: Callable | None = None,
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
            if decode_fn is not None:
                with torch.no_grad():
                    preds = _decode_tensor(
                        preds,
                        decode_fn,
                        n_members=n_members if n_members and n_members > 1 else None,
                    )
                    if trues is not None:
                        trues = _decode_tensor(trues, decode_fn, n_members=None)
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
    model: (
        EncoderProcessorDecoderEnsemble
        | EncoderProcessorDecoder
        | ProcessorModel
        | ProcessorModelEnsemble
    ),
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
    model: (
        EncoderProcessorDecoderEnsemble
        | EncoderProcessorDecoder
        | ProcessorModel
        | ProcessorModelEnsemble
    ),
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
    model: (
        EncoderProcessorDecoderEnsemble
        | EncoderProcessorDecoder
        | ProcessorModel
        | ProcessorModelEnsemble
    ),
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


def _extract_processor_state_dict(
    checkpoint_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only the processor sub-module weights with the prefix stripped."""
    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    return {
        k[len("processor.") :]: v
        for k, v in state_dict.items()
        if k.startswith("processor.")
    }


def _load_autoencoder_config_from_cache(cache_dir: Path) -> DictConfig | None:
    """Load the autoencoder config saved alongside cached latents, if present."""
    config_path = cache_dir / "autoencoder_config.yaml"
    if not config_path.exists():
        log.debug("No autoencoder_config.yaml found in %s", cache_dir)
        return None
    try:
        loaded = OmegaConf.load(config_path)
        if not isinstance(loaded, DictConfig):
            return None
        return loaded
    except Exception as exc:
        log.warning("Could not load autoencoder config from %s: %s", config_path, exc)
        return None


def _try_build_decode_fn(
    cfg: DictConfig,
) -> "tuple[Any, Any] | tuple[None, None]":
    """Try to load a decoder from the cached-latents autoencoder config.

    Returns ``(decoder_module, decode_fn)`` on success, or ``(None, None)``
    if the autoencoder config / checkpoint cannot be found or loaded.

    If the autoencoder was trained with ``use_normalization: true``, the
    returned ``decode_fn`` also applies denormalization so that all metrics
    are computed on unnormalized (raw) data — matching the ambient eval path.
    """
    data_path = cfg.get("datamodule", {}).get("data_path")
    if not data_path:
        return None, None

    ae_cfg = _load_autoencoder_config_from_cache(Path(data_path))
    if ae_cfg is None:
        return None, None

    try:
        _, decoder = setup_autoencoder_components(ae_cfg, {})
        decoder.eval()
        log.info(
            "Loaded decoder (%s) from autoencoder config for data-space evaluation.",
            type(decoder).__name__,
        )

        # If the autoencoder was trained on normalized data, wrap decode_fn to
        # also denormalize so metrics are in raw (unnormalized) space.
        ae_dm_cfg = ae_cfg.get("datamodule", {})
        norm = None
        if ae_dm_cfg.get("use_normalization", False):
            norm_path = ae_dm_cfg.get("normalization_path")
            if norm_path:
                # Resolve env-var interpolations that OmegaConf may leave in the string
                resolved_norm_path = OmegaConf.to_container(
                    OmegaConf.create({"p": norm_path}), resolve=True
                )
                if isinstance(resolved_norm_path, Mapping):
                    norm_path = resolved_norm_path.get("p", norm_path)
                with open(norm_path) as f:
                    norm_stats = yaml.safe_load(f)
                norm = ZScoreNormalization(
                    norm_stats.get("stats", {}),
                    norm_stats.get("core_field_names", []),
                    norm_stats.get("constant_field_names", []),
                )
                log.info(
                    "Autoencoder was trained with normalization — "
                    "decode_fn will also denormalize to raw data space."
                )
            else:
                log.warning(
                    "use_normalization=true in autoencoder config but no "
                    "normalization_path found — metrics may be on normalized scale."
                )

        if norm is not None:

            def decode_fn(x):
                return norm.denormalize_flattened(decoder.decode(x), "variable")
        else:

            def decode_fn(x):
                return decoder.decode(x)

        return decoder, decode_fn
    except Exception as exc:
        log.warning(
            "Could not build decoder from autoencoder config — "
            "falling back to latent-space evaluation: %s",
            exc,
        )
        return None, None


def _is_processor_only_checkpoint(checkpoint_payload: Mapping[str, Any]) -> bool:
    """Return True if the checkpoint contains only processor weights.

    Processor-only checkpoints (from ``autocast train processor``) have no
    ``encoder_decoder.*`` keys, in contrast to full EPD checkpoints.
    """
    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    if not isinstance(state_dict, Mapping):
        return False
    return not any(k.startswith("encoder_decoder.") for k in state_dict)


def run_evaluation(cfg: DictConfig, work_dir: Path | None = None) -> None:  # noqa: PLR0912, PLR0915
    """Run evaluation using an already-composed config."""
    logging.basicConfig(level=logging.INFO)

    # cfg will override the default of float32 matmul precision to "high" that's set
    # here to maintain backward compatibility where this was set but not configurable.
    apply_float32_matmul_precision(cfg, default="high")

    umask_value = cfg.get("umask")
    if umask_value is not None:
        os.umask(int(str(umask_value), 8))
        log.info("Applied process umask %s", umask_value)

    work_dir = resolve_hydra_work_dir(work_dir)

    # Get eval config
    eval_cfg = cfg.get("eval", {})
    eval_batch_size: int = eval_cfg.get("batch_size", 1)
    max_test_batches = eval_cfg.get("max_test_batches")
    max_rollout_batches = _resolve_rollout_batch_limit(eval_cfg)
    log.info(
        "Batch limits: max_test_batches=%s, max_rollout_batches=%s",
        max_test_batches,
        max_rollout_batches,
    )

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

    # Load checkpoint payload early to detect model type
    log.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    processor_only = _is_processor_only_checkpoint(checkpoint_payload)

    # Setup datamodule and resolve config
    datamodule, cfg, stats = setup_datamodule(cfg)

    # Override model n_members from eval config if specified
    if "n_members" in eval_cfg:
        with open_dict(cfg.model):
            cfg.model.n_members = eval_cfg.n_members
        log.info(
            "Overriding model.n_members with %s from eval config", eval_cfg.n_members
        )

    # Setup model and weights based on checkpoint type and datamodule.
    #
    # Mode 1 - ambient/data-space EPD from two checkpoints:
    #   Triggered when the processor ckpt is processor-only AND the config
    #   contains `autoencoder_checkpoint` AND the datamodule yields raw Batch
    #   objects (i.e. not cached latents).  The encoder+decoder are loaded from
    #   `autoencoder_checkpoint`; only the processor weights come from the
    #   processor checkpoint.  Typical invocation:
    #     autocast eval --workdir <ae_workdir> eval.checkpoint=<processor.ckpt>
    #
    # Mode 2 - data-space eval via processor + decoder (cached latents):
    #   Triggered when the processor ckpt is processor-only AND the datamodule
    #   yields EncodedBatch objects (cached latents).  The processor is loaded
    #   from the processor ckpt; the decoder is loaded from `autoencoder_config.yaml`
    #   saved in the cache directory by `autocast cache-latents`.
    #   Predictions and ground-truth latents are both decoded before metrics.
    #
    # Fallback - latent-space processor eval:
    #   No `autoencoder_checkpoint` / no `autoencoder_config.yaml` available.
    #   Metrics are computed in latent space.

    example_batch = stats.get("example_batch")

    # Stateless encoders/decoders (e.g. PermuteConcat/ChannelsLast) contribute no
    # `encoder_decoder.*` params, so a full EPD checkpoint can look processor-only.
    if (
        processor_only
        and isinstance(example_batch, Batch)
        and not cfg.get("autoencoder_checkpoint")
        and cfg.get("model", {}).get("encoder") is not None
        and cfg.get("model", {}).get("decoder") is not None
    ):
        log.info(
            "Checkpoint contains no encoder_decoder.* params, but datamodule "
            "returns raw Batch and model has encoder+decoder config. "
            "Assuming full EPD checkpoint with stateless encoder/decoder."
        )
        processor_only = False

    log.info(
        "Checkpoint type: %s",
        "processor-only" if processor_only else "encoder-processor-decoder",
    )

    decode_fn = None  # optional callable: latent tensor → data-space tensor
    decoder_module = None  # keep reference for device placement

    if processor_only:
        if isinstance(example_batch, Batch) and cfg.get("autoencoder_checkpoint"):
            # Mode 1: reconstruct full EPD; encoder+decoder from autoencoder_checkpoint
            log.info(
                "Mode 1 eval: reconstructing full EPD from autoencoder checkpoint "
                "+ processor checkpoint."
            )
            model = setup_epd_model(cfg, stats, datamodule=datamodule)
            processor_sd = _extract_processor_state_dict(checkpoint_payload)
            load_result = model.processor.load_state_dict(processor_sd, strict=True)
        else:
            model = setup_processor_model(cfg, stats, datamodule=datamodule)
            load_result = model.load_state_dict(
                extract_state_dict(checkpoint_payload), strict=True
            )
            if isinstance(example_batch, EncodedBatch):
                # Mode 2: try to load decoder for data-space evaluation
                decoder_module, decode_fn = _try_build_decode_fn(cfg)
                if decode_fn is not None:
                    log.info(
                        "Mode 2 eval: processor (cached latents) + decoder → "
                        "data-space metrics."
                    )
                else:
                    log.info(
                        "Mode fallback: no decoder found — evaluating in latent space."
                    )
    else:
        model = setup_epd_model(cfg, stats, datamodule=datamodule)
        load_result = model.load_state_dict(
            extract_state_dict(checkpoint_payload), strict=True
        )

    if load_result.missing_keys or load_result.unexpected_keys:
        msg = (
            "Checkpoint parameters do not match the instantiated model. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}."
        )
        raise RuntimeError(msg)

    # Get eval parameters from config
    metrics_list = eval_cfg.get("metrics", DEFAULT_EVAL_METRICS)
    batch_indices = eval_cfg.get("batch_indices", [])

    # Get number of ensemble members from config if available
    n_members = cfg.get("model", {}).get("n_members", 1)

    # Setup Fabric accelerator/device management.
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

    # Setup model and loader with Fabric
    log.info("Model configuration n_members: %s", n_members)
    log.info("Model class: %s", type(model))

    is_processor_model = isinstance(model, ProcessorModel)
    model = fabric.setup_module(model)
    _mark_forward_methods_if_available(model, methods=("rollout",))
    model.eval()
    if decoder_module is not None:
        decoder_module.to(fabric.device)
    test_loader = _limit_batches(
        fabric.setup_dataloaders(datamodule.test_dataloader()),
        max_test_batches,
    )

    # Evaluation

    compute_coverage = eval_cfg.get("compute_coverage", False)
    test_metric_fns: dict[str, Callable[[], Metric]] = {}

    metric_registry = dict(AVAILABLE_METRICS)
    has_ensemble = bool(n_members and n_members > 1)
    if has_ensemble:
        metric_registry.update(AVAILABLE_METRICS_ENSEMBLE)

    for name in metrics_list:
        if _should_skip_metric(name):
            log.info("Skipping metric '%s' due to memory cost.", name)
            continue
        if name in AVAILABLE_METRICS:
            test_metric_fns[name] = AVAILABLE_METRICS[name]
        elif name in AVAILABLE_METRICS_ENSEMBLE:
            if has_ensemble:
                test_metric_fns[name] = AVAILABLE_METRICS_ENSEMBLE[name]
            else:
                log.info(
                    "Skipping ensemble metric '%s' because n_members <= 1.",
                    name,
                )
        else:
            log.warning("Metric %s not found in available metrics", name)

    if (n_members > 1) or compute_coverage:

        def coverage_factory() -> Metric:
            return MultiCoverage(coverage_levels=eval_cfg.get("coverage_levels", None))

        test_metric_fns["coverage"] = coverage_factory

    log.info("Computing test metrics: %s", list(test_metric_fns.keys()))

    # Use metric_windows from config (apply to all metrics)
    test_windows = _map_windows(eval_cfg.get("metric_windows", None))

    # Build predict_fn.
    # - Mode 1 / plain EPD: model(batch) already returns decoded tensor; trues
    #   are taken from batch.output_fields by compute_metrics_from_dataloader.
    # - Mode 2: decode both latent predictions and latent ground truth so
    #   metrics are computed in data space.
    # - Latent fallback: return (latent_pred, latent_true) directly.
    predict_fn = _build_eval_predict_fn(
        model,
        is_processor_model=is_processor_model,
        decode_fn=decode_fn,
        n_members=n_members if n_members and n_members > 1 else None,
    )

    test_metrics_results, _, test_per_batch_rows = compute_metrics_from_dataloader(
        dataloader=test_loader,
        metric_fns=test_metric_fns,
        predict_fn=predict_fn,
        windows=test_windows,
        return_per_batch=True,
        device=fabric.device,
    )

    if test_per_batch_rows:
        test_per_batch_rows = _gather_per_batch_rows(test_per_batch_rows, fabric=fabric)

    # Process and save test metrics
    test_rows = _process_metrics_results(
        test_metrics_results,
        per_batch_rows=test_per_batch_rows,  # pyright: ignore[reportArgumentType]
        log_prefix="Test",
        plot_dir=work_dir,
    )

    evaluation_rows: list[dict[str, float | str]] = []
    evaluation_rows.extend(test_rows)
    evaluation_rows.extend(
        _evaluation_metadata_rows(
            checkpoint_payload=checkpoint_payload,
            model=model,  # pyright: ignore[reportArgumentType]
        )
    )
    benchmark_rows = _collect_benchmark_rows(
        eval_cfg=eval_cfg,
        cfg=cfg,
        stats=stats,
        model=model,  # pyright: ignore[reportArgumentType]
        checkpoint_path=checkpoint_path,
        device=str(fabric.device),
        eval_batch_size=eval_batch_size,
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

            rollout_loader = _limit_batches(
                rollout_test_loader,
                max_rollout_batches,
            )
            _render_rollouts(
                model,  # pyright: ignore[reportArgumentType]
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
                decode_fn=decode_fn,
            )

        # Prepare metric functions for rollouts
        rollout_metric_fns: dict[str, Callable[[], Metric]] = {}

        if compute_rollout_metrics:
            for name in metrics_list:
                if _should_skip_metric(name):
                    log.info("Skipping rollout metric '%s' due to memory cost.", name)
                    continue
                if name in AVAILABLE_METRICS:
                    rollout_metric_fns[name] = AVAILABLE_METRICS[name]
                elif name in AVAILABLE_METRICS_ENSEMBLE:
                    if has_ensemble:
                        rollout_metric_fns[name] = AVAILABLE_METRICS_ENSEMBLE[name]
                    else:
                        log.info(
                            "Skipping ensemble rollout metric '%s' because "
                            "n_members <= 1.",
                            name,
                        )
                else:
                    msg = f"Metric {name} not found in available metrics"
                    log.warning(msg)

        if compute_rollout_coverage and n_members and n_members > 1:
            log.info("Adding rollout coverage to metrics...")

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

            def rollout_predict(batch):
                preds, trues = model.rollout(
                    batch,
                    stride=rollout_stride,
                    max_rollout_steps=max_rollout_steps,
                    free_running_only=eval_cfg.get("free_running_only", True),
                    n_members=n_members if n_members and n_members > 1 else None,
                )
                if decode_fn is not None:
                    with torch.no_grad():
                        preds = _decode_tensor(
                            preds,
                            decode_fn,
                            n_members=n_members
                            if n_members and n_members > 1
                            else None,
                        )
                        if trues is not None:
                            trues = _decode_tensor(trues, decode_fn, n_members=None)
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

            # Process and log results
            rollout_csv_rows = _process_metrics_results(
                rollout_metrics_per_window,
                per_batch_rows=rollout_per_batch_rows,  # pyright: ignore[reportArgumentType]
                log_prefix="Rollout",
                plot_dir=csv_path.parent,
            )

            # Save rollout metrics to CSV
            rollout_csv_path = csv_path.parent / "rollout_metrics.csv"
            rollout_combined_rows = [*rollout_csv_rows]
            rollout_metric_rows, rollout_metadata_rows = (
                _split_metric_and_metadata_rows(rollout_combined_rows)
            )

            if fabric.global_rank == 0 and rollout_metric_rows:
                _write_csv(rollout_metric_rows, rollout_csv_path)
                log.info("Wrote rollout metrics to %s", rollout_csv_path)

            rollout_metadata_csv_path = csv_path.parent / "rollout_metadata.csv"
            if fabric.global_rank == 0 and rollout_metadata_rows:
                _write_csv(rollout_metadata_rows, rollout_metadata_csv_path)
                log.info("Wrote rollout metadata to %s", rollout_metadata_csv_path)

            # Per-timestep, per-channel rollout metrics (rows=metrics, cols=timestep)
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
                        per_timestep_metric_fns[name] = (
                            _build_per_timestep_metric_factory(metric_cls)
                        )

            if per_timestep_metric_fns:
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
                if per_timestep_results and fabric.global_rank == 0:
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
                        out_path = (
                            csv_path.parent
                            / f"rollout_metrics_per_timestep_channel_{c}.csv"
                        )
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
                    out_path_all = (
                        csv_path.parent / "rollout_metrics_per_timestep_channel_all.csv"
                    )
                    df_all.to_csv(out_path_all)
                    log.info(
                        "Wrote rollout metrics per timestep (channel all) to %s",
                        out_path_all,
                    )

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


if __name__ == "__main__":
    main()
