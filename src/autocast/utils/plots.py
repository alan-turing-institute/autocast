from collections.abc import Callable, Iterable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib import animation
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from torchmetrics import Metric

from autocast.metrics.coverage import MultiCoverage
from autocast.types import Tensor, TensorBTSC, TensorBTSCM


def plot_spatiotemporal_video(  # noqa: PLR0915, PLR0912
    true: TensorBTSC,
    pred: TensorBTSC | None = None,
    pred_uq: TensorBTSC | None = None,
    batch_idx: int = 0,
    fps: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    save_path: str | None = None,
    title: str = "Ground Truth vs Prediction",
    pred_uq_label: str = "Prediction UQ",
    colorbar_mode: Literal["none", "row", "column", "all"] = "none",
    colorbar_mode_uq: Literal["none", "row"] = "none",
    channel_names: list[str] | None = None,
):
    """Create a video comparing ground truth and predicted spatiotemporal time series.

    Parameters
    ----------
    true: array_like (B, T, W, H, C)
        Ground-truth tensor.
    pred: array_like
        Optional predicted tensor of shape (B, T, W, H, C).
    batch_idx: int
        Which batch index to visualize (default: 0).
    fps: int, optional
        Frames per second for the video (default: 5).
    vmin: float, optional
        Minimum value for color scale (default: auto from data).
    vmax: float, optional
        Maximum value for color scale (default: auto from data).
    cmap: str, optional
        Colormap to use (default: "viridis").
    save_path: str, optional
        Optional path to save the video (e.g., "output.mp4").
    title: str, optional
        Title for the video (default: "Ground Truth vs Prediction").
    colorbar_mode: {"none", "row", "column", "all"}
        Select how colorbars (and underlying color scales) are shared for the
        first two rows (true vs prediction):
        - "none": every subplot gets its own colorbar (default).
        - "row": a single colorbar per row (first two rows only).
        - "column": a single colorbar per column (true/pred share per channel).
        - "all": one colorbar shared across the first two rows.
    channel_names: list[str] | None
        Optional list of channel names for titles.

    Returns
    -------
    animation.FuncAnimation
        Animation object that can be displayed in notebooks.
    """
    colorbar_mode_str = colorbar_mode.lower()
    valid_modes = {"none", "row", "column", "all"}
    if colorbar_mode_str not in valid_modes:
        raise ValueError(
            "Invalid colorbar_mode "
            f"'{colorbar_mode}'. Expected one of {sorted(valid_modes)}."
        )

    true_batch = true[batch_idx]
    pred_batch = pred[batch_idx] if pred is not None else None
    pred_uq_batch = pred_uq[batch_idx] if pred_uq is not None else None

    # Extract dims and move to CPU
    T, *_, C = true_batch.shape
    true_batch = true_batch.detach().cpu().numpy()
    pred_batch = pred_batch.detach().cpu().numpy()
    if pred_uq_batch is not None:
        pred_uq_batch = pred_uq_batch.detach().cpu().numpy()

    # Calculate difference
    diff_batch = true_batch - pred_batch

    # Set-up rows
    primary_rows = [true_batch, pred_batch]
    n_primary_rows = len(primary_rows)

    def _range_from_arrays(arrays):
        min_val = vmin if vmin is not None else min(float(arr.min()) for arr in arrays)
        max_val = vmax if vmax is not None else max(float(arr.max()) for arr in arrays)
        return min_val, max_val

    norms: list[list[Normalize | None]] = [[None] * C for _ in range(n_primary_rows)]

    if colorbar_mode_str == "column":
        for ch in range(C):
            channel_arrays = [row[:, :, :, ch] for row in primary_rows]
            min_val, max_val = _range_from_arrays(channel_arrays)
            norm = Normalize(vmin=min_val, vmax=max_val)
            for row_idx in range(n_primary_rows):
                norms[row_idx][ch] = norm
    elif colorbar_mode_str == "row":
        for row_idx, row in enumerate(primary_rows):
            min_val, max_val = _range_from_arrays([row])
            norm = Normalize(vmin=min_val, vmax=max_val)
            for ch in range(C):
                norms[row_idx][ch] = norm
    elif colorbar_mode_str == "all":
        min_val, max_val = _range_from_arrays(primary_rows)
        norm = Normalize(vmin=min_val, vmax=max_val)
        for row_idx in range(n_primary_rows):
            for ch in range(C):
                norms[row_idx][ch] = norm
    else:  # "none"
        for row_idx, row in enumerate(primary_rows):
            for ch in range(C):
                min_val, max_val = _range_from_arrays([row[:, :, :, ch]])
                norms[row_idx][ch] = Normalize(vmin=min_val, vmax=max_val)

    diff_max = float(np.abs(diff_batch).max())
    diff_span = diff_max if diff_max > 0 else 1e-9
    diff_norm = TwoSlopeNorm(vmin=-diff_span, vcenter=0, vmax=diff_span)

    rows_to_plot: list[tuple[np.ndarray | Tensor | None, str, str]] = [
        (true_batch, "Ground Truth", cmap),
    ]
    if pred is not None:
        rows_to_plot.append((pred_batch, "Prediction", cmap))
        rows_to_plot.append((diff_batch, "Difference (True - Pred)", "RdBu"))
    if pred_uq is not None:
        rows_to_plot.append((pred_uq_batch, pred_uq_label, "inferno"))
    total_rows = len(rows_to_plot)

    fig = plt.figure(figsize=(C * 4, total_rows * 4))
    gs = GridSpec(total_rows, C, figure=fig, hspace=0.3, wspace=0.3)

    axes = []
    images = []

    for row_idx, (data, row_label, row_cmap) in enumerate(rows_to_plot):
        row_axes = []
        row_images = []

        for ch in range(C):
            ax = fig.add_subplot(gs[row_idx, ch])

            if data is None:
                msg = "Data for plotting cannot be None."
                raise ValueError(msg)
            frame0 = rearrange(data[0, :, :, ch], "w h -> h w")

            if row_idx < n_primary_rows:
                norm = norms[row_idx][ch]
            elif row_idx == len(rows_to_plot) - 1 and pred_uq_batch is not None:
                uq_min = (
                    float(pred_uq_batch[..., ch].min())
                    if colorbar_mode_uq == "none"
                    else float(pred_uq_batch.min())
                )
                uq_max = (
                    float(pred_uq_batch[..., ch].max())
                    if colorbar_mode_uq == "none"
                    else float(pred_uq_batch.max())
                )
                uq_norm = Normalize(vmin=uq_min, vmax=uq_max)
                norm = uq_norm
            else:
                norm = diff_norm
            im = ax.imshow(frame0, cmap=row_cmap, aspect="auto", norm=norm)

            if row_idx == 0:
                (
                    ax.set_title(f"Channel {ch}")
                    if channel_names is None
                    else ax.set_title(f"{channel_names[ch]}")
                )
            if ch == 0:
                ax.set_ylabel(row_label)

            row_axes.append(ax)
            row_images.append(im)

        axes.append(row_axes)
        images.append(row_images)

    def _attach_colorbars():
        for row_idx, row_axes in enumerate(axes):
            for ch_idx, ax in enumerate(row_axes):
                fig.colorbar(
                    images[row_idx][ch_idx],
                    ax=ax,
                    fraction=0.046,
                    pad=0.04,
                )

    _attach_colorbars()

    suptitle_text = fig.suptitle("", fontsize=14, fontweight="bold")

    def update(frame):
        for ch in range(C):
            images[0][ch].set_array(true_batch[frame, :, :, ch])
            if pred is not None:
                images[1][ch].set_array(pred_batch[frame, :, :, ch])
                images[2][ch].set_array(diff_batch[frame, :, :, ch])
            if pred_uq_batch is not None:
                images[3][ch].set_array(pred_uq_batch[frame, :, :, ch])
        suptitle_text.set_text(
            f"{title} - Batch {batch_idx} - Time Step: {frame}/{T - 1}"
        )
        return [img for row in images for img in row] + [suptitle_text]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=False, repeat=True
    )

    if save_path:
        Writer = (
            animation.writers["pillow"]
            if save_path.endswith(".gif")
            else animation.writers["ffmpeg"]
        )
        writer = Writer(fps=fps, metadata={"artist": "autoemulate"}, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Video saved to {save_path}")

    plt.close()
    return anim


def compute_metrics_from_dataloader(
    dataloader: Iterable,
    metric_fns: dict[str, Callable[[], Metric]],
    predict_fn: Callable,
    windows: list[tuple[int, int] | None] | None = None,
    return_tensors: bool = False,
    return_per_batch: bool = False,
) -> tuple[
    dict[None | tuple[int, int], dict[str, Metric]],
    tuple[TensorBTSCM, TensorBTSC] | None,
    list[dict[str, float | str]] | None,
]:
    """
    Compute metrics from a dataloader by running model forward passes.

    Parameters
    ----------
    dataloader: Iterable
        DataLoader that yields batches.
    metric_fns: dict[str, Callable[[], Metric]]
        Dictionary of functions that return fresh metric instances, keyed by metric
        name.
    predict_fn: Callable
        Custom function (batch) -> (preds, trues) for cases like rollout or simply
        the model forward. Should return a tuple of (preds, trues) tensors or a
        single tensor of predictions (in which case trues will be taken from batch).
    windows: list[tuple[int, int] | None], optional
        List of (t_start, t_end) windows to evaluate. None means use all timesteps.
        If multiple windows provided, evaluates each independently.
    return_tensors: bool
        If True, also return concatenated (pred, true) tensors.
    return_per_batch: bool
        If True, also return a list of dictionaries containing metrics for each batch.

    Returns
    -------
    tuple[
        dict[None | tuple[int, int], dict[str, Metric]],
        tuple[TensorBTSCM, TensorBTSC] | None,
        list[dict[str, float | str]] | None,
    ]
        The populated metrics, optionally the tensors, and optionally per-batch metrics.
    """
    metrics_per_window = {
        window: {name: fn() for name, fn in metric_fns.items()}
        for window in (windows or [None])
    }
    all_preds = [] if return_tensors else None
    all_trues = [] if return_tensors else None
    per_batch_rows = [] if return_per_batch else None

    def _get_val(m):
        """Extract scalar values safely."""
        try:
            val = m.compute()
            if val.numel() == 1:
                return float(val.item())
            if hasattr(val, "mean"):
                return float(val.mean().item())
        except Exception:
            pass
        return None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            result = predict_fn(batch)
            if result is None:
                continue

            # Extract preds/trues
            preds, trues = (
                result
                if (isinstance(result, tuple) and len(result) == 2)
                # TODO: consider making generic over batch type with outputs() method
                else (result, getattr(batch, "output_fields", None))
            )
            if not (isinstance(preds, Tensor) and isinstance(trues, Tensor)):
                continue

            # Move to CPU for metric computation
            preds, trues = preds.cpu(), trues.cpu()

            # Get metrics for each window
            for window, metrics_dict in metrics_per_window.items():
                p, t = (
                    (preds, trues)
                    if window is None
                    else (
                        preds[:, window[0] : window[1]],
                        trues[:, window[0] : window[1]],
                    )
                )
                if p.numel() == 0 or t.numel() == 0:
                    continue

                # Update accumulated metrics
                for metric in metrics_dict.values():
                    metric.update(p, t)

                # Store tensors if requested
                if all_preds is not None:
                    all_preds.append(p)
                if all_trues is not None:
                    all_trues.append(t)

                # Per-batch metrics
                if per_batch_rows is not None:
                    row = {
                        "window": f"{window[0]}-{window[1]}" if window else "all",
                        "batch_idx": batch_idx,
                    }
                    for name, fn in metric_fns.items():
                        m = fn()
                        m.update(p, t)
                        if (val := _get_val(m)) is not None:
                            row[name] = val
                    per_batch_rows.append(row)

    # Concatenate tensors if needed
    tensors = None
    if all_preds is not None and all_trues is not None:
        tensors = (torch.cat(all_preds, dim=0), torch.cat(all_trues, dim=0))

    return metrics_per_window, tensors, per_batch_rows


def compute_coverage_scores_from_dataloader(
    dataloader: Iterable,
    predict_fn: Callable,
    coverage_levels: list[float] | None = None,
    windows: list[tuple[int, int] | None] | None = None,
    return_tensors: bool = False,
) -> tuple[
    dict[None | tuple[int, int], MultiCoverage], tuple[TensorBTSCM, TensorBTSC] | None
]:
    """
    Compute coverage scores from a dataloader by running model forward passes.

    Parameters
    ----------
    dataloader: DataLoader
        DataLoader that yields batches.
    model: nn.Module, optional
        Model with forward(batch) that returns predictions with ensemble dimension.
        Either model or predict_fn must be provided.
    predict_fn: Callable, optional
        Custom function (batch) -> (preds, trues) for cases like rollout.
        Either model or predict_fn must be provided.
    coverage_levels: list[float], optional
        Coverage levels to evaluate (default: 0.05 to 0.95).
    windows: list[tuple[int, int] | None], optional
        List of (t_start, t_end) windows to evaluate. None means use all timesteps.
        If multiple windows provided, evaluates each independently.
    return_tensors: bool
        If True, also return concatenated (pred, true) tensors.

    Returns
    -------
    tuple[
        dict[None | tuple[int, int], MultiCoverage],
        tuple[TensorBTSCM, TensorBTSC] | None,
    ]
        The populated MultiCoverage metric and optionally the tensors.
    """
    coverage_levels_ = (
        coverage_levels or np.linspace(0.05, 0.95, 10, endpoint=True).tolist()
    )

    def metric_fn() -> MultiCoverage:
        return MultiCoverage(coverage_levels=coverage_levels_)

    metrics_per_window_dict, tensors, _ = compute_metrics_from_dataloader(
        dataloader=dataloader,
        metric_fns={"coverage": metric_fn},
        predict_fn=predict_fn,
        windows=windows,
        return_tensors=return_tensors,
        return_per_batch=False,
    )

    metrics_per_window = {k: v["coverage"] for k, v in metrics_per_window_dict.items()}

    return metrics_per_window, tensors  # type: ignore since only metric is coverage


def plot_coverage(
    pred: TensorBTSCM,
    true: TensorBTSC,
    coverage_levels: list[float] | None = None,
    save_path: str | None = None,
    title: str = "Coverage plot",
):
    """
    Plot reliability diagram showing expected vs observed coverage.

    This is a convenience wrapper around MultiCoverage.plot().

    Parameters
    ----------
    pred: TensorBTSCM
        Ensemble predictions (last dimension is ensemble members).
    true: TensorBTSC
        Ground truth tensor.
    coverage_levels: list[float], optional
        Coverage levels to evaluate (default: 0.05 to 0.95).
    save_path: str, optional
        Path to save the plot.
    title: str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    coverage_levels_ = (
        coverage_levels or np.linspace(0.05, 0.95, 10, endpoint=True).tolist()
    )

    # Create metric, update with data, and plot
    metric = MultiCoverage(coverage_levels=coverage_levels_)
    metric.update(pred, true)
    return metric.plot(save_path=save_path, title=title)
