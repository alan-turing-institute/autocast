from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from matplotlib import animation
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec


def plot_spatiotemporal_video(  # noqa: PLR0915, PLR0912
    true,
    pred,
    batch_idx=0,
    fps=5,
    vmin=None,
    vmax=None,
    cmap="viridis",
    save_path=None,
    title="Ground Truth vs Prediction",
    colorbar_mode: Literal["none", "row", "column", "all"] = "none",
    channel_names=None,
):
    """Create a video comparing ground truth and predicted spatiotemporal time series.

    Parameters
    ----------
    true: array_like (B, T, W, H, C)
        Ground-truth tensor.
    pred: array_like
        Predicted tensor of shape (B, T, W, H, C).
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
    pred_batch = pred[batch_idx]

    T, _, _, C = true_batch.shape

    if hasattr(true_batch, "detach"):
        true_batch = true_batch.detach().cpu().numpy()
        pred_batch = pred_batch.detach().cpu().numpy()

    diff_batch = true_batch - pred_batch

    primary_rows = [true_batch, pred_batch]
    n_primary_rows = len(primary_rows)

    def _range_from_arrays(arrays):
        min_val = vmin if vmin is not None else min(float(arr.min()) for arr in arrays)
        max_val = vmax if vmax is not None else max(float(arr.max()) for arr in arrays)
        return min_val, max_val

    norms = [[None] * C for _ in range(n_primary_rows)]

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

    rows_to_plot = [
        (true_batch, "Ground Truth", cmap),
        (pred_batch, "Prediction", cmap),
        (diff_batch, "Difference (True - Pred)", "RdBu"),
    ]
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

            frame0 = rearrange(data[0, :, :, ch], "w h -> h w")

            im = ax.imshow(
                frame0,
                cmap=row_cmap,
                aspect="auto",
                norm=norms[row_idx][ch] if row_idx < n_primary_rows else diff_norm,
            )

            if row_idx == 0:
                ax.set_title(
                    f"Channel {ch}"
                ) if channel_names is None else ax.set_title(f"{channel_names[ch]}")
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

    fig.suptitle(f"{title} - Batch {batch_idx}", fontsize=14, fontweight="bold")

    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=12)

    def update(frame):
        for ch in range(C):
            images[0][ch].set_array(true_batch[frame, :, :, ch])
            images[1][ch].set_array(pred_batch[frame, :, :, ch])
            images[2][ch].set_array(diff_batch[frame, :, :, ch])
        time_text.set_text(f"Time Step: {frame}/{T - 1}")
        return [img for row in images for img in row] + [time_text]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=True, repeat=True
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
