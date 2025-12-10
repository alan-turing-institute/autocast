import matplotlib.pyplot as plt
from einops import rearrange
from matplotlib import animation
from matplotlib.gridspec import GridSpec


def plot_spatiotemporal_video(
    true,
    pred,
    batch_idx=0,
    fps=5,
    vmin=None,
    vmax=None,
    cmap="viridis",
    save_path=None,
    title="Ground Truth vs Prediction",
):
    """Create a video comparing ground truth and predicted spatiotemporal time series.

    Parameters
    ----------
    true: array_like
        Ground truth tensor of shape (B, T, W, H, C)
    pred: array_like
        Predicted tensor of shape (B, T, W, H, C)
    batch_idx: int
        Which batch index to visualize (default: 0)
    fps: int, optional
        Frames per second for the video (default: 5)
    vmin: float, optional
        Minimum value for color scale (default: auto from data)
    vmax: float, optional
        Maximum value for color scale (default: auto from data)
    cmap: str, optional
        Colormap to use (default: "viridis")
    save_path: str, optional
        Optional path to save the video (e.g., "output.mp4")
    title: str, optional
        Title for the video (default: "Ground Truth vs Prediction")

    Returns
    -------
    animation.FuncAnimation
        Animation object that can be displayed in notebooks
    """
    # Extract the selected batch
    true_batch = true[batch_idx]  # (T, W, H, C)
    pred_batch = pred[batch_idx]  # (T, W, H, C)

    T, _W, _H, C = true_batch.shape

    # Convert to numpy if needed
    if hasattr(true_batch, "detach"):
        true_batch = true_batch.detach().cpu().numpy()
        pred_batch = pred_batch.detach().cpu().numpy()

    # Determine color scale limits
    if vmin is None:
        vmin = min(true_batch.min(), pred_batch.min())
    if vmax is None:
        vmax = max(true_batch.max(), pred_batch.max())

    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(C * 4, 8))
    gs = GridSpec(2, C, figure=fig, hspace=0.3, wspace=0.3)

    # Create subplots: 2 rows (true, pred) x C columns (channels)
    axes = []
    images = []

    for row_idx, (data, row_label) in enumerate(
        [(true_batch, "Ground Truth"), (pred_batch, "Prediction")]
    ):
        row_axes = []
        row_images = []
        for ch_idx in range(C):
            ax = fig.add_subplot(gs[row_idx, ch_idx])

            # Initialize with first time step
            im = ax.imshow(
                rearrange(data[0, :, :, ch_idx], "w h -> h w"),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                aspect="auto",
            )

            # Set labels
            if ch_idx == 0:
                ax.set_ylabel(row_label, fontsize=12)

            if row_idx == 0:
                ax.set_title(f"Channel {ch_idx}", fontsize=10)

            if row_idx == 1:
                ax.set_xlabel("W", fontsize=9)

            if ch_idx == 0:
                ax.set_ylabel(f"{row_label}\nH", fontsize=9)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            row_axes.append(ax)
            row_images.append(im)

        axes.append(row_axes)
        images.append(row_images)

    # Add main title
    fig.suptitle(f"{title} - Batch {batch_idx}", fontsize=14, fontweight="bold")

    # Time text
    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=12)

    def update(frame):
        """Update function for animation."""
        # Update ground truth row
        for ch_idx in range(C):
            images[0][ch_idx].set_array(true_batch[frame, :, :, ch_idx])
            images[1][ch_idx].set_array(pred_batch[frame, :, :, ch_idx])

        time_text.set_text(f"Time Step: {frame}/{T - 1}")

        return [item for sublist in images for item in sublist] + [time_text]

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=True, repeat=True
    )

    # Save if path provided
    if save_path:
        Writer = (
            animation.writers["pillow"]
            if save_path.endswith(".gif")
            else animation.writers["ffmpeg"]
        )
        writer = Writer(fps=fps, metadata={"artist": "autocast"}, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Video saved to {save_path}")

    plt.close()

    return anim
