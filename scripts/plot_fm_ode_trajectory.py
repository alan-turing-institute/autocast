"""Plot the FM ODE trajectory for one window from a flow-matching EPD run.

Reconstructs the ambient-space EPD model from a processor-only checkpoint by
re-using the eval pipeline's autoencoder injection + datamodule swap, then
runs the flow-matching ODE step-by-step (Euler), capturing each intermediate
latent. Each intermediate is decoded + denormalised and saved as one image
tile per (intermediate, output-step, channel) so the panels can be assembled
in Inkscape into something like the mock-up in methods_sketch_00.pdf.

Example:
    python scripts/plot_fm_ode_trajectory.py \
        --run-dir outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3 \
        --batch-index 0 --sample-index 0 --use-ema \
        --intermediate-stride 5 --channel 0

Deeper rollout (visualise FM ODE at the 12th 4-step window, i.e. after 48 steps):

    python scripts/plot_fm_ode_trajectory.py \
        --rollout-window 12 --use-ema --intermediate-stride 10 --channel 0

Per-channel colormaps (smoke=viridis, u/v=RdBu_r):

    python scripts/plot_fm_ode_trajectory.py --use-ema --all-channels \
        --cmap viridis,RdBu_r,RdBu_r --symmetric-cmaps RdBu_r
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.scripts.eval.encoder_processor_decoder import (
    _maybe_inject_encoder_decoder_from_autoencoder_checkpoint,
    _maybe_swap_to_ambient_datamodule,
    _extract_processor_state_dict,
)
from autocast.scripts.execution import load_checkpoint_payload
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.types import Batch

log = logging.getLogger("plot_fm_ode_trajectory")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3"
        ),
        help="Run directory containing resolved_config.yaml and processor.ckpt.",
    )
    p.add_argument(
        "--checkpoint-name",
        default="processor.ckpt",
        help="Checkpoint filename inside --run-dir.",
    )
    p.add_argument(
        "--config-name",
        default="resolved_config.yaml",
        help="Config filename inside --run-dir.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for tiles. Defaults to <run-dir>/figures/fm_ode_traj."
        ),
    )
    p.add_argument("--batch-index", type=int, default=0, help="Test batch to draw.")
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample within the chosen batch.",
    )
    p.add_argument(
        "--intermediate-stride",
        type=int,
        default=5,
        help="Save every Nth ODE step (plus first and last).",
    )
    p.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel to render (0=smoke, 1=u, 2=v for CNS).",
    )
    p.add_argument(
        "--all-channels",
        action="store_true",
        help="Render all output channels (overrides --channel).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device, e.g. 'cuda' or 'cpu' (default: auto).",
    )
    p.add_argument("--use-ema", action="store_true", help="Use EMA processor weights.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the noise z0.")
    p.add_argument(
        "--format",
        default="pdf",
        choices=("pdf", "png"),
        help="Output file format per tile.",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help=(
            "Matplotlib colormap. Either a single name applied to all rendered "
            "channels, or a comma-separated list aligned with channel order, "
            "e.g. 'viridis,RdBu_r,RdBu_r' for smoke,u,v."
        ),
    )
    p.add_argument(
        "--symmetric-cmaps",
        default="",
        help=(
            "Comma-separated list of colormap names that should use symmetric "
            "limits around zero (e.g. 'RdBu_r,seismic'). Empty by default."
        ),
    )
    p.add_argument(
        "--rollout-window",
        type=int,
        default=0,
        help=(
            "Output window (0-indexed, length n_steps_output frames each) at "
            "which to capture the FM ODE intermediates. 0 = the first window "
            "(uses the test_dataloader). >0 = autoregressive rollout to that "
            "window first; uses the rollout_test_dataloader so ground truth "
            "is available for the deeper window too."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
    )
    return p.parse_args()


def resolve_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_sample(batch: Batch, idx: int) -> Batch:
    """Return a length-1 Batch containing only sample `idx`."""

    def _take(t: torch.Tensor | None) -> torch.Tensor | None:
        return t[idx : idx + 1] if t is not None else None

    return Batch(
        input_fields=batch.input_fields[idx : idx + 1].clone(),
        output_fields=batch.output_fields[idx : idx + 1].clone(),
        constant_scalars=_take(batch.constant_scalars),
        constant_fields=_take(batch.constant_fields),
        boundary_conditions=_take(batch.boundary_conditions),
    )


def to_device(batch: Batch, device: torch.device) -> Batch:
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
        boundary_conditions=(
            batch.boundary_conditions.to(device)
            if batch.boundary_conditions is not None
            else None
        ),
    )


@torch.no_grad()
def run_fm_with_intermediates(
    processor: FlowMatchingProcessor,
    z_input: torch.Tensor,
    global_cond: torch.Tensor | None,
    capture_every: int,
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """Re-implements ``FlowMatchingProcessor.map`` capturing intermediate z.

    Returns the final z (B, T_out, *spatial, C_out) and a list of
    (step_index, z_snapshot) including the initial noise (step=0) and the
    final state (step=flow_ode_steps).
    """
    batch_size = z_input.shape[0]
    device, dtype = z_input.device, z_input.dtype

    spatial_shape = tuple(z_input.shape[2:-1])
    z_shape = (
        batch_size,
        processor.n_steps_output,
        *spatial_shape,
        processor.n_channels_out,
    )
    z = torch.randn(z_shape, device=device, dtype=dtype)
    t = torch.zeros(batch_size, device=device, dtype=dtype)

    snapshots: list[tuple[int, torch.Tensor]] = [(0, z.clone())]

    dt = torch.tensor(1.0 / processor.flow_ode_steps, device=device, dtype=dtype)
    for k in range(processor.flow_ode_steps):
        v = processor.flow_field(z, t, z_input, global_cond)
        z = z + dt * v
        t = t + dt
        step = k + 1
        if step == processor.flow_ode_steps or step % capture_every == 0:
            snapshots.append((step, z.clone()))

    return z, snapshots


def latents_to_ambient(
    model, z_latent: torch.Tensor, denormalise: bool = True
) -> torch.Tensor:
    """Decode latent (B, T_out, *spatial, C_lat) → ambient + denormalise."""
    decoded = model.encoder_decoder.decoder.decode(z_latent)
    if denormalise:
        decoded = model.denormalize_tensor(decoded)
    return decoded


def save_tile(
    image: np.ndarray,
    out_path: Path,
    *,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
    dpi: int,
) -> None:
    """Render a 2-D array as a square axis-free image tile."""
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def channel_indices(args: argparse.Namespace, n_channels: int) -> list[int]:
    if args.all_channels:
        return list(range(n_channels))
    if args.channel >= n_channels:
        msg = f"--channel={args.channel} but tensor has only {n_channels} channels."
        raise ValueError(msg)
    return [args.channel]


def parse_cmap_list(spec: str, channels: list[int]) -> dict[int, str]:
    """Map each rendered channel index to a colormap name.

    `spec` is either a single name (applied to all channels) or a
    comma-separated list whose length matches len(channels).
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) == 0:
        msg = "--cmap must not be empty."
        raise ValueError(msg)
    if len(parts) == 1:
        return {c: parts[0] for c in channels}
    if len(parts) != len(channels):
        msg = (
            f"--cmap='{spec}' has {len(parts)} entries but {len(channels)} channels "
            f"are being rendered ({channels})."
        )
        raise ValueError(msg)
    return dict(zip(channels, parts, strict=True))


def parse_symmetric_set(spec: str) -> set[str]:
    return {p.strip() for p in spec.split(",") if p.strip()}


@torch.no_grad()
def advance_batch_free_running(model, batch: Batch) -> Batch:
    """Free-running step: feed the model's predicted last frame back as input.

    Mirrors ``EncoderProcessorDecoder._advance_batch`` for the case where
    stride == n_steps_output (the typical FM setup): the new input is the
    last ``n_steps_input`` frames of the prediction. Constants, BCs, and
    global cond carriers are preserved.
    """
    pred = model(batch)
    n_in = batch.input_fields.shape[1]
    new_input = pred[:, -n_in:, ...].detach()
    return Batch(
        input_fields=new_input,
        output_fields=batch.output_fields,
        constant_scalars=batch.constant_scalars,
        constant_fields=batch.constant_fields,
        boundary_conditions=batch.boundary_conditions,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    run_dir: Path = args.run_dir.resolve()
    cfg_path = run_dir / args.config_name
    ckpt_path = run_dir / args.checkpoint_name
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else run_dir / "figures" / "fm_ode_traj"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading config from %s", cfg_path)
    cfg = OmegaConf.load(cfg_path)
    if not isinstance(cfg, DictConfig):
        msg = f"Expected DictConfig from {cfg_path}, got {type(cfg).__name__}"
        raise TypeError(msg)

    # Eval-style augmentations: we want the ambient datamodule + injected AE.
    eval_block = cfg.get("eval")
    if eval_block is None or eval_block.get("checkpoint") is None:
        # Mirror the eval CLI's resolution: `eval.checkpoint` references the
        # processor checkpoint relative to the run dir.
        with open_dict(cfg):
            if "eval" not in cfg:
                cfg.eval = OmegaConf.create({})
            cfg.eval.checkpoint = str(ckpt_path)
    if cfg.get("autoencoder_checkpoint") is None:
        # In Mode 1 eval the eval CLI prints the autoencoder path but the
        # resolved_config.yaml may still record null. Try the eval-config
        # sibling file as a backup.
        eval_cfg_path = run_dir / "eval" / "resolved_eval_config.yaml"
        if eval_cfg_path.exists():
            eval_cfg = OmegaConf.load(eval_cfg_path)
            ae_ckpt = eval_cfg.get("autoencoder_checkpoint") if isinstance(
                eval_cfg, DictConfig
            ) else None
            if ae_ckpt is not None:
                with open_dict(cfg):
                    cfg.autoencoder_checkpoint = ae_ckpt
                log.info("Picked up autoencoder_checkpoint from %s", eval_cfg_path)

    if cfg.get("autoencoder_checkpoint") is None:
        msg = (
            "Could not resolve autoencoder_checkpoint from resolved_config.yaml or "
            "eval/resolved_eval_config.yaml; pass via Hydra override would be nice "
            "but for this script we just abort."
        )
        raise RuntimeError(msg)

    # Inject AE encoder/decoder configs (Mode 1 of eval).
    cfg = _maybe_inject_encoder_decoder_from_autoencoder_checkpoint(cfg)

    # Build a probe datamodule first so we can detect that the cached-latent
    # datamodule is the active one and trigger the swap to ambient.
    probe_datamodule, cfg, probe_stats = setup_datamodule(cfg)
    cfg = _maybe_swap_to_ambient_datamodule(
        cfg,
        eval_mode="ambient",
        example_batch=probe_stats.get("example_batch"),
    )
    datamodule, cfg, stats = setup_datamodule(cfg)
    log.info("Final example batch type: %s", type(stats["example_batch"]).__name__)

    # Build the EPD with AE weights loaded; then load processor weights.
    model = setup_epd_model(cfg, stats, datamodule=datamodule)
    log.info("Loading processor checkpoint from %s", ckpt_path)
    payload = load_checkpoint_payload(ckpt_path)
    proc_sd = _extract_processor_state_dict(payload, use_ema=args.use_ema)
    load_result = model.processor.load_state_dict(proc_sd, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        msg = (
            f"Processor load mismatch: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
        raise RuntimeError(msg)

    if not isinstance(model.processor, FlowMatchingProcessor):
        msg = (
            "This script targets FlowMatchingProcessor specifically; "
            f"got {type(model.processor).__name__}."
        )
        raise TypeError(msg)

    device = resolve_device(args.device)
    log.info("Using device: %s", device)
    model.to(device)
    model.eval()

    # Choose the dataloader. Window 0 is fine on the standard test loader
    # (n_steps_output frames of GT). Deeper windows need the rollout loader,
    # which yields a single window per trajectory in full_trajectory_mode so
    # we get GT for every output frame in the trajectory.
    datamodule.setup(stage="test")
    if args.rollout_window <= 0:
        loader = datamodule.test_dataloader()
        loader_name = "test_dataloader"
    else:
        loader = datamodule.rollout_test_dataloader()
        loader_name = "rollout_test_dataloader"
    log.info("Using %s for window=%s", loader_name, args.rollout_window)

    batch = None
    seen = 0
    for i, b in enumerate(loader):
        seen = i + 1
        if i == args.batch_index:
            batch = b
            break
    if batch is None:
        msg = (
            f"{loader_name} exhausted before batch_index={args.batch_index} "
            f"(loader has {seen} batches)."
        )
        raise IndexError(msg)
    if not isinstance(batch, Batch):
        raise TypeError(f"Expected ambient Batch, got {type(batch).__name__}")
    if args.sample_index >= batch.input_fields.shape[0]:
        msg = (
            f"--sample-index={args.sample_index} but batch only has "
            f"{batch.input_fields.shape[0]} samples."
        )
        raise IndexError(msg)

    sample = select_sample(batch, args.sample_index)
    sample = to_device(sample, device)

    n_t_out = int(model.processor.n_steps_output)
    n_t_in = sample.input_fields.shape[1]
    if args.rollout_window > 0:
        # In full-trajectory mode `output_fields` holds the entire GT
        # trajectory after the initial input, so we can slice the GT for
        # any output window. For the visualised window we also need the
        # *predicted* input frame at that point — produce it via free-
        # running rollout for the preceding (rollout_window) windows.
        gt_start = args.rollout_window * n_t_out
        gt_end = gt_start + n_t_out
        if gt_end > sample.output_fields.shape[1]:
            msg = (
                f"--rollout-window={args.rollout_window} requires GT frames "
                f"up to index {gt_end - 1}, but the trajectory only has "
                f"{sample.output_fields.shape[1]} output frames."
            )
            raise IndexError(msg)
        target_fields_window = sample.output_fields[:, gt_start:gt_end, ...]

        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        log.info(
            "Free-running rollout for %s windows before capture...",
            args.rollout_window,
        )
        current = sample
        for _ in range(args.rollout_window):
            current = advance_batch_free_running(model, current)
        capture_sample = current
    else:
        target_fields_window = sample.output_fields
        capture_sample = sample
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    # Encode → latent input + global conditioning at the chosen window.
    with torch.no_grad():
        z_input, global_cond = model.encoder_decoder.encoder.encode_with_cond(
            capture_sample
        )
    log.info(
        "Encoded latent shape: %s; global_cond shape: %s",
        tuple(z_input.shape),
        tuple(global_cond.shape) if global_cond is not None else None,
    )

    capture_every = max(args.intermediate_stride, 1)
    log.info(
        "Rolling out flow ODE with %s steps; capturing every %s.",
        model.processor.flow_ode_steps,
        capture_every,
    )
    _, snapshots = run_fm_with_intermediates(
        model.processor, z_input, global_cond, capture_every=capture_every
    )

    # Decode + denormalise each snapshot, plus the input & ground truth output.
    decoded_snapshots: list[tuple[int, np.ndarray]] = []
    with torch.no_grad():
        for step, z_snap in snapshots:
            ambient = latents_to_ambient(model, z_snap, denormalise=True)
            decoded_snapshots.append((step, ambient.detach().cpu().numpy()))

        input_ambient = (
            model.denormalize_tensor(capture_sample.input_fields).detach().cpu().numpy()
        )
        target_ambient = (
            model.denormalize_tensor(target_fields_window).detach().cpu().numpy()
        )

    # decoded shapes: (1, T_out, H, W, C). Pull channel(s).
    n_t_out_pred = decoded_snapshots[0][1].shape[1]
    n_channels = decoded_snapshots[0][1].shape[-1]
    chans = channel_indices(args, n_channels)
    cmap_for = parse_cmap_list(args.cmap, chans)
    symmetric_cmaps = parse_symmetric_set(args.symmetric_cmaps)
    log.info(
        "Snapshots: %d; T_out=%d; rendering channels: %s; cmaps: %s",
        len(decoded_snapshots),
        n_t_out_pred,
        chans,
        cmap_for,
    )

    # Determine shared color limits per channel from the *target/final* output
    # so the input, ground truth, and final prediction share a scale. Earlier
    # noisy intermediates will inevitably land outside this range — that's
    # part of the visual story. Channels whose colormap is in the symmetric
    # set get limits centered on zero.
    final_pred = decoded_snapshots[-1][1]
    color_limits: dict[int, tuple[float, float]] = {}
    for c in chans:
        all_vals = np.concatenate(
            [
                final_pred[..., c].ravel(),
                target_ambient[..., c].ravel(),
                input_ambient[..., c].ravel(),
            ]
        )
        vmin = float(np.percentile(all_vals, 2))
        vmax = float(np.percentile(all_vals, 98))
        if cmap_for[c] in symmetric_cmaps:
            extent = max(abs(vmin), abs(vmax))
            vmin, vmax = -extent, extent
        if vmin == vmax:
            vmax = vmin + 1.0
        color_limits[c] = (vmin, vmax)

    suffix = args.format
    win_tag = f"w{args.rollout_window:02d}"

    # 1. Input field tiles for the visualised window.
    for t_idx in range(input_ambient.shape[1]):
        for c in chans:
            vmin, vmax = color_limits[c]
            save_tile(
                input_ambient[0, t_idx, ..., c],
                out_dir / f"{win_tag}_input_t{t_idx:02d}_c{c}.{suffix}",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap_for[c],
                dpi=args.dpi,
            )

    # 2. Ground truth output tiles for the visualised window.
    for t_idx in range(target_ambient.shape[1]):
        for c in chans:
            vmin, vmax = color_limits[c]
            save_tile(
                target_ambient[0, t_idx, ..., c],
                out_dir / f"{win_tag}_truth_t{t_idx:02d}_c{c}.{suffix}",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap_for[c],
                dpi=args.dpi,
            )

    # 3. ODE intermediates: one tile per (ode_step, output timestep, channel).
    for step, decoded in decoded_snapshots:
        for t_idx in range(decoded.shape[1]):
            for c in chans:
                vmin, vmax = color_limits[c]
                save_tile(
                    decoded[0, t_idx, ..., c],
                    out_dir
                    / f"{win_tag}_ode_step{step:03d}_t{t_idx:02d}_c{c}.{suffix}",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap_for[c],
                    dpi=args.dpi,
                )

    # 4. Contact sheet, one per rendered channel.
    for contact_c in chans:
        n_cols = decoded_snapshots[0][1].shape[1]  # T_out
        n_rows = len(decoded_snapshots)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows), squeeze=False
        )
        vmin, vmax = color_limits[contact_c]
        for row, (step, decoded) in enumerate(decoded_snapshots):
            for col in range(n_cols):
                ax = axes[row, col]
                ax.imshow(
                    decoded[0, col, ..., contact_c],
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap_for[contact_c],
                    origin="lower",
                )
                ax.set_axis_off()
                if col == 0:
                    ax.set_title(f"k={step}", fontsize=8, loc="left")
        fig.suptitle(
            (
                f"FM ODE trajectory — channel {contact_c}, "
                f"batch {args.batch_index}, sample {args.sample_index}, "
                f"rollout-window {args.rollout_window}"
            ),
            fontsize=10,
        )
        fig.tight_layout()
        contact_path = out_dir / f"{win_tag}_contact_sheet_c{contact_c}.{suffix}"
        fig.savefig(contact_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote contact sheet: %s", contact_path)
    log.info("All tiles written to %s", out_dir)


if __name__ == "__main__":
    main()
