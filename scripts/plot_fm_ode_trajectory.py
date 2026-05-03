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
        --batch-index 0 --sample-index 0 \
        --intermediate-stride 5 --channel 0

Multiple rollout depths (windows are 0-indexed, n_steps_output frames each):

    python scripts/plot_fm_ode_trajectory.py \
        --rollout-windows 0,12,24 --intermediate-stride 10 --channel 0

Multiple noise seeds — different ensemble members (saved with s{seed} tags):

    python scripts/plot_fm_ode_trajectory.py \
        --rollout-windows 0,12 --seeds 0,1,2

Per-channel colormaps (smoke=viridis, u/v=RdBu_r):

    python scripts/plot_fm_ode_trajectory.py --all-channels \
        --cmap viridis,RdBu_r,RdBu_r --symmetric-cmaps RdBu_r
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
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
            "Output directory for tiles. If omitted, defaults to "
            "<run-dir>/figures/fm_ode_traj_<timestamp>[_<label>] so re-runs "
            "do not overwrite previous outputs."
        ),
    )
    p.add_argument(
        "--label",
        default="",
        help=(
            "Optional label folded into the auto-generated output directory "
            "name, e.g. --label cns64_w0-12-24."
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
    p.add_argument(
        "--use-ema",
        action="store_true",
        help=(
            "Use EMA processor weights. Off by default — main eval comparisons "
            "do not use EMA, so this script matches that."
        ),
    )
    p.add_argument(
        "--seeds",
        default="0",
        help=(
            "Comma-separated list of noise seeds. Each seed produces an "
            "independent ensemble member. Output filenames are tagged s{seed}."
        ),
    )
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
        "--rollout-windows",
        default="0",
        help=(
            "Comma-separated list of output windows (0-indexed, n_steps_output "
            "frames each) at which to capture the FM ODE intermediates. 0 is "
            "the first window. For any window >0 we use the rollout test "
            "dataloader (full-trajectory mode) so ground truth is available "
            "at any depth."
        ),
    )
    p.add_argument(
        "--latent-channels",
        default="all",
        help=(
            "Comma-separated list of latent channels to render in latent-space "
            "plots, or 'all' for every channel (typically 8). 'none' disables "
            "latent-space output."
        ),
    )
    p.add_argument(
        "--latent-cmap",
        default="RdBu_r",
        help=(
            "Colormap for latent-space tiles. Single name or comma-separated "
            "list aligned with --latent-channels."
        ),
    )
    p.add_argument(
        "--latent-symmetric",
        action="store_true",
        default=True,
        help=(
            "Use symmetric color limits around zero for latent-space tiles "
            "(latents are roughly zero-centred). On by default."
        ),
    )
    p.add_argument(
        "--no-latent-symmetric",
        dest="latent_symmetric",
        action="store_false",
        help="Disable symmetric latent color limits.",
    )
    p.add_argument(
        "--const-names",
        default="buoyancy_y,smoothness,noise_scale,smoke_diffusivity",
        help=(
            "Comma-separated names for the global-cond constant scalars, "
            "used in filenames for the constants heatmap tiles."
        ),
    )
    p.add_argument(
        "--const-cmap",
        default="cividis",
        help="Matplotlib colormap for constant-scalar heatmap tiles.",
    )
    p.add_argument(
        "--no-constants",
        action="store_true",
        help="Skip the constant-scalars heatmap output.",
    )
    p.add_argument(
        "--no-latent",
        action="store_true",
        help="Skip the latent-space ODE intermediate output.",
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

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"fm_ode_traj_{timestamp}"
        if args.label:
            sanitized = "".join(
                ch if ch.isalnum() or ch in "-_." else "_" for ch in args.label
            )
            suffix = f"{suffix}_{sanitized}"
        out_dir = run_dir / "figures" / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

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

    # Parse plan: which windows × which seeds.
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds is empty.")
    windows = sorted({int(w) for w in args.rollout_windows.split(",") if w.strip()})
    if not windows:
        raise ValueError("--rollout-windows is empty.")
    log.info("Plan: windows=%s, seeds=%s", windows, seeds)

    # Choose the dataloader. Window 0 only is fine on the standard test
    # loader; any deeper window needs full trajectories, so we always use
    # the rollout loader when max(windows) > 0.
    datamodule.setup(stage="test")
    if max(windows) <= 0:
        loader = datamodule.test_dataloader()
        loader_name = "test_dataloader"
    else:
        loader = datamodule.rollout_test_dataloader()
        loader_name = "rollout_test_dataloader"
    log.info("Using %s", loader_name)

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

    base_sample = select_sample(batch, args.sample_index)
    base_sample = to_device(base_sample, device)

    n_t_out = int(model.processor.n_steps_output)

    # Sanity-check trajectory length covers the deepest window's GT slice.
    gt_end = (max(windows) + 1) * n_t_out
    if gt_end > base_sample.output_fields.shape[1]:
        msg = (
            f"Deepest window {max(windows)} requires GT frames up to "
            f"{gt_end - 1}, but the trajectory only has "
            f"{base_sample.output_fields.shape[1]} output frames."
        )
        raise IndexError(msg)

    # Channel selection + colormap setup.
    n_channels = base_sample.input_fields.shape[-1]
    chans = channel_indices(args, n_channels)
    cmap_for = parse_cmap_list(args.cmap, chans)
    symmetric_cmaps = parse_symmetric_set(args.symmetric_cmaps)

    # Compute global ambient color limits per channel from the union of all
    # visualised-window GTs + the input frame, so the same scale applies
    # across windows/seeds. Noisy intermediates will saturate against this.
    target_slices_per_window: dict[int, np.ndarray] = {}
    with torch.no_grad():
        input_ambient_np = (
            model.denormalize_tensor(base_sample.input_fields).detach().cpu().numpy()
        )
        for w in windows:
            tgt = base_sample.output_fields[:, w * n_t_out : (w + 1) * n_t_out]
            target_slices_per_window[w] = (
                model.denormalize_tensor(tgt).detach().cpu().numpy()
            )

    color_limits: dict[int, tuple[float, float]] = {}
    for c in chans:
        all_vals = np.concatenate(
            [input_ambient_np[..., c].ravel()]
            + [arr[..., c].ravel() for arr in target_slices_per_window.values()]
        )
        vmin = float(np.percentile(all_vals, 2))
        vmax = float(np.percentile(all_vals, 98))
        if cmap_for[c] in symmetric_cmaps:
            extent = max(abs(vmin), abs(vmax))
            vmin, vmax = -extent, extent
        if vmin == vmax:
            vmax = vmin + 1.0
        color_limits[c] = (vmin, vmax)

    # Latent setup.
    n_latent_channels = int(model.processor.n_channels_out)
    latent_chans = parse_latent_channel_spec(args.latent_channels, n_latent_channels)
    latent_cmap_for: dict[int, str] = {}
    if latent_chans and not args.no_latent:
        latent_cmap_for = parse_cmap_list(args.latent_cmap, latent_chans)
    log.info(
        "Channels: ambient=%s (cmaps=%s), latent=%s (cmaps=%s)",
        chans,
        cmap_for,
        latent_chans,
        latent_cmap_for,
    )

    # Compute global latent color limits per latent channel from each
    # window's encoded GT (this is the destination of the FM trajectory).
    latent_targets_per_window: dict[int, np.ndarray] = {}
    if latent_chans and not args.no_latent:
        with torch.no_grad():
            for w in windows:
                tgt = base_sample.output_fields[
                    :, w * n_t_out : (w + 1) * n_t_out
                ]
                # Encoder consumes Batch.input_fields, so masquerade the
                # target window as the input.
                fake_batch = Batch(
                    input_fields=tgt,
                    output_fields=tgt,
                    constant_scalars=base_sample.constant_scalars,
                    constant_fields=base_sample.constant_fields,
                    boundary_conditions=base_sample.boundary_conditions,
                )
                z_tgt = model.encoder_decoder.encoder.encode(fake_batch)
                if isinstance(z_tgt, tuple):
                    z_tgt = z_tgt[0]
                latent_targets_per_window[w] = z_tgt.detach().cpu().numpy()

    latent_color_limits: dict[int, tuple[float, float]] = {}
    for lc in latent_chans:
        all_vals = np.concatenate(
            [arr[..., lc].ravel() for arr in latent_targets_per_window.values()]
        ) if latent_targets_per_window else np.array([0.0, 1.0])
        vmin = float(np.percentile(all_vals, 2))
        vmax = float(np.percentile(all_vals, 98))
        if args.latent_symmetric:
            extent = max(abs(vmin), abs(vmax))
            vmin, vmax = -extent, extent
        if vmin == vmax:
            vmax = vmin + 1.0
        latent_color_limits[lc] = (vmin, vmax)

    # Constants: render once (sample-invariant across seeds/windows).
    if not args.no_constants and base_sample.constant_scalars is not None:
        write_constants_tiles(
            base_sample=base_sample,
            full_batch_constants=batch.constant_scalars,
            ambient_spatial_shape=tuple(base_sample.input_fields.shape[2:-1]),
            const_names=[s.strip() for s in args.const_names.split(",") if s.strip()],
            cmap=args.const_cmap,
            out_dir=out_dir,
            suffix=args.format,
            dpi=args.dpi,
        )
    elif base_sample.constant_scalars is None:
        log.info("No constant_scalars on this sample — skipping constants tiles.")

    capture_every = max(args.intermediate_stride, 1)
    log.info(
        "FM ODE: %s steps total; capturing every %s.",
        model.processor.flow_ode_steps,
        capture_every,
    )

    # Outer loop: seeds (independent ensemble members).
    # Inner loop: windows in ascending order (incremental advance).
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        log.info("=== seed=%s ===", seed)

        current = base_sample
        last_window = 0
        for window in windows:
            for _ in range(window - last_window):
                current = advance_batch_free_running(model, current)
            last_window = window

            visualise_window(
                model=model,
                capture_sample=current,
                target_window_ambient=target_slices_per_window[window],
                target_window_latent=latent_targets_per_window.get(window),
                window=window,
                seed=seed,
                chans=chans,
                cmap_for=cmap_for,
                color_limits=color_limits,
                latent_chans=latent_chans,
                latent_cmap_for=latent_cmap_for,
                latent_color_limits=latent_color_limits,
                capture_every=capture_every,
                out_dir=out_dir,
                suffix=args.format,
                dpi=args.dpi,
                batch_index=args.batch_index,
                sample_index=args.sample_index,
                emit_latent=not args.no_latent,
            )

    log.info("All tiles written to %s", out_dir)


def parse_latent_channel_spec(spec: str, n_latent: int) -> list[int]:
    s = spec.strip().lower()
    if s == "none" or s == "":
        return []
    if s == "all":
        return list(range(n_latent))
    out = []
    for part in spec.split(","):
        if not part.strip():
            continue
        idx = int(part)
        if not (0 <= idx < n_latent):
            msg = f"--latent-channels={spec!r} contains out-of-range index {idx} (n_latent={n_latent})."
            raise ValueError(msg)
        out.append(idx)
    return out


def write_constants_tiles(
    *,
    base_sample: Batch,
    full_batch_constants: torch.Tensor | None,
    ambient_spatial_shape: tuple[int, ...],
    const_names: list[str],
    cmap: str,
    out_dir: Path,
    suffix: str,
    dpi: int,
) -> None:
    """Render each global-cond constant as a uniform-color square tile.

    Color limits per constant are taken from the min/max across the loaded
    batch (so e.g. a buoyancy of 0.72 sits sensibly between 0.2 and 0.8 if
    the batch covers that range). Constants whose batch has no spread fall
    back to ±0.5 around the value so the tile is still visible.
    """
    cs = base_sample.constant_scalars
    if cs is None:
        return
    values = cs.detach().cpu().numpy().reshape(-1)  # (n_const,)
    n_const = values.shape[0]

    if full_batch_constants is not None and full_batch_constants.dim() == 2:
        batch_cs = full_batch_constants.detach().cpu().numpy()  # (B, n_const)
    else:
        batch_cs = None

    if len(const_names) < n_const:
        const_names = const_names + [f"c{i}" for i in range(len(const_names), n_const)]
    elif len(const_names) > n_const:
        const_names = const_names[:n_const]

    h, w = ambient_spatial_shape[:2]
    fig, axes = plt.subplots(1, n_const, figsize=(2.0 * n_const, 2.4), squeeze=False)
    for i in range(n_const):
        val = float(values[i])
        if batch_cs is not None and batch_cs.shape[0] > 1:
            vmin = float(batch_cs[:, i].min())
            vmax = float(batch_cs[:, i].max())
        else:
            vmin, vmax = val, val
        if vmin == vmax:
            vmin = val - 0.5
            vmax = val + 0.5
        tile = np.full((h, w), val, dtype=np.float32)
        save_tile(
            tile,
            out_dir / f"const_{i}_{const_names[i]}.{suffix}",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            dpi=dpi,
        )
        ax = axes[0, i]
        ax.imshow(tile, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
        ax.set_axis_off()
        ax.set_title(f"{const_names[i]}\n{val:.4g}", fontsize=9)
    fig.suptitle("Constant scalars (global conditioning)", fontsize=10)
    fig.tight_layout()
    summary_path = out_dir / f"const_summary.{suffix}"
    fig.savefig(summary_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote constant scalars summary: %s", summary_path)


def visualise_window(
    *,
    model,
    capture_sample: Batch,
    target_window_ambient: np.ndarray,
    target_window_latent: np.ndarray | None,
    window: int,
    seed: int,
    chans: list[int],
    cmap_for: dict[int, str],
    color_limits: dict[int, tuple[float, float]],
    latent_chans: list[int],
    latent_cmap_for: dict[int, str],
    latent_color_limits: dict[int, tuple[float, float]],
    capture_every: int,
    out_dir: Path,
    suffix: str,
    dpi: int,
    batch_index: int,
    sample_index: int,
    emit_latent: bool,
) -> None:
    """Capture FM ODE intermediates at the current window and write tiles."""
    log.info("Capturing seed=%s window=%s", seed, window)

    with torch.no_grad():
        z_input, global_cond = model.encoder_decoder.encoder.encode_with_cond(
            capture_sample
        )

    _, snapshots = run_fm_with_intermediates(
        model.processor, z_input, global_cond, capture_every=capture_every
    )

    decoded_snapshots: list[tuple[int, np.ndarray]] = []
    latent_snapshots: list[tuple[int, np.ndarray]] = []
    with torch.no_grad():
        for step, z_snap in snapshots:
            ambient = latents_to_ambient(model, z_snap, denormalise=True)
            decoded_snapshots.append((step, ambient.detach().cpu().numpy()))
            if emit_latent and latent_chans:
                latent_snapshots.append((step, z_snap.detach().cpu().numpy()))

        input_ambient = (
            model.denormalize_tensor(capture_sample.input_fields).detach().cpu().numpy()
        )
        z_input_np = z_input.detach().cpu().numpy()

    tag = f"s{seed:03d}_w{window:02d}"

    # 1. Ambient input frame (the visualised window's input).
    for t_idx in range(input_ambient.shape[1]):
        for c in chans:
            vmin, vmax = color_limits[c]
            save_tile(
                input_ambient[0, t_idx, ..., c],
                out_dir / f"{tag}_ambient_input_t{t_idx:02d}_c{c}.{suffix}",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap_for[c],
                dpi=dpi,
            )

    # 2. Ambient ground-truth output for this window.
    for t_idx in range(target_window_ambient.shape[1]):
        for c in chans:
            vmin, vmax = color_limits[c]
            save_tile(
                target_window_ambient[0, t_idx, ..., c],
                out_dir / f"{tag}_ambient_truth_t{t_idx:02d}_c{c}.{suffix}",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap_for[c],
                dpi=dpi,
            )

    # 3. Ambient ODE intermediates.
    for step, decoded in decoded_snapshots:
        for t_idx in range(decoded.shape[1]):
            for c in chans:
                vmin, vmax = color_limits[c]
                save_tile(
                    decoded[0, t_idx, ..., c],
                    out_dir
                    / f"{tag}_ambient_ode_step{step:03d}_t{t_idx:02d}_c{c}.{suffix}",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap_for[c],
                    dpi=dpi,
                )

    # 4. Ambient contact sheet, one per channel.
    for contact_c in chans:
        write_contact_sheet(
            decoded_snapshots,
            channel=contact_c,
            cmap=cmap_for[contact_c],
            color_limits=color_limits[contact_c],
            title=(
                f"FM ODE (ambient) — c{contact_c}, batch {batch_index}, "
                f"sample {sample_index}, window {window}, seed {seed}"
            ),
            out_path=out_dir / f"{tag}_ambient_contact_c{contact_c}.{suffix}",
            dpi=dpi,
        )

    if not emit_latent or not latent_chans:
        return

    # 5. Latent input (encoder output for the visualised window's input).
    for t_idx in range(z_input_np.shape[1]):
        for lc in latent_chans:
            vmin, vmax = latent_color_limits[lc]
            save_tile(
                z_input_np[0, t_idx, ..., lc],
                out_dir / f"{tag}_latent_input_t{t_idx:02d}_lc{lc}.{suffix}",
                vmin=vmin,
                vmax=vmax,
                cmap=latent_cmap_for[lc],
                dpi=dpi,
            )

    # 6. Latent ground truth (encoder output for the GT window).
    if target_window_latent is not None:
        for t_idx in range(target_window_latent.shape[1]):
            for lc in latent_chans:
                vmin, vmax = latent_color_limits[lc]
                save_tile(
                    target_window_latent[0, t_idx, ..., lc],
                    out_dir
                    / f"{tag}_latent_truth_t{t_idx:02d}_lc{lc}.{suffix}",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=latent_cmap_for[lc],
                    dpi=dpi,
                )

    # 7. Latent ODE intermediates (this is where the ODE actually runs).
    for step, z_snap_np in latent_snapshots:
        for t_idx in range(z_snap_np.shape[1]):
            for lc in latent_chans:
                vmin, vmax = latent_color_limits[lc]
                save_tile(
                    z_snap_np[0, t_idx, ..., lc],
                    out_dir
                    / f"{tag}_latent_ode_step{step:03d}_t{t_idx:02d}_lc{lc}.{suffix}",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=latent_cmap_for[lc],
                    dpi=dpi,
                )

    # 8. Latent contact sheet, one per latent channel.
    for contact_lc in latent_chans:
        write_contact_sheet(
            latent_snapshots,
            channel=contact_lc,
            cmap=latent_cmap_for[contact_lc],
            color_limits=latent_color_limits[contact_lc],
            title=(
                f"FM ODE (latent) — lc{contact_lc}, batch {batch_index}, "
                f"sample {sample_index}, window {window}, seed {seed}"
            ),
            out_path=out_dir / f"{tag}_latent_contact_lc{contact_lc}.{suffix}",
            dpi=dpi,
        )


def write_contact_sheet(
    snapshots: list[tuple[int, np.ndarray]],
    *,
    channel: int,
    cmap: str,
    color_limits: tuple[float, float],
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    n_cols = snapshots[0][1].shape[1]  # T_out
    n_rows = len(snapshots)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows), squeeze=False
    )
    vmin, vmax = color_limits
    for row, (step, arr) in enumerate(snapshots):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.imshow(
                arr[0, col, ..., channel],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                origin="lower",
            )
            ax.set_axis_off()
            if col == 0:
                ax.set_title(f"k={step}", fontsize=8, loc="left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote contact sheet: %s", out_path)


if __name__ == "__main__":
    main()
