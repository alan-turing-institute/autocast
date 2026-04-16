#!/usr/bin/env python3
"""Time per-epoch training duration for each model/dataset configuration.

Runs a short training (default: 3 epochs) per configuration, saves a
checkpoint so that TrainingTimerCallback's per-epoch times can be
extracted, and writes the results to a JSON file.

Usage:
    python scripts/time_epochs.py [OPTIONS]
    # or via uv:
    uv run python scripts/time_epochs.py [OPTIONS]

Options:
    -n, --num-epochs NUM      Number of timing epochs (default: 3)
    -o, --output FILE         Output JSON path (default: epoch_timings.json)
    -c, --configs FILE        File listing Hydra override lines (default: scripts/timing_configs.txt)
    --extra OVERRIDES         Extra Hydra overrides (space-separated, appended to every run)
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Allow running without torch installed in the outer script environment;
# torch is only needed to extract checkpoint timing data.
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def parse_configs(path: Path) -> list[str]:
    """Read config lines, skipping blanks and comments."""
    lines = []
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def extract_epoch_times_from_checkpoint(ckpt_path: Path) -> list[float] | None:
    """Extract per-epoch times from TrainingTimerCallback in a checkpoint."""
    if not HAS_TORCH or not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        callbacks = ckpt.get("callbacks", {})
        for key, state in callbacks.items():
            if "TrainingTimerCallback" in key and "epoch_times_s" in state:
                times = state["epoch_times_s"]
                if times:
                    return times
    except Exception as exc:  # noqa: BLE001
        print(f"    (could not read checkpoint: {exc})")
    return None


def run_timing(
    config_line: str,
    num_epochs: int,
    work_dir: Path,
    extra_overrides: list[str],
) -> dict | None:
    """Run one timing experiment and return results dict, or None on failure."""
    label = config_line.replace(" ", "_")
    run_dir = work_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint path for extracting TrainingTimerCallback data
    ckpt_path = run_dir / "timing.ckpt"

    overrides = shlex.split(config_line)
    cmd = [
        "uv", "run", "train_encoder_processor_decoder",
        *overrides,
        f"trainer.max_epochs={num_epochs}",
        "trainer.max_time=null",
        # Keep checkpointing so we can extract per-epoch times
        "trainer.enable_checkpointing=false",
        "trainer.callbacks=[]",
        "logging.wandb.enabled=false",
        "output.skip_test=true",
        "output.save_config=false",
        # Save a final checkpoint with timing data
        f"output.checkpoint_path={ckpt_path}",
        f"hydra.run.dir={run_dir}",
        *extra_overrides,
    ]

    stdout_log = run_dir / "stdout.log"
    print(f"  Command: {' '.join(cmd[:6])}... (see {stdout_log})")

    wall_start = time.monotonic()
    try:
        with open(stdout_log, "w") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1h safety timeout per timing run
            )
        wall_end = time.monotonic()

        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode}) — see {stdout_log}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 1h — see {stdout_log}")
        return None

    wall_total = wall_end - wall_start

    # Try to get per-epoch times from the checkpoint
    epoch_times = extract_epoch_times_from_checkpoint(ckpt_path)

    if epoch_times and len(epoch_times) >= num_epochs:
        # Use callback-measured per-epoch times (excludes startup overhead)
        seconds_per_epoch = sum(epoch_times) / len(epoch_times)
        source = "callback"
    else:
        # Fall back to wall-clock (includes startup — conservative)
        seconds_per_epoch = wall_total / num_epochs
        epoch_times = None
        source = "wallclock"

    result = {
        "config_overrides": config_line,
        "num_timing_epochs": num_epochs,
        "wall_total_seconds": round(wall_total, 2),
        "seconds_per_epoch": round(seconds_per_epoch, 2),
        "timing_source": source,
    }
    if epoch_times:
        result["epoch_times_s"] = [round(t, 2) for t in epoch_times]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num-epochs", type=int, default=3)
    parser.add_argument("-o", "--output", default="epoch_timings.json")
    parser.add_argument("-c", "--configs", default="scripts/timing_configs.txt")
    parser.add_argument("--extra", default="", help="Extra Hydra overrides (space-separated)")
    args = parser.parse_args()

    configs_path = Path(args.configs)
    if not configs_path.exists():
        print(f"Error: {configs_path} not found.", file=sys.stderr)
        print("Create it with one config override per line. See scripts/timing_configs.txt.", file=sys.stderr)
        sys.exit(1)

    config_lines = parse_configs(configs_path)
    if not config_lines:
        print("No configurations found in config file.", file=sys.stderr)
        sys.exit(1)

    extra = shlex.split(args.extra) if args.extra else []

    print(f"Timing {len(config_lines)} configuration(s), {args.num_epochs} epochs each")
    print(f"Output: {args.output}")
    print("─" * 70)

    results = []
    with tempfile.TemporaryDirectory(prefix="autocast_timing_") as tmpdir:
        work_dir = Path(tmpdir)
        for i, config_line in enumerate(config_lines, 1):
            print(f"\n[{i}/{len(config_lines)}] {config_line}")
            result = run_timing(config_line, args.num_epochs, work_dir, extra)
            if result:
                spe = result["seconds_per_epoch"]
                src = result["timing_source"]
                print(f"  OK: ~{spe:.1f} s/epoch ({src}), {result['wall_total_seconds']:.1f}s total")
                results.append(result)

    print("\n" + "─" * 70)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {len(results)} timing(s) to: {output_path}")

    if results:
        print(f"\nNext: python scripts/compute_cosine_epochs.py -i {args.output}")


if __name__ == "__main__":
    main()
