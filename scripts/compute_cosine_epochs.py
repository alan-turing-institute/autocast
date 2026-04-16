#!/usr/bin/env python3
"""Compute max_epochs for each timed configuration given a wall-clock budget.

Reads the JSON output from ``time_epochs.sh`` and prints, for each
configuration, the recommended ``trainer.max_epochs`` value so that
training (and the half-period cosine LR schedule) completes within
the given budget.

Usage:
    python scripts/compute_cosine_epochs.py [OPTIONS]

Options:
    -i, --input FILE      Timing JSON from time_epochs.sh (default: epoch_timings.json)
    -b, --budget HOURS    Wall-clock budget in hours (default: 24)
    -m, --margin FRACTION Safety margin fraction subtracted from budget (default: 0.02)
    -o, --output FILE     Write results JSON here (optional)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def compute_max_epochs(
    seconds_per_epoch: float,
    budget_hours: float,
    margin: float = 0.02,
) -> dict:
    """Return max_epochs and related info for one configuration."""
    budget_seconds = budget_hours * 3600
    usable_seconds = budget_seconds * (1.0 - margin)
    max_epochs = int(math.floor(usable_seconds / seconds_per_epoch))
    expected_hours = (max_epochs * seconds_per_epoch) / 3600
    headroom_hours = budget_hours - expected_hours
    return {
        "max_epochs": max_epochs,
        "expected_hours": round(expected_hours, 2),
        "headroom_hours": round(headroom_hours, 2),
        "budget_hours": budget_hours,
        "margin": margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        default="epoch_timings.json",
        help="Timing JSON from time_epochs.sh",
    )
    parser.add_argument(
        "-b",
        "--budget",
        type=float,
        default=24.0,
        help="Wall-clock budget in hours",
    )
    parser.add_argument(
        "-m",
        "--margin",
        type=float,
        default=0.02,
        help="Safety margin fraction (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write results JSON here (optional)",
    )
    args = parser.parse_args()

    timings_path = Path(args.input)
    if not timings_path.exists():
        print(
            f"Error: {timings_path} not found. Run time_epochs.sh first.",
            file=sys.stderr,
        )
        sys.exit(1)

    timings = json.loads(timings_path.read_text())

    results = []
    print()
    print(
        f"{'Config':<55} {'s/epoch':>8} {'max_epochs':>11} {'est. hrs':>9} {'headroom':>9}"
    )
    print("─" * 95)

    for entry in timings:
        label = entry["config_overrides"]
        spe = entry["seconds_per_epoch"]
        info = compute_max_epochs(spe, args.budget, args.margin)
        result = {**entry, **info}
        results.append(result)

        print(
            f"{label:<55} {spe:>8.1f} {info['max_epochs']:>11d} "
            f"{info['expected_hours']:>9.1f} {info['headroom_hours']:>9.1f}"
        )

    print("─" * 95)
    print()
    print("Recommended Hydra overrides for each config:")
    print()
    for r in results:
        max_time_str = f"{int(args.budget):02d}:00:00:00"
        print(f"  # {r['config_overrides']}")
        print(
            f"  trainer.max_epochs={r['max_epochs']} "
            f"trainer.max_time={max_time_str} "
            f"optimizer=adamw_half"
        )
        print()

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Wrote detailed results to: {out_path}")


if __name__ == "__main__":
    main()
