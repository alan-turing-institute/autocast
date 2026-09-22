r"""Calibration-trajectory-count (cal-B) sufficiency sweep for the new set.

Answers: how many calibration trajectories are enough for reporting sharpness
to settle -- i.e. where does interval width (Winkler) flatten and coverage
land near nominal, as a function of the calibration-pool size K. Ported from
July's ``sweep_calB_sufficiency.py``: a FIXED ~50-trajectory held-out test set
(split off once, seeded, held constant for the whole sweep), then, per K in a
grid, ``n_draws`` random disjoint K-sized calibration subsets drawn from the
fixed calibration pool (never resampling the test set). Conformal is fit and
scored at every draw; EMOS is fit once per K (on the first draw's subset --
EMOS fitting is ~100x slower than conformal and barely varies across draws
since ``per=TIME`` already pools over batch x space) and used as a parametric
reference. Raw needs no calibration at all.

Usage
-----
.. code-block:: bash

    python -m autocast.scripts.conformal.sufficiency \\
        --new new.pt --out eval_conformal/ [--balance-by-scalars] \\
        [--k-grid 9,12,15,20,30,50,75,100] [--n-draws 20] \\
        [--split-seed 20260709] [--threads N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from autouq.calibrators.conformal import Ensemble

from autocast.scripts.conformal.calibrate import fit_emos_per_frame
from autocast.scripts.conformal.data import (
    DEFAULT_SPLIT_SEED,
    PredictionDump,
    balanced_split_by_scalars,
    default_device,
    draw_calibration_subset,
    fixed_split,
    load_prediction_dump,
)
from autocast.scripts.conformal.scoring import (
    NOMINAL_ALPHA,
    Method,
    per_lead_coverage,
    per_lead_winkler,
    raw_interval_multi,
)
from autocast.types import Tensor

#: Calibration-set sizes swept (as in the July 2026 runs). 9 is deliberately
#: the conformal hard floor for alpha=0.1 (`ConformalCalibrator.score_quantile`
#: needs `ceil((n+1)*0.9) <= n`) -- unlike `calibrate.py`, this sweep only ever
#: evaluates the single nominal alpha, never the full reliability grid, so
#: that floor is the only constraint on K.
DEFAULT_K_GRID: tuple[int, ...] = (9, 12, 15, 20, 30, 50, 75, 100)
DEFAULT_N_DRAWS = 20


def _fit_raw_single(pred_test: Tensor) -> tuple[Tensor, Tensor]:
    lower, upper = raw_interval_multi(pred_test, [1.0 - NOMINAL_ALPHA])
    return lower[..., 0], upper[..., 0]


def _fit_conformal_single(
    true_cal: Tensor, pred_cal: Tensor, pred_test: Tensor
) -> tuple[Tensor, Tensor]:
    """Fit the shipped conformal calibrator, evaluated at one alpha only."""
    ensemble = Ensemble(mode="std")
    ensemble.calibrate(true_cal, pred_cal)
    intervals = ensemble.predict(pred_test, alphas=NOMINAL_ALPHA)
    return intervals[..., 0, 0], intervals[..., 1, 0]


def _fit_emos_single(
    true_cal: Tensor, pred_cal: Tensor, pred_test: Tensor
) -> tuple[Tensor, Tensor]:
    """`EMOS(per=TIME)`, fit frame by frame and evaluated at one alpha only."""
    emos = fit_emos_per_frame(true_cal, pred_cal)
    intervals = emos.predict(pred_test, alphas=NOMINAL_ALPHA)
    return intervals[..., 0, 0], intervals[..., 1, 0]


def _summarize(values: np.ndarray, n_draws_used: int) -> dict[str, Any]:
    """Per-lead mean/std plus lead-averaged mean/std, over the draw axis."""
    per_draw_lead_avg = values.mean(axis=1)
    return {
        "n_draws_used": n_draws_used,
        "per_lead_mean": values.mean(axis=0).tolist(),
        "per_lead_std": values.std(axis=0).tolist(),
        "lead_avg_mean": float(per_draw_lead_avg.mean()),
        "lead_avg_std": float(per_draw_lead_avg.std()),
    }


def run_sufficiency(
    new_dump: PredictionDump,
    *,
    balance_by_scalars: bool = False,
    split_seed: int = DEFAULT_SPLIT_SEED,
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
    n_draws: int = DEFAULT_N_DRAWS,
) -> dict[str, Any]:
    """Run the sweep and return July's ``result.json`` schema as a dict."""
    if balance_by_scalars:
        if new_dump.constant_scalars is None:
            msg = "--balance-by-scalars requires constant_scalars in the new-set dump."
            raise ValueError(msg)
        split = balanced_split_by_scalars(new_dump.constant_scalars, seed=split_seed)
    else:
        split = fixed_split(new_dump.n_trajectories, seed=split_seed)
    pool_idx, test_idx = split.calibration, split.test
    pool_size = int(pool_idx.shape[0])
    true_test, pred_test = new_dump.trues[test_idx], new_dump.preds[test_idx]

    ks = sorted(k for k in k_grid if k <= pool_size)
    if not ks:
        msg = f"no K in grid {k_grid} fits within pool_size={pool_size}."
        raise ValueError(msg)

    raw_lo, raw_hi = _fit_raw_single(pred_test)
    raw_cov = per_lead_coverage(true_test, raw_lo, raw_hi)
    raw_wink = per_lead_winkler(true_test, raw_lo, raw_hi, NOMINAL_ALPHA)

    by_k: dict[str, Any] = {}
    for k in ks:
        conformal_cov_draws: list[np.ndarray] = []
        conformal_wink_draws: list[np.ndarray] = []
        emos_cov = emos_wink = None
        for draw_idx in range(n_draws):
            cal_idx = draw_calibration_subset(pool_idx, k, draw_idx, seed0=split_seed)
            true_cal, pred_cal = new_dump.trues[cal_idx], new_dump.preds[cal_idx]

            conformal_lo, conformal_hi = _fit_conformal_single(
                true_cal, pred_cal, pred_test
            )
            conformal_cov_draws.append(
                per_lead_coverage(true_test, conformal_lo, conformal_hi)
            )
            conformal_wink_draws.append(
                per_lead_winkler(true_test, conformal_lo, conformal_hi, NOMINAL_ALPHA)
            )

            if draw_idx == 0:
                emos_lo, emos_hi = _fit_emos_single(true_cal, pred_cal, pred_test)
                emos_cov = per_lead_coverage(true_test, emos_lo, emos_hi)
                emos_wink = per_lead_winkler(true_test, emos_lo, emos_hi, NOMINAL_ALPHA)

        if emos_cov is None or emos_wink is None:
            msg = "n_draws must be >= 1 so the reference EMOS fit runs at least once."
            raise RuntimeError(msg)
        by_k[str(k)] = {
            Method.RAW.value: {
                "cov": _summarize(raw_cov[None, :], 1),
                "wink": _summarize(raw_wink[None, :], 1),
            },
            Method.EMOS.value: {
                "cov": _summarize(emos_cov[None, :], 1),
                "wink": _summarize(emos_wink[None, :], 1),
            },
            Method.CONFORMAL.value: {
                "cov": _summarize(np.stack(conformal_cov_draws), n_draws),
                "wink": _summarize(np.stack(conformal_wink_draws), n_draws),
            },
        }

    return {
        "alpha": NOMINAL_ALPHA,
        "b_total": new_dump.n_trajectories,
        "test_size": int(test_idx.shape[0]),
        "pool_size": pool_size,
        "split_seed": split_seed,
        "balanced_by_scalars": balance_by_scalars,
        "k_grid": ks,
        "k_grid_requested": list(k_grid),
        "n_draws": n_draws,
        "by_K": by_k,
    }


def summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per ``(K, method)``: mean/std coverage and Winkler over frames."""
    rows = []
    for k in result["k_grid"]:
        for method in (Method.RAW.value, Method.EMOS.value, Method.CONFORMAL.value):
            entry = result["by_K"][str(k)][method]
            rows.append(
                {
                    "K": k,
                    "method": method,
                    "coverage_mean": entry["cov"]["lead_avg_mean"],
                    "coverage_std": entry["cov"]["lead_avg_std"],
                    "winkler_mean": entry["wink"]["lead_avg_mean"],
                    "winkler_std": entry["wink"]["lead_avg_std"],
                }
            )
    return rows


def per_frame_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per ``(K, method, frame)``: per-lead coverage and Winkler."""
    rows = []
    for k in result["k_grid"]:
        for method in (Method.RAW.value, Method.EMOS.value, Method.CONFORMAL.value):
            entry = result["by_K"][str(k)][method]
            n_frames = len(entry["cov"]["per_lead_mean"])
            rows.extend(
                {
                    "K": k,
                    "method": method,
                    "frame": frame,
                    "coverage_mean": entry["cov"]["per_lead_mean"][frame],
                    "coverage_std": entry["cov"]["per_lead_std"][frame],
                    "winkler_mean": entry["wink"]["per_lead_mean"][frame],
                    "winkler_std": entry["wink"]["per_lead_std"][frame],
                }
                for frame in range(n_frames)
            )
    return rows


def sufficiency(
    *,
    new_path: Path,
    out_dir: Path,
    balance_by_scalars: bool = False,
    split_seed: int = DEFAULT_SPLIT_SEED,
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
    n_draws: int = DEFAULT_N_DRAWS,
    device: str | None = None,
) -> None:
    """Run the sweep and write ``data_sufficiency/`` under ``out_dir``.

    ``device`` defaults to CUDA if available; see
    :func:`autocast.scripts.conformal.calibrate.calibrate`'s docstring for
    the same convention (float32 matmul precision, no bf16/``torch.compile``).
    """
    resolved_device = device if device is not None else default_device()
    torch.set_float32_matmul_precision("high")
    new_dump = load_prediction_dump(new_path).to(resolved_device)
    result = run_sufficiency(
        new_dump,
        balance_by_scalars=balance_by_scalars,
        split_seed=split_seed,
        k_grid=k_grid,
        n_draws=n_draws,
    )

    sufficiency_dir = out_dir / "data_sufficiency"
    sufficiency_dir.mkdir(parents=True, exist_ok=True)
    (sufficiency_dir / "sufficiency.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(summary_rows(result)).to_csv(
        sufficiency_dir / "sufficiency.csv", index=False
    )
    pd.DataFrame(per_frame_rows(result)).to_csv(
        sufficiency_dir / "sufficiency_per_frame.csv", index=False
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--new", required=True, type=Path, help="new-set prediction .pt"
    )
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--balance-by-scalars",
        action="store_true",
        help="stratify the new-set split by distinct constant_scalars rows",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--k-grid",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_GRID),
        help="comma-separated calibration-pool-size grid",
    )
    parser.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    parser.add_argument(
        "--threads", type=int, default=None, help="torch.set_num_threads"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu' (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    k_grid = tuple(int(k) for k in args.k_grid.split(","))
    sufficiency(
        new_path=args.new,
        out_dir=args.out,
        balance_by_scalars=args.balance_by_scalars,
        split_seed=args.split_seed,
        k_grid=k_grid,
        n_draws=args.n_draws,
        device=args.device,
    )


if __name__ == "__main__":
    main()
