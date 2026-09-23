"""Plot Figure-4-style calibration comparisons for the main paper checkpoints.

Use the raw forecasts from the calibration export by default, so calibration
methods are compared on the same forecast realisation. No evaluation is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autocast.scripts import plot_dataset_comparisons as plots

RUNS = {
    "CRPS": {
        "AD": "crps_ad64_vit_azula_large_bed4611_da01a04",
        "CNS": "crps_cns64_vit_azula_large_bed4611_c99f534",
        "GS": "crps_gs64_vit_azula_large_bed4611_828a161",
        "GPE": "crps_gpe64_vit_azula_large_bed4611_e0a6df5",
    },
    "FM": {
        "AD": "diff_ad64_flow_matching_vit_09490da_dae1382",
        "CNS": "diff_cns64_flow_matching_vit_09490da_636fcc3",
        "GS": "diff_gs64_flow_matching_vit_09490da_7e9e331",
        "GPE": "diff_gpe64_flow_matching_vit_09490da_47bf39a",
    },
}
METHODS = ("raw", "EMOS", "conformal")
DATASETS = ["AD", "CNS", "GS", "GPE"]
COVERAGE_METRICS = ["coverage_0.9", "coverage_0.5", "coverage_0.1"]
WINDOWS = ("0-4", "6-12", "13-30", "31-99")
# Recovered by rendering the original Figure 4's eight historical evaluations.
# Keep its exact limits rather than expanding them for the calibrated curves.
REFERENCE_COVERAGE_YLIM = (-0.914115395769477, 0.914115395769477)
REFERENCE_RESULTS = Path("outputs/2026-05-15_collated")
REFERENCE_FIGURE = REFERENCE_RESULTS / (
    "2026-05-19_final_plots/main_comparison_m8_complete_no_fm_amb_best_winkler/"
    "paper_uq_reliability_by_lead_time.pdf"
)


def fingerprint(path: Path, root: Path) -> dict[str, str]:
    """Record a path and its exact contents without resolving data symlinks."""
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def check_coverage(eval_dir: Path) -> list[Path]:
    """Require complete, finite coverage curves and all 100 forecast frames."""
    lead_path = eval_dir / "rollout_metrics_per_timestep_channel_all.csv"
    lead = pd.read_csv(lead_path, index_col=0)
    selected = lead.loc[COVERAGE_METRICS].to_numpy(dtype=float)
    if (
        list(lead.columns) != [str(i) for i in range(100)]
        or not np.isfinite(selected).all()
        or not ((selected >= 0) & (selected <= 1)).all()
    ):
        raise ValueError(f"Invalid or incomplete lead-time coverage: {lead_path}")
    paths = [lead_path]
    for window in WINDOWS:
        path = eval_dir / f"rollout_coverage_window_{window}.csv"
        curve = pd.read_csv(path)
        values = curve["observed_mean"].to_numpy(dtype=float)
        if (
            len(curve) != 19
            or not np.allclose(curve["coverage_level"], np.arange(1, 20) / 20)
            or not np.isfinite(values).all()
            or not ((values >= 0) & (values <= 1)).all()
        ):
            raise ValueError(f"Invalid or incomplete window coverage: {path}")
        # Both Figure 4 panels must use the same frame/window convention.
        start, stop = map(int, window.split("-"))
        for metric in COVERAGE_METRICS:
            nominal = float(metric.split("_")[1])
            observed = values[np.isclose(curve["coverage_level"], nominal)][0]
            expected = lead.loc[metric].iloc[start:stop].mean()
            if not np.isclose(observed, expected, atol=1e-6):
                raise ValueError(f"Window/lead-time coverage mismatch: {path}")
        paths.append(path)
    return paths


def collect_inputs(
    results: Path, split: str, methods: tuple[str, ...] = METHODS
) -> tuple[pd.DataFrame, dict]:
    """Select main runs and audit differences from the historical raw curves."""
    rows, inputs, audit = [], [], []
    for model, datasets in RUNS.items():
        for dataset, run in datasets.items():
            run_dir = results / run
            cal_dir = run_dir / "eval_conformal"
            manifest_path = cal_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            inputs.append(manifest_path)
            for subset in ("paper_valid", "paper_test", "new"):
                inputs.append(
                    cal_dir / "predictions" / subset / "resolved_eval_config.yaml"
                )
            historical = run_dir / (
                "eval_best_multiwinkler_from0p25" if model == "CRPS" else "eval"
            )
            differences = {}
            # Historical results use the paper test set. Do not present a
            # different-test-set discrepancy as a repeated-evaluation check.
            if split.endswith("__test-paper"):
                for window in WINDOWS:
                    filename = f"rollout_coverage_window_{window}.csv"
                    raw_path = cal_dir / split / "raw" / filename
                    old_path = historical / filename
                    raw = pd.read_csv(raw_path).set_index("coverage_level")[
                        "observed_mean"
                    ]
                    old = pd.read_csv(old_path).set_index("coverage_level")[
                        "observed_mean"
                    ]
                    differences[window] = float((raw - old).abs().max())
                    inputs.append(old_path)
            audit.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "run": run,
                    "trajectory_counts": {
                        key: value["n_trajectories"]
                        for key, value in manifest["inputs"].items()
                    },
                    "calibration_trajectories": (
                        len(manifest["split"]["new_calibration_idx"])
                        if split.startswith("calib-new__")
                        else manifest["inputs"]["paper_valid"]["n_trajectories"]
                    ),
                    "test_trajectories": (
                        len(manifest["split"]["new_test_idx"])
                        if split.endswith("__test-new")
                        else manifest["inputs"]["paper_test"]["n_trajectories"]
                    ),
                    "historical_test_set_matches": split.endswith("__test-paper"),
                    "max_absolute_raw_vs_historical_coverage": differences or None,
                }
            )
            for method in methods:
                eval_subdir = f"eval_conformal/{split}/{method}"
                inputs.extend(check_coverage(run_dir / eval_subdir))
                rows.append(
                    {
                        "model": model,
                        "dataset_label": dataset,
                        "run_path": run,
                        "eval_subdir": eval_subdir,
                        "plot_group": f"{model}_{method}",
                    }
                )
    provenance = {
        "baseline": "raw forecasts from the same eval_conformal export",
        "calibration_test_pairing": split,
        "methods": list(methods),
        "runs": audit,
        "inputs": [fingerprint(path, results) for path in sorted(set(inputs))],
    }
    return pd.DataFrame(rows), provenance


def styles_for(
    models: list[str], *, combined: bool, methods: tuple[str, ...] = METHODS
) -> tuple[dict, list[str]]:
    """Keep line styles tied to models and distinguish calibration by colour."""
    palette = plt.get_cmap("tab10")
    styles = {}
    for model in models:
        for method in methods:
            if method == "raw":
                label = f"{model} (main)"
                hue = 0 if model == "CRPS" else 1
            else:
                label = "EMOS" if method == "EMOS" else "Conformal prediction"
                if combined:
                    label = f"{model} + {method}"
                hue = 2 if method == "EMOS" else 4
                if methods == ("raw", "conformal"):
                    label = f"{model} + CP"
                    hue = 0 if model == "CRPS" else 1
            color = palette(hue)
            if methods == ("raw", "conformal") and method == "conformal":
                color = plots._base_then_dark_variant(color, 1, 2)
            styles[f"{model}_{method}"] = {
                "color": color,
                "label": label,
                "linestyle": "--" if model == "CRPS" else "-",
            }
    # Matplotlib fills legend columns first: keep one model per legend row.
    order = [
        styles[f"{model}_{method}"]["label"] for method in methods for model in models
    ]
    return styles, order


def main() -> None:
    """Validate inputs, render the selected comparisons and archive their provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("outputs/2026-07-24_collated")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cp-only",
        action="store_true",
        help="Render one combined original/CP comparison without EMOS.",
    )
    parser.add_argument(
        "--split",
        choices=[
            f"calib-{cal}__test-{test}"
            for cal in ("paper-valid", "new")
            for test in ("paper", "new")
        ],
        default="calib-new__test-paper",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    methods = ("raw", "conformal") if args.cp_only else METHODS
    frame, provenance = collect_inputs(args.results_dir, args.split, methods)
    args.output_dir.mkdir(parents=True)
    plots.FIGURE_FORMATS[:] = ["pdf", "png"]
    comparisons = (
        (("combined_cp_calibration", ["CRPS", "FM"]),)
        if args.cp_only
        else (
            ("crps_calibration", ["CRPS"]),
            ("fm_calibration", ["FM"]),
            ("combined_calibration", ["CRPS", "FM"]),
        )
    )
    for name, models in comparisons:
        output = args.output_dir / name
        output.mkdir()
        styles, order = styles_for(models, combined=len(models) == 2, methods=methods)
        plots.plot_paper_uq_reliability_figure(
            cast(pd.DataFrame, frame[frame["model"].isin(models)]),
            args.results_dir,
            output,
            styles,
            COVERAGE_METRICS,
            dataset_order=DATASETS,
            hue_order=order,
            legend_ncol=4 if args.cp_only else 3,
            coverage_delta_ylim=REFERENCE_COVERAGE_YLIM,
        )
    repo = Path(__file__).resolve().parents[1]
    reference_inputs = []
    for model, runs in RUNS.items():
        for run in runs.values():
            eval_subdir = (
                "eval_best_multiwinkler_from0p25" if model == "CRPS" else "eval"
            )
            path = (
                REFERENCE_RESULTS
                / run
                / eval_subdir
                / "rollout_metrics_per_timestep_channel_all.csv"
            )
            reference_inputs.append(fingerprint(repo / path, repo))
    provenance["rhs_axis_reference"] = {
        "figure": fingerprint(repo / REFERENCE_FIGURE, repo),
        "inputs": reference_inputs,
        "ylim": REFERENCE_COVERAGE_YLIM,
        "visible_yticks": [-0.5, 0.0, 0.5],
        "xlim": [-4.95, 103.95],
        "overflow": (
            "Boundary triangles mark the largest excursion in each contiguous "
            "off-scale segment; curve values are not clamped."
        ),
    }
    provenance["off_scale_values"] = []
    lo, hi = REFERENCE_COVERAGE_YLIM
    for row in frame.to_dict(orient="records"):
        path = (
            args.results_dir
            / row["run_path"]
            / row["eval_subdir"]
            / "rollout_metrics_per_timestep_channel_all.csv"
        )
        lead = pd.read_csv(path, index_col=0)
        for metric in COVERAGE_METRICS:
            delta = lead.loc[metric] / float(metric.split("_")[1]) - 1
            provenance["off_scale_values"].append(
                {
                    "model": row["model"],
                    "dataset": row["dataset_label"],
                    "method": row["plot_group"].split("_", 1)[1],
                    "metric": metric,
                    "below": int((delta < lo).sum()),
                    "above": int((delta > hi).sum()),
                    "min": float(delta.min()),
                    "max": float(delta.max()),
                }
            )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    source_paths = (
        Path(__file__).resolve(),
        Path(plots.__file__).resolve(),
        repo / "uv.lock",
    )
    sources_match_head = True
    for path in source_paths:
        committed = subprocess.run(
            ["git", "show", f"{head}:{path.relative_to(repo)}"],
            cwd=repo,
            capture_output=True,
            check=False,
        )
        if committed.returncode or committed.stdout != path.read_bytes():
            sources_match_head = False
    provenance.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "results_root": str(args.results_dir),
            "plotting_base_commit": head,
            "plotting_commit": head if sources_match_head else None,
            "note": (
                "Recipe, renderer and lockfile exactly match the recorded commit."
                if sources_match_head
                else "Rendering sources differ from HEAD; exact snapshots are archived."
            ),
            "command": [
                "uv",
                "run",
                "--frozen",
                "--no-sync",
                "python",
                "scripts/plot_calibration_comparisons.py",
                "--results-dir",
                str(args.results_dir),
                "--output-dir",
                str(args.output_dir),
                "--split",
                args.split,
                *(["--cp-only"] if args.cp_only else []),
            ],
        }
    )
    snapshot = args.output_dir / "plotting_source"
    snapshot.mkdir()
    provenance["code"] = []
    for path in source_paths:
        shutil.copy2(path, snapshot / path.name)
        provenance["code"].append(fingerprint(path, repo))
    provenance["outputs"] = [
        fingerprint(path, args.output_dir)
        for path in sorted(args.output_dir.glob("*/*"))
        if path.suffix in {".pdf", ".png"}
    ]
    (args.output_dir / "plotting_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
