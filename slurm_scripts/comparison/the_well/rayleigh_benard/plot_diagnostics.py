"""Build balanced RB figures and tables from complete trajectory-level outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from slurm_scripts.comparison.the_well.rayleigh_benard.complete_diagnostics import (
    LEVELS,
    METRICS,
    sha256,
)

COVERAGES = [f"coverage_{q:g}" for q in LEVELS]
COLORS = ["#2879ad", "#2ca02c", "#ff7f0e", "#173d65"]


def trajectory_stats(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Aggregate within independent trajectories before estimating SEs."""
    columns = METRICS + COVERAGES
    per_trajectory = frame.groupby(["trajectory_idx", *group])[columns].mean()
    if group:
        grouped = per_trajectory.groupby(group)
        counts = np.asarray(grouped.size(), dtype=float)
        mean = cast(pd.DataFrame, grouped.mean())
        se = cast(pd.DataFrame, grouped.std(ddof=1)).div(np.sqrt(counts), axis=0)
    else:
        values = per_trajectory.to_numpy(dtype=float)
        mean = pd.DataFrame([values.mean(axis=0)], columns=pd.Index(columns))
        se = pd.DataFrame(
            [values.std(axis=0, ddof=1) / np.sqrt(len(values))],
            columns=pd.Index(columns),
        )
    result = mean.add_suffix("_mean").join(se.add_suffix("_se"))
    return result.reset_index(drop=not group)


def calibration_mae_stats(frame: pd.DataFrame) -> tuple[float, float]:
    """Use the joint covariance across levels for the paper's delta-method SE."""
    values = frame.groupby("trajectory_idx")[COVERAGES].mean().to_numpy()
    observed = values.mean(axis=0)
    gradient = np.sign(observed - LEVELS) / len(LEVELS)
    covariance = np.cov(values, rowvar=False, ddof=1) / len(values)
    variance = gradient @ covariance @ gradient
    return float(np.abs(observed - LEVELS).mean()), float(np.sqrt(max(0, variance)))


def validate_rows(frame: pd.DataFrame, *, horizon: int, paired: bool) -> None:
    """Reject duplicate, incomplete, nonfinite or misidentified results."""
    ids = ["selection_position", "lead", "channel"]
    if (
        frame.duplicated(ids).any()
        or not np.isfinite(frame[METRICS + COVERAGES].to_numpy()).all()
    ):
        msg = "Duplicate identities or nonfinite metric results"
        raise ValueError(msg)
    expected = 512 * 4 * 4 if paired else 64 * horizon * 4
    if len(frame) != expected or frame.trajectory_idx.nunique() != 64:
        msg = "Expected complete 64-trajectory production results"
        raise ValueError(msg)
    if set(frame.channel) != set(range(4)) or set(frame.lead) != set(
        range(1, horizon + 1)
    ):
        msg = "Changed channels or lead indices"
        raise ValueError(msg)
    counts = frame.groupby("selection_position").size()
    if len(counts) != (512 if paired else 64) or not bool(
        (np.asarray(counts) == horizon * 4).all()
    ):
        msg = "Incomplete window or trajectory"
        raise ValueError(msg)
    mapping = frame.groupby("selection_position").trajectory_idx.nunique()
    if not mapping.eq(1).all():
        msg = "A selection position maps to multiple physical trajectories"
        raise ValueError(msg)
    if (
        paired
        and not frame.groupby("trajectory_idx").selection_position.nunique().eq(8).all()
    ):
        msg = "Expected eight contexts per trajectory"
        raise ValueError(msg)
    coverage = frame[COVERAGES].to_numpy()
    if (
        coverage.min() < 0
        or coverage.max() > 1
        or np.diff(coverage, axis=1).min() < -1e-7
    ):
        msg = "Invalid or nonmonotonic coverage levels"
        raise ValueError(msg)


def load_results(manifest: Path) -> tuple[dict, dict]:
    """Verify completion, provenance, hashes and exact identities across models."""
    state = json.loads(manifest.read_text())
    results = {}
    identities = {}
    gpu = None
    for name, entry in state["models"].items():
        folder = manifest.parent / name / "production"
        if not (folder / "COMPLETE").exists():
            raise ValueError(f"Production evaluation is incomplete: {name}")
        record = json.loads((folder / "provenance.json").read_text())
        if (
            record["pilot"]
            or record["manifest_sha256"] != sha256(manifest)
            or record["global_step"] != entry["expected_step"]
            or record["members"] != 16
            or record["horizon"] != state["horizon"]
        ):
            msg = "Mismatched endpoint or evaluation protocol"
            raise ValueError(msg)
        cohort = yaml.safe_load((manifest.parent / "cohort.yaml").read_text())
        if (
            record["rollout_trajectories"] != cohort["indices"]
            or record["cohort_sha256"] != state["cohort_sha256"]
            or record["windows_sha256"] != state["windows_sha256"]
            or record["rollout_seed"] != state["rollout_seed"]
            or record["paired_seed"] != state["paired_seed"]
        ):
            msg = "Changed sampling identities or seeds"
            raise ValueError(msg)
        for path, checksum in record["files"].items():
            if sha256(folder / path) != checksum:
                raise ValueError(f"Output hash changed: {name}/{path}")
        frames = {
            task: pd.read_csv(folder / f"{task}.csv") for task in ["paired", "rollout"]
        }
        for task, frame in frames.items():
            validate_rows(
                frame,
                horizon=4 if task == "paired" else state["horizon"],
                paired=task == "paired",
            )
            fields = ["selection_position", "trajectory_idx", "lead", "channel"]
            if task == "paired":
                fields += ["file", "trajectory_in_file", "context_frame"]
            current = frame.sort_values(by=fields).loc[:, fields].reset_index(drop=True)
            if task in identities and not current.equals(identities[task]):
                raise ValueError(f"Unpaired {task} identities for {name}")
            identities[task] = current
        latency = json.loads((folder / "latency.json").read_text())
        if (
            latency["members"] != 16
            or latency["repeats"] != 50
            or latency["warmup"] != 5
        ):
            msg = "Incomplete latency benchmark"
            raise ValueError(msg)
        if gpu is not None and gpu != latency["gpu"]:
            msg = "Latency GPU types differ"
            raise ValueError(msg)
        gpu = latency["gpu"]
        results[name] = {**frames, "latency": latency, "provenance": record}
    return state, results


def build(manifest: Path, output: Path) -> None:  # noqa: PLR0915
    """Render publication-style plots with one-SE trajectory bands and tables."""
    state, results = load_results(manifest)
    output.mkdir(parents=True, exist_ok=False)
    styles = {
        name: (
            entry["label"],
            COLORS[i],
            "-."
            if entry["expected_step"] < 262144
            else ("-" if "fm" in name else "--"),
        )
        for i, (name, entry) in enumerate(state["models"].items())
    }
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    tables, latencies = [], []
    for name, data in results.items():
        for task in ["paired", "rollout"]:
            stats = trajectory_stats(data[task], [])
            stats["coverage_mae_mean"], stats["coverage_mae_se"] = (
                calibration_mae_stats(data[task])
            )
            stats.insert(0, "model", name)
            stats.insert(1, "task", task)
            tables.append(stats)
            for group, suffix in [
                (["lead"], "pooled"),
                (["lead", "channel"], "by_channel"),
            ]:
                trajectory_stats(data[task], group).to_csv(
                    output / f"{name}-{task}-{suffix}.csv", index=False
                )
        timings = np.array(data["latency"]["seconds_per_16_member_forecast"])
        if len(timings) != 50 or not np.isfinite(timings).all() or (timings <= 0).any():
            msg = "Invalid measured latency samples"
            raise ValueError(msg)
        latencies.append(
            {
                "model": name,
                "mean_ms_per_ensemble": timings.mean() * 1000,
                "median_ms_per_ensemble": np.median(timings) * 1000,
                "std_ms_per_ensemble": timings.std(ddof=1) * 1000,
                "gpu": data["latency"]["gpu"],
            }
        )
    pd.concat(tables).to_csv(output / "overall_mean_se.csv", index=False)
    pd.DataFrame(latencies).to_csv(output / "inference_latency.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    for name, data in results.items():
        label, color, linestyle = styles[name]
        stats = trajectory_stats(data["paired"], []).iloc[0]
        mean = np.array([stats[f"{key}_mean"] for key in COVERAGES])
        se = np.array([stats[f"{key}_se"] for key in COVERAGES])
        ax.plot(LEVELS, mean, label=label, color=color, ls=linestyle)
        ax.fill_between(LEVELS, mean - se, mean + se, color=color, alpha=0.13)
    ax.plot([0, 1], [0, 1], color=".5", ls=":")
    ax.set(
        xlim=(0, 1), ylim=(0, 1), xlabel="Nominal coverage", ylabel="Empirical coverage"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "single_window_reliability.png", dpi=240)
    plt.close(fig)
    # Match the paper's four reliability windows exactly. All model outputs
    # share this prefix even when the diagnostic campaign retains 180 frames.
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))
    for row, (start, stop) in enumerate([(0, 4), (6, 12), (13, 30), (31, 99)]):
        for name, data in results.items():
            label, color, linestyle = styles[name]
            frame = data["rollout"]
            window = frame[frame.lead.gt(start) & frame.lead.le(stop)]
            stats = trajectory_stats(window, []).iloc[0]
            mean = np.array([stats[f"{key}_mean"] for key in COVERAGES])
            se = np.array([stats[f"{key}_se"] for key in COVERAGES])
            axes[row, 0].plot(LEVELS, mean, color=color, ls=linestyle, label=label)
            axes[row, 0].fill_between(
                LEVELS, mean - se, mean + se, color=color, alpha=0.13
            )
            curve = trajectory_stats(frame[frame.lead.le(100)], ["lead"])
            metric = f"coverage_{[0.9, 0.5, 0.1][row]:g}" if row < 3 else "vrmse_v2"
            mean = curve[f"{metric}_mean"] - ([0.9, 0.5, 0.1][row] if row < 3 else 0)
            se = curve[f"{metric}_se"]
            axes[row, 1].plot(curve.lead - 1, mean, color=color, ls=linestyle)
            axes[row, 1].fill_between(
                curve.lead - 1, mean - se, mean + se, color=color, alpha=0.13
            )
        axes[row, 0].plot([0, 1], [0, 1], color=".5", ls=":")
        axes[row, 0].set(
            title=f"Forecast indices [{start}:{stop})", xlim=(0, 1), ylim=(0, 1)
        )
        axes[row, 1].set(
            title=f"Coverage error: {[90, 50, 10][row]}%" if row < 3 else "VRMSE",
            xlabel="Forecast index",
        )
        if row < 3:
            axes[row, 1].axhline(0, color=".5", ls=":")
    limits = max(abs(v) for ax in axes[:3, 1] for v in ax.get_ylim())
    for ax in axes[:3, 1]:
        ax.set_ylim(-limits, limits)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "rollout_reliability_mean_se.png", dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(7, 1, figsize=(7, 13))
    for ax, metric in zip(
        axes,
        [
            "vrmse_v2",
            "crps",
            "ssr",
            "energy",
            "psrmse_high",
            "psrmse_mid",
            "psrmse_low",
        ],
        strict=True,
    ):
        for name, data in results.items():
            label, color, linestyle = styles[name]
            curve = trajectory_stats(data["rollout"], ["lead"])
            mean, se = curve[f"{metric}_mean"], curve[f"{metric}_se"]
            ax.plot(curve.lead - 1, mean, label=label, color=color, ls=linestyle)
            ax.fill_between(
                curve.lead - 1, mean - se, mean + se, color=color, alpha=0.13
            )
        ax.set(ylabel=metric, xlabel="Forecast index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "appendix_metrics_mean_se.png", dpi=240)
    plt.close(fig)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
                "source_commit": state["source_commit"],
                "models": list(results),
                "statistics": (
                    "Trajectory is sampling unit; bands +/-1 SE; "
                    "conditional on frozen checkpoints"
                ),
                "input_provenance": {
                    name: data["provenance"] for name, data in results.items()
                },
                "files": {
                    p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Expose comparison generation without running any model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
