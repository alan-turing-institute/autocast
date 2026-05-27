"""Plot comparison summaries from Autocast evaluation outputs."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure

plt.switch_backend("Agg")

log = logging.getLogger(__name__)

DEFAULT_PLOT_METRICS = ("vrmse", "coverage", "crps", "ssr")
METRIC_LABELS = {
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "vrmse": "VRMSE",
    "crps": "CRPS",
    "coverage": "Coverage",
    "ssr": "SSR",
}
DATASET_LABELS = {
    "ad64": "AD64",
    "gray_scott": "GS64",
    "gs64": "GS64",
    "cns64": "CNS64",
    "conditioned_navier_stokes": "CNS64",
    "shallow_water": "SW64",
    "sw64": "SW64",
}
EVAL_METRIC_FILES = (
    "evaluation_metrics.csv",
    "rollout_metrics.csv",
    "rollout_metrics_per_timestep_channel_all.csv",
)


@dataclass(frozen=True)
class RunTarget:
    """A single run/evaluation directory selected for plotting."""

    path: Path
    ref: str
    relative_path: str
    eval_subdir: str
    label: str | None = None


def normalize_eval_subdir(eval_subdir: str | None) -> str:
    """Normalize an evaluation subdirectory name."""
    if not eval_subdir:
        return "eval"
    return eval_subdir.strip().strip("/") or "eval"


def _is_eval_metrics_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in EVAL_METRIC_FILES)


def _eval_sort_key(eval_subdir: str) -> tuple[int, str]:
    return (0 if eval_subdir == "eval" else 1, eval_subdir)


def _available_eval_subdirs(run_dir: Path) -> list[str]:
    """Return evaluation subdirs that contain metrics for a run."""
    if not run_dir.exists():
        return []
    return sorted(
        [
            child.name
            for child in run_dir.iterdir()
            if child.name.startswith("eval") and _is_eval_metrics_dir(child)
        ],
        key=_eval_sort_key,
    )


def _select_default_eval_subdir(eval_subdirs: Sequence[str]) -> str:
    """Select one eval directory per run to avoid accidental duplicates."""
    if "eval" in eval_subdirs:
        return "eval"
    return eval_subdirs[0]


def _format_available_eval_subdirs(eval_subdirs: Sequence[str], limit: int = 3) -> str:
    if len(eval_subdirs) <= limit:
        return ", ".join(eval_subdirs)
    visible = ", ".join(eval_subdirs[:limit])
    return f"{visible}, ... (+{len(eval_subdirs) - limit})"


def _discover_run_targets(results_dir: Path) -> list[RunTarget]:
    """Discover run directories and select one eval output from each."""
    targets: list[RunTarget] = []
    for config_path in sorted(results_dir.rglob("resolved_config.yaml")):
        run_dir = config_path.parent
        try:
            rel_parts = run_dir.relative_to(results_dir).parts
        except ValueError:
            rel_parts = ()
        if "plots" in rel_parts:
            continue

        eval_subdirs = _available_eval_subdirs(run_dir)
        if not eval_subdirs:
            continue
        eval_subdir = _select_default_eval_subdir(eval_subdirs)
        relative_path = str(run_dir.relative_to(results_dir))
        ref = run_dir.name
        if eval_subdir != "eval":
            ref = f"{ref}::eval={eval_subdir}"
        targets.append(
            RunTarget(
                path=run_dir,
                ref=ref,
                relative_path=relative_path,
                eval_subdir=eval_subdir,
            )
        )
    return targets


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _dataset_from_config(config: dict[str, Any]) -> str | None:
    datamodule = config.get("datamodule")
    if isinstance(datamodule, str):
        return datamodule
    if isinstance(datamodule, dict):
        dataset = datamodule.get("dataset")
        if isinstance(dataset, str) and dataset:
            return dataset
        data_path = datamodule.get("data_path")
        if data_path:
            return Path(str(data_path)).name
    dataset = config.get("dataset")
    return dataset if isinstance(dataset, str) and dataset else None


def dataset_label_from_module(dataset_module: str | None) -> str | None:
    """Return a compact display label for a dataset module or path name."""
    if not dataset_module:
        return None
    key = Path(str(dataset_module)).name.lower().replace("-", "_")
    return DATASET_LABELS.get(key, key.upper())


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _numeric_row_values(row: pd.Series, skip: Iterable[str]) -> dict[str, float]:
    skip_set = set(skip)
    values: dict[str, float] = {}
    for col, value in row.items():
        if str(col) in skip_set:
            continue
        parsed = _as_finite_float(value)
        if parsed is not None:
            values[str(col)] = parsed
    return values


def _first_overall_row(df: pd.DataFrame) -> pd.Series | None:
    selected = df
    if "window" in selected.columns:
        mask = cast(pd.Series, selected["window"].astype(str).eq("all"))
        selected = cast(pd.DataFrame, selected.loc[mask])
    if "batch_idx" in selected.columns:
        mask = cast(pd.Series, selected["batch_idx"].astype(str).eq("all"))
        selected = cast(pd.DataFrame, selected.loc[mask])
    if selected.empty:
        return None
    return selected.iloc[0]


def load_single_run_metrics(
    run_dir: Path,
    *,
    eval_subdir: str | None = "eval",
    run_ref: str | None = None,
    run_path: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Load scalar metrics and metadata for one run."""
    eval_name = normalize_eval_subdir(eval_subdir)
    eval_dir = run_dir / eval_name
    config = _read_yaml_mapping(run_dir / "resolved_config.yaml")
    dataset = _dataset_from_config(config)

    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_path": run_path or run_dir.name,
        "run_ref": run_ref or run_dir.name,
        "eval_subdir": eval_name,
        "dataset": dataset,
        "dataset_label": dataset_label_from_module(dataset) or "unknown",
        "plot_group": label or run_ref or run_dir.name,
    }

    eval_csv = eval_dir / "evaluation_metrics.csv"
    if eval_csv.exists():
        overall = _first_overall_row(pd.read_csv(eval_csv))
        if overall is not None:
            for metric, value in _numeric_row_values(
                overall, skip=("window", "batch_idx")
            ).items():
                row[f"overall_{metric}"] = value

    rollout_csv = eval_dir / "rollout_metrics.csv"
    if rollout_csv.exists():
        rollout = pd.read_csv(rollout_csv)
        if "window" in rollout.columns:
            if "batch_idx" in rollout.columns:
                rollout = rollout[rollout["batch_idx"].astype(str) == "all"]
            for _, metric_row in rollout.iterrows():
                window = str(metric_row["window"])
                for metric, value in _numeric_row_values(
                    metric_row, skip=("window", "batch_idx")
                ).items():
                    row[f"{metric}_{window}"] = value

    return row


def _resolve_run_dir(results_dir: Path, run_id: str) -> Path:
    direct = results_dir / run_id
    if direct.exists():
        return direct
    matches = sorted(path for path in results_dir.rglob(run_id) if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        msg = f"Run id {run_id!r} matched multiple directories"
        raise ValueError(msg)
    msg = f"Run id {run_id!r} was not found under {results_dir}"
    raise FileNotFoundError(msg)


def _parse_run_spec(results_dir: Path, tokens: Sequence[str]) -> RunTarget:
    if not tokens:
        msg = "--run requires at least a run id"
        raise ValueError(msg)
    run_id = tokens[0]
    label_parts: list[str] = []
    eval_subdir = "eval"
    for token in tokens[1:]:
        if token.startswith("eval="):
            eval_subdir = normalize_eval_subdir(token.split("=", 1)[1])
        else:
            label_parts.append(token)
    run_dir = _resolve_run_dir(results_dir, run_id)
    relative_path = str(run_dir.relative_to(results_dir))
    label = " ".join(label_parts) if label_parts else None
    ref = run_id if eval_subdir == "eval" else f"{run_id}::eval={eval_subdir}"
    return RunTarget(
        path=run_dir,
        ref=ref,
        relative_path=relative_path,
        eval_subdir=eval_subdir,
        label=label,
    )


def load_runs(
    results_dir: Path,
    *,
    run_specs: Sequence[Sequence[str]] | None = None,
    run_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load selected runs from a results directory."""
    targets: list[RunTarget] = []
    if run_specs:
        targets.extend(_parse_run_spec(results_dir, spec) for spec in run_specs)
    elif run_ids:
        targets.extend(_parse_run_spec(results_dir, [run_id]) for run_id in run_ids)
    else:
        targets.extend(_discover_run_targets(results_dir))

    rows = [
        load_single_run_metrics(
            target.path,
            eval_subdir=target.eval_subdir,
            run_ref=target.ref,
            run_path=target.relative_path,
            label=target.label,
        )
        for target in targets
    ]
    return pd.DataFrame(rows)


def _contains_any(series: pd.Series, values: Sequence[str]) -> pd.Series:
    requested = [value.lower() for value in values]
    lowered = series.fillna("").astype(str).str.lower()
    return lowered.apply(lambda value: any(item in value for item in requested))


def filter_runs(
    df: pd.DataFrame,
    *,
    datasets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    filters: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply simple CLI filters to a run table."""
    out = df.copy()
    if datasets and "dataset_label" in out.columns:
        mask = _contains_any(cast(pd.Series, out["dataset_label"]), datasets)
        out = cast(pd.DataFrame, out.loc[mask])
    if models and "run_name" in out.columns:
        mask = _contains_any(cast(pd.Series, out["run_name"]), models)
        out = cast(pd.DataFrame, out.loc[mask])
    for expr in filters or ():
        if "=" not in expr:
            msg = f"Filter {expr!r} must use column=value"
            raise ValueError(msg)
        column, value = [part.strip() for part in expr.split("=", 1)]
        if column not in out.columns:
            msg = f"Filter column {column!r} is not present"
            raise ValueError(msg)
        mask = cast(pd.Series, out[column].astype(str).eq(value))
        out = cast(pd.DataFrame, out.loc[mask])
    return out


def _ordered_unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _metric_label(metric: str) -> str:
    if metric.startswith("coverage_"):
        return f"Coverage {metric.removeprefix('coverage_')}"
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _style_by_group(groups: Sequence[str]) -> dict[str, dict[str, str]]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return {
        group: {"color": colors[idx % len(colors)], "label": group}
        for idx, group in enumerate(groups)
    }


def _save_fig(
    fig: Figure,
    out_dir: Path,
    name: str,
    formats: Sequence[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_overall_metric(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    *,
    figure_formats: Sequence[str] = ("png",),
) -> Figure | None:
    """Plot one grouped bar chart for an overall metric."""
    column = f"overall_{metric}"
    if column not in df.columns:
        log.warning("Skipping %s; column %s is not present", metric, column)
        return None
    data = cast(pd.DataFrame, df.dropna(subset=[column]).copy())
    if data.empty:
        log.warning("Skipping %s; no finite values", metric)
        return None

    datasets = _ordered_unique(cast(pd.Series, data["dataset_label"]))
    groups = _ordered_unique(cast(pd.Series, data["plot_group"]))
    styles = _style_by_group(groups)
    summary = cast(
        pd.DataFrame,
        data.groupby(["dataset_label", "plot_group"], sort=False)[column]
        .mean()
        .reset_index(),
    )

    x = np.arange(len(datasets))
    width = min(0.8 / max(len(groups), 1), 0.35)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * len(datasets)), 4.0))
    for idx, group in enumerate(groups):
        offsets = x + (idx - (len(groups) - 1) / 2) * width
        values = []
        for dataset in datasets:
            match = summary[
                (summary["dataset_label"] == dataset) & (summary["plot_group"] == group)
            ]
            match_values = cast(pd.Series, match[column])
            values.append(float(match_values.iloc[0]) if not match.empty else np.nan)
        ax.bar(
            offsets,
            values,
            width=width,
            color=styles[group]["color"],
            label=styles[group]["label"],
        )

    if metric == "ssr":
        ax.axhline(1.0, color="0.3", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(f"Overall {_metric_label(metric)}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save_fig(fig, out_dir, f"overall_{metric}", figure_formats)
    return fig


def _load_timestep_metric(eval_dir: Path, metric: str) -> pd.Series | None:
    csv_path = eval_dir / "rollout_metrics_per_timestep_channel_all.csv"
    if not csv_path.exists():
        return None
    table = pd.read_csv(csv_path, index_col=0)
    if metric not in table.index:
        return None
    raw = table.loc[metric]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[0]
    raw_series = cast(pd.Series, raw)
    values = pd.Series(
        pd.to_numeric(raw_series, errors="coerce"),
        index=raw_series.index,
    ).dropna()
    if values.empty:
        return None
    with suppress(ValueError):
        values.index = [int(col) for col in values.index]
    return values


def plot_lead_time_metric(
    df: pd.DataFrame,
    results_dir: Path,
    metric: str,
    out_dir: Path,
    *,
    figure_formats: Sequence[str] = ("png",),
) -> Figure | None:
    """Plot one rollout lead-time curve per selected run."""
    groups = _ordered_unique(cast(pd.Series, df["plot_group"]))
    styles = _style_by_group(groups)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plotted = 0
    for _, row in df.iterrows():
        eval_dir = results_dir / str(row["run_path"]) / str(row["eval_subdir"])
        series = _load_timestep_metric(eval_dir, metric)
        if series is None:
            continue
        group = str(row["plot_group"])
        ax.plot(
            series.index,
            series.to_numpy(dtype=float),
            color=styles[group]["color"],
            marker="o",
            linewidth=1.8,
            label=group,
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        log.warning("Skipping lead-time %s; no timestep data found", metric)
        return None

    if metric == "ssr":
        ax.axhline(1.0, color="0.3", linestyle=":", linewidth=1)
    ax.set_xlabel("Lead time")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(f"Lead-time {_metric_label(metric)}")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    ax.grid(alpha=0.25)
    _save_fig(fig, out_dir, f"lead_time_{metric}", figure_formats)
    return fig


def write_run_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Write the selected run table used by the generated plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "selected_runs.csv", index=False)


def format_run_table(df: pd.DataFrame) -> str:
    """Format selected runs for ``--list`` output."""
    columns = [
        col
        for col in ("run_path", "eval_subdir", "dataset_label", "plot_group")
        if col in df.columns
    ]
    metric_columns = sorted(col for col in df.columns if col.startswith("overall_"))
    return df[columns + metric_columns].to_string(index=False)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add plotting arguments to an argparse parser."""
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Root directory containing run outputs",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for plots (defaults to <results-dir>/plots/<name>)",
    )
    parser.add_argument("--name", default="comparison", help="Default plot folder name")
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        choices=("png", "pdf"),
        default=["png"],
        help="Figure file formats to write",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        action="append",
        help="Add one run: --run <id> [label words...] [eval=<subdir>]",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Run directory names to include when --run labels are not needed",
    )
    parser.add_argument("--datasets", nargs="+", help="Filter by dataset label")
    parser.add_argument("--models", nargs="+", help="Filter by run name substring")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Exact column filter in column=value form; repeat as needed",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_PLOT_METRICS),
        help="Overall metrics to plot",
    )
    parser.add_argument(
        "--lead-time-metrics",
        nargs="+",
        default=[],
        help="Rollout per-timestep metrics to plot",
    )
    parser.add_argument("--list", action="store_true", help="Print selected runs")
    parser.add_argument("--sort", help="Column to sort selected runs by")
    parser.add_argument("--reverse", action="store_true", help="Reverse sort order")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate small comparison plots from Autocast evaluation outputs."
    )
    add_arguments(parser)
    return parser


def _argument_error(
    parser: argparse.ArgumentParser | None,
    message: str,
) -> None:
    if parser is not None:
        parser.error(message)
    raise ValueError(message)


def run_from_args(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    """Run plotting from parsed command-line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_dir / "plots" / args.name
    )

    df = load_runs(results_dir, run_specs=args.run, run_ids=args.runs)
    df = filter_runs(
        df,
        datasets=args.datasets,
        models=args.models,
        filters=args.filter,
    )
    if args.sort:
        if args.sort not in df.columns:
            _argument_error(parser, f"--sort column {args.sort!r} is not present")
        df = df.sort_values(args.sort, ascending=not args.reverse)
    if df.empty:
        _argument_error(parser, "no runs matched the requested selection")

    write_run_table(df, output_dir)
    if args.list:
        print(format_run_table(df))
        return

    for metric in args.metrics:
        plot_overall_metric(
            df,
            metric,
            output_dir,
            figure_formats=args.figure_formats,
        )
    for metric in args.lead_time_metrics:
        plot_lead_time_metric(
            df,
            results_dir,
            metric,
            output_dir,
            figure_formats=args.figure_formats,
        )

    log.info("Wrote plots to %s", output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the plotting CLI."""
    parser = build_parser()
    run_from_args(parser.parse_args(argv), parser=parser)


if __name__ == "__main__":
    main()
