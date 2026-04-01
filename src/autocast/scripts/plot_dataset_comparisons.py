from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import yaml

mpl.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DATASET_LABEL_OVERRIDES = {
    "ad64": "AD64",
    "adm32": "ADM32",
    "cns64": "CNS64",
    "gpe_ri_high_complexity": "GPE-RI high",
    "gpe_ri_low_complexity": "GPE-RI low",
    "gpehc64": "GPEHC64",
    "gpelc64": "GPELC64",
    "gs64": "GS64",
    "lb128x32": "LB128x32",
    "rd64": "RD64",
    "sw2d464": "SW2D464",
    "sw2d64": "SW2D64",
}

ARCHITECTURE_PREFIXES = (
    "flow_matching_vit",
    "flow_matching_large",
    "flow_matching",
    "fno_concat",
    "fno",
    "vit_latent",
    "vit_large",
    "vit",
    "diffusion_vit",
    "diffusion",
    "unet_large",
    "unet",
    "swin",
    "unet_azula_large",
    "unet_azula_small",
    "unet_large_concat",
    "unet_small_concat",
    "swin_large",
    "swin_small",
    "vit_azula_large",
    "vit_azula_small",
)

MODEL_FAMILY_DISPLAY_LABELS = {
    "flow_matching_vit": "FM ViT",
    "fno_concat": "FNO",
    "flow_matching": "FM U-Net",
    "flow_matching_large": "FM U-Net (large)",
    "vit_large": "ViT (large)",
    "vit": "ViT (small)",
    "vit_latent": "ViT latent",
    "diffusion": "Diffusion U-Net",
    "diffusion_vit": "Diffusion ViT",
    "unet_azula_large": "Azula U-Net (large)",
    "unet_azula_small": "Azula U-Net (small)",
    "unet_large_concat": "U-Net (large)",
    "unet_small_concat": "U-Net (small)",
    "swin_large": "Swin (large)",
    "swin_small": "Swin (small)",
    "vit_azula_large": "Azula ViT (large)",
    "vit_azula_small": "Azula ViT (small)",
}

ROLL_WINDOWS = ["0-1", "0-4", "6-12", "13-30", "31-99"]
WINDOW_ROWS = ["all", *ROLL_WINDOWS]
MODEL_SCALE_PARAM_COL = "params_processor_total"

# ---------------------------------------------------------------------------
# Core Utilities
# ---------------------------------------------------------------------------


def resolve_results_root(outputs_dir: str) -> Path:  # noqa: D103
    p = Path(outputs_dir).expanduser()
    return (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()


def dataset_label_from_module(dataset_module: str | None) -> str | None:
    if not dataset_module or pd.isna(dataset_module):
        return None
    if dataset_module in DATASET_LABEL_OVERRIDES:
        return DATASET_LABEL_OVERRIDES[dataset_module]
    return str(dataset_module).replace("_", " ").title()


def _dataset_candidates() -> list[str]:
    base = set(DATASET_LABEL_OVERRIDES.keys())
    return sorted(base, key=len, reverse=True)


def arch_key_from_processor_segment(segment: str) -> str:
    seg = str(segment)
    if not seg:
        return "unknown"
    for pref in ARCHITECTURE_PREFIXES:
        if seg == pref or seg.startswith(pref + "_"):
            return pref
    return seg.split("_")[0]


def parse_loss_dataset_arch(
    run_name: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not run_name:
        return None, None, None
    m = re.match(r"^(diff|crps)_(.+)_([0-9a-f]{7}|[a-z]+)_[0-9a-f]{7}$", str(run_name))
    if not m:
        return None, None, None
    loss, mid = m.group(1), m.group(2)
    for ds in _dataset_candidates():
        pfx = ds + "_"
        if mid.startswith(pfx):
            return loss, ds, mid[len(pfx) :]
    return loss, None, mid


def normalize_dataset_module(dataset: str | None) -> str | None:
    if pd.isna(dataset) or not dataset:
        return None
    m = re.match(r"^(?P<base>.+)_(?P<suffix>[0-9a-f]{6,}|\d{6,})$", str(dataset))
    return m.group("base") if m else str(dataset)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_single_run_metrics(run_dir: Path) -> dict:
    row = {"run_name": run_dir.name, "run_path": run_dir.name, "dataset": None}
    eval_dir = run_dir / "eval"

    # Evaluation metrics
    p_eval = eval_dir / "evaluation_metrics.csv"
    if p_eval.exists():
        em = pd.read_csv(p_eval)
        if not em.empty and "window" in em.columns:
            em["window"] = em["window"].astype(str)
            if "batch_idx" in em.columns:
                em["batch_idx"] = em["batch_idx"].astype(str)
                ov = em[(em["window"] == "all") & (em["batch_idx"] == "all")]
                if ov.empty:
                    ov = em[em["window"] == "all"]
            else:
                ov = em[em["window"] == "all"]
            if not ov.empty:
                rr = ov.iloc[0]
                for col in em.columns:
                    if col in {"window", "batch_idx"}:
                        continue
                    row[f"overall_{col}"] = pd.to_numeric(rr[col], errors="coerce")

    # Rollout metrics
    p_roll = eval_dir / "rollout_metrics.csv"
    if p_roll.exists():
        rm = pd.read_csv(p_roll)
        if not rm.empty and "window" in rm.columns:
            rm["window"] = rm["window"].astype(str)
            if "batch_idx" in rm.columns:
                rm = rm[rm["batch_idx"].astype(str) == "all"]
            for _, rr in rm.iterrows():
                w = rr["window"]
                for col in rm.columns:
                    if col in {"window", "batch_idx"}:
                        continue
                    row[f"{col}_{w}"] = pd.to_numeric(rr[col], errors="coerce")

    # Evaluation metadata (Params)
    p_meta = eval_dir / "evaluation_metadata.csv"
    if p_meta.exists():
        md = pd.read_csv(p_meta)
        if not md.empty and {"category", "metric", "value"}.issubset(md.columns):
            params = md[md["category"] == "params"]
            for metric, out_col in [
                ("model_total", "params_model_total"),
                ("processor_total", "params_processor_total"),
            ]:
                s = params.loc[params["metric"] == metric, "value"]
                if not s.empty:
                    row[out_col] = pd.to_numeric(s.iloc[0], errors="coerce")
    return row


def assign_model_scale(df_in: pd.DataFrame) -> pd.Series:
    out = pd.Series("large", index=df_in.index, dtype="object")
    if MODEL_SCALE_PARAM_COL not in df_in.columns:
        return out
    params = pd.to_numeric(df_in[MODEL_SCALE_PARAM_COL], errors="coerce")
    group_cols = [
        c for c in ["dataset_module", "loss_family", "arch_key"] if c in df_in.columns
    ]
    if len(group_cols) < 2:
        return out

    for _, g in df_in.groupby(group_cols, dropna=False):
        idx = g.index
        p = params.loc[idx]
        vals = sorted(p.dropna().unique().tolist())
        if len(vals) <= 1:
            out.loc[idx] = "large"
            continue
        low, high = vals[0], vals[-1]
        out.loc[p[p == low].index] = "small"
        out.loc[p[p == high].index] = "large"
        for ii in p[(p != low) & (p != high)].index:
            val = p.loc[ii]
            out.loc[ii] = (
                "small"
                if pd.notna(val) and abs(val - low) <= abs(val - high)
                else "large"
            )
    return out


def load_config_metadata(run_dir: Path) -> dict:
    row = {"run_name": run_dir.name}
    p_config = run_dir / "resolved_config.yaml"
    if p_config.exists():
        try:
            with open(p_config) as f:
                cfg = yaml.safe_load(f)
            if cfg:
                row["n_steps_output"] = cfg.get("datamodule", {}).get("n_steps_output")
                row["batch_size"] = cfg.get("datamodule", {}).get("batch_size")
                row["dataset"] = Path(
                    cfg.get("datamodule", {}).get("data_path", "")
                ).name
                row["loss_func"] = (
                    cfg.get("model", {})
                    .get("loss_func", {})
                    .get("_target_", "")
                    .split(".")[-1]
                )
                row["processor"] = (
                    cfg.get("model", {})
                    .get("processor", {})
                    .get("_target_", "")
                    .split(".")[-1]
                )
                inj = cfg.get("model", {}).get("input_noise_injector", {})
                row["noise_injector"] = (
                    inj.get("_target_", "").split(".")[-1] if inj else "None"
                )
                row["noise_channels"] = inj.get("n_noise_channels") if inj else 0
                row["lr"] = cfg.get("optimizer", {}).get("learning_rate")
        except Exception:
            pass
    return row


# ---------------------------------------------------------------------------
# Coloring / Styling
# ---------------------------------------------------------------------------


def _mix_with_white(color, frac):
    r, g, b, a = color
    return (r + (1.0 - r) * frac, g + (1.0 - g) * frac, b + (1.0 - b) * frac, a)


def _mix_with_black(color, frac):
    r, g, b, a = color
    scale = max(0.0, 1.0 - frac)
    return (r * scale, g * scale, b * scale, a)


def _rgb_luminance(rgba):
    return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]


def get_hue_and_lightness(
    pg: str, df_in: pd.DataFrame, explicit_groups: list[list[str]] | None
):
    """
    Returns (base_color, assigned_color, is_diff, scale).
    If explicit_groups is used, different groups get different base_colors.
    Within a group, lightness varies.
    """
    parts = pg.split("__")
    arch = parts[0]
    loss = parts[1] if len(parts) > 1 else "unknown"
    scale = parts[2] if len(parts) > 2 else "large"

    if explicit_groups:
        # Find which group index contains a run resolving to this pg.
        # This requires tracking which run names belong to which group and their pg.
        # We handle this mapping explicitly in build_family_style.
        pass

    # Default notebook scheme fallback
    cmap = plt.get_cmap("tab20")
    hues = sorted({p.split("__")[0] for p in df_in["plot_group"].unique()})
    h_idx = hues.index(arch) % 10 if arch in hues else 0
    c0, c1 = cmap(2 * h_idx), cmap(2 * h_idx + 1)
    light, dark = (c0, c1) if _rgb_luminance(c0) > _rgb_luminance(c1) else (c1, c0)
    color = light if scale == "small" else dark
    return color, loss == "diff", scale


def plot_group_display_label(pg: str) -> str:
    parts = pg.split("__")
    arch, loss, scale = (
        parts[0],
        parts[1] if len(parts) > 1 else "?",
        parts[2] if len(parts) > 2 else "?",
    )
    run_name = parts[3] if len(parts) > 3 else ""
    model_l = MODEL_FAMILY_DISPLAY_LABELS.get(arch, arch)
    loss_l = {"crps": "CRPS", "diff": "FM"}.get(loss, loss)

    h = ""
    if run_name:
        m = re.search(r"_[0-9a-f]{7}$", run_name)
        h = f" [{run_name[-7:]}]" if m else f" [{run_name[:7]}]"

    return f"{model_l} · {scale} · {loss_l}{h}"


def extract_valid_plot_groups_from_run_names(
    run_names: list[str], df_in: pd.DataFrame
) -> set[str]:
    # Given a list of run dir names, find their corresponding plot_groups in df_in
    return set(df_in[df_in["run_name"].isin(run_names)]["plot_group"].unique())


def build_family_style(
    df_in: pd.DataFrame, explicit_groups: list[list[str]] | None = None
) -> dict:
    styles = {}
    present = sorted(df_in["plot_group"].dropna().unique())

    if explicit_groups and len(explicit_groups) > 0:
        base_cmap = plt.get_cmap("tab10")
        for i, group_runs in enumerate(explicit_groups):
            base_color = base_cmap(i % 10)
            group_pgs = extract_valid_plot_groups_from_run_names(group_runs, df_in)
            group_pgs = sorted(list(group_pgs))
            if not group_pgs:
                continue

            # Vary lightness across the group's plot groups
            n = len(group_pgs)
            for j, pg in enumerate(group_pgs):
                if pg in styles:
                    continue  # Conflict resolution (first claims)
                # j=0 -> normal, j=1 -> lighter, j=2 -> darker etc
                if n == 1:
                    c = base_color
                elif n == 2:
                    c = (
                        _mix_with_white(base_color, 0.4)
                        if j == 0
                        else _mix_with_black(base_color, 0.2)
                    )
                else:
                    frac = (j / (n - 1)) - 0.5  # -0.5 to 0.5
                    c = (
                        _mix_with_white(base_color, frac * 1.2)
                        if frac > 0
                        else _mix_with_black(base_color, -frac * 0.8)
                    )

                parts = pg.split("__")
                loss = parts[1] if len(parts) > 1 else "unknown"
                styles[pg] = {
                    "color": c,
                    "label": plot_group_display_label(pg),
                    "marker": "^" if loss == "diff" else "o",
                    "linestyle": "-" if loss == "diff" else "--",
                }

    # Fallback for remaining
    for pg in present:
        if pg not in styles:
            c, is_diff, scale = get_hue_and_lightness(pg, df_in, None)
            styles[pg] = {
                "color": c,
                "label": plot_group_display_label(pg),
                "marker": "^" if is_diff else "o",
                "linestyle": "-" if is_diff else "--",
            }
    return styles


# ---------------------------------------------------------------------------
# Plotting Implementations
# ---------------------------------------------------------------------------


def save_fig(fig, out_dir: Path, name: str):
    p = out_dir / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)


def grouped_bar(
    df_in: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    styles: dict,
):
    if metric not in df_in.columns:
        return
    d = df_in.dropna(subset=[metric]).copy()
    if d.empty:
        return

    group_col = "plot_group"
    g = (
        d.groupby(["dataset_label", group_col], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    datasets = sorted(g["dataset_label"].unique())
    families = sorted(g[group_col].unique())
    x = np.arange(len(datasets), dtype=float)
    width = 0.8 / max(1, len(families))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(datasets) + 2.0), 3.8))
    all_positive = []

    for i, fam in enumerate(families):
        vals = pd.to_numeric(
            g[g[group_col] == fam].set_index("dataset_label").reindex(datasets)[metric],
            errors="coerce",
        ).to_numpy(dtype=float)
        all_positive.extend(v for v in vals if np.isfinite(v) and v > 0)
        offs = (i - (len(families) - 1) / 2.0) * width
        style = styles.get(fam, {"color": "k", "label": fam})
        ax.bar(
            x + offs,
            vals,
            width=width,
            label=style["label"],
            color=style["color"],
            edgecolor="0.25",
            linewidth=0.6,
            linestyle=style.get("linestyle", "-"),
        )

    if all_positive and min(all_positive) > 0:
        ax.set_yscale("log")

    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=0 if len(datasets) <= 4 else 30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    # Add compact legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            frameon=False,
        )

    save_fig(fig, out_dir, f"{metric}.png")


def plot_coverage_calibration_panel(
    df_in: pd.DataFrame, results_root: Path, out_dir: Path, styles: dict
):
    curves = []
    base = df_in.dropna(
        subset=["run_path", "dataset_label", "plot_group"]
    ).drop_duplicates()
    for _, r in base.iterrows():
        for w in WINDOW_ROWS:
            fn = (
                "test_coverage_window_all.csv"
                if w == "all"
                else f"rollout_coverage_window_{w}.csv"
            )
            p = results_root / r["run_path"] / "eval" / fn
            if p.exists():
                c = pd.read_csv(p)
                if not c.empty and {"coverage_level", "observed_mean"}.issubset(
                    c.columns
                ):
                    c["window"] = w
                    c["dataset_label"] = r["dataset_label"]
                    c["plot_group"] = r["plot_group"]
                    c["run_path"] = r["run_path"]
                    curves.append(c)

    if not curves:
        return
    cur_panel = pd.concat(curves, ignore_index=True)
    datasets = sorted(cur_panel["dataset_label"].unique())
    groups = sorted(cur_panel["plot_group"].unique())

    nrows, ncols = len(WINDOW_ROWS), max(1, len(datasets))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 2.3 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, w in enumerate(WINDOW_ROWS):
        for j, ds_label in enumerate(datasets):
            ax = axes[i][j]
            ax.plot([0, 1], [0, 1], "--", color="k", alpha=0.6)
            sub = cur_panel[
                (cur_panel["window"] == w) & (cur_panel["dataset_label"] == ds_label)
            ]
            for fam in groups:
                sf = sub[sub["plot_group"] == fam]
                if sf.empty:
                    continue
                st = styles.get(fam, {"color": "k", "linestyle": "-"})

                # Plot family mean
                mean_curve = (
                    sf.groupby("coverage_level", as_index=False)["observed_mean"]
                    .mean()
                    .sort_values("coverage_level")
                )
                ax.plot(
                    mean_curve["coverage_level"],
                    mean_curve["observed_mean"],
                    color=st["color"],
                    lw=2,
                    linestyle=st.get("linestyle", "-"),
                )
            if i == 0:
                ax.set_title(ds_label)
            if j == 0:
                ax.set_ylabel(w)
            ax.grid(alpha=0.2)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=styles[g]["color"],
            ls=styles[g].get("linestyle", "-"),
            lw=2,
            label=styles[g]["label"],
        )
        for g in groups
        if g in styles
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    save_fig(fig, out_dir, "coverage_calibration_panel.png")


def plot_lead_time_panel(
    df_in: pd.DataFrame,
    metrics: list[str],
    results_root: Path,
    out_dir: Path,
    name: str,
    styles: dict,
):
    rows = []
    base = df_in.dropna(
        subset=["run_path", "dataset_label", "plot_group"]
    ).drop_duplicates()
    for r in base.itertuples(index=False):
        p = (
            results_root
            / str(r.run_path)
            / "eval"
            / "rollout_metrics_per_timestep_channel_0.csv"
        )
        if not p.exists():
            continue
        raw = pd.read_csv(p, index_col=0)
        if raw.empty:
            continue
        long = raw.reset_index().rename(columns={raw.index.name or "index": "metric"})
        long = long.melt(
            id_vars="metric", var_name="timestep", value_name="value"
        ).dropna()
        long["timestep"] = pd.to_numeric(long["timestep"], errors="coerce")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long[long["metric"].isin(metrics)].copy()
        if long.empty:
            continue
        long["dataset_label"] = r.dataset_label
        long["plot_group"] = r.plot_group
        rows.append(long)

    if not rows:
        return
    metrics_long = pd.concat(rows, ignore_index=True).dropna(subset=["value"])
    datasets = sorted(metrics_long["dataset_label"].unique())
    groups = sorted(metrics_long["plot_group"].unique())

    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 2.7 * nrows),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for r, metric in enumerate(metrics):
        sub = metrics_long[metrics_long["metric"] == metric]
        for c, ds_label in enumerate(datasets):
            ax = axes[r][c]
            ds = sub[sub["dataset_label"] == ds_label]
            is_cov = metric.startswith("coverage_")
            vals = []
            for fam in groups:
                sf = ds[ds["plot_group"] == fam]
                if sf.empty:
                    continue
                agg = (
                    sf.groupby("timestep", as_index=False)["value"]
                    .agg(["mean", "std", "count"])
                    .sort_values("timestep")
                )
                st = styles.get(fam, {"color": "k"})

                m = (
                    agg["mean"].clip(lower=0, upper=1)
                    if is_cov
                    else agg["mean"].clip(lower=1e-6)
                )
                if not is_cov:
                    vals.extend(m.dropna().tolist())
                ax.plot(
                    agg["timestep"],
                    m,
                    color=st["color"],
                    lw=2,
                    linestyle=st.get("linestyle", "-"),
                )
                if (agg["count"] > 1).any():
                    y1 = (
                        (agg["mean"] - agg["std"].fillna(0)).clip(lower=0, upper=1)
                        if is_cov
                        else (agg["mean"] - agg["std"].fillna(0)).clip(lower=1e-6)
                    )
                    y2 = (
                        (agg["mean"] + agg["std"].fillna(0)).clip(lower=0, upper=1)
                        if is_cov
                        else (agg["mean"] + agg["std"].fillna(0)).clip(lower=1e-6)
                    )
                    if not is_cov:
                        vals.extend(y1.dropna().tolist())
                        vals.extend(y2.dropna().tolist())
                    ax.fill_between(
                        agg["timestep"], y1, y2, color=st["color"], alpha=0.15
                    )

            if r == 0:
                ax.set_title(ds_label)
            if r == nrows - 1:
                ax.set_xlabel("Lead time")
            if c == 0:
                ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
            if is_cov:
                ax.set_ylim(0, 1)
            elif vals:
                ax.set_yscale("log")
                ymin = max(1e-6, min(vals) * 0.8) if vals else 1e-6
                ymax = max(vals) * 1.25 if vals else 10.0
                if not np.isfinite(ymin) or np.isnan(ymin):
                    ymin = 1e-6
                if not np.isfinite(ymax) or np.isnan(ymax):
                    ymax = 10.0
                ax.set_ylim(bottom=ymin, top=ymax)
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=styles[g]["color"],
            ls=styles[g].get("linestyle", "-"),
            lw=2,
            label=styles[g]["label"],
        )
        for g in groups
        if g in styles
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset comparison plots (extracted from notebook)"
    )
    parser.add_argument(
        "--results-dir", required=True, help="Path to collated results folder"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save PNG plots (defaults to <results-dir>/plots/<name>)",
    )
    parser.add_argument(
        "--name",
        default="comparison_plots",
        help="Leaf directory name inside plots if --output-dir is not given",
    )
    parser.add_argument(
        "--run-group",
        action="append",
        nargs="+",
        help="Specify a group of runs. E.g. --run-group run1 run2 --run-group run3 run4",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Explicit list of run directory names to include (if not using --run-group)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Filter to specific dataset modules (e.g., ad64 gs64)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Filter to specific model families (e.g., fno_concat unet_large_concat)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["vrmse", "coverage"],
        help="Metrics to plot for rollouts and overall (default: vrmse coverage)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print a summarized table of run metadata and exit",
    )
    parser.add_argument(
        "--key",
        action="append",
        help="Key to filter the metadata table via OR matching. Use with --value.",
    )
    parser.add_argument(
        "--value",
        action="append",
        help="Value matching the --key for filtering the metadata table.",
    )
    args = parser.parse_args()

    results_dir = resolve_results_root(args.results_dir)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = results_dir / "plots" / args.name

    if not args.list:
        out_dir.mkdir(parents=True, exist_ok=True)

    explicit_groups = args.run_group or []
    # Gather explicitly requested runs, or fallback to auto-discover
    if explicit_groups:
        run_dirs = [results_dir / r for g in explicit_groups for r in g]
    elif args.runs:
        run_dirs = [results_dir / r for r in args.runs]
    else:
        run_dirs = sorted(
            [
                p
                for p in results_dir.iterdir()
                if p.is_dir() and (p / "eval" / "evaluation_metrics.csv").exists()
            ]
        )

    print(f"Loading {len(run_dirs)} runs...")
    rows = []
    for d in run_dirs:
        if not d.exists():
            print(f"Warning: requested run not found: {d}")
            continue
        try:
            r = load_single_run_metrics(d)
        except Exception as e:
            print(f"Error loading {d}: {e}")
            continue
        rows.append(r)

    if not rows:
        print("No valid runs to process.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    _parsed = df["run_name"].map(parse_loss_dataset_arch)
    df["loss_family"] = _parsed.map(lambda x: x[0])
    df["dataset_module"] = _parsed.map(lambda x: x[1]).fillna(
        df["dataset"].map(normalize_dataset_module)
    )
    df["arch_segment"] = _parsed.map(lambda x: x[2])
    df["arch_key"] = df["arch_segment"].map(arch_key_from_processor_segment)
    df["dataset_label"] = df["dataset_module"].map(dataset_label_from_module)
    df["model_scale"] = assign_model_scale(df)

    if args.list:
        m_rows = []
        for rd in run_dirs:
            m_rows.append(load_config_metadata(rd))
        mdf = pd.DataFrame(m_rows)
        merge_cols = [
            c
            for c in [
                "run_name",
                "params_processor_total",
                "train_total_s",
                "dataset_label",
                "model_scale",
            ]
            if c in df.columns
        ]
        merged = (
            pd.merge(df[merge_cols], mdf, on="run_name", how="inner")
            if not df.empty
            else mdf
        )

        if "train_total_s" in merged.columns:
            merged["train_hrs"] = (
                pd.to_numeric(merged["train_total_s"], errors="coerce") / 3600
            ).round(1)

        if "params_processor_total" in merged.columns:
            merged["params_M"] = (
                pd.to_numeric(merged["params_processor_total"], errors="coerce") / 1e6
            ).round(1).astype(str) + "M"

        show_cols = [
            "run_name",
            "dataset_label",
            "processor",
            "model_scale",
            "params_M",
            "n_steps_output",
            "loss_func",
            "noise_injector",
            "noise_channels",
            "batch_size",
            "lr",
            "train_hrs",
        ]
        show_cols = [c for c in show_cols if c in merged.columns]

        renames = {
            "run_name": "Run",
            "dataset_label": "Dataset",
            "processor": "Model",
            "model_scale": "Scale",
            "params_M": "Params",
            "n_steps_output": "N_Out",
            "loss_func": "Loss",
            "noise_injector": "NoiseInj",
            "noise_channels": "NoiseC",
            "batch_size": "BS",
            "lr": "LR",
            "train_hrs": "Train_hr",
        }
        merged = merged.rename(columns=renames)

        # Apply Key/Value filtering using OR logic across filters
        if args.key and args.value:
            if len(args.key) != len(args.value):
                print(
                    "Error: --key and --value must be provided the same number of times."
                )
                sys.exit(1)

            mask = pd.Series(False, index=merged.index)
            valid_filters = 0
            for k, v in zip(args.key, args.value):
                col = renames.get(k, k)  # map parameter name to display name if used
                if col in merged.columns:
                    mask = mask | (merged[col].astype(str) == str(v))
                    valid_filters += 1
                else:
                    print(f"Warning: Metadata filter key '{k}' not found.")

            if valid_filters > 0:
                merged = merged[mask]

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", 40)
        print("\n--- Available Runs Metadata ---\n")
        print(
            merged[[r for c, r in renames.items() if r in merged.columns]].to_string(
                index=False
            )
        )
        sys.exit(0)

    # Construct plot_group (includes run_name to guarantee no multi-run averaging)
    df["plot_group"] = (
        df["arch_key"].astype(str)
        + "__"
        + df["loss_family"].astype(str)
        + "__"
        + df["model_scale"].astype(str)
        + "__"
        + df["run_name"].astype(str)
    )

    # Apply dataset and model filters
    if args.datasets:
        df = df[df["dataset_module"].isin(args.datasets)]
    if args.models:
        df = df[df["arch_key"].isin(args.models)]

    if df.empty:
        print("No valid runs remain after applying filters.")
        sys.exit(1)

    # Styling
    styles = build_family_style(df, explicit_groups)

    print(
        f"Found {len(df['dataset_label'].unique())} datasets and {len(df['plot_group'].unique())} model variants."
    )

    # Render overall bars
    for m in args.metrics:
        grouped_bar(
            df, f"overall_{m}", f"Overall {m.upper()}", m.upper(), out_dir, styles
        )

    # Render window bars
    for m in args.metrics:
        for w in ROLL_WINDOWS:
            grouped_bar(
                df, f"{m}_{w}", f"{m.upper()} in window {w}", m.upper(), out_dir, styles
            )

    # Render Calibration panel
    plot_coverage_calibration_panel(df, results_dir, out_dir, styles)

    # Render lead-time panels
    err_metric = [m for m in args.metrics if "coverage" not in m]
    cov_metric = [m for m in args.metrics if "coverage" in m]
    # Add common rollout variants if we only provided generic 'coverage'
    if "coverage" in args.metrics and "coverage_0.9" not in cov_metric:
        cov_metric.extend(["coverage_0.9", "coverage_0.5"])
        err_metric.append("rmse")

    if err_metric:
        plot_lead_time_panel(
            df, err_metric, results_dir, out_dir, "lead_time_panel_error.png", styles
        )
    if cov_metric:
        plot_lead_time_panel(
            df, cov_metric, results_dir, out_dir, "lead_time_panel_coverage.png", styles
        )

    print("Finished generating plots.")


if __name__ == "__main__":
    main()
