from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import matplotlib as mpl
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from matplotlib.lines import Line2D

mpl.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DATASET_LABEL_OVERRIDES = {
    "ad64": "AD",
    "adm32": "ADM32",
    "cns64": "CNS",
    "gpe_ri_high_complexity": "GPE-H",
    "gpe_ri_low_complexity": "GPE-L",
    "gpe_laser_only_wake": "GPE",
    "gpe_laser_wake_only": "GPE",
    "gpe64": "GPE",
    "gpehc64": "GPEHC64",
    "gpelc64": "GPELC64",
    "gs64": "GS",
    "lb128x32": "LB",
    "rd64": "RD",
    "shallow_water2d_128": "SW",
    "sw2d464": "SW4",
    "sw2d64": "SW",
}

# Canonical grid resolution for each known dataset module.
DATASET_RESOLUTION: dict[str, str] = {
    "ad64": "64x64",
    "adm32": "64x64",
    "cns64": "64x64",
    "gpe_ri_high_complexity": "64x64",
    "gpe_ri_low_complexity": "64x64",
    "gpe_laser_only_wake": "64x64",
    "gpe_laser_wake_only": "64x64",
    "gpe64": "64x64",
    "gpehc64": "64x64",
    "gpelc64": "64x64",
    "gs64": "64x64",
    "lb128x32": "128x32",
    "rd64": "64x64",
    "shallow_water2d_128": "128x128",
    "sw2d464": "64x64",
    "sw2d64": "64x64",
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
WINDOW_ROWS_OVERALL_AND_ROLLOUT = ["all", "0-4", "6-12", "13-30", "31-99"]
WINDOW_ROWS_OVERALL_ONLY = ["all"]
WINDOW_ROWS_ROLLOUT_ONLY = ["0-4", "6-12", "13-30", "31-99"]
MODEL_SCALE_PARAM_COL = "params_processor_total"
DEFAULT_EVAL_SUBDIR = "eval"
DEFAULT_PLOT_METRICS: tuple[str, ...] = ("vrmse", "coverage", "crps", "ssr")
DISPERSION_METRICS: set[str] = {"ssr"}
DEFAULT_TICK_LABEL_FONT_SIZE = 10.0
DEFAULT_AXIS_LABEL_FONT_SIZE = 10.0
DEFAULT_LEGEND_FONT_SIZE = 8.0
SINGLE_STEP_RESULTS_TABLE_METRICS: tuple[tuple[str, str], ...] = (
    ("overall_vrmse", "VRMSE"),
    ("overall_crps", "CRPS"),
    ("overall_ssr", "SSR"),
    ("model_latency_ms_per_sample", "Inference latency (ms/sample)"),
    ("train_mean_epoch_s", "Training time (s/epoch)"),
)
SINGLE_STEP_RESULTS_LATEX_HEADERS: dict[str, str] = {
    "Dataset": "Dataset",
    "Model": "Model",
    "VRMSE": "VRMSE",
    "CRPS": "CRPS",
    "SSR": "SSR",
    "Inference latency (ms/sample)": r"\shortstack{Inference\\latency\\(ms/sample)}",
    "Training time (s/epoch)": r"\shortstack{Training\\time\\(s/epoch)}",
}
SINGLE_STEP_RESULTS_LOWER_IS_BETTER = {
    "VRMSE",
    "CRPS",
    "Inference latency (ms/sample)",
    "Training time (s/epoch)",
}
SINGLE_STEP_RESULTS_TARGET_METRICS = {"SSR": 1.0}


def _is_dispersion_metric(metric: str) -> bool:
    """Return True for ratio-style dispersion diagnostics."""
    return metric in DISPERSION_METRICS


def _overall_or_window_bar_ylim(
    metric: str,
    error_ylim: tuple[float | None, float | None] | None,
) -> tuple[float | None, float | None] | None:
    """Apply error y-limits only to error-like scalar metrics."""
    if "coverage" in metric or _is_dispersion_metric(metric):
        return None
    return error_ylim


def _overall_or_window_bar_yscale(metric: str) -> str:
    """Choose a y-scale for scalar overall/window bar charts."""
    if _is_dispersion_metric(metric):
        return "linear"
    return "auto"


def _overall_or_window_bar_ref_value(metric: str) -> float | None:
    """Return the expected reference value for scalar bar charts."""
    if metric == "ssr":
        return 1.0
    return None


def _derive_lead_time_metrics(metrics: list[str]) -> tuple[list[str], list[str]]:
    """Split scalar plot metrics into error and coverage lead-time rows."""
    err_metric = [
        m for m in metrics if "coverage" not in m and not _is_dispersion_metric(m)
    ]
    cov_metric = [m for m in metrics if "coverage" in m]
    dispersion_metric = [m for m in metrics if _is_dispersion_metric(m)]

    # Add common rollout variants if we only provided generic 'coverage'.
    if "coverage" in metrics and "coverage_0.9" not in cov_metric:
        cov_metric.extend(["coverage_0.9", "coverage_0.5"])
        err_metric.append("rmse")

    return err_metric, [*cov_metric, *dispersion_metric]


def _resolve_font_size(
    base_size: float,
    font_scale: float = 1.0,
    explicit_font_size: float | None = None,
) -> float:
    """Resolve configurable plot font sizes without changing defaults."""
    if explicit_font_size is not None:
        return float(explicit_font_size)
    return float(base_size) * float(font_scale)


def _apply_tick_label_style(ax: Axes, tick_label_scale: float = 1.0) -> None:
    """Apply optional tick-label scaling to an axis."""
    if tick_label_scale == 1.0:
        return
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=_resolve_font_size(DEFAULT_TICK_LABEL_FONT_SIZE, tick_label_scale),
    )


def _apply_axis_label_style(ax: Axes, axis_label_scale: float = 1.0) -> None:
    """Apply optional axis-label scaling to an axis."""
    if axis_label_scale == 1.0:
        return
    font_size = _resolve_font_size(DEFAULT_AXIS_LABEL_FONT_SIZE, axis_label_scale)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)


def _set_shared_ylabel(
    fig: FigureBase,
    label: str,
    axis_label_scale: float = 1.0,
) -> None:
    """Set a panel-level y label when repeated row labels would be noisy."""
    if not hasattr(fig, "supylabel"):
        return
    text = fig.supylabel(label)
    if axis_label_scale != 1.0:
        text.set_fontsize(
            _resolve_font_size(DEFAULT_AXIS_LABEL_FONT_SIZE, axis_label_scale)
        )


def _set_shared_xlabel(
    fig: FigureBase,
    label: str,
    axis_label_scale: float = 1.0,
) -> None:
    """Set a panel-level x label when repeated labels would be noisy."""
    if not hasattr(fig, "supxlabel"):
        return
    text = fig.supxlabel(label)
    if axis_label_scale != 1.0:
        text.set_fontsize(
            _resolve_font_size(DEFAULT_AXIS_LABEL_FONT_SIZE, axis_label_scale)
        )


def _scaled_figsize(width: float, height: float, figure_scale: float = 1.0):
    return (width * figure_scale, height * figure_scale)


def _metric_axis_label(metric: str) -> str:
    """Format metric keys as publication-facing axis labels."""
    if metric.startswith("coverage_"):
        level = metric.split("_", 1)[1]
        return f"COVERAGE {level}"
    return metric.upper()


def _coverage_window_axis_label(window: str, short: bool = False) -> str:
    """Format coverage calibration y labels from rollout window values."""
    prefix = "Emp. cov." if short else "Empirical coverage"
    if window == "all":
        return prefix
    window_label = str(window).replace("-", ":")
    return f"{prefix} ({window_label})"


# ---------------------------------------------------------------------------
# Lead-time panel group presets
# ---------------------------------------------------------------------------
# Each preset targets a qualitatively distinct aspect of ensemble forecast
# quality and produces a dedicated lead-time panel (rows=metrics, cols=datasets).
# Fields:
#   metrics:  row keys (must match rows in rollout_metrics_per_timestep_channel_0.csv)
#   name:     output PNG filename
#   title:    short human-readable group title (used for legend/caption)
#   yscale:   'log' (default for error-like scores) or 'linear' (for ratios)
#   ref:      optional y reference line (e.g. 1.0 for SSR's ideal dispersion)
METRIC_GROUP_PRESETS: dict[str, dict[str, object]] = {
    "probabilistic_scores": {
        "metrics": ["crps", "fcrps", "afcrps", "winkler"],
        "name": "lead_time_panel_probabilistic_scores.png",
        "title": "Marginal proper scoring rules (CRPS family + Winkler)",
        "yscale": "log",
        "ref": None,
    },
    "coherence": {
        "metrics": ["energy"],
        "name": "lead_time_panel_coherence.png",
        "title": "Spatiotemporal coherence (Energy Score / multivariate CRPS)",
        "yscale": "log",
        "ref": None,
    },
    "coherence_plus": {
        "metrics": ["energy", "variogram"],
        "name": "lead_time_panel_coherence_plus.png",
        "title": "Spatiotemporal coherence (Energy + Variogram, memory-intensive)",
        "yscale": "log",
        "ref": None,
    },
    "dispersion": {
        "metrics": ["ssr"],
        "name": "lead_time_panel_dispersion.png",
        "title": "Dispersion (Spread-Skill Ratio, ideal=1.0)",
        "yscale": "linear",
        "ref": 1.0,
    },
    "physics_ps": {
        "metrics": [
            "psrmse_low",
            "psrmse_mid",
            "psrmse_high",
            "psrmse_tail",
        ],
        "name": "lead_time_panel_physics_ps.png",
        "title": "Physics: power-spectrum RMSE by band",
        "yscale": "log",
        "ref": None,
    },
    "physics_pscc": {
        "metrics": [
            "pscc_low",
            "pscc_mid",
            "pscc_high",
            "pscc_tail",
        ],
        "name": "lead_time_panel_physics_pscc.png",
        "title": "Physics: power-spectrum cross-correlation RMSE by band",
        "yscale": "log",
        "ref": None,
    },
}

DEFAULT_METRIC_GROUPS: tuple[str, ...] = (
    "probabilistic_scores",
    "coherence",
    "physics_ps",
)


# ---------------------------------------------------------------------------
# Filter expression parser
# ---------------------------------------------------------------------------
# Supports: KEY=VALUE, AND, OR, parentheses.
#   e.g. "Scale=large AND (Dataset=SW2D64 OR Dataset=CNS64)"


def _tokenize_filter(expr: str) -> list[str]:
    """Split a filter expression into tokens.

    Recognised tokens: ``(``, ``)``, ``AND``, ``OR``, ``KEY=VALUE``.
    """
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in " \t":
            i += 1
            continue
        if c in "()":
            tokens.append(c)
            i += 1
            continue
        # Read a word (could be AND/OR or a KEY=VALUE atom).
        j = i
        while j < len(expr) and expr[j] not in " \t()":
            j += 1
        tokens.append(expr[i:j])
        i = j
    return tokens


def _parse_filter_expr(
    tokens: list[str],
    pos: int,
    df: pd.DataFrame,
    col_aliases: dict[str, str],
) -> tuple[pd.Series, int]:
    """Recursive-descent parser.  Returns (boolean mask, next position)."""
    mask, pos = _parse_or(tokens, pos, df, col_aliases)
    return mask, pos


def _parse_or(
    tokens: list[str],
    pos: int,
    df: pd.DataFrame,
    col_aliases: dict[str, str],
) -> tuple[pd.Series, int]:
    left, pos = _parse_and(tokens, pos, df, col_aliases)
    while pos < len(tokens) and tokens[pos].upper() == "OR":
        pos += 1  # consume OR
        right, pos = _parse_and(tokens, pos, df, col_aliases)
        left = left | right
    return left, pos


def _parse_and(
    tokens: list[str],
    pos: int,
    df: pd.DataFrame,
    col_aliases: dict[str, str],
) -> tuple[pd.Series, int]:
    left, pos = _parse_atom(tokens, pos, df, col_aliases)
    while pos < len(tokens) and tokens[pos].upper() == "AND":
        pos += 1  # consume AND
        right, pos = _parse_atom(tokens, pos, df, col_aliases)
        left = left & right
    return left, pos


def _parse_atom(
    tokens: list[str],
    pos: int,
    df: pd.DataFrame,
    col_aliases: dict[str, str],
) -> tuple[pd.Series, int]:
    if pos >= len(tokens):
        msg = "Unexpected end of filter expression."
        raise ValueError(msg)
    tok = tokens[pos]
    if tok == "(":
        pos += 1  # consume (
        mask, pos = _parse_or(tokens, pos, df, col_aliases)
        if pos >= len(tokens) or tokens[pos] != ")":
            msg = "Missing closing ')' in filter expression."
            raise ValueError(msg)
        pos += 1  # consume )
        return mask, pos
    if "=" not in tok:
        msg = f"Expected KEY=VALUE but got '{tok}' in filter expression."
        raise ValueError(msg)
    key, value = tok.split("=", 1)

    # Resolve keys case-insensitively across aliases and dataframe columns.
    key_cf = key.casefold()
    col = col_aliases.get(key, key)
    if col not in df.columns:
        alias_keys_cf = {k.casefold(): k for k in col_aliases}
        if key_cf in alias_keys_cf:
            col = col_aliases[alias_keys_cf[key_cf]]
        else:
            cols_cf = {c.casefold(): c for c in df.columns}
            if key_cf in cols_cf:
                col = cols_cf[key_cf]

    if col not in df.columns:
        msg = f"Metadata filter key '{key}' not found."
        raise ValueError(msg)
    return df[col].astype(str) == value, pos + 1


def apply_filter_expr(
    expr: str,
    df: pd.DataFrame,
    col_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply a boolean filter expression to *df* and return the filtered frame.

    Examples::

        apply_filter_expr("Scale=large", df)
        apply_filter_expr("Scale=large AND Dataset=SW2D64", df)
        apply_filter_expr("(Scale=large OR Scale=small) AND Dataset=SW2D64", df)
    """
    aliases = col_aliases or {}
    tokens = _tokenize_filter(expr)
    if not tokens:
        return df
    mask, pos = _parse_filter_expr(tokens, 0, df, aliases)
    if pos != len(tokens):
        msg = (
            f"Unexpected token '{tokens[pos]}' at position {pos} in filter expression."
        )
        raise ValueError(msg)
    return cast(pd.DataFrame, df[mask])


# ---------------------------------------------------------------------------
# Core Utilities
# ---------------------------------------------------------------------------


def resolve_results_root(outputs_dir: str) -> Path:  # noqa: D103
    p = Path(outputs_dir).expanduser()
    return (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()


def normalize_eval_subdir(eval_subdir: str | None) -> str:
    """Normalize an eval subdir value and fall back to the default."""
    if eval_subdir is None:
        return DEFAULT_EVAL_SUBDIR
    normalized = str(eval_subdir).strip().strip("/").strip()
    if not normalized:
        return DEFAULT_EVAL_SUBDIR

    token = normalized.casefold()
    # Friendly aliases for --run ... eval=<token>
    if token in {"eval", "ambient", "default"}:
        return DEFAULT_EVAL_SUBDIR
    if token in {"latent", "eval_latent"}:
        return "eval_latent"
    return normalized


def _apply_explicit_order(
    items: list[str],
    explicit_order: list[str] | None,
) -> list[str]:
    """Return *items* ordered by *explicit_order*, rest appended sorted."""
    if not explicit_order:
        return sorted(items)
    item_set = set(items)
    ordered = [x for x in explicit_order if x in item_set]
    remaining = sorted(x for x in items if x not in set(explicit_order))
    return ordered + remaining


def _order_groups_by_label(
    groups: list[str],
    styles: dict,
    hue_order: list[str] | None,
) -> list[str]:
    """Sort *groups* (plot_group keys) so their display labels follow *hue_order*."""
    if not hue_order:
        return sorted(groups)
    label_rank = {label: i for i, label in enumerate(hue_order)}

    def _key(pg: str) -> tuple[int, str]:
        label = styles.get(pg, {}).get("label", pg)
        rank = label_rank.get(label, len(hue_order))
        return (rank, label)

    return sorted(groups, key=_key)


def dataset_label_from_module(dataset_module: str | None) -> str | None:
    """Return a human-readable label for a dataset module name."""
    if not dataset_module or pd.isna(dataset_module):
        return None
    if dataset_module in DATASET_LABEL_OVERRIDES:
        return DATASET_LABEL_OVERRIDES[dataset_module]
    return str(dataset_module).replace("_", " ").title()


def _dataset_candidates() -> list[str]:
    base = set(DATASET_LABEL_OVERRIDES.keys())
    return sorted(base, key=len, reverse=True)


def arch_key_from_processor_segment(segment: str) -> str:
    """Map a processor name segment to a canonical architecture key."""
    seg = str(segment)
    if not seg:
        return "unknown"
    for pref in sorted(ARCHITECTURE_PREFIXES, key=len, reverse=True):
        if seg == pref or seg.startswith(pref + "_"):
            return pref
    return seg.split("_")[0]


def parse_loss_dataset_arch(
    run_name: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Parse a run directory name into (loss_family, dataset_module, arch_segment)."""
    if not run_name:
        return None, None, None
    m = re.match(
        r"^(diff|crps|epd)_(.+)_([0-9a-f]{7}|[a-z]+)_[0-9a-f]{7}(?:_[a-z0-9]+)*$",
        str(run_name),
    )
    if not m:
        return None, None, None
    loss, mid = m.group(1), m.group(2)

    # Try known datasets
    for ds in _dataset_candidates():
        pfx = ds + "_"
        if mid.startswith(pfx):
            return loss, ds, mid[len(pfx) :]

    # Fallback using known architectures
    for arch in sorted(ARCHITECTURE_PREFIXES, key=len, reverse=True):
        sfx = "_" + arch
        if sfx in mid:
            idx = mid.index(sfx)
            return loss, mid[:idx], mid[idx + 1 :]

    # Ultimate fallback
    if "_" in mid:
        s1, s2 = mid.split("_", 1)
        return loss, s1, s2

    return loss, None, mid


def normalize_dataset_module(dataset: str | None) -> str | None:
    """Strip trailing hex/numeric suffixes from a dataset module path component."""
    if dataset is None or not dataset:
        return None
    try:
        if pd.isna(dataset):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    m = re.match(r"^(?P<base>.+)_(?P<suffix>[0-9a-f]{6,}|\d{6,})$", str(dataset))
    return m.group("base") if m else str(dataset)


def dataset_module_from_data_path(data_path: str | None) -> str | None:
    """Infer dataset module from data_path (including cached latents paths)."""
    if not data_path:
        return None
    p = Path(str(data_path))

    # Most direct case: explicit dataset directory.
    direct = normalize_dataset_module(p.name)
    if direct and direct not in {"cached_latents", "latents", "encoded"}:
        return direct

    # Cached latents case: parent directory usually encodes source AE run,
    # e.g. ae_cns64_e011131_b3e6f6b -> cns64
    parent = p.parent.name if p.name in {"cached_latents", "latents", "encoded"} else ""
    if parent.startswith("ae_"):
        toks = parent.split("_")[1:]
        # Strip trailing run/hash suffixes while preserving dataset tokens.
        while toks and re.fullmatch(r"[0-9a-f]{6,}|\d{6,}", toks[-1]):
            toks.pop()
        if toks:
            return normalize_dataset_module("_".join(toks))
    return None


def resolution_from_run_name(run_name: str | None) -> str | None:
    """Infer a model resolution token from run_name when present."""
    if not run_name:
        return None
    # Common form: ..._vit_640_<hash>_<hash> or ..._vit_128_<hash>_<hash>
    m = re.search(r"_vit_(\d{2,5})(?:_|$)", str(run_name))
    if m:
        return m.group(1)
    return None


def _format_spatial_resolution(vals: list[int] | tuple[int, ...]) -> str | None:
    """Format 2D spatial resolution as 'W x H' with larger dim first."""
    if len(vals) < 2:
        return None
    a, b = int(vals[0]), int(vals[1])
    hi, lo = (a, b) if a >= b else (b, a)
    return f"{hi}x{lo}"


def dataset_grid_resolution(
    dataset_module: str | None, data_path: str | None
) -> str | None:
    """Infer canonical grid resolution from dataset identifier or data path."""
    ds = str(dataset_module or "")

    # Explicit lookup from known dataset modules.
    if ds in DATASET_RESOLUTION:
        return DATASET_RESOLUTION[ds]

    # Direct token like lb128x32
    m = re.search(r"(\d+)x(\d+)", ds)
    if m:
        return _format_spatial_resolution([int(m.group(1)), int(m.group(2))])

    # Heuristic: names ending in 64 are typically 64x64 grids.
    if ds.endswith("64"):
        return "64x64"

    # 128-grid datasets often include this token in the full path/module name.
    p = str(data_path or "")
    if "_128" in ds or "_128" in p:
        return "128x128"

    return None


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_single_run_metrics(  # noqa: PLR0912, PLR0915
    run_dir: Path,
    eval_subdir: str = DEFAULT_EVAL_SUBDIR,
    run_ref: str | None = None,
) -> dict:
    """Load evaluation metrics and rollout metrics from a single run directory."""
    resolved_eval_subdir = normalize_eval_subdir(eval_subdir)
    row = {
        "run_name": run_ref or run_dir.name,
        "run_id": run_dir.name,
        "run_path": run_dir.name,
        "eval_subdir": resolved_eval_subdir,
        "dataset": None,
    }
    eval_dir = run_dir / resolved_eval_subdir

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
            rt = md[md["category"] == "runtime_train"]
            for metric, out_col in [
                ("mean_epoch_s", "train_mean_epoch_s"),
                ("total_s", "train_total_s"),
                ("min_epoch_s", "train_min_epoch_s"),
                ("max_epoch_s", "train_max_epoch_s"),
            ]:
                s = rt.loc[rt["metric"] == metric, "value"]
                if not s.empty:
                    row[out_col] = pd.to_numeric(s.iloc[0], errors="coerce")

            params = md[md["category"] == "params"]
            for metric, out_col in [
                ("model_total", "params_model_total"),
                ("model_trainable", "params_model_trainable"),
                ("processor_total", "params_processor_total"),
                ("encoder_total", "params_encoder_total"),
                ("decoder_total", "params_decoder_total"),
            ]:
                s = params.loc[params["metric"] == metric, "value"]
                if not s.empty:
                    row[out_col] = pd.to_numeric(s.iloc[0], errors="coerce")

    # Benchmark metrics (inference/rollout latency, throughput, memory, flops)
    p_bench = eval_dir / "benchmark_metrics.csv"
    if p_bench.exists():
        bm = pd.read_csv(p_bench)
        if not bm.empty and {"benchmark_type", "metric", "value"}.issubset(bm.columns):
            for _, rr in bm.iterrows():
                prefix = str(rr["benchmark_type"])
                metric = str(rr["metric"])
                row[f"{prefix}_{metric}"] = pd.to_numeric(rr["value"], errors="coerce")

    # Training metadata (dataset hints for cached latents)
    p_config = run_dir / "resolved_config.yaml"
    if p_config.exists():
        try:
            with open(p_config) as f:
                cfg = yaml.safe_load(f)
            if cfg:
                data_path = cfg.get("datamodule", {}).get("data_path", "")
                row["dataset"] = (
                    Path(data_path).name if data_path else row.get("dataset")
                )
                row["dataset_from_data_path"] = dataset_module_from_data_path(data_path)
        except Exception:
            pass
    return row


def assign_model_scale(df_in: pd.DataFrame) -> pd.Series:
    """Label each run as 'small' or 'large' relative to its architecture group."""
    out = pd.Series("large", index=df_in.index, dtype="object")

    # If the architecture segment already encodes size intent, trust it.
    explicit_scale = pd.Series(None, index=df_in.index, dtype="object")
    if "arch_segment" in df_in.columns:
        seg = cast(pd.Series, df_in["arch_segment"].fillna(""))
        seg_s = seg.astype(str)
        explicit_scale = pd.Series(
            np.where(
                seg_s.str.contains(r"(?:^|_)small(?:_|$)"),
                "small",
                np.where(seg_s.str.contains(r"(?:^|_)large(?:_|$)"), "large", ""),
            ),
            index=df_in.index,
            dtype="object",
        )
        explicit_scale = explicit_scale.replace("", np.nan)
        out.loc[explicit_scale == "small"] = "small"
        out.loc[explicit_scale == "large"] = "large"

    if MODEL_SCALE_PARAM_COL not in df_in.columns:
        return out
    params = cast(
        pd.Series, pd.to_numeric(df_in[MODEL_SCALE_PARAM_COL], errors="coerce")
    )
    group_cols = [
        c for c in ["dataset_module", "loss_family", "arch_key"] if c in df_in.columns
    ]
    if len(group_cols) < 2:
        return out

    for _, g in df_in.groupby(group_cols, dropna=False):
        idx = g.index
        pending_idx = explicit_scale.loc[idx][explicit_scale.loc[idx].isna()].index
        if len(pending_idx) == 0:
            continue

        p = cast(pd.Series, params.loc[pending_idx])
        vals = sorted(p.dropna().unique().tolist())
        if len(vals) <= 1:
            out.loc[pending_idx] = "large"
            continue
        low, high = vals[0], vals[-1]
        low_idx = cast(pd.Series, p[p == low]).index
        high_idx = cast(pd.Series, p[p == high]).index
        out.loc[low_idx] = "small"
        out.loc[high_idx] = "large"
        mid_mask = (p != low) & (p != high)
        for ii in cast(pd.Series, p[mid_mask]).index:
            val = p.loc[ii]
            out.loc[ii] = (
                "small"
                if pd.notna(val) and abs(float(val) - low) <= abs(float(val) - high)
                else "large"
            )
    return out


def _resolve_resolution(
    row: dict[str, object],
    proc: dict,
    data_path: str,
) -> str | None:
    """Resolve grid resolution from processor config or dataset heuristics."""
    res = None
    proc_sr = proc.get("spatial_resolution")
    if isinstance(proc_sr, (list, tuple)):
        vals = [int(v) for v in proc_sr[:2] if v is not None]
        res = _format_spatial_resolution(vals)
    if not res:
        res = dataset_grid_resolution(
            cast(str | None, row.get("dataset_from_data_path")),
            data_path,
        )
    if not res:
        _, ds_from_name, _ = parse_loss_dataset_arch(
            cast(str | None, row.get("run_name"))
        )
        if ds_from_name:
            res = dataset_grid_resolution(ds_from_name, data_path)
    return res


def _extract_config_fields(cfg: dict, row: dict[str, object]) -> None:
    """Populate *row* with fields extracted from a resolved Hydra config."""
    datamodule = cfg.get("datamodule", {})
    row["n_steps_input"] = datamodule.get("n_steps_input")
    row["n_steps_output"] = datamodule.get("n_steps_output")
    row["batch_size"] = datamodule.get("batch_size")
    data_path = datamodule.get("data_path", "")
    row["dataset"] = Path(data_path).name
    row["dataset_from_data_path"] = dataset_module_from_data_path(data_path)
    row["loss_func"] = (
        cfg.get("model", {}).get("loss_func", {}).get("_target_", "").split(".")[-1]
    )
    proc = cfg.get("model", {}).get("processor", {})
    row["processor"] = proc.get("_target_", "").split(".")[-1]

    if isinstance(proc, dict):
        row["resolution"] = _resolve_resolution(row, proc, data_path)

    inj = cfg.get("model", {}).get("input_noise_injector", {})
    row["noise_injector"] = inj.get("_target_", "").split(".")[-1] if inj else "None"

    # Noise channels from processor or injector
    nc = proc.get("n_noise_channels")
    if nc is None and inj:
        nc = inj.get("n_noise_channels")
    row["noise_channels"] = int(nc) if nc is not None else 0

    # ODE / sampling steps (flow_ode_steps or sampler_steps)
    ode = proc.get("flow_ode_steps") or proc.get("sampler_steps")
    if ode is not None:
        row["ode_steps"] = int(ode)

    row["lr"] = cfg.get("optimizer", {}).get("learning_rate")

    # GPU count from trainer config
    trainer = cfg.get("trainer", {})
    devices = trainer.get("devices", 1)
    num_nodes = trainer.get("num_nodes", 1)
    try:
        n_gpus = int(devices) * int(num_nodes)
    except (TypeError, ValueError):
        n_gpus = 1
    row["n_gpus"] = n_gpus

    bs = row.get("batch_size")
    if bs is not None:
        assert isinstance(bs, int | str | float), (
            "Batch size must be an integer, string, or float."
        )
        row["eff_batch_size"] = int(bs) * n_gpus


def load_config_metadata(  # noqa: PLR0912
    run_dir: Path,
    eval_subdir: str = DEFAULT_EVAL_SUBDIR,
    run_ref: str | None = None,
) -> dict[str, object]:
    """Load training config metadata (LR, batch size, noise, etc.)."""
    resolved_eval_subdir = normalize_eval_subdir(eval_subdir)
    row: dict[str, object] = {
        "run_name": run_ref or run_dir.name,
        "run_id": run_dir.name,
        "eval_subdir": resolved_eval_subdir,
    }
    p_config = run_dir / "resolved_config.yaml"
    if p_config.exists():
        try:
            with open(p_config) as f:
                cfg = yaml.safe_load(f)
            if cfg:
                _extract_config_fields(cfg, row)
        except Exception:
            pass

    # Run date: inspect only local directory names to avoid following symlinks.
    _date_found = False
    for _anc in (run_dir, *run_dir.parents):
        dm = re.match(r"(\d{4}-\d{2}-\d{2})$", _anc.name)
        if dm:
            row["run_date"] = dm.group(1)
            _date_found = True
            break
    if not _date_found and p_config.exists():
        mtime = p_config.stat().st_mtime
        row["run_date"] = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )

    # Training time from evaluation_metadata.csv
    p_meta = run_dir / resolved_eval_subdir / "evaluation_metadata.csv"
    if p_meta.exists():
        try:
            md = pd.read_csv(p_meta)
            if not md.empty and {"category", "metric", "value"}.issubset(md.columns):
                rt = md[md["category"] == "runtime_train"]
                s = rt.loc[rt["metric"] == "total_s", "value"]
                if not s.empty:
                    row["train_total_s"] = pd.to_numeric(s.iloc[0], errors="coerce")
                s = rt.loc[rt["metric"] == "mean_epoch_s", "value"]
                if not s.empty:
                    row["train_mean_epoch_s"] = pd.to_numeric(
                        s.iloc[0], errors="coerce"
                    )

                params = md[md["category"] == "params"]
                p = params.loc[params["metric"] == "processor_total", "value"]
                if not p.empty:
                    row["params_processor_total"] = pd.to_numeric(
                        p.iloc[0], errors="coerce"
                    )
        except Exception:
            pass

    # Inference latency from benchmark_metrics.csv
    p_bench = run_dir / resolved_eval_subdir / "benchmark_metrics.csv"
    if p_bench.exists():
        try:
            bm = pd.read_csv(p_bench)
            if not bm.empty and {
                "benchmark_type",
                "metric",
                "value",
            }.issubset(bm.columns):
                lat = bm.loc[
                    (bm["benchmark_type"].astype(str) == "model")
                    & (bm["metric"].astype(str) == "latency_ms_per_sample"),
                    "value",
                ]
                if not lat.empty:
                    row["model_latency_ms_per_sample"] = pd.to_numeric(
                        lat.iloc[0], errors="coerce"
                    )
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
    """Get hue and lightness for a plot group.

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
    hues = sorted(
        {p.split("__")[0] for p in cast(pd.Series, df_in["plot_group"]).unique()}
    )
    h_idx = hues.index(arch) % 10 if arch in hues else 0
    c0, c1 = cmap(2 * h_idx), cmap(2 * h_idx + 1)
    light, dark = (c0, c1) if _rgb_luminance(c0) > _rgb_luminance(c1) else (c1, c0)
    color = light if scale == "small" else dark
    return color, loss == "diff", scale


def plot_group_display_label(pg: str) -> str:
    """Return a human-readable label for a plot_group key."""
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


def _run_name_from_plot_group(pg: str) -> str | None:
    """Extract run_name from a plot_group token when present."""
    parts = pg.split("__")
    if len(parts) > 3 and parts[3]:
        return parts[3]
    return None


def _run_ref_aliases(run_name: str) -> list[str]:
    """Return lookup aliases for a run ref with optional eval/index suffixes."""
    aliases = [run_name]
    base = run_name.split("#", 1)[0]
    if base not in aliases:
        aliases.append(base)
    core = base.split("::eval=", 1)[0]
    if core not in aliases:
        aliases.append(core)
    return aliases


def _style_label_for_plot_group(
    pg: str,
    custom_label_by_run: dict[str, str] | None,
) -> str:
    """Return custom label for a run when provided, else the default label."""
    if custom_label_by_run:
        run_name = _run_name_from_plot_group(pg)
        if run_name:
            for alias in _run_ref_aliases(run_name):
                if alias in custom_label_by_run:
                    return custom_label_by_run[alias]
    return plot_group_display_label(pg)


def extract_valid_plot_groups_from_run_names(
    run_names: list[str], df_in: pd.DataFrame
) -> set[str]:
    """Return the plot_group keys that correspond to the given run directory names."""
    # Given a list of run dir names, find their corresponding plot_groups in df_in
    mask = cast(pd.Series, df_in["run_name"].isin(run_names))
    if "run_id" in df_in.columns:
        mask = cast(pd.Series, mask | df_in["run_id"].isin(run_names))
    return set(cast(pd.Series, df_in[mask]["plot_group"]).unique())


def _shade_variant(base_color, idx: int, total: int):
    """Return a deterministic lightness variant for a color within a family."""
    if total <= 1:
        return base_color
    if total == 2:
        return (
            _mix_with_white(base_color, 0.22)
            if idx == 0
            else _mix_with_black(base_color, 0.22)
        )
    frac = (idx / (total - 1)) - 0.5  # -0.5..0.5
    # Positive frac → darker, negative → lighter (first item = lightest).
    return (
        _mix_with_black(base_color, frac * 0.6)
        if frac > 0
        else _mix_with_white(base_color, -frac * 0.8)
    )


def _build_label_color_map(
    labels: list[str],
) -> dict[str, tuple]:
    """Map labels to colors, grouping by prefix before '('."""
    base_cmap = plt.get_cmap("tab10")

    def _prefix(lbl: str) -> str:
        idx = lbl.find("(")
        return lbl[:idx].strip() if idx >= 0 else lbl.strip()

    prefixes = sorted({_prefix(lb) for lb in labels})
    pfx_hue = {p: i for i, p in enumerate(prefixes)}

    hue_members: dict[int, list[str]] = {}
    seen: set[str] = set()
    for lb in labels:
        if lb in seen:
            continue
        seen.add(lb)
        hi = pfx_hue[_prefix(lb)]
        hue_members.setdefault(hi, []).append(lb)

    result: dict[str, tuple] = {}
    for hi, members in hue_members.items():
        base = base_cmap(hi % 10)
        for j, lb in enumerate(members):
            result[lb] = _shade_variant(base, j, len(members))
    return result


def _apply_explicit_group_styles(
    styles: dict,
    df_in: pd.DataFrame,
    explicit_groups: list[list[str]],
    custom_label_by_run: dict[str, str] | None,
    uniform_group_color: bool,
    group_hues: list[int] | None,
) -> None:
    """Assign styles from explicit --run-group definitions."""
    base_cmap = plt.get_cmap("tab10")

    hue_family: dict[int, list[int]] = {}
    if group_hues:
        for gi, hi in enumerate(group_hues):
            hue_family.setdefault(hi, []).append(gi)

    for i, group_runs in enumerate(explicit_groups):
        if group_hues and i < len(group_hues):
            hue_idx = group_hues[i]
            base_color = base_cmap(hue_idx % 10)
            siblings = hue_family.get(hue_idx, [i])
            c = _shade_variant(base_color, siblings.index(i), len(siblings))
        else:
            base_color = base_cmap(i % 10)
            c = base_color

        group_pgs = extract_valid_plot_groups_from_run_names(group_runs, df_in)
        group_pgs = sorted(group_pgs)
        for j, pg in enumerate(group_pgs):
            if pg in styles:
                continue
            run_c = (
                c
                if (uniform_group_color or group_hues)
                else _shade_variant(base_color, j, len(group_pgs))
            )
            parts = pg.split("__")
            loss = parts[1] if len(parts) > 1 else "unknown"
            styles[pg] = {
                "color": run_c,
                "label": _style_label_for_plot_group(pg, custom_label_by_run),
                "marker": "^" if loss == "diff" else "o",
                "linestyle": "-" if loss == "diff" else "--",
            }


def _apply_run_hue_styles(
    styles: dict,
    pgs: list[str],
    hue_group_by_run: dict[str, int],
    custom_label_by_run: dict[str, str] | None,
    uniform_run_hue_color: bool,
) -> None:
    """Assign styles using per-run hue group indices from --run <id> [label] <hue>."""
    base_cmap = plt.get_cmap("tab10")

    # Group plot_groups by hue index, preserving --runs order within each group.
    hue_members: dict[int, list[str]] = {}
    no_hue: list[str] = []
    for pg in pgs:
        rn = _run_name_from_plot_group(pg)
        hue_idx: int | None = None
        if rn:
            for alias in _run_ref_aliases(rn):
                if alias in hue_group_by_run:
                    hue_idx = hue_group_by_run[alias]
                    break
        if hue_idx is not None:
            hue_members.setdefault(hue_idx, []).append(pg)
            continue
        no_hue.append(pg)

    for hi, members in sorted(hue_members.items()):
        base_color = base_cmap(hi % 10)
        # Assign shades by unique label (first-seen), not by run index.
        # This keeps color/lightness stable across datasets when the label matches.
        member_labels = [
            str(_style_label_for_plot_group(pg, custom_label_by_run)) for pg in members
        ]
        unique_labels = list(dict.fromkeys(member_labels))
        label_to_color = {
            lb: (
                base_color
                if uniform_run_hue_color
                else _shade_variant(base_color, j, len(unique_labels))
            )
            for j, lb in enumerate(unique_labels)
        }

        for pg, lb in zip(members, member_labels, strict=False):
            parts = pg.split("__")
            loss = parts[1] if len(parts) > 1 else "unknown"
            styles[pg] = {
                "color": label_to_color[lb],
                "label": lb,
                "marker": "^" if loss == "diff" else "o",
                "linestyle": "-" if loss == "diff" else "--",
            }

    # Runs without an explicit hue get sequential colors after the last hue.
    next_hi = max(hue_members.keys(), default=-1) + 1
    for i, pg in enumerate(no_hue):
        parts = pg.split("__")
        loss = parts[1] if len(parts) > 1 else "unknown"
        styles[pg] = {
            "color": base_cmap((next_hi + i) % 10),
            "label": _style_label_for_plot_group(pg, custom_label_by_run),
            "marker": "^" if loss == "diff" else "o",
            "linestyle": "-" if loss == "diff" else "--",
        }


def _apply_label_color_styles(
    styles: dict,
    pgs: list[str],
    custom_label_by_run: dict[str, str],
) -> None:
    """Assign styles by label — same label gets same color."""
    pg_label = {}
    for pg in pgs:
        rn = _run_name_from_plot_group(pg)
        lbl: str | None = None
        if rn:
            for alias in _run_ref_aliases(rn):
                if alias in custom_label_by_run:
                    lbl = custom_label_by_run[alias]
                    break
        pg_label[pg] = (
            lbl
            if lbl is not None
            else _style_label_for_plot_group(pg, custom_label_by_run)
        )
    label_color = _build_label_color_map(list(pg_label.values()))
    for pg in pgs:
        lb = pg_label[pg]
        parts = pg.split("__")
        loss = parts[1] if len(parts) > 1 else "unknown"
        styles[pg] = {
            "color": label_color[lb],
            "label": _style_label_for_plot_group(pg, custom_label_by_run),
            "marker": "^" if loss == "diff" else "o",
            "linestyle": "-" if loss == "diff" else "--",
        }


def build_family_style(
    df_in: pd.DataFrame,
    explicit_groups: list[list[str]] | None = None,
    custom_label_by_run: dict[str, str] | None = None,
    uniform_group_color: bool = False,
    uniform_run_hue_color: bool = False,
    group_hues: list[int] | None = None,
    color_by_label: bool = False,
    hue_group_by_run: dict[str, int] | None = None,
) -> dict:
    """Build a style dict (color, label, marker, linestyle) for each plot_group."""
    styles: dict = {}
    _present_raw = cast(pd.Series, df_in["plot_group"].dropna())
    # Preserve first-seen order (reflects --runs order) instead of sorting.
    present = list(dict.fromkeys(_present_raw.tolist()))

    if explicit_groups and len(explicit_groups) > 0:
        _apply_explicit_group_styles(
            styles,
            df_in,
            explicit_groups,
            custom_label_by_run,
            uniform_group_color,
            group_hues,
        )

    remaining = [pg for pg in present if pg not in styles]
    if hue_group_by_run and remaining:
        _apply_run_hue_styles(
            styles,
            remaining,
            hue_group_by_run,
            custom_label_by_run,
            uniform_run_hue_color,
        )
        remaining = []
    elif color_by_label and custom_label_by_run and remaining:
        _apply_label_color_styles(styles, remaining, custom_label_by_run)
        remaining = []

    # Fallback for remaining plot groups: each run gets its own hue.
    cmap = plt.get_cmap("tab20")
    for i, pg in enumerate(sorted(remaining)):
        parts = pg.split("__")
        loss = parts[1] if len(parts) > 1 else "unknown"
        styles[pg] = {
            "color": cmap(i % 20),
            "label": _style_label_for_plot_group(pg, custom_label_by_run),
            "marker": "^" if loss == "diff" else "o",
            "linestyle": "-" if loss == "diff" else "--",
        }
    return styles


def build_custom_label_map(
    label_run_pairs: list[list[str]] | None,
    valid_run_names: set[str] | None = None,
) -> dict[str, str]:
    """Build run_name -> custom label map from ``--label-run`` entries.

    A label value of "None" (case-insensitive) means: keep default auto label.
    """
    if not label_run_pairs:
        return {}

    custom: dict[str, str] = {}
    for pair in label_run_pairs:
        if len(pair) != 2:
            msg = "Each --label-run must provide exactly two values: <run_id> <label>."
            raise ValueError(msg)

        run_name, label = pair[0], pair[1]
        if valid_run_names is not None and run_name not in valid_run_names:
            msg = (
                f"--label-run references unknown run_id '{run_name}'. "
                "Ensure it is included via --runs/--run-group or exists in "
                "auto-discovery."
            )
            raise ValueError(msg)

        norm = str(label).strip()
        if norm.casefold() == "none":
            continue
        custom[run_name] = norm

    return custom


# ---------------------------------------------------------------------------
# Plotting Implementations
# ---------------------------------------------------------------------------


def save_fig(fig, out_dir: Path, name: str):
    """Save a matplotlib figure to disk and close it."""
    p = out_dir / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)


def _dedup_legend_handles(groups: list, styles: dict) -> list[Line2D]:
    """Build legend handles with duplicate labels removed."""
    handles = []
    seen: set[str] = set()
    for g in groups:
        if g not in styles:
            continue
        lbl = str(styles[g]["label"])
        if lbl in seen:
            continue
        seen.add(lbl)
        handles.append(
            Line2D(
                [0],
                [0],
                color=styles[g]["color"],
                ls=styles[g].get("linestyle", "-"),
                lw=2,
                label=lbl,
            )
        )
    return handles


def build_single_step_results_table(
    df_in: pd.DataFrame,
    styles: dict,
    dataset_order: list[str] | None = None,
    hue_order: list[str] | None = None,
) -> pd.DataFrame:
    """Build the CSV/LaTeX table from the same grouped means as bar plots."""
    if "dataset_label" not in df_in.columns or "plot_group" not in df_in.columns:
        return pd.DataFrame()

    data = df_in.copy()
    metric_cols = [src for src, _ in SINGLE_STEP_RESULTS_TABLE_METRICS]
    available = [c for c in metric_cols if c in data.columns]
    if not available:
        return pd.DataFrame()

    for col in available:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    grouped = (
        data.groupby(["dataset_label", "plot_group"], dropna=False)[available]
        .mean()
        .reset_index()
    )
    datasets = _apply_explicit_order(
        list(grouped["dataset_label"].dropna().unique()), dataset_order
    )

    rows: list[dict[str, object]] = []
    for ds_label in datasets:
        ds_rows = grouped[grouped["dataset_label"] == ds_label]
        groups = _order_groups_by_label(
            cast(pd.Series, ds_rows["plot_group"].astype(str)).dropna().tolist(),
            styles,
            hue_order,
        )
        for group in groups:
            row_match = ds_rows[ds_rows["plot_group"].astype(str) == str(group)]
            if row_match.empty:
                continue
            raw = row_match.iloc[0]
            style = styles.get(group, {"label": group})
            table_row: dict[str, object] = {
                "Dataset": ds_label,
                "Model": str(style.get("label", group)),
            }
            for src, label in SINGLE_STEP_RESULTS_TABLE_METRICS:
                table_row[label] = raw[src] if src in raw.index else np.nan
            rows.append(table_row)

    columns = [
        "Dataset",
        "Model",
        *[label for _, label in SINGLE_STEP_RESULTS_TABLE_METRICS],
    ]
    return pd.DataFrame(rows, columns=columns)


def _latex_escape(value: object) -> str:
    """Escape plain text for insertion into a LaTeX table."""
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _format_latex_table_value(value: object) -> str:
    """Format table cells for the compact LaTeX output."""
    if isinstance(value, str):
        return _latex_escape(value)
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _latex_escape(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.1e}"


def _best_latex_cells_by_dataset(table: pd.DataFrame) -> set[tuple[int, str]]:
    """Return table cells that should be highlighted as best-in-dataset."""
    best_cells: set[tuple[int, str]] = set()
    if "Dataset" not in table.columns:
        return best_cells

    for _, dataset_rows in table.groupby("Dataset", sort=False):
        for metric in [label for _, label in SINGLE_STEP_RESULTS_TABLE_METRICS]:
            if metric not in dataset_rows.columns:
                continue
            values = pd.to_numeric(dataset_rows[metric], errors="coerce")
            values = cast(pd.Series, values.dropna())
            if values.empty:
                continue
            if metric in SINGLE_STEP_RESULTS_TARGET_METRICS:
                target = SINGLE_STEP_RESULTS_TARGET_METRICS[metric]
                scores = (values - target).abs()
                best_score = scores.min()
                winners = scores[scores == best_score].index
            elif metric in SINGLE_STEP_RESULTS_LOWER_IS_BETTER:
                best_value = values.min()
                winners = values[values == best_value].index
            else:
                continue
            for idx in winners:
                best_cells.add((int(idx), metric))
    return best_cells


def render_single_step_results_latex(table: pd.DataFrame) -> str:
    """Render the results table as a linewidth-bounded tabularx fragment."""
    columns = table.columns.tolist()
    align = (
        "@{}" + ("l" * min(2, len(columns))) + ("X" * max(0, len(columns) - 2)) + "@{}"
    )
    header = " & ".join(
        SINGLE_STEP_RESULTS_LATEX_HEADERS.get(col, _latex_escape(col))
        for col in columns
    )
    best_cells = _best_latex_cells_by_dataset(table)
    body = []
    for idx, row in table.iterrows():
        cells = []
        for col in columns:
            cell = _format_latex_table_value(row[col])
            if cell and (int(idx), col) in best_cells:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        body.append(" & ".join(cells) + r" \\")
    lines = [
        r"% Requires \usepackage{booktabs,tabularx}",
        rf"\begin{{tabularx}}{{\linewidth}}{{{align}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabularx}",
        "",
    ]
    return "\n".join(lines)


def write_single_step_results_table(
    df_in: pd.DataFrame,
    out_dir: Path,
    styles: dict,
    dataset_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    stem: str = "single_step_overall_results",
) -> pd.DataFrame:
    """Write single-step overall results as CSV and LaTeX."""
    table = build_single_step_results_table(
        df_in,
        styles,
        dataset_order=dataset_order,
        hue_order=hue_order,
    )
    if table.empty:
        print("No single-step overall results table available.")
        return table

    csv_path = out_dir / f"{stem}.csv"
    tex_path = out_dir / f"{stem}.tex"
    table.to_csv(csv_path, index=False, float_format="%.6g")
    tex_path.write_text(render_single_step_results_latex(table), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    return table


def grouped_bar(  # noqa: PLR0912, PLR0915
    df_in: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    styles: dict,
    y_scale: str = "auto",
    dataset_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    ax: Axes | None = None,
    save: bool = True,
    ylim: tuple[float | None, float | None] | None = None,
    show_legend: bool = True,
    ref_value: float | None = None,
    tick_label_scale: float = 1.0,
    axis_label_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    legend_font_size: float | None = None,
    figure_scale: float = 1.0,
    short_axis_labels: bool = False,
    shared_axis_labels: bool = False,
    height_scale: float = 1.0,
) -> FigureBase | None:
    """Render a grouped bar chart of a metric across datasets and model families.

    When ``ax`` is provided, draw into it and skip figure creation/saving.
    ``ylim`` (low, high) overrides automatic y-axis limits; either bound may be
    ``None`` to keep the auto-computed value on that side.
    ``ref_value`` draws a dotted horizontal reference line at that value.
    """
    _ = (short_axis_labels, shared_axis_labels, height_scale)
    if metric not in df_in.columns:
        return None
    d = df_in.dropna(subset=[metric]).copy()
    if d.empty:
        return None

    group_col = "plot_group"
    g = (
        d.groupby(["dataset_label", group_col], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    datasets = _apply_explicit_order(list(g["dataset_label"].unique()), dataset_order)
    _families = _order_groups_by_label(list(g[group_col].unique()), styles, hue_order)
    x = np.arange(len(datasets), dtype=float)

    # Width should reflect groups present per dataset, not global families across all
    # datasets. Otherwise datasets with fewer groups render overly narrow bars.
    present_by_dataset: dict[str, list[str]] = {}
    max_present = 1
    for ds in datasets:
        ds_groups = _order_groups_by_label(
            cast(
                pd.Series,
                g.loc[g["dataset_label"] == ds, group_col].astype(str),  # type: ignore  # noqa: PGH003
            )
            .dropna()
            .unique()
            .tolist(),
            styles,
            hue_order,
        )
        present_by_dataset[ds] = ds_groups
        max_present = max(max_present, len(ds_groups))

    width = 0.8 / max(1, max_present)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=_scaled_figsize(
                max(8.0, 0.9 * len(datasets) + 2.0),
                3.8,
                figure_scale,
            )
        )
    else:
        fig = ax.figure
    all_positive = []

    seen_labels: set[str] = set()
    for ds_idx, ds in enumerate(datasets):
        ds_rows = g[g["dataset_label"] == ds].set_index(group_col)
        ds_groups = present_by_dataset.get(ds, [])
        for j, fam in enumerate(ds_groups):
            if fam not in ds_rows.index:
                continue
            raw_val = ds_rows.loc[fam, metric]
            raw_scalar = raw_val.iloc[0] if isinstance(raw_val, pd.Series) else raw_val
            val_ser = cast(
                pd.Series,
                pd.to_numeric(pd.Series([raw_scalar]), errors="coerce"),
            )
            if val_ser.empty or pd.isna(val_ser.iloc[0]):
                continue
            val = float(val_ser.iloc[0])
            if not np.isfinite(val):
                continue

            if val > 0:
                all_positive.append(val)

            offs = (j - (len(ds_groups) - 1) / 2.0) * width
            style = styles.get(fam, {"color": "k", "label": fam})
            label = str(style["label"])
            if label in seen_labels:
                label = "_nolegend_"
            else:
                seen_labels.add(label)

            ax.bar(
                float(x[ds_idx]) + offs,
                val,
                width=width,
                label=label,
                color=style["color"],
                edgecolor="0.25",
                linewidth=0.6,
                linestyle=style.get("linestyle", "-"),
            )

    if y_scale == "linear":
        ax.set_yscale("linear")
    elif all_positive and min(all_positive) > 0:
        ax.set_yscale("log")

    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=0 if len(datasets) <= 4 else 30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    _apply_tick_label_style(ax, tick_label_scale)
    _apply_axis_label_style(ax, axis_label_scale)
    if ref_value is not None:
        ax.axhline(
            ref_value,
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.65,
        )

    if ylim is not None:
        lo, hi = ylim
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(
            bottom=lo if lo is not None else cur_lo,
            top=hi if hi is not None else cur_hi,
        )

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=_resolve_font_size(
                    DEFAULT_LEGEND_FONT_SIZE,
                    legend_font_scale,
                    legend_font_size,
                ),
                frameon=False,
            )

    if save:
        save_fig(fig, out_dir, f"{metric}.png")
        return None
    return fig


def plot_coverage_calibration_panel(  # noqa: PLR0912, PLR0915
    df_in: pd.DataFrame,
    results_root: Path,
    out_dir: Path,
    styles: dict,
    dataset_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    axes: np.ndarray | None = None,
    fig: FigureBase | None = None,
    save: bool = True,
    show_legend: bool = True,
    window_rows: list[str] | None = None,
    name: str = "coverage_calibration_panel.png",
    tick_label_scale: float = 1.0,
    axis_label_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    legend_font_size: float | None = None,
    figure_scale: float = 1.0,
    short_axis_labels: bool = False,
    shared_axis_labels: bool = False,
    height_scale: float = 1.0,
) -> FigureBase | None:
    """Plot the coverage calibration panel (rollout windows x datasets).

    When ``axes`` (2D grid with shape (len(window_rows), n_datasets)) is given,
    draw into that grid instead of creating a new figure.
    """
    rows = window_rows if window_rows is not None else WINDOW_ROWS
    curves = []
    base = df_in.dropna(
        subset=["run_path", "dataset_label", "plot_group"]
    ).drop_duplicates()
    for _, r in base.iterrows():
        for w in rows:
            fn = (
                "test_coverage_window_all.csv"
                if w == "all"
                else f"rollout_coverage_window_{w}.csv"
            )
            eval_subdir = normalize_eval_subdir(cast(str | None, r.get("eval_subdir")))
            p = results_root / str(r["run_path"]) / eval_subdir / fn
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
        return None
    cur_panel = pd.concat(curves, ignore_index=True)
    datasets = _apply_explicit_order(
        list(cur_panel["dataset_label"].unique()), dataset_order
    )
    groups = _order_groups_by_label(
        list(cur_panel["plot_group"].unique()), styles, hue_order
    )

    nrows, ncols = len(rows), max(1, len(datasets))
    if axes is None:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=_scaled_figsize(
                4.0 * ncols,
                2.3 * nrows * height_scale,
                figure_scale,
            ),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
    else:
        # axes supplied by caller; make sure it's a 2D array
        axes = np.atleast_2d(axes)
        if fig is None:
            fig = axes[0][0].figure

    assert axes is not None
    assert fig is not None

    for i, w in enumerate(rows):
        for j, ds_label in enumerate(datasets):
            ax = axes[i][j]
            ax.plot([0, 1], [0, 1], "--", color="k", alpha=0.6)
            sub = cur_panel[
                (cur_panel["window"] == w) & (cur_panel["dataset_label"] == ds_label)
            ]
            for fam in groups:
                sf = cast(pd.DataFrame, sub[sub["plot_group"] == fam])
                if sf.empty:
                    continue
                st = styles.get(fam, {"color": "k", "linestyle": "-"})

                # Plot each run's curve individually (no averaging)
                mean_curve = cast(
                    pd.DataFrame,
                    cast(
                        pd.DataFrame,
                        sf.groupby("coverage_level", as_index=False)[
                            "observed_mean"
                        ].mean(),
                    ).sort_values(by="coverage_level"),
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
            if i == nrows - 1:
                ax.set_xlabel(
                    "" if shared_axis_labels else r"Expected coverage (1 - $\alpha$)"
                )
            if j == 0:
                ax.set_ylabel(
                    _coverage_window_axis_label(str(w), short=short_axis_labels)
                )
            ax.grid(alpha=0.2)
            _apply_tick_label_style(ax, tick_label_scale)
            _apply_axis_label_style(ax, axis_label_scale)

    if shared_axis_labels:
        _set_shared_xlabel(fig, r"Expected coverage (1 - $\alpha$)", axis_label_scale)
    if show_legend:
        legend_handles = _dedup_legend_handles(groups, styles)
        fig.legend(
            legend_handles,
            [str(h.get_label()) for h in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=4,
            fontsize=_resolve_font_size(
                DEFAULT_LEGEND_FONT_SIZE,
                legend_font_scale,
                legend_font_size,
            ),
            frameon=False,
        )
    if save:
        plt.tight_layout(rect=(0, 0, 1, 0.975))
        save_fig(fig, out_dir, name)
        return None
    return fig


def plot_lead_time_panel(  # noqa: PLR0912, PLR0915
    df_in: pd.DataFrame,
    metrics: list[str],
    results_root: Path,
    out_dir: Path,
    name: str,
    styles: dict,
    dataset_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    axes: np.ndarray | None = None,
    fig: FigureBase | None = None,
    save: bool = True,
    error_ylim: tuple[float | None, float | None] | None = None,
    show_legend: bool = True,
    yscale: str | None = None,
    ref_value: float | None = None,
    coverage_delta: bool = False,
    tick_label_scale: float = 1.0,
    axis_label_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    legend_font_size: float | None = None,
    figure_scale: float = 1.0,
    short_axis_labels: bool = False,
    shared_axis_labels: bool = False,
    height_scale: float = 1.0,
) -> FigureBase | None:
    """Plot per-metric, per-dataset lead-time curves as a panel figure.

    ``error_ylim`` (low, high) overrides auto y-limits for non-coverage rows;
    coverage rows remain fixed at [0, 1]. Either bound may be ``None``.
    When ``axes`` is provided, draw into that pre-made grid with shape
    (len(metrics_to_plot), n_datasets).

    ``yscale`` forces the y-axis scale for non-coverage rows (``'log'`` or
    ``'linear'``). When omitted, the historical default of log-scale is used.
    ``ref_value`` draws a dashed horizontal reference line at that y value on
    every non-coverage subplot (e.g. 1.0 for SSR's ideal dispersion).
    """
    rows: list[pd.DataFrame] = []
    base = df_in.dropna(
        subset=["run_path", "dataset_label", "plot_group"]
    ).drop_duplicates()
    for r in base.itertuples(index=False):
        run_path = getattr(r, "run_path", None)
        eval_subdir = normalize_eval_subdir(getattr(r, "eval_subdir", None))
        ds_label = getattr(r, "dataset_label", None)
        pg = getattr(r, "plot_group", None)
        if run_path is None or ds_label is None or pg is None:
            continue
        p = (
            results_root
            / str(run_path)
            / eval_subdir
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
        long = cast(pd.DataFrame, long[long["metric"].isin(metrics)].copy())
        if long.empty:
            continue
        long["dataset_label"] = ds_label
        long["plot_group"] = pg
        rows.append(long)

    if not rows:
        return None
    metrics_long = pd.concat(rows, ignore_index=True).dropna(subset=["value"])

    available_metrics = set(metrics_long["metric"].dropna().astype(str).unique())
    metrics_to_plot = [m for m in metrics if m in available_metrics]
    missing_metrics = [m for m in metrics if m not in available_metrics]
    if missing_metrics:
        print(
            "Skipping lead-time metrics not present in per-timestep file: "
            + ", ".join(missing_metrics)
        )
    if not metrics_to_plot:
        return None

    datasets = _apply_explicit_order(
        list(metrics_long["dataset_label"].unique()), dataset_order
    )
    groups = _order_groups_by_label(
        list(metrics_long["plot_group"].unique()), styles, hue_order
    )

    nrows, ncols = len(metrics_to_plot), len(datasets)
    shared_ylabel = (
        r"Rel. $\Delta$ empirical coverage"
        if coverage_delta and short_axis_labels
        else r"$\Delta$ empirical coverage (proportional)"
        if coverage_delta
        else None
    )
    if axes is None:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=_scaled_figsize(
                3.6 * ncols,
                2.7 * nrows * height_scale,
                figure_scale,
            ),
            sharex=True,
            sharey=False,
            squeeze=False,
        )
    else:
        axes = np.atleast_2d(axes)
        if fig is None:
            fig = axes[0][0].figure

    assert axes is not None
    assert fig is not None

    for r, metric in enumerate(metrics_to_plot):
        sub = metrics_long[metrics_long["metric"] == metric]
        for c, ds_label in enumerate(datasets):
            ax = axes[r][c]
            ds = sub[sub["dataset_label"] == ds_label]
            is_cov = metric.startswith("coverage_")
            is_ssr = metric == "ssr"
            is_cov_delta = bool(coverage_delta and is_cov)
            cov_target: float | None = None
            if is_cov_delta:
                try:
                    cov_target = float(metric.split("_", 1)[1])
                except Exception:
                    # If the metric name doesn't encode its nominal level,
                    # fall back to the raw coverage series.
                    is_cov_delta = False
                    cov_target = None
            # Per-metric y treatment: SSR is a dimensionless reliability
            # diagnostic with ideal=1.0, so it renders on a linear axis with
            # a dashed reference line regardless of the panel-wide settings.
            row_yscale = "linear" if (is_ssr or is_cov_delta) else yscale
            row_ref = 1.0 if (is_ssr and ref_value is None) else ref_value
            if is_cov_delta:
                row_ref = 0.0
            vals: list[float] = []
            for fam in groups:
                sf = cast(pd.DataFrame, ds[ds["plot_group"] == fam])
                if sf.empty:
                    continue
                agg = cast(
                    pd.DataFrame,
                    sf.groupby("timestep", as_index=False)["value"]
                    .agg(["mean", "std", "count"])
                    .set_index("timestep")
                    .sort_index()
                    .reset_index(),
                )
                st = styles.get(fam, {"color": "k"})
                mean = cast(pd.Series, agg["mean"])
                std = cast(pd.Series, agg["std"]).fillna(0)

                if is_cov_delta and cov_target is not None:
                    m = (mean / cov_target) - 1.0
                    vals.extend(m.dropna().tolist())
                else:
                    m = (
                        mean.clip(lower=0, upper=1)
                        if is_cov
                        else mean.clip(lower=1e-06)
                    )
                if (not is_cov) and (not is_cov_delta):
                    vals.extend(m.dropna().tolist())
                ax.plot(
                    agg["timestep"],
                    m,
                    color=st["color"],
                    lw=2,
                    linestyle=st.get("linestyle", "-"),
                )
                if (agg["count"] > 1).any():
                    if is_cov_delta and cov_target is not None:
                        y1 = ((mean - std) / cov_target) - 1.0
                        y2 = ((mean + std) / cov_target) - 1.0
                        vals.extend(y1.dropna().tolist())
                        vals.extend(y2.dropna().tolist())
                    elif is_cov:
                        y1 = (mean - std).clip(lower=0, upper=1)
                        y2 = (mean + std).clip(lower=0, upper=1)
                    else:
                        y1 = (mean - std).clip(lower=1e-6)
                        y2 = (mean + std).clip(lower=1e-6)
                    if (not is_cov) and (not is_cov_delta):
                        vals.extend(y1.dropna().tolist())
                        vals.extend(y2.dropna().tolist())
                    ax.fill_between(
                        agg["timestep"], y1, y2, color=st["color"], alpha=0.15
                    )

            if r == 0:
                ax.set_title(ds_label)
            if r == nrows - 1:
                ax.set_xlabel("" if shared_axis_labels else "Lead time")
            if c == 0:
                if is_cov_delta and cov_target is not None:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel(_metric_axis_label(metric))
            ax.grid(alpha=0.25)
            _apply_tick_label_style(ax, tick_label_scale)
            _apply_axis_label_style(ax, axis_label_scale)
            use_log = (row_yscale or "log") == "log"
            if is_cov and not is_cov_delta:
                ax.set_ylim(0, 1)
            elif is_cov_delta:
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    max_abs = max(abs(v) for v in finite)
                    pad = max(1e-6, 0.1 * max_abs)
                    lim = max(0.02, max_abs + pad)
                else:
                    lim = 0.1
                ax.set_yscale("linear")
                ax.set_ylim(-lim, lim)
            elif vals:
                if use_log:
                    ax.set_yscale("log")
                    ymin = max(1e-6, min(vals) * 0.8)
                    ymax = max(vals) * 1.25
                    if not np.isfinite(ymin) or np.isnan(ymin):
                        ymin = 1e-6
                    if not np.isfinite(ymax) or np.isnan(ymax):
                        ymax = 10.0
                else:
                    ax.set_yscale("linear")
                    finite = [v for v in vals if np.isfinite(v)]
                    if finite:
                        lo_v, hi_v = min(finite), max(finite)
                        pad = max(
                            1e-6, 0.1 * (hi_v - lo_v if hi_v > lo_v else abs(hi_v))
                        )
                        ymin, ymax = lo_v - pad, hi_v + pad
                        if row_ref is not None:
                            ymin = min(ymin, row_ref - pad)
                            ymax = max(ymax, row_ref + pad)
                    else:
                        ymin, ymax = 0.0, 1.0
                ax.set_ylim(bottom=ymin, top=ymax)
            if (not is_cov or is_cov_delta) and row_ref is not None:
                ax.axhline(
                    row_ref,
                    color="black",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.6,
                )
            # Apply user override for error rows (skip SSR: ratio diagnostic)
            if not is_cov and not is_ssr and error_ylim is not None:
                lo, hi = error_ylim
                cur_lo, cur_hi = ax.get_ylim()
                ax.set_ylim(
                    bottom=lo if lo is not None else cur_lo,
                    top=hi if hi is not None else cur_hi,
                )
    if shared_ylabel is not None:
        _set_shared_ylabel(fig, shared_ylabel, axis_label_scale)
    if shared_axis_labels:
        _set_shared_xlabel(fig, "Lead time", axis_label_scale)
    if show_legend:
        handles = _dedup_legend_handles(groups, styles)
        fig.legend(
            handles,
            [str(h.get_label()) for h in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=4,
            fontsize=_resolve_font_size(
                DEFAULT_LEGEND_FONT_SIZE,
                legend_font_scale,
                legend_font_size,
            ),
            frameon=False,
        )
    if save:
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        save_fig(fig, out_dir, name)
        return None
    return fig


# ---------------------------------------------------------------------------
# Training curves: wandb API
# ---------------------------------------------------------------------------


def _training_history_cache_path(
    run_dir: Path,
    eval_subdir: str = DEFAULT_EVAL_SUBDIR,
) -> Path:
    return run_dir / normalize_eval_subdir(eval_subdir) / "training_history.csv"


def _load_training_history_cache(
    run_dir: Path,
    eval_subdir: str = DEFAULT_EVAL_SUBDIR,
) -> pd.DataFrame | None:
    p = _training_history_cache_path(run_dir, eval_subdir=eval_subdir)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if df.empty or not {"epoch", "metric", "value"}.issubset(df.columns):
        return None
    return df


def _save_training_history_cache(
    run_dir: Path,
    df: pd.DataFrame,
    eval_subdir: str = DEFAULT_EVAL_SUBDIR,
) -> None:
    p = _training_history_cache_path(run_dir, eval_subdir=eval_subdir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    except OSError:
        pass


def parse_training_metrics_from_wandb(  # noqa: PLR0911, PLR0912, PLR0915
    run_dir: Path,
    eval_subdir: str = DEFAULT_EVAL_SUBDIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch per-epoch training metrics from wandb for a single run.

    Uses ``resolved_config.yaml`` for the wandb project/entity/display name,
    resolves the run via the public API, and scans its full history. Assumes
    ``WANDB_API_KEY`` is set. Results are cached to
    ``<run_dir>/<eval_subdir>/training_history.csv`` to avoid repeat API calls.

    Returns a long-format frame with columns ``epoch, metric, value``. Empty
    on any failure.
    """
    empty = pd.DataFrame(columns=pd.Index(["epoch", "metric", "value"]))

    if not force_refresh:
        cached = _load_training_history_cache(run_dir, eval_subdir=eval_subdir)
        if cached is not None:
            return cached

    config_p = run_dir / "resolved_config.yaml"
    if not config_p.exists():
        return empty

    try:
        with open(config_p) as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        return empty

    wb = (
        ((cfg.get("logging") or {}).get("wandb") or {}) if isinstance(cfg, dict) else {}
    )
    project = wb.get("project") or "autocast"
    entity = wb.get("entity")
    display_name = wb.get("name") or run_dir.name

    try:
        api = wandb.Api(timeout=60)
    except Exception as e:
        print(f"wandb API init failed ({e}); skipping {display_name}.")
        return empty

    if not entity:
        entity = getattr(api, "default_entity", None)
    if not entity:
        print(
            f"wandb: no entity configured for {display_name}; "
            "set logging.wandb.entity or WANDB_ENTITY."
        )
        return empty

    path = f"{entity}/{project}"
    try:
        runs = list(api.runs(path=path, filters={"display_name": display_name}))
    except Exception as e:
        print(f"wandb runs() failed for {display_name} ({path}): {e}")
        return empty

    if not runs:
        print(f"wandb: no run found for display_name={display_name!r} in {path}.")
        return empty
    if len(runs) > 1:
        # Prefer the most recently finished run for determinism.
        runs.sort(key=lambda r: getattr(r, "created_at", ""), reverse=True)

    wrun = runs[0]
    try:
        rows = list(wrun.scan_history())
    except Exception as e:
        print(f"wandb scan_history failed for {display_name}: {e}")
        return empty

    if not rows:
        return empty

    hist = pd.DataFrame(rows)
    # Find an epoch-like column; prefer an explicit 'epoch' log key.
    epoch_col: str | None = None
    for c in ("epoch", "trainer/global_epoch", "_step"):
        if c in hist.columns:
            epoch_col = c
            break
    if epoch_col is None:
        return empty

    metric_cols = [
        c
        for c in hist.columns
        if c != epoch_col
        and not c.startswith("_")
        and pd.api.types.is_numeric_dtype(hist[c])
    ]
    if not metric_cols:
        return empty

    sub = hist[[epoch_col, *metric_cols]].copy()
    sub[epoch_col] = pd.to_numeric(sub[epoch_col], errors="coerce")
    sub = cast(pd.DataFrame, sub[pd.notna(sub[epoch_col])])
    long = sub.melt(id_vars=epoch_col, var_name="metric", value_name="value").dropna(
        subset=["value"]
    )
    long = long.rename(columns={epoch_col: "epoch"})
    # Collapse multi-log-per-epoch to a single representative value.
    grouped = cast(
        pd.DataFrame,
        long.groupby(["epoch", "metric"], as_index=False).agg(value=("value", "mean")),
    )
    long = cast(
        pd.DataFrame,
        grouped.sort_values(by="metric").sort_values(by="epoch", kind="stable"),
    ).reset_index(drop=True)
    long["epoch"] = long["epoch"].astype(int, errors="ignore")

    _save_training_history_cache(run_dir, long, eval_subdir=eval_subdir)
    return long


def load_training_metrics(
    run_dir: Path,
    eval_subdir: str = DEFAULT_EVAL_SUBDIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load per-epoch training metrics for a single run from wandb."""
    return parse_training_metrics_from_wandb(
        run_dir,
        eval_subdir=eval_subdir,
        force_refresh=force_refresh,
    )


def plot_training_curves(  # noqa: PLR0912, PLR0915
    df_in: pd.DataFrame,
    metrics: list[str],
    results_root: Path,
    out_dir: Path,
    name: str,
    styles: dict,
    dataset_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    y_scale: str = "log",
    ylim: tuple[float | None, float | None] | None = None,
    axes: np.ndarray | None = None,
    fig: FigureBase | None = None,
    save: bool = True,
    show_legend: bool = True,
    force_refresh: bool = False,
    tick_label_scale: float = 1.0,
    axis_label_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    legend_font_size: float | None = None,
    figure_scale: float = 1.0,
    short_axis_labels: bool = False,
    shared_axis_labels: bool = False,
    height_scale: float = 1.0,
) -> FigureBase | None:
    """Plot per-metric, per-dataset training curves (epoch vs metric value).

    Training metrics are loaded via :func:`load_training_metrics` from wandb.
    ``metrics`` names should match logged keys, e.g. ``val_loss``, ``val_vrmse``.
    """
    _ = short_axis_labels
    rows: list[pd.DataFrame] = []
    base = df_in.dropna(
        subset=["run_path", "dataset_label", "plot_group"]
    ).drop_duplicates()
    for r in base.itertuples(index=False):
        run_path = getattr(r, "run_path", None)
        eval_subdir = normalize_eval_subdir(getattr(r, "eval_subdir", None))
        ds_label = getattr(r, "dataset_label", None)
        pg = getattr(r, "plot_group", None)
        if run_path is None or ds_label is None or pg is None:
            continue
        run_dir = results_root / str(run_path)
        if not run_dir.exists():
            continue
        tm = load_training_metrics(
            run_dir,
            eval_subdir=eval_subdir,
            force_refresh=force_refresh,
        )
        if tm.empty:
            continue
        tm = cast(pd.DataFrame, tm[tm["metric"].isin(metrics)].copy())
        if tm.empty:
            continue
        tm["dataset_label"] = ds_label
        tm["plot_group"] = pg
        rows.append(tm)

    if not rows:
        print("No training metrics available from wandb history.")
        return None

    long = pd.concat(rows, ignore_index=True).dropna(subset=["value"])
    available = set(long["metric"].astype(str).unique())
    metrics_to_plot = [m for m in metrics if m in available]
    missing = [m for m in metrics if m not in available]
    if missing:
        print("Skipping training metrics not found in logs: " + ", ".join(missing))
    if not metrics_to_plot:
        return None

    datasets = _apply_explicit_order(
        list(long["dataset_label"].unique()), dataset_order
    )
    groups = _order_groups_by_label(
        list(long["plot_group"].unique()), styles, hue_order
    )

    nrows, ncols = len(metrics_to_plot), len(datasets)
    if axes is None:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=_scaled_figsize(
                3.6 * ncols,
                2.7 * nrows * height_scale,
                figure_scale,
            ),
            sharex=False,
            sharey=False,
            squeeze=False,
        )
    else:
        axes = np.atleast_2d(axes)
        if fig is None:
            fig = axes[0][0].figure

    assert axes is not None
    assert fig is not None

    for r_idx, metric in enumerate(metrics_to_plot):
        sub = long[long["metric"] == metric]
        for c_idx, ds_label in enumerate(datasets):
            ax = axes[r_idx][c_idx]
            ds = sub[sub["dataset_label"] == ds_label]
            vals: list[float] = []
            for fam in groups:
                sf = cast(pd.DataFrame, ds[ds["plot_group"] == fam])
                if sf.empty:
                    continue
                agg = cast(
                    pd.DataFrame,
                    sf.groupby("epoch", as_index=False).agg(value=("value", "mean")),
                )
                agg = cast(pd.DataFrame, agg.sort_values(by="epoch"))
                st = styles.get(fam, {"color": "k"})
                ax.plot(
                    agg["epoch"],
                    agg["value"],
                    color=st["color"],
                    lw=2,
                    linestyle=st.get("linestyle", "-"),
                )
                vals.extend(agg["value"].dropna().tolist())

            if r_idx == 0:
                ax.set_title(ds_label)
            if r_idx == nrows - 1:
                ax.set_xlabel("" if shared_axis_labels else "Epoch")
            if c_idx == 0:
                ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
            _apply_tick_label_style(ax, tick_label_scale)
            _apply_axis_label_style(ax, axis_label_scale)

            positive = [v for v in vals if np.isfinite(v) and v > 0]
            if y_scale == "log" and positive:
                ax.set_yscale("log")
            elif y_scale == "linear":
                ax.set_yscale("linear")

            if ylim is not None:
                lo, hi = ylim
                cur_lo, cur_hi = ax.get_ylim()
                ax.set_ylim(
                    bottom=lo if lo is not None else cur_lo,
                    top=hi if hi is not None else cur_hi,
                )

    if shared_axis_labels:
        _set_shared_xlabel(fig, "Epoch", axis_label_scale)
    if show_legend:
        handles = _dedup_legend_handles(groups, styles)
        fig.legend(
            handles,
            [str(h.get_label()) for h in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=4,
            fontsize=_resolve_font_size(
                DEFAULT_LEGEND_FONT_SIZE,
                legend_font_scale,
                legend_font_size,
            ),
            frameon=False,
        )
    if save:
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        save_fig(fig, out_dir, name)
        return None
    return fig


# ---------------------------------------------------------------------------
# Panel composite figure
# ---------------------------------------------------------------------------


def plot_panel_figure(  # noqa: PLR0915
    df_in: pd.DataFrame,
    results_root: Path,
    out_dir: Path,
    styles: dict,
    dataset_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    overall_metrics: tuple[str, ...] = DEFAULT_PLOT_METRICS,
    error_metrics: list[str] | None = None,
    coverage_metrics: list[str] | None = None,
    training_metrics: list[str] | None = None,
    training_yscale: str = "log",
    training_ylim: tuple[float | None, float | None] | None = None,
    training_refresh: bool = False,
    error_ylim: tuple[float | None, float | None] | None = None,
    coverage_ylim: tuple[float | None, float | None] | None = None,
    name: str = "panel_figure.png",
    tick_label_scale: float = 1.0,
    axis_label_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    legend_font_size: float | None = None,
    figure_scale: float = 1.0,
    short_axis_labels: bool = False,
    shared_axis_labels: bool = False,
) -> None:
    """Render a panel composite figure with a reserved notes area.

    Layout::

        ┌──── notes (~20% x ~20%) ────┬───────────────────────────────┐
        │                              │ global legend                 │
        ├────────────┬─────────────────┼───────────────┬───────────────┤
        │ training   │ overall         │ coverage      │ lead-time     │
        │ curves     │ metrics         │ calibration   │ (err + cov)   │
        │ (far left) │                 │ panel         │ panel         │
        └────────────┴─────────────────┴───────────────┴───────────────┘
    """
    overall_metrics = tuple(overall_metrics or DEFAULT_PLOT_METRICS)
    error_metrics = error_metrics or ["vrmse"]
    coverage_metrics = coverage_metrics or ["coverage_0.9", "coverage_0.5"]
    if training_metrics is None:
        training_metrics = ["val_loss", "train_loss"]
    training_metrics = list(training_metrics)
    include_training_panel = len(training_metrics) > 0

    datasets = _apply_explicit_order(
        list(df_in["dataset_label"].dropna().unique()), dataset_order
    )
    n_ds = max(1, len(datasets))

    # Keep output shape approximately 16:10 and avoid over-compact single-dataset
    # renders by using a fixed minimum canvas with gentle dataset-driven scaling.
    panel_scale = max(1.0, np.sqrt(n_ds / 2.0))
    fig = plt.figure(
        figsize=_scaled_figsize(16.0 * panel_scale, 10.0 * panel_scale, figure_scale)
    )
    plot_style_kwargs = {
        "tick_label_scale": tick_label_scale,
        "axis_label_scale": axis_label_scale,
        "legend_font_scale": legend_font_scale,
        "legend_font_size": legend_font_size,
        "figure_scale": figure_scale,
        "short_axis_labels": short_axis_labels,
        "shared_axis_labels": shared_axis_labels,
    }
    if include_training_panel:
        outer = fig.add_gridspec(
            5,
            5,
            width_ratios=[1.3 * n_ds, 1.0, 1.1 * n_ds, 1.1 * n_ds, 1.0 * n_ds],
            height_ratios=[0.2, 0.8, 1.0, 1.0, 1.0],
            wspace=0.28,
            hspace=0.24,
        )
        left_spec = outer[1:, 1]
        middle_spec = outer[1:, 2:4]
        right_spec = outer[1:, 4]
    else:
        outer = fig.add_gridspec(
            5,
            4,
            width_ratios=[1.0, 1.1 * n_ds, 1.1 * n_ds, 1.0 * n_ds],
            height_ratios=[0.2, 0.8, 1.0, 1.0, 1.0],
            wspace=0.28,
            hspace=0.24,
        )
        left_spec = outer[1:, 0]
        middle_spec = outer[1:, 1:3]
        right_spec = outer[1:, 3]

    # Top-left reserved notes region (intentionally blank).
    notes_ax = fig.add_subplot(outer[0, 0])
    notes_ax.set_axis_off()

    # --- Far-left: training curves panel --------------------------------
    if include_training_panel:
        train = fig.add_subfigure(outer[1:, 0])
        n_train = max(1, len(training_metrics))
        tr_axes = train.subplots(
            n_train, n_ds, sharex=True, sharey=False, squeeze=False
        )
        _ = plot_training_curves(
            df_in,
            training_metrics,
            results_root,
            out_dir,
            "panel_training_curves.png",
            styles,
            dataset_order=dataset_order,
            hue_order=hue_order,
            y_scale=training_yscale,
            ylim=training_ylim,
            axes=tr_axes,
            fig=train,
            save=False,
            show_legend=False,
            force_refresh=training_refresh,
            **plot_style_kwargs,
        )

    # --- Left-middle: stacked overall-metric bar charts -----------------
    left = fig.add_subfigure(left_spec)
    n_overall = max(1, len(overall_metrics))
    left_axes = np.asarray(left.subplots(n_overall, 1, squeeze=False)).reshape(-1)
    for i, m in enumerate(overall_metrics):
        grouped_bar(
            df_in,
            f"overall_{m}",
            f"Overall {m.upper()}",
            m.upper(),
            out_dir,
            styles,
            y_scale=_overall_or_window_bar_yscale(m),
            dataset_order=dataset_order,
            hue_order=hue_order,
            ax=left_axes[i],
            save=False,
            ylim=_overall_or_window_bar_ylim(m, error_ylim),
            show_legend=False,
            ref_value=_overall_or_window_bar_ref_value(m),
            **plot_style_kwargs,
        )
    left.suptitle("")

    # --- Middle: coverage calibration panel -----------------------------
    middle = fig.add_subfigure(middle_spec)
    nrows_cov = len(WINDOW_ROWS)
    mid_axes = middle.subplots(nrows_cov, n_ds, sharex=True, sharey=True, squeeze=False)
    plot_coverage_calibration_panel(
        df_in,
        results_root,
        out_dir,
        styles,
        dataset_order=dataset_order,
        hue_order=hue_order,
        axes=mid_axes,
        fig=middle,
        save=False,
        show_legend=False,
        **plot_style_kwargs,
    )

    # --- Right: lead-time error (top) + coverage (bottom) ---------------
    right = fig.add_subfigure(right_spec)
    n_err = len(error_metrics)
    n_cov = len(coverage_metrics)
    right_top, right_bot = right.subfigures(
        2,
        1,
        height_ratios=[max(1, n_err), max(1, n_cov)],
        hspace=0.08,
    )
    rt_axes = right_top.subplots(n_err, n_ds, sharex=True, sharey=False, squeeze=False)
    plot_lead_time_panel(
        df_in,
        error_metrics,
        results_root,
        out_dir,
        "slide_lead_time_error.png",
        styles,
        dataset_order=dataset_order,
        hue_order=hue_order,
        axes=rt_axes,
        fig=right_top,
        save=False,
        error_ylim=error_ylim,
        show_legend=False,
        **plot_style_kwargs,
    )
    rb_axes = right_bot.subplots(n_cov, n_ds, sharex=True, sharey=False, squeeze=False)
    plot_lead_time_panel(
        df_in,
        coverage_metrics,
        results_root,
        out_dir,
        "slide_lead_time_coverage.png",
        styles,
        dataset_order=dataset_order,
        hue_order=hue_order,
        axes=rb_axes,
        fig=right_bot,
        save=False,
        error_ylim=None,
        show_legend=False,
        **plot_style_kwargs,
    )
    if coverage_ylim is not None:
        lo, hi = coverage_ylim
        rb_arr = np.asarray(rb_axes)
        for r, metric in enumerate(coverage_metrics):
            # SSR is a ratio diagnostic (linear, ref=1.0) mixed into the
            # coverage panel; do not squash it onto the coverage [0, 1] axis.
            if metric == "ssr":
                continue
            for ax in rb_arr[r]:
                cur_lo, cur_hi = ax.get_ylim()
                ax.set_ylim(
                    bottom=lo if lo is not None else cur_lo,
                    top=hi if hi is not None else cur_hi,
                )

    # --- Global legend --------------------------------------------------
    # Collect groups present anywhere for the legend.
    all_groups = list(df_in["plot_group"].dropna().unique())
    ordered_groups = _order_groups_by_label(all_groups, styles, hue_order)
    handles = _dedup_legend_handles(ordered_groups, styles)
    if handles:
        fig.legend(
            handles,
            [str(h.get_label()) for h in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(6, len(handles)),
            fontsize=_resolve_font_size(9.0, legend_font_scale, legend_font_size),
            frameon=False,
        )

    save_fig(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------


def main():  # noqa: PLR0912, PLR0915
    """Entry point: parse CLI args and generate all comparison plots."""
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
        "--run",
        action="append",
        nargs="+",
        metavar="RUN_ID",
        help=(
            "Add a run, with optional label: "
            '--run <id> ["label"] [hue] [eval=<subdir>]. '
            "Repeat for each run. Integer hue overrides --color-by-label "
            "for that run and groups runs into a shared hue family."
        ),
    )
    parser.add_argument(
        "--run-group",
        action="append",
        nargs="+",
        help="Group of runs sharing a color hue. Repeat for multiple groups.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Run directory names to include (alternative to --run).",
    )
    parser.add_argument(
        "--label-run",
        action="append",
        nargs=2,
        metavar=("RUN_ID", "LABEL"),
        help="(Deprecated, use --run) Assign a custom legend label.",
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
        default=list(DEFAULT_PLOT_METRICS),
        help=(
            "Metrics to plot for rollouts and overall "
            f"(default: {' '.join(DEFAULT_PLOT_METRICS)})"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print a summarized table of run metadata and exit",
    )
    parser.add_argument(
        "--key",
        action="append",
        help="(Deprecated, use --filter) Key for metadata filtering.",
    )
    parser.add_argument(
        "--value",
        action="append",
        help="(Deprecated, use --filter) Value matching the --key.",
    )
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        help=(
            "Boolean filter expression with AND, OR, and parentheses. "
            'e.g. --filter "Scale=large AND (Dataset=SW2D64 OR '
            'Dataset=CNS64)"'
        ),
    )
    parser.add_argument(
        "--sort",
        dest="sort_col",
        help=(
            "Column name (display name) to sort the --list table by, "
            "e.g. --sort Date or --sort Train_hr"
        ),
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the sort order when --sort is used",
    )
    parser.add_argument(
        "--uniform-group-color",
        action="store_true",
        help="Use the same color for all runs in a --run-group",
    )
    parser.add_argument(
        "--uniform-run-hue-color",
        action="store_true",
        help=(
            "Use the same color for runs sharing an integer hue in --run "
            "(otherwise shades vary within each hue group)."
        ),
    )
    parser.add_argument(
        "--group-hues",
        type=int,
        nargs="+",
        help=(
            "Map each --run-group to a hue index (0-based). "
            "Groups sharing a hue get shade variants. "
            "e.g. --group-hues 0 1 0 1"
        ),
    )
    parser.add_argument(
        "--color-by-label",
        action="store_true",
        help=(
            "Derive colors from --label-run mappings: same label = same color. "
            "Hue families from prefix before '('. "
            "No --run-group needed."
        ),
    )
    parser.add_argument(
        "--dataset-order",
        nargs="+",
        help=(
            "Display order for datasets on the x-axis (by label). "
            'e.g. --dataset-order "SW" "CNS64"'
        ),
    )
    parser.add_argument(
        "--error-ylim",
        nargs=2,
        metavar=("LOW", "HIGH"),
        help=(
            "Fixed y-axis range for error metrics (VRMSE, RMSE, etc.) on both "
            "bar charts and lead-time panels. Use 'auto' on either side to "
            "keep the automatic bound, e.g. --error-ylim 1e-3 auto"
        ),
    )
    parser.add_argument(
        "--coverage-ylim",
        nargs=2,
        metavar=("LOW", "HIGH"),
        help=(
            "Fixed y-axis range for coverage lead-time panels "
            "(defaults to 0-1). Coverage bars/calibration are unaffected."
        ),
    )
    parser.add_argument(
        "--lead-time-error-metrics",
        nargs="+",
        help=(
            "Metrics shown as rows in the lead-time error panel and top of "
            "the combined lead-time panel (default derives from --metrics)."
        ),
    )
    parser.add_argument(
        "--lead-time-coverage-metrics",
        nargs="+",
        help=(
            "Coverage metrics shown as rows in the lead-time coverage panel "
            "and bottom of the combined lead-time panel. "
            "e.g. --lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1"
        ),
    )
    parser.add_argument(
        "--lead-time-coverage-delta",
        action="store_true",
        help=(
            "Also render an additional lead-time coverage panel where each "
            "coverage_<p> curve is transformed to coverage / p - 1, i.e. "
            "proportional deviation from the nominal coverage level. Output: "
            "lead_time_panel_coverage_delta.png"
        ),
    )
    parser.add_argument(
        "--combined-lead-time",
        action="store_true",
        help=(
            "Also render a single lead-time panel stacking error metrics "
            "(top rows) and coverage metrics (bottom rows)."
        ),
    )
    parser.add_argument(
        "--no-ssr-in-coverage",
        dest="include_ssr_in_coverage",
        action="store_false",
        default=True,
        help=(
            "Suppress the default inclusion of 'ssr' as a row in "
            "the lead-time coverage panel (ssr renders with a linear axis "
            "and a reference line at 1.0)."
        ),
    )
    parser.add_argument(
        "--metric-groups",
        nargs="+",
        default=list(DEFAULT_METRIC_GROUPS),
        choices=[*METRIC_GROUP_PRESETS.keys(), "all", "none"],
        help=(
            "Lead-time panel groups to render. "
            "Presets: "
            + ", ".join(METRIC_GROUP_PRESETS.keys())
            + ". Use 'all' for every preset or 'none' to skip. "
            f"Default: {' '.join(DEFAULT_METRIC_GROUPS)}."
        ),
    )
    parser.add_argument(
        "--training-metrics",
        nargs="+",
        help=(
            "Training-curve metrics read from wandb history, "
            "e.g. --training-metrics val_loss train_loss. "
            "Omit to skip training curve plots."
        ),
    )
    parser.add_argument(
        "--training-yscale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scale for training-curve plots (default: log).",
    )
    parser.add_argument(
        "--training-ylim",
        nargs=2,
        metavar=("LOW", "HIGH"),
        help=(
            "Fixed y-axis range for training-curve plots. "
            "Use 'auto' on either side to keep the automatic bound."
        ),
    )
    parser.add_argument(
        "--training-refresh",
        action="store_true",
        help=(
            "Force re-fetching training history from wandb even if a local "
            "cache (<run>/<eval_subdir>/training_history.csv) already exists."
        ),
    )
    parser.add_argument(
        "--panel-figure",
        action="store_true",
        help=(
            "Also render a composite panel with overall bars, coverage "
            "calibration, lead-time panels, training curves, and a blank "
            "top-left notes area (~20% x ~20%)."
        ),
    )
    parser.add_argument(
        "--panel-figure-no-training",
        action="store_true",
        help=(
            "Also render a panel variant without the training-curves column. "
            "Can be combined with --panel-figure to save both variants."
        ),
    )
    parser.add_argument(
        "--coverage-panel-overall-rollout-windows",
        action="store_true",
        help=(
            "Also render coverage calibration panel variants for: "
            "all plus rollout windows, all only, and rollout windows only "
            "(using 0-4, 6-12, 13-30, 31-99; excluding the 0-1 rollout window)."
        ),
    )
    parser.add_argument(
        "--tick-label-scale",
        type=float,
        default=1.0,
        help="Scale tick label font sizes for publication plots (default: 1.0).",
    )
    parser.add_argument(
        "--axis-label-scale",
        type=float,
        default=1.0,
        help="Scale axis label font sizes for publication plots (default: 1.0).",
    )
    parser.add_argument(
        "--legend-font-scale",
        type=float,
        default=1.0,
        help="Scale legend font sizes for publication plots (default: 1.0).",
    )
    parser.add_argument(
        "--legend-font-size",
        type=float,
        help="Explicit legend font size; overrides --legend-font-scale.",
    )
    parser.add_argument(
        "--figure-scale",
        type=float,
        default=1.0,
        help="Scale generated figure sizes without changing layout (default: 1.0).",
    )
    parser.add_argument(
        "--short-axis-labels",
        action="store_true",
        help="Use compact axis labels for dense publication grid plots.",
    )
    parser.add_argument(
        "--shared-axis-labels",
        action="store_true",
        help="Use panel-level shared x/y labels where repeated labels are redundant.",
    )
    parser.add_argument(
        "--coverage-panel-height-scale",
        type=float,
        default=1.0,
        help="Scale standalone coverage-calibration panel height (default: 1.0).",
    )
    args = parser.parse_args()
    for style_arg in (
        "tick_label_scale",
        "axis_label_scale",
        "legend_font_scale",
        "figure_scale",
        "coverage_panel_height_scale",
    ):
        if getattr(args, style_arg) <= 0:
            msg = f"--{style_arg.replace('_', '-')} must be positive"
            raise SystemExit(msg)
    if args.legend_font_size is not None and args.legend_font_size <= 0:
        msg = "--legend-font-size must be positive"
        raise SystemExit(msg)

    def _parse_ylim(pair: list[str] | None) -> tuple[float | None, float | None] | None:
        if not pair:
            return None

        def _one(v: str) -> float | None:
            if v is None or str(v).lower() in {"auto", "none", "-"}:
                return None
            try:
                return float(v)
            except ValueError as exc:
                msg = f"Invalid ylim value: {v!r} (expected number or 'auto')"
                raise SystemExit(msg) from exc

        return (_one(pair[0]), _one(pair[1]))

    error_ylim = _parse_ylim(args.error_ylim)
    coverage_ylim = _parse_ylim(args.coverage_ylim)
    training_ylim = _parse_ylim(args.training_ylim)

    results_dir = resolve_results_root(args.results_dir)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = results_dir / "plots" / args.name

    if not args.list:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Merge --run entries into --runs / --label-run for unified handling.
    # Each --run entry is: <id> [label] [hue] [eval=<subdir>] in any order
    # for optional fields, with backward-compatible support for:
    #   <id>
    #   <id> <label>
    #   <id> <hue>
    #   <id> <label> <hue>
    hue_group_by_run: dict[str, int] = {}
    eval_subdir_by_run: dict[str, str] = {}
    run_order: list[str] = []

    def _parse_run_entry(
        entry: list[str],
    ) -> tuple[str, str | None, int | None, str | None]:
        if not entry:
            msg = "Error: --run requires at least <id>."
            raise ValueError(msg)

        run_id = entry[0]
        label: str | None = None
        hue: int | None = None
        eval_subdir: str | None = None

        for tok in entry[1:]:
            tok_s = str(tok)
            tok_l = tok_s.lower()
            if tok_l.startswith("eval="):
                if eval_subdir is not None:
                    msg = (
                        f"Error: duplicate eval subdir in --run {run_id!r}. "
                        "Use one eval=<subdir> token."
                    )
                    raise ValueError(msg)
                eval_value = tok_s.split("=", 1)[1]
                if not eval_value.strip():
                    msg = (
                        f"Error: empty eval subdir in --run {run_id!r}. "
                        "Use eval=<subdir>."
                    )
                    raise ValueError(msg)
                eval_subdir = eval_value
                continue

            if hue is None and tok_s.lstrip("-").isdigit():
                hue = int(tok_s)
                continue

            if label is None:
                label = tok_s
                continue

            msg = (
                f"Error: unrecognized extra token {tok_s!r} in --run {run_id!r}. "
                "Expected optional [label] [hue] and/or eval=<subdir>."
            )
            raise ValueError(msg)

        return run_id, label, hue, eval_subdir

    if args.run:
        merged_runs: list[str] = []
        merged_labels: list[list[str]] = list(args.label_run or [])
        parsed_entries: list[tuple[str, str | None, int | None, str]] = []
        for entry in args.run:
            try:
                run_id, label, hue, run_eval_subdir = _parse_run_entry(entry)
            except ValueError as e:
                print(e)
                sys.exit(1)
            parsed_entries.append(
                (run_id, label, hue, normalize_eval_subdir(run_eval_subdir))
            )
            merged_runs.append(run_id)
            eval_subdir_by_run[run_id] = normalize_eval_subdir(run_eval_subdir)

        # Build stable per-entry refs so repeated run_ids with different eval/labels
        # are treated as distinct series in style/legend/hue mapping.
        ref_counts: dict[str, int] = {}
        for run_id, label, hue, run_eval_subdir in parsed_entries:
            base_ref = run_id
            if run_eval_subdir != DEFAULT_EVAL_SUBDIR:
                base_ref = f"{run_id}::eval={run_eval_subdir}"
            n_seen = ref_counts.get(base_ref, 0)
            ref_counts[base_ref] = n_seen + 1
            run_ref = base_ref if n_seen == 0 else f"{base_ref}#{n_seen + 1}"
            run_order.append(run_ref)
            if label is not None:
                merged_labels.append([run_ref, label])
            if hue is not None:
                hue_group_by_run[run_ref] = hue
        # --run takes precedence; append any extra --runs
        if args.runs:
            for run_id in args.runs:
                merged_runs.append(run_id)
                run_order.append(run_id)
        args.runs = merged_runs
        args.label_run = merged_labels or None
    elif args.runs:
        run_order = list(args.runs)

    explicit_groups = args.run_group or []
    # Gather explicitly requested runs, or fallback to auto-discover
    if explicit_groups:
        run_targets = [
            (results_dir / r, DEFAULT_EVAL_SUBDIR, r)
            for g in explicit_groups
            for r in g
        ]
    elif args.runs:
        run_targets = []
        if args.run:
            ref_counts = {}
            for entry in args.run:
                run_id, _, _, run_eval_subdir = _parse_run_entry(entry)
                eval_subdir = normalize_eval_subdir(run_eval_subdir)
                base_ref = run_id
                if eval_subdir != DEFAULT_EVAL_SUBDIR:
                    base_ref = f"{run_id}::eval={eval_subdir}"
                n_seen = ref_counts.get(base_ref, 0)
                ref_counts[base_ref] = n_seen + 1
                run_ref = base_ref if n_seen == 0 else f"{base_ref}#{n_seen + 1}"
                run_targets.append((results_dir / run_id, eval_subdir, run_ref))
            if args.runs:
                for run_id in args.runs[len(args.run) :]:
                    run_targets.append(
                        (results_dir / run_id, DEFAULT_EVAL_SUBDIR, run_id)
                    )
        else:
            run_targets = [
                (
                    results_dir / r,
                    normalize_eval_subdir(
                        eval_subdir_by_run.get(r, DEFAULT_EVAL_SUBDIR)
                    ),
                    r,
                )
                for r in args.runs
            ]
    else:
        run_targets = sorted(
            [
                (p, DEFAULT_EVAL_SUBDIR, p.name)
                for p in results_dir.iterdir()
                if p.is_dir() and (p / "eval" / "evaluation_metrics.csv").exists()
            ],
            key=lambda x: x[0],
        )

    valid_run_names = {
        name for rd, _, run_ref in run_targets for name in (run_ref, rd.name)
    }

    try:
        custom_label_by_run = build_custom_label_map(
            args.label_run,
            valid_run_names=valid_run_names,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.list:
        print(f"Loading {len(run_targets)} runs...")
        m_rows = []
        for rd, run_eval_subdir, run_ref in run_targets:
            m_rows.append(
                load_config_metadata(rd, eval_subdir=run_eval_subdir, run_ref=run_ref)
            )
        mdf = pd.DataFrame(m_rows)

        if mdf.empty:
            print("No valid runs to process.")
            sys.exit(1)

        _parsed = mdf["run_id"].map(parse_loss_dataset_arch)
        mdf["loss_family"] = _parsed.map(
            lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else None
        )
        parsed_dataset = _parsed.map(
            lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else None
        )
        parsed_dataset = cast(
            pd.Series,
            parsed_dataset.where(
                ~parsed_dataset.isin(["cached_latents", "latents", "encoded"]),
                np.nan,
            ),
        )
        mdf["dataset_module"] = parsed_dataset.fillna(
            mdf.get("dataset_from_data_path")
        ).fillna(mdf["dataset"].map(normalize_dataset_module))
        mdf["dataset_label"] = mdf["dataset_module"].map(dataset_label_from_module)
        mdf["arch_segment"] = _parsed.map(
            lambda x: x[2] if isinstance(x, tuple) and len(x) > 2 else None
        )
        mdf["arch_key"] = mdf["arch_segment"].map(arch_key_from_processor_segment)
        mdf["model_scale"] = assign_model_scale(mdf)

        merged = mdf.copy()

        if "train_total_s" in merged.columns:
            _train_s = cast(
                pd.Series,
                pd.to_numeric(merged["train_total_s"], errors="coerce"),
            )
            merged["train_hrs"] = (_train_s / 3600.0).round(1)

        if "model_latency_ms_per_sample" in merged.columns:
            _infer_ms = cast(
                pd.Series,
                pd.to_numeric(merged["model_latency_ms_per_sample"], errors="coerce"),
            )
            merged["infer_ms"] = _infer_ms.round(2)

        if "params_processor_total" in merged.columns:
            _params = pd.to_numeric(merged["params_processor_total"], errors="coerce")
            merged["params_M"] = (
                (_params / 1e6).round(1).astype(str) + "M"  # type: ignore[operator]
            )

        show_cols = [
            "run_name",
            "dataset_label",
            "processor",
            "resolution",
            "model_scale",
            "params_M",
            "n_steps_input",
            "n_steps_output",
            "loss_func",
            "noise_injector",
            "noise_channels",
            "ode_steps",
            "n_gpus",
            "batch_size",
            "eff_batch_size",
            "lr",
            "train_hrs",
            "train_mean_epoch_s",
            "infer_ms",
            "run_date",
        ]
        show_cols = [c for c in show_cols if c in merged.columns]

        renames = {
            "run_name": "Run",
            "dataset_label": "Dataset",
            "processor": "Model",
            "resolution": "Resolution",
            "model_scale": "Scale",
            "params_M": "Params",
            "n_steps_input": "N_In",
            "n_steps_output": "N_Out",
            "loss_func": "Loss",
            "noise_injector": "NoiseInj",
            "noise_channels": "NoiseC",
            "ode_steps": "ODE",
            "n_gpus": "GPUs",
            "batch_size": "BS",
            "eff_batch_size": "EffBS",
            "lr": "LR",
            "train_hrs": "Train_hr",
            "train_mean_epoch_s": "Epoch_s",
            "infer_ms": "Infer_ms",
            "run_date": "Date",
        }
        merged = merged.rename(columns=renames)

        # Build filter expression from legacy --key/--value or --filter.
        filter_expr = args.filter_expr or ""
        if not filter_expr and args.key and args.value:
            if len(args.key) != len(args.value):
                print(
                    "Error: --key and --value must be given the same number of times."
                )
                sys.exit(1)
            parts = [f"{k}={v}" for k, v in zip(args.key, args.value, strict=False)]
            filter_expr = " OR ".join(parts)

        if filter_expr:
            merged = apply_filter_expr(
                filter_expr,
                merged,
                col_aliases=renames,
            )

        if args.sort_col:
            if args.sort_col in merged.columns:
                merged = cast(
                    pd.DataFrame,
                    merged.sort_values(
                        args.sort_col,
                        ascending=not args.reverse,
                        na_position="last",
                    ),
                )
            else:
                available = ", ".join(merged.columns.tolist())
                print(
                    f"Warning: --sort column '{args.sort_col}' not found. "
                    f"Available columns: {available}"
                )

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", 40)
        print("\n--- Available Runs Metadata ---\n")
        disp_cols = [r for c, r in renames.items() if r in merged.columns]
        print(cast(pd.DataFrame, merged[disp_cols]).to_string(index=False))
        sys.exit(0)

    print(f"Loading {len(run_targets)} runs...")
    rows = []
    for d, run_eval_subdir, run_ref in run_targets:
        if not d.exists():
            print(f"Warning: requested run not found: {d}")
            continue
        try:
            r = load_single_run_metrics(
                d,
                eval_subdir=run_eval_subdir,
                run_ref=run_ref,
            )
        except Exception as e:
            print(f"Error loading {d}: {e}")
            continue
        rows.append(r)

    if not rows:
        print("No valid runs to process.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    _parsed = df["run_id"].map(parse_loss_dataset_arch)
    df["loss_family"] = _parsed.map(
        lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else None
    )
    parsed_dataset = _parsed.map(
        lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else None
    )
    parsed_dataset = cast(
        pd.Series,
        parsed_dataset.where(
            ~parsed_dataset.isin(["cached_latents", "latents", "encoded"]), np.nan
        ),
    )
    df["dataset_module"] = parsed_dataset.fillna(
        df.get("dataset_from_data_path")
    ).fillna(df["dataset"].map(normalize_dataset_module))
    df["arch_segment"] = _parsed.map(
        lambda x: x[2] if isinstance(x, tuple) and len(x) > 2 else None
    )
    df["arch_key"] = df["arch_segment"].map(arch_key_from_processor_segment)
    df["dataset_label"] = df["dataset_module"].map(dataset_label_from_module)
    df["model_scale"] = assign_model_scale(df)

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
        df = cast(pd.DataFrame, df[df["dataset_module"].isin(args.datasets)])
    if args.models:
        df = cast(pd.DataFrame, df[df["arch_key"].isin(args.models)])

    if cast(pd.DataFrame, df).empty:
        print("No valid runs remain after applying filters.")
        sys.exit(1)

    # Styling
    df = cast(pd.DataFrame, df)
    if "train_total_s" in df.columns:
        _train_s = cast(
            pd.Series,
            pd.to_numeric(df["train_total_s"], errors="coerce"),
        )
        df["train_hrs"] = _train_s / 3600.0
    styles = build_family_style(
        df,
        explicit_groups,
        custom_label_by_run=custom_label_by_run,
        uniform_group_color=args.uniform_group_color,
        uniform_run_hue_color=args.uniform_run_hue_color,
        group_hues=args.group_hues,
        color_by_label=args.color_by_label,
        hue_group_by_run=hue_group_by_run or None,
    )

    n_ds = df["dataset_label"].nunique()
    n_mv = df["plot_group"].nunique()
    print(f"Found {n_ds} datasets and {n_mv} model variants.")

    ds_order = args.dataset_order

    # Derive hue order from --runs order: first unique label seen wins.
    hu_order: list[str] | None = None
    if run_order and custom_label_by_run:
        seen: list[str] = []
        seen_set: set[str] = set()
        for run in run_order:
            label = custom_label_by_run.get(run)
            if label and label not in seen_set:
                seen.append(label)
                seen_set.add(label)
        if seen:
            hu_order = seen

    plot_style_kwargs = {
        "tick_label_scale": args.tick_label_scale,
        "axis_label_scale": args.axis_label_scale,
        "legend_font_scale": args.legend_font_scale,
        "legend_font_size": args.legend_font_size,
        "figure_scale": args.figure_scale,
        "short_axis_labels": args.short_axis_labels,
        "shared_axis_labels": args.shared_axis_labels,
    }
    coverage_plot_style_kwargs = {
        **plot_style_kwargs,
        "height_scale": args.coverage_panel_height_scale,
    }
    write_single_step_results_table(
        df,
        out_dir,
        styles,
        dataset_order=ds_order,
        hue_order=hu_order,
    )

    # Render overall bars
    for m in args.metrics:
        grouped_bar(
            df,
            f"overall_{m}",
            f"Overall {m.upper()}",
            m.upper(),
            out_dir,
            styles,
            y_scale=_overall_or_window_bar_yscale(m),
            dataset_order=ds_order,
            hue_order=hu_order,
            ylim=_overall_or_window_bar_ylim(m, error_ylim),
            ref_value=_overall_or_window_bar_ref_value(m),
            **plot_style_kwargs,
        )

    # Render window bars
    for m in args.metrics:
        for w in ROLL_WINDOWS:
            grouped_bar(
                df,
                f"{m}_{w}",
                f"{m.upper()} in window {w}",
                m.upper(),
                out_dir,
                styles,
                y_scale=_overall_or_window_bar_yscale(m),
                dataset_order=ds_order,
                hue_order=hu_order,
                ylim=_overall_or_window_bar_ylim(m, error_ylim),
                ref_value=_overall_or_window_bar_ref_value(m),
                **plot_style_kwargs,
            )

    # Render Calibration panel
    plot_coverage_calibration_panel(
        df,
        results_dir,
        out_dir,
        styles,
        dataset_order=ds_order,
        hue_order=hu_order,
        **coverage_plot_style_kwargs,
    )
    if args.coverage_panel_overall_rollout_windows:
        plot_coverage_calibration_panel(
            df,
            results_dir,
            out_dir,
            styles,
            dataset_order=ds_order,
            hue_order=hu_order,
            window_rows=WINDOW_ROWS_OVERALL_AND_ROLLOUT,
            name="coverage_calibration_panel_overall_rollout_windows.png",
            **coverage_plot_style_kwargs,
        )
        plot_coverage_calibration_panel(
            df,
            results_dir,
            out_dir,
            styles,
            dataset_order=ds_order,
            hue_order=hu_order,
            window_rows=WINDOW_ROWS_OVERALL_ONLY,
            name="coverage_calibration_panel_overall_only.png",
            **coverage_plot_style_kwargs,
        )
        plot_coverage_calibration_panel(
            df,
            results_dir,
            out_dir,
            styles,
            dataset_order=ds_order,
            hue_order=hu_order,
            window_rows=WINDOW_ROWS_ROLLOUT_ONLY,
            name="coverage_calibration_panel_rollout_windows_only.png",
            **coverage_plot_style_kwargs,
        )

    # Render efficiency bars (training/inference), if available.
    for metric, title, ylabel in [
        ("train_hrs", "Training time total (hours)", "hours"),
        ("train_mean_epoch_s", "Training time per epoch (seconds)", "seconds"),
        (
            "model_latency_ms_per_sample",
            "Inference latency per sample (ms)",
            "ms",
        ),
        (
            "model_throughput_samples_per_sec",
            "Inference throughput (samples/s)",
            "samples/s",
        ),
    ]:
        is_train_metric = metric in {"train_hrs", "train_mean_epoch_s"}
        grouped_bar(
            df,
            metric,
            title,
            ylabel,
            out_dir,
            styles,
            y_scale="linear" if is_train_metric else "auto",
            dataset_order=ds_order,
            hue_order=hu_order,
            **plot_style_kwargs,
        )

    # Render lead-time panels
    err_metric, cov_metric = _derive_lead_time_metrics(list(args.metrics))

    # Explicit overrides
    if args.lead_time_error_metrics:
        err_metric = list(args.lead_time_error_metrics)
    if args.lead_time_coverage_metrics:
        cov_metric = list(args.lead_time_coverage_metrics)

    # SSR is a dimensionless dispersion diagnostic (ideal=1.0) that belongs
    # alongside the coverage rows in the lead-time coverage panel by default.
    if args.include_ssr_in_coverage and "ssr" not in cov_metric:
        cov_metric = [*cov_metric, "ssr"]
    elif not args.include_ssr_in_coverage:
        cov_metric = [m for m in cov_metric if m != "ssr"]

    if err_metric:
        plot_lead_time_panel(
            df,
            err_metric,
            results_dir,
            out_dir,
            "lead_time_panel_error.png",
            styles,
            dataset_order=ds_order,
            hue_order=hu_order,
            error_ylim=error_ylim,
            **plot_style_kwargs,
        )
    if cov_metric:
        plot_lead_time_panel(
            df,
            cov_metric,
            results_dir,
            out_dir,
            "lead_time_panel_coverage.png",
            styles,
            dataset_order=ds_order,
            hue_order=hu_order,
            **plot_style_kwargs,
        )
        if args.lead_time_coverage_delta:
            delta_metrics = [m for m in cov_metric if m.startswith("coverage_")]
            if delta_metrics:
                plot_lead_time_panel(
                    df,
                    delta_metrics,
                    results_dir,
                    out_dir,
                    "lead_time_panel_coverage_delta.png",
                    styles,
                    dataset_order=ds_order,
                    hue_order=hu_order,
                    coverage_delta=True,
                    **plot_style_kwargs,
                )

    # Combined lead-time panel (error rows on top, coverage rows below).
    if args.combined_lead_time and (err_metric or cov_metric):
        combined_metrics = list(err_metric) + list(cov_metric)
        plot_lead_time_panel(
            df,
            combined_metrics,
            results_dir,
            out_dir,
            "lead_time_panel_combined.png",
            styles,
            dataset_order=ds_order,
            hue_order=hu_order,
            error_ylim=error_ylim,
            **plot_style_kwargs,
        )

    # Per-group lead-time panels (SSR, coherence, balancing coverage, physics)
    group_selection = list(args.metric_groups or [])
    if "none" in group_selection:
        group_selection = []
    elif "all" in group_selection:
        group_selection = list(METRIC_GROUP_PRESETS.keys())
    # Dedup while preserving order
    seen_groups: set[str] = set()
    ordered_groups: list[str] = []
    for g in group_selection:
        if g in METRIC_GROUP_PRESETS and g not in seen_groups:
            ordered_groups.append(g)
            seen_groups.add(g)
    for group_id in ordered_groups:
        preset = METRIC_GROUP_PRESETS[group_id]
        group_metrics = cast(list[str], preset["metrics"])
        group_name = cast(str, preset["name"])
        group_yscale = cast(str, preset.get("yscale", "log"))
        group_ref = cast("float | None", preset.get("ref"))
        plot_lead_time_panel(
            df,
            group_metrics,
            results_dir,
            out_dir,
            group_name,
            styles,
            dataset_order=ds_order,
            hue_order=hu_order,
            error_ylim=None,
            yscale=group_yscale,
            ref_value=group_ref,
            **plot_style_kwargs,
        )

    # Training curves
    if args.training_metrics:
        plot_training_curves(
            df,
            list(args.training_metrics),
            results_dir,
            out_dir,
            "training_curves.png",
            styles,
            dataset_order=ds_order,
            hue_order=hu_order,
            y_scale=args.training_yscale,
            ylim=training_ylim,
            force_refresh=args.training_refresh,
            **plot_style_kwargs,
        )

    # Composite panel figure(s)
    panel_kwargs = {
        "dataset_order": ds_order,
        "hue_order": hu_order,
        "overall_metrics": tuple(args.metrics or DEFAULT_PLOT_METRICS),
        "error_metrics": err_metric or ["vrmse"],
        "coverage_metrics": cov_metric
        or ["coverage_0.9", "coverage_0.5", "coverage_0.1"],
        "training_yscale": args.training_yscale,
        "training_ylim": training_ylim,
        "training_refresh": args.training_refresh,
        "error_ylim": error_ylim,
        "coverage_ylim": coverage_ylim,
        **plot_style_kwargs,
    }
    if args.panel_figure:
        plot_panel_figure(
            df,
            results_dir,
            out_dir,
            styles,
            training_metrics=list(args.training_metrics)
            if args.training_metrics
            else ["val_loss", "train_loss"],
            name="panel_figure.png",
            **panel_kwargs,
        )
    if args.panel_figure_no_training:
        plot_panel_figure(
            df,
            results_dir,
            out_dir,
            styles,
            training_metrics=[],
            name="panel_figure_no_training.png",
            **panel_kwargs,
        )

    print("Finished generating plots.")


if __name__ == "__main__":
    main()
