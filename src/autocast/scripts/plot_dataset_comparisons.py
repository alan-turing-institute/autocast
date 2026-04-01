from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import matplotlib as mpl
import yaml
from matplotlib.lines import Line2D

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
    "gpe_laser_only_wake": "Gpe Laser Only Wake",
    "gpehc64": "GPEHC64",
    "gpelc64": "GPELC64",
    "gs64": "GS64",
    "lb128x32": "LB128x32",
    "rd64": "RD64",
    "shallow_water2d_128": "Shallow Water2D 128",
    "sw2d464": "SW2D464",
    "sw2d64": "SW2D64",
}

# Canonical grid resolution for each known dataset module.
DATASET_RESOLUTION: dict[str, str] = {
    "ad64": "64x64",
    "adm32": "64x64",
    "cns64": "64x64",
    "gpe_ri_high_complexity": "64x64",
    "gpe_ri_low_complexity": "64x64",
    "gpe_laser_only_wake": "64x64",
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
MODEL_SCALE_PARAM_COL = "params_processor_total"


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
        r"^(diff|crps|epd)_(.+)_([0-9a-f]{7}|[a-z]+)_[0-9a-f]{7}$", str(run_name)
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


def load_single_run_metrics(run_dir: Path) -> dict:  # noqa: PLR0912, PLR0915
    """Load evaluation metrics and rollout metrics from a single run directory."""
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


def load_config_metadata(run_dir: Path) -> dict[str, object]:  # noqa: PLR0912
    """Load training config metadata (LR, batch size, noise, etc.)."""
    row: dict[str, object] = {"run_name": run_dir.name}
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
    p_meta = run_dir / "eval" / "evaluation_metadata.csv"
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
    p_bench = run_dir / "eval" / "benchmark_metrics.csv"
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


def _style_label_for_plot_group(
    pg: str,
    custom_label_by_run: dict[str, str] | None,
) -> str:
    """Return custom label for a run when provided, else the default label."""
    if custom_label_by_run:
        run_name = _run_name_from_plot_group(pg)
        if run_name and run_name in custom_label_by_run:
            return custom_label_by_run[run_name]
    return plot_group_display_label(pg)


def extract_valid_plot_groups_from_run_names(
    run_names: list[str], df_in: pd.DataFrame
) -> set[str]:
    """Return the plot_group keys that correspond to the given run directory names."""
    # Given a list of run dir names, find their corresponding plot_groups in df_in
    return set(
        cast(pd.Series, df_in[df_in["run_name"].isin(run_names)]["plot_group"]).unique()
    )


def _shade_variant(base_color, idx: int, total: int):
    """Return a deterministic lightness variant for a color within a family."""
    if total <= 1:
        return base_color
    if total == 2:
        return (
            _mix_with_white(base_color, 0.22)
            if idx == 0
            else _mix_with_black(base_color, 0.18)
        )
    frac = (idx / (total - 1)) - 0.5  # -0.5..0.5
    return (
        _mix_with_white(base_color, frac * 0.8)
        if frac > 0
        else _mix_with_black(base_color, -frac * 0.6)
    )


def build_family_style(
    df_in: pd.DataFrame,
    explicit_groups: list[list[str]] | None = None,
    custom_label_by_run: dict[str, str] | None = None,
) -> dict:
    """Build a style dict (color, label, marker, linestyle) for each plot_group."""
    styles = {}
    _present_raw = cast(pd.Series, df_in["plot_group"].dropna())
    present = sorted(_present_raw.unique().tolist())

    if explicit_groups and len(explicit_groups) > 0:
        base_cmap = plt.get_cmap("tab10")
        for i, group_runs in enumerate(explicit_groups):
            base_color = base_cmap(i % 10)
            group_pgs = extract_valid_plot_groups_from_run_names(group_runs, df_in)
            group_pgs = sorted(group_pgs)
            if not group_pgs:
                continue

            # Vary lightness across the group's plot groups
            n = len(group_pgs)
            for j, pg in enumerate(group_pgs):
                if pg in styles:
                    continue  # Conflict resolution (first claims)
                c = _shade_variant(base_color, j, n)

                parts = pg.split("__")
                loss = parts[1] if len(parts) > 1 else "unknown"
                styles[pg] = {
                    "color": c,
                    "label": _style_label_for_plot_group(pg, custom_label_by_run),
                    "marker": "^" if loss == "diff" else "o",
                    "linestyle": "-" if loss == "diff" else "--",
                }

    # Fallback for remaining plot groups: each run gets its own hue.
    remaining = [pg for pg in present if pg not in styles]
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
    run_dirs: list[Path], labels: list[str] | None
) -> dict[str, str]:
    """Build run_name -> custom label map from CLI labels.

    A label value of "None" (case-insensitive) means: keep default auto label.
    """
    if not labels:
        return {}
    if len(labels) != len(run_dirs):
        msg = (
            "--labels count must match selected run count. "
            f"Got {len(labels)} labels for {len(run_dirs)} runs."
        )
        raise ValueError(msg)

    custom: dict[str, str] = {}
    for run_dir, label in zip(run_dirs, labels, strict=True):
        norm = str(label).strip()
        if norm.casefold() == "none":
            continue
        custom[run_dir.name] = norm
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


def grouped_bar(
    df_in: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    styles: dict,
    y_scale: str = "auto",
):
    """Render a grouped bar chart of a metric across datasets and model families."""
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
        vals = cast(
            np.ndarray,
            cast(
                pd.Series,
                pd.to_numeric(
                    g[g[group_col] == fam]
                    .set_index("dataset_label")
                    .reindex(datasets)[metric],
                    errors="coerce",
                ),
            ).to_numpy(dtype=np.dtype("float64")),  # type: ignore[arg-type]
        )
        all_positive.extend(float(v) for v in vals if np.isfinite(v) and v > 0)
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

    if y_scale == "linear":
        ax.set_yscale("linear")
    elif all_positive and min(all_positive) > 0:
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
    """Plot the coverage calibration panel (rollout windows x datasets)."""
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
                    ).sort_values("coverage_level"),
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
        Line2D(
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
        [str(h.get_label()) for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    save_fig(fig, out_dir, "coverage_calibration_panel.png")


def plot_lead_time_panel(  # noqa: PLR0912, PLR0915
    df_in: pd.DataFrame,
    metrics: list[str],
    results_root: Path,
    out_dir: Path,
    name: str,
    styles: dict,
):
    """Plot per-metric, per-dataset lead-time curves as a panel figure."""
    rows = []
    base = df_in.dropna(
        subset=["run_path", "dataset_label", "plot_group"]
    ).drop_duplicates()
    for r in base.itertuples(index=False):
        run_path = getattr(r, "run_path", None)
        ds_label = getattr(r, "dataset_label", None)
        pg = getattr(r, "plot_group", None)
        if run_path is None or ds_label is None or pg is None:
            continue
        p = (
            results_root
            / str(run_path)
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
        long = cast(pd.DataFrame, long[long["metric"].isin(metrics)].copy())
        if long.empty:
            continue
        long["dataset_label"] = ds_label
        long["plot_group"] = pg
        rows.append(long)

    if not rows:
        return
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
        return

    datasets = sorted(metrics_long["dataset_label"].unique())
    groups = sorted(metrics_long["plot_group"].unique())

    nrows, ncols = len(metrics_to_plot), len(datasets)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 2.7 * nrows),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for r, metric in enumerate(metrics_to_plot):
        sub = metrics_long[metrics_long["metric"] == metric]
        for c, ds_label in enumerate(datasets):
            ax = axes[r][c]
            ds = sub[sub["dataset_label"] == ds_label]
            is_cov = metric.startswith("coverage_")
            vals = []
            for fam in groups:
                sf = cast(pd.DataFrame, ds[ds["plot_group"] == fam])
                if sf.empty:
                    continue
                agg = cast(
                    pd.DataFrame,
                    cast(
                        pd.DataFrame,
                        sf.groupby("timestep", as_index=False)["value"].agg(
                            ["mean", "std", "count"]
                        ),
                    ).sort_values("timestep"),
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
        Line2D(
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
        [str(h.get_label()) for h in handles],
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
        "--run-group",
        action="append",
        nargs="+",
        help="Group of runs sharing a color hue. Repeat for multiple groups.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Run directory names to include (alternative to --run-group).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help=(
            "Custom legend labels in run order (from --runs or flattened --run-group). "
            "Use 'None' to keep default auto label for a specific run."
        ),
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

    try:
        custom_label_by_run = build_custom_label_map(run_dirs, args.labels)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.list:
        print(f"Loading {len(run_dirs)} runs...")
        m_rows = []
        for rd in run_dirs:
            m_rows.append(load_config_metadata(rd))
        mdf = pd.DataFrame(m_rows)

        if mdf.empty:
            print("No valid runs to process.")
            sys.exit(1)

        _parsed = mdf["run_name"].map(parse_loss_dataset_arch)
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
    )

    n_ds = df["dataset_label"].nunique()
    n_mv = df["plot_group"].nunique()
    print(f"Found {n_ds} datasets and {n_mv} model variants.")

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
        )

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
