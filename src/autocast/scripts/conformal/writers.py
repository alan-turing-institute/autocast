"""Write the fixed conformal-calibration output layout.

Every path and column name here is part of the reviewed, fixed layout (see
this package's __init__.py module map docstrings) -- do not rename anything
without checking with the owner first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from autocast.scripts.conformal.scoring import (
    LEVELS,
    NOMINAL_ALPHA,
    NOMINAL_LEVEL,
    WINDOWS,
    FittedMethod,
    coverage_reliability_table,
    per_frame_coverage_calibration_error,
    per_frame_excess_kurtosis,
    per_frame_ingredients,
    per_frame_member_metric_values,
    per_frame_observed_coverage,
    per_lead_winkler,
    rank_histogram_per_frame,
    window_row,
)
from autocast.types import TensorBTSC

#: Constant `batch_idx` column value in `rollout_metrics.csv` (paper format:
#: every row is already fully reduced over the batch dimension).
_BATCH_IDX_ALL = "all"


def _paper_window_label(window: tuple[int, int]) -> str:
    """Return a "start-end" filename/column label, matching the paper's CSVs."""
    start, end = window
    return f"{start}-{end}"


def _windows_within(n_frames: int) -> list[tuple[int, int]]:
    """`WINDOWS` clipped to `[0, n_frames)`, dropping any that become empty.

    Production dumps always have T=100 frames, so every configured window
    fits. Test dumps are often shorter; mirrors
    ``autocast.metrics.trajectory.TrajectoryMetricAccumulator.update``'s own
    windowing convention (clip ``end``, skip when ``start >= end``) rather
    than silently scoring an empty slice.
    """
    clipped = ((start, min(end, n_frames)) for start, end in WINDOWS)
    return [(start, end) for start, end in clipped if start < end]


def write_rollout_metrics_csv(
    out_dir: Path, fitted: FittedMethod, true: TensorBTSC
) -> None:
    """Write ``rollout_metrics.csv``: one row per window, paper format."""
    rows = []
    for window in _windows_within(true.shape[1]):
        row: dict[str, Any] = {
            "window": _paper_window_label(window),
            "batch_idx": _BATCH_IDX_ALL,
        }
        row.update(window_row(fitted, true, window))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "rollout_metrics.csv", index=False)


def write_rollout_coverage_window_csvs(
    out_dir: Path, fitted: FittedMethod, true: TensorBTSC
) -> None:
    """Write one ``rollout_coverage_window_<w>.csv`` per window."""
    for start, end in _windows_within(true.shape[1]):
        true_w = true[:, start:end]
        lower_w = fitted.lower[:, start:end]
        upper_w = fitted.upper[:, start:end]
        levels, observed_means, observed_channels = coverage_reliability_table(
            true_w, lower_w, upper_w, LEVELS
        )
        data: dict[str, Any] = {
            "coverage_level": levels,
            "observed_mean": observed_means,
        }
        for channel in range(observed_channels.shape[1]):
            data[f"channel_{channel}"] = observed_channels[:, channel].tolist()
        label = _paper_window_label((start, end))
        pd.DataFrame(data).to_csv(
            out_dir / f"rollout_coverage_window_{label}.csv", index=False
        )


def per_level_coverage_rows(observed: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame coverage at each of :data:`LEVELS`, averaged over channels.

    Named ``coverage_<level>`` as in the eval's own
    ``rollout_metrics_per_timestep_channel_all.csv``
    (`autocast.utils.plots.compute_metrics_per_timestep_from_dataloader`).

    Parameters
    ----------
    observed
        Observed coverage, shape ``(T, C, len(LEVELS))``.
    """
    per_level = observed.mean(axis=1)  # (T, len(LEVELS))
    return {f"coverage_{level}": per_level[:, i] for i, level in enumerate(LEVELS)}


def write_per_timestep_csv(
    out_dir: Path, fitted: FittedMethod, true: TensorBTSC
) -> None:
    """Write ``rollout_metrics_per_timestep_channel_all.csv`` (rows=metric, cols=frame).

    Every metric is computed for all ``T`` frames in one broadcast call (no
    Python loop over frames) via the ``per_frame_*`` family in
    :mod:`autocast.scripts.conformal.scoring`. ``coverage`` is the calibration
    error; the ``coverage_<level>`` rows after it are the coverage itself.
    Raw/EMOS (``fitted.samples is not None``) additionally get an ``exkurt``
    row.
    """
    level_index = LEVELS.index(NOMINAL_LEVEL)
    columns: dict[str, np.ndarray] = per_frame_member_metric_values(fitted, true)
    columns["winkler"] = per_lead_winkler(
        true,
        fitted.lower[..., level_index],
        fitted.upper[..., level_index],
        NOMINAL_ALPHA,
    )
    observed = per_frame_observed_coverage(true, fitted.lower, fitted.upper)
    columns["coverage"] = per_frame_coverage_calibration_error(observed, LEVELS)
    columns.update(per_level_coverage_rows(observed))
    if fitted.samples is not None:
        columns["exkurt"] = per_frame_excess_kurtosis(true, fitted.samples)

    frame_df = pd.DataFrame(columns).T
    frame_df.columns = pd.Index([str(frame) for frame in range(true.shape[1])])
    frame_df.to_csv(out_dir / "rollout_metrics_per_timestep_channel_all.csv")


def write_per_frame_ingredients_csv(
    out_dir: Path, fitted: FittedMethod, true: TensorBTSC
) -> None:
    """Write ``per_frame_ingredients.csv`` (additive components per frame)."""
    per_frame_ingredients(fitted, true).to_csv(
        out_dir / "per_frame_ingredients.csv", index=False
    )


def write_rank_histogram_csv(
    out_dir: Path, fitted: FittedMethod, true: TensorBTSC
) -> None:
    """Write ``rank_histogram.csv`` (raw/EMOS only; rows=frame, cols=rank)."""
    if fitted.samples is None:
        msg = "rank_histogram.csv requires ensemble members; conformal has none."
        raise ValueError(msg)
    stacked = rank_histogram_per_frame(true, fitted.samples)  # (T, M+1)
    columns = pd.Index([str(rank) for rank in range(stacked.shape[1])])
    pd.DataFrame(stacked, columns=columns).to_csv(
        out_dir / "rank_histogram.csv", index_label="frame"
    )


def write_method_outputs(
    method_dir: Path, fitted: FittedMethod, true: TensorBTSC
) -> None:
    """Write every per-method CSV (``raw/``, ``EMOS/``, or ``conformal/``)."""
    method_dir.mkdir(parents=True, exist_ok=True)
    write_rollout_metrics_csv(method_dir, fitted, true)
    write_rollout_coverage_window_csvs(method_dir, fitted, true)
    write_per_timestep_csv(method_dir, fitted, true)
    write_per_frame_ingredients_csv(method_dir, fitted, true)
    if fitted.samples is not None:
        write_rank_histogram_csv(method_dir, fitted, true)


def write_coverage_map(
    out_dir: Path, coverage_by_method: dict[str, torch.Tensor]
) -> None:
    """Write ``coverage_map.pt`` from precomputed per-method coverage slices.

    Takes already-reduced ``(H, W, ..., C)`` tensors (see
    :func:`autocast.scripts.conformal.scoring.coverage_map_slice`), not full
    :class:`FittedMethod` objects -- the caller computes and drops each
    method's much larger reliability-grid tensor before moving to the next
    method, so this function must not be handed one that's still alive.
    """
    torch.save(coverage_by_method, out_dir / "coverage_map.pt")


def write_bands(out_dir: Path, conformal: FittedMethod) -> None:
    """Write ``bands.pt``: conformal lower/upper bands at the nominal level."""
    level_index = LEVELS.index(NOMINAL_LEVEL)
    torch.save(
        {
            "alpha": NOMINAL_ALPHA,
            "lower": conformal.lower[..., level_index].float().cpu(),
            "upper": conformal.upper[..., level_index].float().cpu(),
        },
        out_dir / "bands.pt",
    )


def write_calibrator(
    out_dir: Path, conformal_multiplier: torch.Tensor, emos_state: dict[str, Any]
) -> None:
    """Write ``calibrator.pt``: conformal multipliers ``[T,H,W,C]`` + EMOS state.

    ``emos_state`` holds the fitted ``EMOS`` object's internal coefficients
    (there is no public accessor for them); see
    :func:`autocast.scripts.conformal.calibrate.emos_state_dict`. Every tensor
    value is moved to CPU here (whatever device the fit ran on) so
    ``calibrator.pt`` loads on any machine, matching every other ``.pt``
    writer in this module.
    """
    portable_emos_state = {
        key: value.float().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in emos_state.items()
    }
    torch.save(
        {
            "conformal_multiplier": conformal_multiplier.float().cpu(),
            "emos": portable_emos_state,
        },
        out_dir / "calibrator.pt",
    )


def write_sample_fields(
    out_dir: Path,
    *,
    truth: TensorBTSC,
    raw: torch.Tensor,
    emos_indep: torch.Tensor,
    emos_ecc: torch.Tensor,
    n_trajectories: int,
    leads: tuple[int, int],
) -> None:
    """Write ``sample_fields.pt``: first/last-frame snapshots, first N trajectories."""
    first_lead, last_lead = leads
    lead_index = torch.tensor([first_lead, last_lead])
    torch.save(
        {
            "leads": leads,
            "truth": truth[:n_trajectories][:, lead_index].float().cpu(),
            "raw": raw[:n_trajectories][:, lead_index].float().cpu(),
            "EMOS_indep": emos_indep[:n_trajectories][:, lead_index].float().cpu(),
            "EMOS_ECC": emos_ecc[:n_trajectories][:, lead_index].float().cpu(),
        },
        out_dir / "sample_fields.pt",
    )


def write_dependence_csv(
    out_dir: Path,
    channel_names: list[str],
    raw_ssr: torch.Tensor,
    indep_ssr: torch.Tensor,
    ecc_ssr: torch.Tensor,
) -> None:
    """Write ``dependence.csv``: per-channel spatial-mean SSR, raw / indep / ECC."""
    rows = [
        {
            "channel": name,
            "raw": float(raw_ssr[i]),
            "indep": float(indep_ssr[i]),
            "ecc": float(ecc_ssr[i]),
        }
        for i, name in enumerate(channel_names)
    ]
    pd.DataFrame(rows).to_csv(out_dir / "dependence.csv", index=False)


def write_summary_csv(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``summary.csv``: one row per method, headline scores + bootstrap std."""
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    """Write the top-level ``manifest.json``."""
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


#: One line per ``calib-X__test-Y/`` folder, in the order `calibrate` writes them.
_COMBINATION_TEXT = {
    "calib-new__test-new": (
        "calibrated on the new set's {new_cal} calibration trajectories, tested "
        "on its {new_test} test trajectories"
    ),
    "calib-new__test-paper": (
        "calibrated on the new set's {new_cal} calibration trajectories, tested "
        "on the paper's {paper_test} test trajectories"
    ),
    "calib-paper-valid__test-new": (
        "calibrated on the paper's {paper_valid} validation trajectories, "
        "tested on the new set's {new_test} test trajectories"
    ),
    "calib-paper-valid__test-paper": (
        "calibrated on the paper's {paper_valid} validation trajectories, "
        "tested on the paper's {paper_test} test trajectories"
    ),
}


def write_readme(out_dir: Path, manifest: dict[str, Any]) -> None:
    """Write ``README.md``: a plain-language guide to the folder, from the manifest.

    Each entry of the manifest's optional ``amendments`` list (``date``,
    ``git_commit``, ``change``) becomes a paragraph under the header, for
    files changed after the run that produced the folder.
    """
    inputs, split = manifest["inputs"], manifest["split"]
    counts = {
        "new": inputs["new"]["n_trajectories"],
        "new_cal": len(split["new_calibration_idx"]),
        "new_test": len(split["new_test_idx"]),
        "paper_valid": inputs["paper_valid"]["n_trajectories"],
        "paper_test": inputs["paper_test"]["n_trajectories"],
    }
    split_rule = (
        "balanced across the parameter settings (`constant_scalars`)"
        if split["balanced_by_scalars"]
        else "at random"
    )
    level = round(100 * (1 - manifest["alpha"]))
    windows = ", ".join(_paper_window_label(tuple(w)) for w in manifest["windows"])
    in_a_window = {f for start, end in manifest["windows"] for f in range(start, end)}
    last_end = max(end for _, end in manifest["windows"])
    outside = ", ".join(str(f) for f in range(last_end + 1) if f not in in_a_window)
    table_rows = "\n".join(
        [
            f"| `new/` | {counts['new']} | New simulated trajectories, split once "
            f"({split_rule}, seed {split['seed']}) into {counts['new_cal']} for "
            f"calibration and {counts['new_test']} for testing |",
            f"| `paper_valid/` | {counts['paper_valid']} | The paper dataset's "
            "validation split: calibration |",
            f"| `paper_test/` | {counts['paper_test']} | The paper dataset's test "
            "split: testing |",
        ]
    )
    combinations = "\n".join(
        f"- `{name}/`: {text.format(**counts)}"
        for name, text in _COMBINATION_TEXT.items()
    )
    amendments = "".join(
        f"\n\nAmended on {entry['date']} at commit `{entry['git_commit']}`: "
        f"{entry['change']}"
        for entry in manifest.get("amendments", [])
    )
    text = f"""# Calibrated forecasts: {out_dir.resolve().parent.name}

Uncertainty calibration of this run's saved ensemble forecasts, made with
`autocast.scripts.conformal` at commit `{manifest["git_commit"]}` (autouq
{manifest["autouq_version"]}) on {manifest["generated_at_utc"][:10]}.
`manifest.json` records the same information in machine-readable form, including
the exact trajectory indices of every split.{amendments}

## Forecasts (`predictions/`)

| Folder | Trajectories | Used for |
|---|---|---|
{table_rows}

`rollout_tensors.pt` holds the forecasts (`preds`, shape [trajectory, frame,
height, width, channel, member]) and the true fields (`trues`); the
`resolved_eval_config.yaml` next to it records the checkpoint and data that
produced them.

## Results: one folder per calibration source and test source

{combinations}

Each holds three methods:

- `raw/`: the model's own ensemble, uncalibrated.
- `EMOS/`: a Gaussian forecast at every pixel, whose mean and variance are
  affine maps of the ensemble's mean and variance, with coefficients fitted
  separately for each frame on the calibration trajectories.
- `conformal/`: conformal bands scaled by the ensemble spread, fitted
  separately for every frame, pixel and channel. There are no forecast members,
  so no CRPS or rank histogram.

In each method folder: `rollout_metrics.csv` (scores per forecast window, in the
paper's format), `rollout_coverage_window_<window>.csv` (coverage at each
nominal level from 5% to 95%), `rollout_metrics_per_timestep_channel_all.csv`
(the same scores frame by frame, plus rows `coverage_0.05` to `coverage_0.95`:
the coverage at each nominal level and frame, averaged over channels, named as in
the eval's own file), `per_frame_ingredients.csv` (per-frame sums
from which any window's scores can be rebuilt exactly) and, for `raw/` and
`EMOS/`, `rank_histogram.csv` (one row per frame). As in the eval's own
`rollout_metrics.csv`, the `coverage` column of the first file, and the `coverage`
row of the third, is the coverage error, not the coverage itself: the gap between
observed and nominal coverage, taken at each frame, level and channel and then
averaged over the window's frames, the 19 levels and the channels. The {level}%
coverage is also in `summary.csv` and `per_frame_ingredients.csv`.

Next to them: `summary.csv` (headline scores at the {level}% level over the
whole forecast, with bootstrap standard deviations over test trajectories),
`coverage_map.pt` ({level}% coverage per pixel and channel, averaged over test
trajectories and all frames), `bands.pt` (the conformal {level}% bands),
`calibrator.pt` (the fitted conformal multipliers and EMOS coefficients),
`sample_fields.pt` (truth, raw, EMOS and EMOS+ECC members for the first few
test trajectories at the first and last frame) and `dependence.csv` (spread
over error of each channel's whole-field average, for raw, EMOS, and EMOS with
ensemble copula coupling).

Forecast windows: {windows}. A window `a-b` covers frames a to b-1, sliced as in
the paper's own CSVs, so a few frames fall in no window ({outside}); the
per-frame files cover every frame, and `per_frame_ingredients.csv` rebuilds any
range exactly.

## Data sufficiency (`data_sufficiency/`)

How coverage and band width change with the number of calibration trajectories:
random subsets of the new set's calibration trajectories, each tested on its
{counts["new_test"]} test trajectories. `sufficiency.json` holds every draw,
`sufficiency.csv` the summary per calibration-set size and
`sufficiency_per_frame.csv` the same per frame.
"""
    (out_dir / "README.md").write_text(text)
