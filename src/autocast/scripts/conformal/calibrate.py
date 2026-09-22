r"""Calibrate saved ensemble forecasts with EMOS and conformal prediction.

For one model's three eval-dump prediction files (the 150-trajectory new set,
the paper's validation split, and the paper's test split), computes all four
calibration-source x test-source combinations and writes::

    <out>/
      manifest.json            inputs (paths, md5), split indices, seeds, versions
      calib-{new,paper-valid}__test-{new,paper}/
        raw/ EMOS/ conformal/  rollout_metrics.csv,
                               rollout_coverage_window_<window>.csv,
                               rollout_metrics_per_timestep_channel_all.csv
                               (the eval's own formats), per_frame_ingredients.csv
        raw/ EMOS/             also rank_histogram.csv
        coverage_map.pt  bands.pt  calibrator.pt  sample_fields.pt
        dependence.csv  summary.csv

Usage
-----
.. code-block:: bash

    python -m autocast.scripts.conformal.calibrate \\
        --new new.pt --paper-valid paper_valid.pt --paper-test paper_test.pt \\
        --out eval_conformal/ [--balance-by-scalars] [--split-seed 20260709] \\
        [--threads N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from autouq.calibrators import ECC, EMOS, AxisRole, ComposedCalibrator
from autouq.calibrators.conformal import Ensemble

from autocast.scripts.conformal import writers
from autocast.scripts.conformal.data import (
    DEFAULT_BALANCED_CALIBRATION_PER_GROUP,
    DEFAULT_BALANCED_TEST_PER_GROUP,
    DEFAULT_SPLIT_SEED,
    CalibrationSource,
    PredictionDump,
    SplitIndices,
    TestSource,
    balanced_split_by_scalars,
    build_manifest,
    default_device,
    fixed_split,
    load_prediction_dump,
)
from autocast.scripts.conformal.scoring import (
    LEVELS,
    NOMINAL_ALPHA,
    NOMINAL_LEVEL,
    WINDOWS,
    FittedMethod,
    Method,
    bootstrap_summary,
    coverage_map_slice,
    raw_interval_multi,
    seeded_generator,
    spatial_mean_spread_skill,
)
from autocast.types import Tensor

_LEVEL_INDEX = LEVELS.index(NOMINAL_LEVEL)

#: Bootstrap draws for `summary.csv` error bars (`calibrate_full_cell.py`'s
#: default).
N_BOOTSTRAP = 200

#: First/last-test-trajectory count kept in `sample_fields.pt`.
N_SAMPLE_FIELD_TRAJECTORIES = 5

_ALPHAS_FOR_LEVELS = [round(1.0 - level, 2) for level in LEVELS]


def _channel_names(n_channels: int) -> list[str]:
    return [f"channel_{index}" for index in range(n_channels)]


def fit_raw(pred_test: Tensor) -> FittedMethod:
    """Empirical ensemble-quantile intervals; the raw ensemble is its own sample."""
    lower, upper = raw_interval_multi(pred_test, LEVELS)
    return FittedMethod(Method.RAW, lower, upper, samples=pred_test)


def fit_emos(
    true_cal: Tensor,
    pred_cal: Tensor,
    pred_test: Tensor,
    *,
    sample_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[FittedMethod, EMOS]:
    """Fit `EMOS(per=TIME)` and evaluate it on the test set at every level.

    ``sample_seed`` makes ``.sample()`` (and therefore every diagnostic
    derived from it -- CRPS/SSR/spread/skill in ``rollout_metrics.csv``,
    rank histograms, excess kurtosis, ``sample_fields.pt``) reproducible
    across runs; ``.predict()``-based coverage/Winkler are already exact
    (no sampling), so only this draw needed seeding.
    """
    emos = EMOS(per=(AxisRole.TIME,))
    emos.calibrate(true_cal.detach(), pred_cal.detach())
    intervals = emos.predict(pred_test, alphas=_ALPHAS_FOR_LEVELS)
    generator = seeded_generator(sample_seed, pred_test.device)
    samples = emos.sample(pred_test, n_members=pred_test.shape[-1], generator=generator)
    fitted = FittedMethod(
        Method.EMOS, intervals[..., 0, :], intervals[..., 1, :], samples=samples
    )
    return fitted, emos


def fit_conformal(
    true_cal: Tensor, pred_cal: Tensor, pred_test: Tensor
) -> tuple[FittedMethod, Ensemble]:
    """Fit the per-pixel/frame/channel conformal `Ensemble(mode="std")` calibrator."""
    ensemble = Ensemble(mode="std")
    ensemble.calibrate(true_cal, pred_cal)
    intervals = ensemble.predict(pred_test, alphas=_ALPHAS_FOR_LEVELS)
    fitted = FittedMethod(
        Method.CONFORMAL, intervals[..., 0, :], intervals[..., 1, :], samples=None
    )
    return fitted, ensemble


def _select_calibration(
    source: CalibrationSource,
    *,
    new_dump: PredictionDump,
    paper_valid_dump: PredictionDump,
    new_split: SplitIndices,
) -> tuple[Tensor, Tensor]:
    if source is CalibrationSource.NEW:
        idx = new_split.calibration
        return new_dump.trues[idx], new_dump.preds[idx]
    return paper_valid_dump.trues, paper_valid_dump.preds


def _select_test(
    source: TestSource,
    *,
    new_dump: PredictionDump,
    paper_test_dump: PredictionDump,
    new_split: SplitIndices,
) -> tuple[Tensor, Tensor]:
    if source is TestSource.NEW:
        idx = new_split.test
        return new_dump.trues[idx], new_dump.preds[idx]
    return paper_test_dump.trues, paper_test_dump.preds


def run_combination(
    out_dir: Path,
    *,
    true_cal: Tensor,
    pred_cal: Tensor,
    true_test: Tensor,
    pred_test: Tensor,
    sample_seed: int = DEFAULT_SPLIT_SEED,
) -> None:
    """Fit raw/EMOS/conformal and write one ``calib-X__test-Y/`` subtree.

    ``sample_seed`` seeds every stochastic ``.sample()`` draw (EMOS's own
    and the EMOS+ECC composition's) so a rerun with the same inputs and seed
    reproduces every diagnostic exactly, not just the calibrated intervals.

    Fits and writes each method's outputs in turn, discarding that method's
    ``(B, T, H, W, C, len(LEVELS))`` interval tensor (via ``del``) before
    moving to the next -- the dominant peak-memory cost, since it is ~19x a
    single-level interval. Measured empirically at H=W=8, C=3 (scaled to the
    real H=W=64 Navier-Stokes case):
    holding all three methods' interval tensors at once would land close to
    90 GB; processing one at a time keeps only the current method's tensor
    live alongside the small state each method needs to keep afterward
    (``emos``'s samples for the ECC/dependence step, the tiny fitted
    ``EMOS``/conformal coefficients for ``calibrator.pt``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_by_method: dict[str, Tensor] = {}
    summary_rows: list[dict[str, float | str]] = []

    raw = fit_raw(pred_test)
    writers.write_method_outputs(out_dir / Method.RAW.value, raw, true_test)
    coverage_by_method[Method.RAW.value] = coverage_map_slice(
        true_test, raw.lower[..., _LEVEL_INDEX], raw.upper[..., _LEVEL_INDEX]
    )
    summary_rows.append(
        {
            "method": Method.RAW.value,
            **bootstrap_summary(
                raw, true_test, n_boot=N_BOOTSTRAP, seed=DEFAULT_SPLIT_SEED
            ),
        }
    )
    del raw  # frees its (..., len(LEVELS)) bands; `pred_test` itself lives on

    emos_fitted, emos = fit_emos(true_cal, pred_cal, pred_test, sample_seed=sample_seed)
    writers.write_method_outputs(out_dir / Method.EMOS.value, emos_fitted, true_test)
    coverage_by_method[Method.EMOS.value] = coverage_map_slice(
        true_test,
        emos_fitted.lower[..., _LEVEL_INDEX],
        emos_fitted.upper[..., _LEVEL_INDEX],
    )
    summary_rows.append(
        {
            "method": Method.EMOS.value,
            **bootstrap_summary(
                emos_fitted, true_test, n_boot=N_BOOTSTRAP, seed=DEFAULT_SPLIT_SEED + 1
            ),
        }
    )
    if emos_fitted.samples is None:
        msg = "EMOS fit must produce ensemble samples."
        raise RuntimeError(msg)
    emos_samples = emos_fitted.samples  # kept alive past the `del` below
    del emos_fitted  # frees its (..., len(LEVELS)) bands, not `emos_samples`

    conformal_fitted, ensemble = fit_conformal(true_cal, pred_cal, pred_test)
    writers.write_method_outputs(
        out_dir / Method.CONFORMAL.value, conformal_fitted, true_test
    )
    coverage_by_method[Method.CONFORMAL.value] = coverage_map_slice(
        true_test,
        conformal_fitted.lower[..., _LEVEL_INDEX],
        conformal_fitted.upper[..., _LEVEL_INDEX],
    )
    writers.write_bands(out_dir, conformal_fitted)
    conformal_multiplier = ensemble.score_quantile(NOMINAL_ALPHA)
    summary_rows.append(
        {
            "method": Method.CONFORMAL.value,
            **bootstrap_summary(
                conformal_fitted,
                true_test,
                n_boot=N_BOOTSTRAP,
                seed=DEFAULT_SPLIT_SEED + 2,
            ),
        }
    )
    del conformal_fitted, ensemble  # frees its bands and calibration scores

    writers.write_coverage_map(out_dir, coverage_by_method)
    writers.write_calibrator(out_dir, conformal_multiplier, emos_state_dict(emos))

    # EMOS+ECC, reusing the already-fitted EMOS marginal (no refit) -- shared
    # by `dependence.csv` and `sample_fields.pt` (D10, D11).
    composed = ComposedCalibrator(emos, ECC())
    ecc_generator = seeded_generator(sample_seed + 1, pred_test.device)
    ecc_samples = composed.sample(
        pred_test, n_members=pred_test.shape[-1], generator=ecc_generator
    )

    n_channels = true_test.shape[-1]
    writers.write_dependence_csv(
        out_dir,
        _channel_names(n_channels),
        raw_ssr=spatial_mean_spread_skill(pred_test, true_test),
        indep_ssr=spatial_mean_spread_skill(emos_samples, true_test),
        ecc_ssr=spatial_mean_spread_skill(ecc_samples, true_test),
    )
    writers.write_sample_fields(
        out_dir,
        truth=true_test,
        raw=pred_test,
        emos_indep=emos_samples,
        emos_ecc=ecc_samples,
        n_trajectories=min(N_SAMPLE_FIELD_TRAJECTORIES, true_test.shape[0]),
        leads=(0, true_test.shape[1] - 1),
    )

    writers.write_summary_csv(out_dir, summary_rows)


def emos_state_dict(emos: EMOS) -> dict[str, Tensor | tuple[str, ...]]:
    """Serialize an ``EMOS``'s fitted coefficients (no public accessor exists)."""
    # `autouq.calibrators.emos.EMOS` has no public accessor for its fitted
    # coefficients, so this reaches into its internal state directly (the
    # library documents these as the fitted parameters in its own docstring).
    beta0, beta1 = emos._beta0, emos._beta1
    raw_gamma0, raw_gamma1 = emos._raw_gamma0, emos._raw_gamma1
    loc, scale = emos._loc, emos._scale
    if (
        beta0 is None
        or beta1 is None
        or raw_gamma0 is None
        or raw_gamma1 is None
        or loc is None
        or scale is None
    ):
        msg = "EMOS.calibrate must be called before serializing its coefficients."
        raise RuntimeError(msg)
    return {
        "per": tuple(role.value for role in emos.per),
        "beta0": beta0,
        "beta1": beta1,
        "raw_gamma0": raw_gamma0,
        "raw_gamma1": raw_gamma1,
        "loc": loc,
        "scale": scale,
    }


def calibrate(
    *,
    new_path: Path,
    paper_valid_path: Path,
    paper_test_path: Path,
    out_dir: Path,
    balance_by_scalars: bool = False,
    split_seed: int = DEFAULT_SPLIT_SEED,
    device: str | torch.device | None = None,
) -> None:
    """Run all four calibration-source x test-source combinations.

    ``device`` defaults to CUDA if available (:func:`autocast.scripts.
    conformal.data.default_device`). Every input tensor is moved to it once,
    right after loading; every fit and score in :func:`run_combination` then
    runs on that device (the ``autouq`` calibrators are plain torch, so this
    needs no calibrator-side changes). Also sets
    ``torch.set_float32_matmul_precision("high")`` -- measured neutral-to-
    positive for this workload on this machine, per the
    ``gpu-job-settings`` guidance; no bf16, no ``torch.compile`` (both
    measured neutral or negative here).
    """
    resolved_device = device if device is not None else default_device()
    torch.set_float32_matmul_precision("high")
    new_dump = load_prediction_dump(new_path).to(resolved_device)
    paper_valid_dump = load_prediction_dump(paper_valid_path).to(resolved_device)
    paper_test_dump = load_prediction_dump(paper_test_path).to(resolved_device)

    if balance_by_scalars:
        if new_dump.constant_scalars is None:
            msg = "--balance-by-scalars requires constant_scalars in the new-set dump."
            raise ValueError(msg)
        new_split = balanced_split_by_scalars(
            new_dump.constant_scalars,
            n_calibration_per_group=DEFAULT_BALANCED_CALIBRATION_PER_GROUP,
            n_test_per_group=DEFAULT_BALANCED_TEST_PER_GROUP,
            seed=split_seed,
        )
    else:
        new_split = fixed_split(new_dump.n_trajectories, seed=split_seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        new_dump=new_dump,
        paper_valid_dump=paper_valid_dump,
        paper_test_dump=paper_test_dump,
        new_split=new_split,
        balanced_by_scalars=balance_by_scalars,
        split_seed=split_seed,
        alpha=NOMINAL_ALPHA,
        levels=list(LEVELS),
        windows=list(WINDOWS),
    )
    writers.write_manifest(out_dir, manifest)

    for calibration_source in CalibrationSource:
        true_cal, pred_cal = _select_calibration(
            calibration_source,
            new_dump=new_dump,
            paper_valid_dump=paper_valid_dump,
            new_split=new_split,
        )
        for test_source in TestSource:
            true_test, pred_test = _select_test(
                test_source,
                new_dump=new_dump,
                paper_test_dump=paper_test_dump,
                new_split=new_split,
            )
            combo_name = f"calib-{calibration_source.value}__test-{test_source.value}"
            combo_dir = out_dir / combo_name
            run_combination(
                combo_dir,
                true_cal=true_cal,
                pred_cal=pred_cal,
                true_test=true_test,
                pred_test=pred_test,
                sample_seed=split_seed,
            )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--new", required=True, type=Path, help="new-set prediction .pt"
    )
    parser.add_argument(
        "--paper-valid",
        required=True,
        type=Path,
        help="paper validation-split prediction .pt",
    )
    parser.add_argument(
        "--paper-test",
        required=True,
        type=Path,
        help="paper test-split prediction .pt",
    )
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--balance-by-scalars",
        action="store_true",
        help="stratify the new-set split by distinct constant_scalars rows",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
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

    calibrate(
        new_path=args.new,
        paper_valid_path=args.paper_valid,
        paper_test_path=args.paper_test,
        out_dir=args.out,
        device=args.device,
        balance_by_scalars=args.balance_by_scalars,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
