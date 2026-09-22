r"""Calibrate saved ensemble forecasts with EMOS and conformal prediction.

For one model's three eval-dump prediction files (the 150-trajectory new set,
the paper's validation split, and the paper's test split), computes all four
calibration-source x test-source combinations and writes::

    <out>/
      README.md                plain-language guide to the folder
      manifest.json            inputs (relative paths, md5), split indices, seeds,
                               versions
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
from collections.abc import Callable, Sequence
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
from autocast.types import Tensor, TensorBTSC, TensorBTSCM

_LEVEL_INDEX = LEVELS.index(NOMINAL_LEVEL)

#: Bootstrap draws for `summary.csv` error bars (`calibrate_full_cell.py`'s
#: default).
N_BOOTSTRAP = 200

#: First/last-test-trajectory count kept in `sample_fields.pt`.
N_SAMPLE_FIELD_TRAJECTORIES = 5

_ALPHAS_FOR_LEVELS = [round(1.0 - level, 2) for level in LEVELS]

#: How many of :data:`LEVELS` a single `.predict()` call requests at once
#: (see :func:`_predict_intervals_chunked`). Both `EMOS.predict` and
#: `Ensemble.predict` build every requested level's interval as several
#: simultaneous full ``(B, T, H, W, C, n_alphas)`` tensors before returning
#: (broadcast intermediates plus the final `torch.stack`); measured
#: empirically (this package's memory investigation) that requesting all 19
#: levels in one call, as `EMOS.predict` does, peaks at ~4.45 GB on a
#: `(10, 100, 64, 64, 3)`-shaped test set, chunks of 4 down to ~2.3 GB, for
#: bit-identical results (each level's interval is a pure per-level function
#: of the already-fitted calibrator, independent of which other levels are
#: requested in the same call).
_PREDICT_LEVEL_CHUNK_SIZE = 4


def _predict_intervals_chunked(
    predict: Callable[[Tensor, Sequence[float]], Tensor],
    pred_test: Tensor,
    alphas: Sequence[float],
    *,
    chunk_size: int = _PREDICT_LEVEL_CHUNK_SIZE,
) -> tuple[Tensor, Tensor]:
    """Call a calibrator's ``predict(pred_test, alphas=...)`` in level chunks.

    Writes directly into preallocated, full-size ``lower``/``upper`` tensors
    (rather than collecting per-chunk pieces and concatenating them at the
    end, which would hold close to the full size twice at the last step) --
    the same "allocate once, fill by block" shape as
    :func:`autocast.scripts.conformal.scoring.per_frame_ingredients`'s frame
    blocking.
    """
    n_levels = len(alphas)
    out_shape = (*pred_test.shape[:-1], n_levels)
    lower = torch.empty(out_shape, dtype=pred_test.dtype, device=pred_test.device)
    upper = torch.empty(out_shape, dtype=pred_test.dtype, device=pred_test.device)
    for start in range(0, n_levels, chunk_size):
        end = min(start + chunk_size, n_levels)
        intervals = predict(pred_test, alphas[start:end])
        lower[..., start:end] = intervals[..., 0, :]
        upper[..., start:end] = intervals[..., 1, :]
    return lower, upper


def _channel_names(n_channels: int) -> list[str]:
    return [f"channel_{index}" for index in range(n_channels)]


def fit_raw(pred_test: Tensor) -> FittedMethod:
    """Empirical ensemble-quantile intervals; the raw ensemble is its own sample."""
    lower, upper = raw_interval_multi(pred_test, LEVELS)
    return FittedMethod(Method.RAW, lower, upper, samples=pred_test)


#: Fitted-coefficient attributes of `autouq`'s `EMOS`, one entry per frame
#: under ``per=TIME`` (see :func:`emos_state_dict` on why they are private).
_EMOS_FITTED_ATTRS = (
    "_beta0",
    "_beta1",
    "_raw_gamma0",
    "_raw_gamma1",
    "_loc",
    "_scale",
)


def fit_emos_per_frame(true_cal: TensorBTSC, pred_cal: TensorBTSCM) -> EMOS:
    """Fit `EMOS(per=TIME)` one frame at a time, returned as a single calibrator.

    Under ``per=TIME`` every frame has its own coefficients and its own
    standardization, so the fitting loss separates into one independent
    problem per frame. `EMOS.calibrate` still solves all frames in one L-BFGS
    run with a single shared line search, and that run stops before most
    frames reach their optimum. Measured on the flow-matching Navier--Stokes
    model: 90% test coverage 0.746 from the joint fit vs 0.852 fitted frame by
    frame, with a lower calibration CRPS for the per-frame fit, and where the
    joint fit stops differs between CPU and CUDA (0.848 vs 0.764 on the same
    forecasts). Fitting each frame separately reaches every frame's optimum,
    which is the optimum of the same model.

    The loop over frames is deliberate: L-BFGS's line search cannot be
    vectorized over independent problems, and one frame fits in well under a
    second. Each frame is fit in float64 and the coefficients are cast back
    to the input dtype, so everything downstream stays in the input
    precision. In float32 some frames' fits stall close to their starting
    point: on the crps_cns64 forecasts 4 of 100 frames ended with a 0.1-2.8%
    higher calibration CRPS than the float64 fit, and on
    ``test_gpu.py``'s synthetic data 2 of 8 frames never moved their
    ``gamma0`` from its initial value, on CPU and CUDA alike.

    Parameters
    ----------
    true_cal
        Calibration truth, ``[B, T, H, W, C]``.
    pred_cal
        Calibration ensemble forecasts, ``[B, T, H, W, C, M]``.

    Returns
    -------
    EMOS
        A calibrator whose per-frame coefficients are the individual fits.
    """
    true_cal, pred_cal = true_cal.detach(), pred_cal.detach()
    frame_fits = []
    for frame in range(true_cal.shape[1]):
        frame_fit = EMOS(per=(AxisRole.TIME,))
        frame_fit.calibrate(
            true_cal[:, frame : frame + 1].double(),
            pred_cal[:, frame : frame + 1].double(),
        )
        frame_fits.append(frame_fit)
    combined = EMOS(per=(AxisRole.TIME,))
    for name in _EMOS_FITTED_ATTRS:
        coefficients = torch.cat([getattr(fit, name) for fit in frame_fits])
        setattr(combined, name, coefficients.to(pred_cal.dtype))
    return combined


def fit_emos(
    true_cal: Tensor,
    pred_cal: Tensor,
    pred_test: Tensor,
    *,
    sample_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[FittedMethod, EMOS]:
    """Fit `EMOS(per=TIME)` and evaluate it on the test set at every level.

    The fit is :func:`fit_emos_per_frame`. ``sample_seed`` makes ``.sample()``
    (and therefore every diagnostic derived from it -- CRPS/SSR/spread/skill
    in ``rollout_metrics.csv``, rank histograms, excess kurtosis,
    ``sample_fields.pt``) reproducible across runs; ``.predict()``-based
    coverage/Winkler are already exact (no sampling), so only this draw needed
    seeding.
    """
    emos = fit_emos_per_frame(true_cal, pred_cal)
    lower, upper = _predict_intervals_chunked(
        emos.predict, pred_test, _ALPHAS_FOR_LEVELS
    )
    generator = seeded_generator(sample_seed, pred_test.device)
    samples = emos.sample(pred_test, n_members=pred_test.shape[-1], generator=generator)
    fitted = FittedMethod(Method.EMOS, lower, upper, samples=samples)
    return fitted, emos


def fit_conformal(
    true_cal: Tensor, pred_cal: Tensor, pred_test: Tensor
) -> tuple[FittedMethod, Ensemble]:
    """Fit the per-pixel/frame/channel conformal `Ensemble(mode="std")` calibrator."""
    ensemble = Ensemble(mode="std")
    ensemble.calibrate(true_cal, pred_cal)
    lower, upper = _predict_intervals_chunked(
        ensemble.predict, pred_test, _ALPHAS_FOR_LEVELS
    )
    fitted = FittedMethod(Method.CONFORMAL, lower, upper, samples=None)
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
    conformal.data.default_device`). The three loaded dumps themselves stay
    on CPU for this function's whole lifetime; only each combination's four
    selected tensors (``true_cal``/``pred_cal``/``true_test``/``pred_test``)
    are moved to ``device``, right before that combination's
    :func:`run_combination` call. Measured directly (this package's memory
    investigation): holding all three dumps on CUDA for all four
    combinations, as an earlier version of this function did, costs ~9.6 GB
    of permanently-resident device memory at the real 150/20/20 x T=100 x
    64x64 x C=3 x M=10 shape -- for data most combinations don't touch
    (``paper_valid``/``paper_test`` combined are only ~2.7 GB of that, so the
    bulk is the 150-trajectory ``new`` dump sitting on-device even during the
    two ``calib-paper-valid__test-paper`` .. combinations that never read
    it). Every fit and score in :func:`run_combination` then runs on
    ``device`` (the ``autouq`` calibrators are plain torch, so this needs no
    calibrator-side changes). Also sets
    ``torch.set_float32_matmul_precision("high")`` -- measured neutral-to-
    positive for this workload on this machine, per the
    ``gpu-job-settings`` guidance; no bf16, no ``torch.compile`` (both
    measured neutral or negative here).
    """
    resolved_device = device if device is not None else default_device()
    torch.set_float32_matmul_precision("high")
    new_dump = load_prediction_dump(new_path)
    paper_valid_dump = load_prediction_dump(paper_valid_path)
    paper_test_dump = load_prediction_dump(paper_test_path)

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
        relative_to=out_dir,
    )
    writers.write_manifest(out_dir, manifest)
    writers.write_readme(out_dir, manifest)

    for calibration_source in CalibrationSource:
        true_cal_cpu, pred_cal_cpu = _select_calibration(
            calibration_source,
            new_dump=new_dump,
            paper_valid_dump=paper_valid_dump,
            new_split=new_split,
        )
        true_cal = true_cal_cpu.to(resolved_device)
        pred_cal = pred_cal_cpu.to(resolved_device)
        for test_source in TestSource:
            true_test_cpu, pred_test_cpu = _select_test(
                test_source,
                new_dump=new_dump,
                paper_test_dump=paper_test_dump,
                new_split=new_split,
            )
            true_test = true_test_cpu.to(resolved_device)
            pred_test = pred_test_cpu.to(resolved_device)
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
