"""CPU/CUDA parity and a realistic-size GPU memory benchmark for this package.

Skips when no CUDA device is present, following this repo's convention (see
``tests/metrics/test_deterministic.py::test_isotropic_binning_device_consistency``).
"""

from __future__ import annotations

import time

import pandas as pd
import pytest
import torch

from autocast.scripts.conformal import calibrate, sufficiency

from .conftest import make_synthetic_dump, save_dump

_HAS_CUDA = torch.cuda.is_available()
_SKIP_REASON = "No CUDA device available for the CPU/CUDA parity test"

#: Peak-memory ceiling for :func:`test_calibrate_and_sufficiency_benchmark`
#: (this package's memory contract: well under this machine's shared
#: CPU/GPU pool at the largest real input shape -- see that test's
#: docstring).
_PEAK_MEMORY_CEILING_GB = 40.0


@pytest.mark.skipif(not _HAS_CUDA, reason=_SKIP_REASON)
def test_cpu_cuda_parity_run_combination(tmp_path):
    """Same inputs/seed on CPU vs CUDA -> same numbers within float32 tolerance.

    Deliberately does NOT compare anything derived from ``.sample()`` (EMOS's
    own draw, and the EMOS+ECC composition's): a CPU ``torch.Generator`` and a
    CUDA one use different underlying RNG algorithms, so the *same seed*
    produces a *different* random sequence per device -- this is documented
    PyTorch behavior, not a bug, and comparing those columns would just be
    comparing two independent random draws (verified empirically: EMOS's
    ``exkurt_last_frame`` differed by ~1.3 between devices with matching
    seeds, while every ``.predict()``-based deterministic value below did
    not). What's compared is exactly the deterministic compute path: `raw`
    (closed-form quantiles) and `conformal` (a closed-form kthvalue
    threshold) in full, and `EMOS`'s ``winkler``/``coverage`` (from
    ``.predict()``, not ``.sample()``) plus its fitted intervals
    (``bands.pt``/``coverage_map.pt``, both ``.predict()``-derived).
    """
    calibration = make_synthetic_dump(
        b_total=25, n_frames=8, height=16, width=16, n_channels=2, n_members=6, seed=60
    )
    test = make_synthetic_dump(
        b_total=12, n_frames=8, height=16, width=16, n_channels=2, n_members=6, seed=61
    )

    def run(device: str, out_dir):
        torch.set_float32_matmul_precision("high")
        calibrate.run_combination(
            out_dir,
            true_cal=calibration["trues"].to(device),
            pred_cal=calibration["preds"].to(device),
            true_test=test["trues"].to(device),
            pred_test=test["preds"].to(device),
            sample_seed=123,
        )

    cpu_dir, cuda_dir = tmp_path / "cpu", tmp_path / "cuda"
    run("cpu", cpu_dir)
    run("cuda", cuda_dir)

    # A loose-but-bounded tolerance even for the deterministic path:
    # `torch.quantile`'s CPU/CUDA implementations can pick different
    # interpolated values on ties with only `n_members=6`, flipping a
    # handful of borderline points' coverage -- a tiny absolute shift that
    # reads as a large relative one against the near-zero "coverage"
    # calibration-error column. Still tight enough to catch a real bug
    # (wrong device, wrong dtype, an outright different interval), which
    # would diverge by orders of magnitude, not ~1%.
    def assert_close(cpu_df: pd.DataFrame, cuda_df: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            cpu_df, cuda_df, check_exact=False, rtol=0.1, atol=5e-3
        )

    for method_dir in ("raw", "conformal"):
        cpu_df = pd.read_csv(cpu_dir / method_dir / "rollout_metrics.csv")
        cuda_df = pd.read_csv(cuda_dir / method_dir / "rollout_metrics.csv")
        columns = list(cpu_df.select_dtypes(include="number").columns)
        assert_close(cpu_df.loc[:, columns], cuda_df.loc[:, columns])

    deterministic_columns = ["winkler", "coverage"]
    cpu_emos = pd.read_csv(cpu_dir / "EMOS" / "rollout_metrics.csv")
    cuda_emos = pd.read_csv(cuda_dir / "EMOS" / "rollout_metrics.csv")
    assert_close(
        cpu_emos.loc[:, deterministic_columns], cuda_emos.loc[:, deterministic_columns]
    )

    cpu_coverage_map = torch.load(cpu_dir / "coverage_map.pt", weights_only=False)
    cuda_coverage_map = torch.load(cuda_dir / "coverage_map.pt", weights_only=False)
    for name in ("raw", "EMOS", "conformal"):
        torch.testing.assert_close(
            cpu_coverage_map[name],
            cuda_coverage_map[name].cpu(),
            rtol=0.1,
            atol=5e-3,
        )


@pytest.mark.skipif(not _HAS_CUDA, reason=_SKIP_REASON)
def test_calibrate_and_sufficiency_benchmark(tmp_path):
    """Peak CUDA memory + wall time at the largest real input shape.

    150/20/20 trajectories, T=100, H=W=64, C=3, M=10 -- the worst-case real
    shape (cns64: three channels, the paper's full trajectory counts), not
    the smaller C=2 shape this benchmark originally shipped with. Deselect
    this test by name for a fast run (it's slow by design, ~3 minutes on
    this machine): ``pytest tests/scripts/conformal --deselect
    tests/scripts/conformal/test_gpu.py::test_calibrate_and_sufficiency_benchmark``.

    Asserts peak CUDA memory stays under :data:`_PEAK_MEMORY_CEILING_GB` --
    a regression guard, not just a report: this package's original
    ``calibrate()`` measured 63.71 GB peak at the smaller C=2 shape (see this
    package's memory investigation), almost entirely from three fixable
    sites (``raw_interval_multi`` calling ``torch.quantile`` with a batched
    multi-level ``q`` tensor -- measured to scale peak memory linearly in
    the *number* of quantiles requested rather than sharing one sort;
    ``EMOS``/``Ensemble.predict()`` being asked for all 19 levels in one
    call; and the three input prediction dumps being held on the device for
    every one of the four calibration x test combinations, not just the
    ones that use them). Measured after the fix, at the real C=3 shape:
    peak ~28.2 GB for ``calibrate()``, wall time within ~1.5x of the
    pre-fix C=2 measurement (~60s) scaled for the extra channel.
    """
    device = "cuda"
    new = make_synthetic_dump(
        b_total=150,
        n_frames=100,
        height=64,
        width=64,
        n_channels=3,
        n_members=10,
        seed=1,
    )
    paper_valid = make_synthetic_dump(
        b_total=20,
        n_frames=100,
        height=64,
        width=64,
        n_channels=3,
        n_members=10,
        seed=2,
    )
    paper_test = make_synthetic_dump(
        b_total=20,
        n_frames=100,
        height=64,
        width=64,
        n_channels=3,
        n_members=10,
        seed=3,
    )
    new_path, pv_path, pt_path = (
        tmp_path / "new.pt",
        tmp_path / "paper_valid.pt",
        tmp_path / "paper_test.pt",
    )
    save_dump(new, new_path)
    save_dump(paper_valid, pv_path)
    save_dump(paper_test, pt_path)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    calibrate.calibrate(
        new_path=new_path,
        paper_valid_path=pv_path,
        paper_test_path=pt_path,
        out_dir=tmp_path / "eval_conformal",
        device=device,
    )
    torch.cuda.synchronize()
    calibrate_seconds = time.time() - start
    calibrate_peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"[benchmark] calibrate(): {calibrate_seconds:.1f}s, "
        f"peak {calibrate_peak_gb:.2f} GB"
    )
    assert calibrate_peak_gb < _PEAK_MEMORY_CEILING_GB

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    sufficiency.sufficiency(
        new_path=new_path, out_dir=tmp_path / "eval_conformal", device=device
    )
    torch.cuda.synchronize()
    sufficiency_seconds = time.time() - start
    sufficiency_peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"[benchmark] sufficiency(): {sufficiency_seconds:.1f}s, "
        f"peak {sufficiency_peak_gb:.2f} GB"
    )
    assert sufficiency_peak_gb < _PEAK_MEMORY_CEILING_GB

    assert (tmp_path / "eval_conformal" / "manifest.json").exists()
    assert (
        tmp_path / "eval_conformal" / "data_sufficiency" / "sufficiency.json"
    ).exists()
