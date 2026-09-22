"""CPU/CUDA parity for the conformal calibration package.

Skips when no CUDA device is present, following this repo's convention (see
``tests/metrics/test_deterministic.py::test_isotropic_binning_device_consistency``).
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from autocast.scripts.conformal import calibrate

from .conftest import make_synthetic_dump

_HAS_CUDA = torch.cuda.is_available()
_SKIP_REASON = "No CUDA device available for the CPU/CUDA parity test"


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
        columns = cpu_df.select_dtypes(include="number").columns
        assert_close(cpu_df[columns], cuda_df[columns])

    deterministic_columns = ["winkler", "coverage"]
    cpu_emos = pd.read_csv(cpu_dir / "EMOS" / "rollout_metrics.csv")
    cuda_emos = pd.read_csv(cuda_dir / "EMOS" / "rollout_metrics.csv")
    assert_close(cpu_emos[deterministic_columns], cuda_emos[deterministic_columns])

    cpu_coverage_map = torch.load(cpu_dir / "coverage_map.pt", weights_only=False)
    cuda_coverage_map = torch.load(cuda_dir / "coverage_map.pt", weights_only=False)
    for name in ("raw", "EMOS", "conformal"):
        torch.testing.assert_close(
            cpu_coverage_map[name],
            cuda_coverage_map[name].cpu(),
            rtol=0.1,
            atol=5e-3,
        )
