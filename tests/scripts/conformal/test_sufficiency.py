import json

import pandas as pd
import pytest

from autocast.scripts.conformal import sufficiency
from autocast.scripts.conformal.data import load_prediction_dump

from .conftest import make_synthetic_dump, save_dump


def test_sufficiency_writes_expected_files(tmp_path):
    new_path = tmp_path / "new.pt"
    save_dump(make_synthetic_dump(b_total=90, n_frames=6, seed=40), new_path)

    out_dir = tmp_path / "eval_conformal"
    sufficiency.sufficiency(
        new_path=new_path,
        out_dir=out_dir,
        k_grid=(9, 15, 30),
        n_draws=3,
        device="cpu",  # tiny synthetic data -- GPU launch overhead dominates
    )

    sufficiency_dir = out_dir / "data_sufficiency"
    result = json.loads((sufficiency_dir / "sufficiency.json").read_text())
    assert result["k_grid"] == [9, 15, 30]
    assert set(result["by_K"]["9"]) == {"raw", "EMOS", "conformal"}
    assert result["by_K"]["30"]["conformal"]["cov"]["n_draws_used"] == 3

    summary = pd.read_csv(sufficiency_dir / "sufficiency.csv")
    assert len(summary) == len({9, 15, 30}) * 3  # 3 methods per K
    per_frame = pd.read_csv(sufficiency_dir / "sufficiency_per_frame.csv")
    assert set(per_frame["frame"]) == set(range(6))


def test_sufficiency_filters_k_grid_to_pool_size(tmp_path):
    # Default test_size=50, min_calibration=9 -> pool = max(b_total-50, 9);
    # b_total=65 gives a pool of exactly 15, so k_grid (9, 15, 100) filters
    # down to [9, 15].
    new_path = tmp_path / "new.pt"
    save_dump(make_synthetic_dump(b_total=65, n_frames=4, seed=41), new_path)
    dump = load_prediction_dump(new_path)

    result = sufficiency.run_sufficiency(dump, k_grid=(9, 15, 100), n_draws=2)

    assert result["k_grid"] == [9, 15]
    assert result["k_grid_requested"] == [9, 15, 100]


def test_sufficiency_usage_errors(tmp_path):
    save_dump(make_synthetic_dump(b_total=30, n_frames=2), tmp_path / "new.pt")
    new_dump = load_prediction_dump(tmp_path / "new.pt")
    with pytest.raises(ValueError, match="requires constant_scalars"):
        sufficiency.run_sufficiency(new_dump, balance_by_scalars=True)
    with pytest.raises(ValueError, match="no K in grid"):
        sufficiency.run_sufficiency(new_dump, k_grid=(500,))
    with pytest.raises(RuntimeError, match="n_draws must be"):
        sufficiency.run_sufficiency(new_dump, k_grid=(9,), n_draws=0)
