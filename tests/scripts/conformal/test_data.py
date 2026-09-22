import pickle

import pytest
import torch

from autocast.scripts.conformal import data

from .conftest import make_grouped_dump


def test_fixed_split_reproduces_expected_permutation():
    """Same seed, same `torch.randperm` -> same split (July's algorithm)."""
    b_total = 150
    generator = torch.Generator().manual_seed(data.DEFAULT_SPLIT_SEED)
    expected_perm = torch.randperm(b_total, generator=generator)
    expected_test = expected_perm[: data.DEFAULT_TEST_SIZE]
    expected_calibration = expected_perm[data.DEFAULT_TEST_SIZE :]

    split = data.fixed_split(b_total)

    assert torch.equal(split.test, expected_test)
    assert torch.equal(split.calibration, expected_calibration)
    assert split.test.shape[0] == data.DEFAULT_TEST_SIZE
    assert split.calibration.shape[0] == b_total - data.DEFAULT_TEST_SIZE


def test_fixed_split_disjoint_and_covers_range():
    split = data.fixed_split(30, test_size=10)
    combined = torch.cat([split.calibration, split.test]).sort().values
    assert torch.equal(combined, torch.arange(30))


def test_fixed_split_clamps_test_size_to_respect_min_calibration():
    split = data.fixed_split(15, test_size=50, min_calibration=9)
    assert split.calibration.shape[0] == 9
    assert split.test.shape[0] == 6


def test_fixed_split_raises_when_too_small():
    with pytest.raises(ValueError, match="too small"):
        data.fixed_split(9, test_size=50, min_calibration=9)


def test_fixed_split_is_deterministic_across_calls():
    first = data.fixed_split(50, test_size=15, seed=123)
    second = data.fixed_split(50, test_size=15, seed=123)
    assert torch.equal(first.calibration, second.calibration)
    assert torch.equal(first.test, second.test)


def test_balanced_split_by_scalars_gives_expected_counts():
    n_groups, n_per_group = 6, 25
    group_ids = torch.arange(n_groups).repeat_interleave(n_per_group)
    constant_scalars = group_ids.float().unsqueeze(-1)

    split = data.balanced_split_by_scalars(constant_scalars)

    assert split.calibration.shape[0] == n_groups * 17
    assert split.test.shape[0] == n_groups * 8
    calibration_set = set(split.calibration.tolist())
    test_set = set(split.test.tolist())
    assert calibration_set.isdisjoint(test_set)

    for group in range(n_groups):
        group_members = set((group_ids == group).nonzero().flatten().tolist())
        assert len(group_members & calibration_set) == 17
        assert len(group_members & test_set) == 8


def test_balanced_split_raises_when_group_too_small():
    constant_scalars = torch.zeros(20, 1)  # one group, 20 members, need 25
    with pytest.raises(ValueError, match="need >="):
        data.balanced_split_by_scalars(constant_scalars)


def test_balanced_split_is_deterministic_across_calls():
    dump = make_grouped_dump(seed=1)
    first = data.balanced_split_by_scalars(dump["constant_scalars"])
    second = data.balanced_split_by_scalars(dump["constant_scalars"])
    assert torch.equal(first.calibration, second.calibration)
    assert torch.equal(first.test, second.test)


def test_draw_calibration_subset_reproducible_and_sized():
    pool_idx = torch.arange(100)
    first = data.draw_calibration_subset(pool_idx, k=20, draw_idx=3)
    second = data.draw_calibration_subset(pool_idx, k=20, draw_idx=3)
    assert torch.equal(first, second)
    assert first.shape[0] == 20
    assert set(first.tolist()).issubset(set(pool_idx.tolist()))


def test_draw_calibration_subset_different_draws_differ():
    pool_idx = torch.arange(100)
    draw0 = data.draw_calibration_subset(pool_idx, k=20, draw_idx=0)
    draw1 = data.draw_calibration_subset(pool_idx, k=20, draw_idx=1)
    assert not torch.equal(draw0, draw1)


def test_draw_calibration_subset_raises_when_k_too_large():
    pool_idx = torch.arange(10)
    with pytest.raises(ValueError, match="exceeds calibration-pool size"):
        data.draw_calibration_subset(pool_idx, k=20, draw_idx=0)


def test_load_prediction_dump_roundtrip(tmp_path):
    payload = {
        "preds": torch.randn(3, 2, 2, 2, 1, 4),
        "trues": torch.randn(3, 2, 2, 2, 1),
        "constant_scalars": torch.randn(3, 1),
        "meta": {"n_members": 4},
    }
    path = tmp_path / "dump.pt"
    torch.save(payload, path)

    dump = data.load_prediction_dump(path)

    assert dump.n_trajectories == 3
    assert dump.md5
    assert torch.equal(dump.preds, payload["preds"])
    assert dump.constant_scalars is not None
    assert torch.equal(dump.constant_scalars, payload["constant_scalars"])
    assert dump.meta["n_members"] == 4


def test_load_prediction_dump_handles_missing_constant_scalars(tmp_path):
    payload = {
        "preds": torch.randn(2, 2, 2, 2, 1, 3),
        "trues": torch.randn(2, 2, 2, 2, 1),
        "constant_scalars": None,
        "meta": {},
    }
    path = tmp_path / "dump.pt"
    torch.save(payload, path)

    dump = data.load_prediction_dump(path)

    assert dump.constant_scalars is None


def test_build_manifest_has_expected_keys(tmp_path):
    payload = {
        "preds": torch.randn(3, 2, 2, 2, 1, 4),
        "trues": torch.randn(3, 2, 2, 2, 1),
        "constant_scalars": None,
        "meta": {},
    }
    new_path = tmp_path / "new.pt"
    valid_path = tmp_path / "valid.pt"
    test_path = tmp_path / "test.pt"
    for path in (new_path, valid_path, test_path):
        torch.save(payload, path)

    new_dump = data.load_prediction_dump(new_path)
    paper_valid_dump = data.load_prediction_dump(valid_path)
    paper_test_dump = data.load_prediction_dump(test_path)
    split = data.fixed_split(3, test_size=1, min_calibration=1)

    manifest = data.build_manifest(
        new_dump=new_dump,
        paper_valid_dump=paper_valid_dump,
        paper_test_dump=paper_test_dump,
        new_split=split,
        balanced_by_scalars=False,
        split_seed=data.DEFAULT_SPLIT_SEED,
        alpha=0.1,
        levels=[0.9],
        windows=[(0, 1)],
    )

    assert manifest["inputs"]["new"]["md5"] == new_dump.md5
    assert manifest["split"]["seed"] == data.DEFAULT_SPLIT_SEED
    assert manifest["alpha"] == 0.1
    assert manifest["levels"] == [0.9]
    assert manifest["windows"] == [[0, 1]]
    assert len(manifest["git_commit"]) == 40
    assert manifest["autouq_version"]
    assert "generated_at_utc" in manifest


def test_git_commit_is_a_full_sha():
    assert len(data.git_commit()) == 40


def test_autouq_version_is_nonempty():
    assert data.autouq_version()


class _NotATensor:
    """Stands in for an arbitrary object a crafted dump could carry."""


def test_load_prediction_dump_refuses_arbitrary_objects(tmp_path):
    payload = {
        "preds": torch.zeros(2, 1, 2, 2, 1, 3),
        "trues": torch.zeros(2, 1, 2, 2, 1),
        "constant_scalars": None,
        "meta": {"split": "test", "extra": _NotATensor()},
    }
    path = tmp_path / "crafted.pt"
    torch.save(payload, path)
    with pytest.raises(pickle.UnpicklingError):
        data.load_prediction_dump(path)
