"""Loading, splitting, and manifest construction for conformal calibration inputs.

Consumes eval-dump prediction files -- ``{"preds": [B,T,H,W,C,M], "trues":
[B,T,H,W,C], "constant_scalars": [B,K] or None, "meta": dict}`` -- and produces
the fixed calibration/test trajectory splits consumed by
:mod:`autocast.scripts.conformal.calibrate` and
:mod:`autocast.scripts.conformal.sufficiency`.

The fixed split reproduces the one used for the earlier (July 2026)
calibration runs exactly, at the same default seed and test-set size, so those
results can be compared directly. What
July called the "assessment" set (held out, default 50 trajectories) is this
package's *test* set; July's "pool" (the remainder, default ~100) is this
package's *calibration* set.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from autocast.types import TensorBC, TensorBTSC, TensorBTSCM

#: Conformal's hard floor for alpha=0.1: order = ceil((n+1)*0.9) <= n needs
#: n >= 9 (`autouq.calibrators.conformal.conformal.ConformalCalibrator
#: .score_quantile`). Used as the default minimum calibration-set size so a
#: fixed split never starves the smallest feasible calibration count.
CONFORMAL_MIN_CALIBRATION = 9

#: Split of the 150-trajectory new set: ~100 calibration / ~50 test.
DEFAULT_TEST_SIZE = 50
DEFAULT_SPLIT_SEED = 20260709

#: Gray-Scott balanced split (D18): 17 calibration + 8 test per pattern,
#: 6 patterns * 25 trajectories/pattern = 150 total.
DEFAULT_BALANCED_CALIBRATION_PER_GROUP = 17
DEFAULT_BALANCED_TEST_PER_GROUP = 8

_MD5_CHUNK_SIZE = 8 * 1024 * 1024


def default_device() -> str:
    """Return "cuda" if a GPU is available, else "cpu" (the ``--device`` default).

    Not `autocast.scripts.execution.resolve_device`, which can also pick MPS:
    MPS has no float64, and the EMOS fit runs in float64.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


class CalibrationSource(StrEnum):
    """Which prediction set the calibration set is drawn from."""

    NEW = "new"
    PAPER_VALID = "paper-valid"


class TestSource(StrEnum):
    """Which prediction set the test set is drawn from."""

    # Not a pytest test class -- the name collision is with pytest's default
    # `Test*` collection pattern, not this enum's meaning.
    __test__ = False

    NEW = "new"
    PAPER = "paper"


@dataclass(frozen=True)
class SplitIndices:
    """Disjoint trajectory-index tensors for the calibration/test roles."""

    calibration: torch.Tensor
    test: torch.Tensor


@dataclass(frozen=True)
class PredictionDump:
    """One loaded eval-dump prediction file."""

    path: Path
    md5: str
    preds: TensorBTSCM
    trues: TensorBTSC
    constant_scalars: TensorBC | None
    meta: dict[str, Any]

    @property
    def n_trajectories(self) -> int:
        """Trajectory count (batch dimension)."""
        return int(self.trues.shape[0])

    def to(self, device: torch.device | str) -> PredictionDump:
        """Return a copy with every tensor moved to ``device`` (once, up front)."""
        return replace(
            self,
            preds=self.preds.to(device),
            trues=self.trues.to(device),
            constant_scalars=(
                self.constant_scalars.to(device)
                if self.constant_scalars is not None
                else None
            ),
        )


def _md5sum(path: Path, chunk_size: int = _MD5_CHUNK_SIZE) -> str:
    """Stream an md5 digest of `path` without holding it in memory at once."""
    digest = hashlib.md5()  # provenance checksum, not a security use
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction_dump(path: str | Path) -> PredictionDump:
    """Load one eval-dump ``.pt`` file into a :class:`PredictionDump`.

    Parameters
    ----------
    path
        Path to a ``.pt`` file with keys ``preds`` ``[B,T,H,W,C,M]``, ``trues``
        ``[B,T,H,W,C]``, ``constant_scalars`` ``[B,K]`` or ``None``, and
        ``meta`` (a dict).

    Returns
    -------
    PredictionDump
        The loaded tensors (cast to float32, CPU) plus the file's md5 digest
        for manifest provenance.

    Loaded with ``weights_only=True``: a dump holds only tensors, strings and
    numbers, so nothing else in a file is ever unpickled.
    """
    resolved = Path(path)
    digest = _md5sum(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    constant_scalars = payload.get("constant_scalars")
    return PredictionDump(
        path=resolved,
        md5=digest,
        preds=payload["preds"].float(),
        trues=payload["trues"].float(),
        constant_scalars=(
            constant_scalars.float() if constant_scalars is not None else None
        ),
        meta=dict(payload.get("meta", {}) or {}),
    )


def fixed_split(
    b_total: int,
    *,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
    min_calibration: int = CONFORMAL_MIN_CALIBRATION,
) -> SplitIndices:
    """Deterministic calibration/test trajectory split.

    Ported from July's ``sufficiency_common.fixed_split``: a single seeded
    permutation of ``[0, b_total)``, split into a ``test_size``-trajectory
    test set and a calibration set holding the remainder. At production scale
    (``b_total`` ~150) this reproduces the plan's ~100 calibration / ~50 test
    split exactly. At small dumps, ``test_size`` is clamped down so the
    calibration set keeps at least ``min_calibration`` trajectories (the
    conformal hard floor for alpha=0.1).

    Parameters
    ----------
    b_total
        Total trajectory count to split.
    test_size
        Target test-set size (clamped down on small dumps).
    seed
        Seed for the permutation; constant across a whole calibration run.
    min_calibration
        Minimum calibration-set size to preserve when clamping ``test_size``.

    Returns
    -------
    SplitIndices
        Disjoint ``calibration``/``test`` trajectory-index tensors covering
        ``[0, b_total)``.
    """
    resolved_test_size = min(test_size, b_total - min_calibration)
    if resolved_test_size <= 0:
        msg = (
            f"b_total={b_total} too small for a nonempty test set leaving >= "
            f"min_calibration={min_calibration} calibration trajectories "
            f"(requested test_size={test_size})."
        )
        raise ValueError(msg)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(b_total, generator=generator)
    test_idx = perm[:resolved_test_size].clone()
    calibration_idx = perm[resolved_test_size:].clone()
    return SplitIndices(calibration=calibration_idx, test=test_idx)


def balanced_split_by_scalars(
    constant_scalars: TensorBC,
    *,
    n_calibration_per_group: int = DEFAULT_BALANCED_CALIBRATION_PER_GROUP,
    n_test_per_group: int = DEFAULT_BALANCED_TEST_PER_GROUP,
    seed: int = DEFAULT_SPLIT_SEED,
) -> SplitIndices:
    """Stratify the calibration/test split by distinct ``constant_scalars`` rows.

    Groups trajectories by distinct rows of ``constant_scalars`` (Gray-Scott:
    6 patterns of 25 trajectories each) and draws a fixed, seeded
    ``n_calibration_per_group``/``n_test_per_group`` split independently
    within each group, so every pattern is represented proportionally in both
    the calibration and test sets (D18).

    Parameters
    ----------
    constant_scalars
        Per-trajectory conditioning scalars, shape ``(B, K)``.
    n_calibration_per_group
        Calibration trajectories to draw from each group.
    n_test_per_group
        Test trajectories to draw from each group.
    seed
        Base seed; each group gets its own generator seeded from this plus
        the group's index (following the ``seed0 + 1000 * k`` convention used
        by July's ``sufficiency_common.draw_cal_subset``), so groups don't
        share draws.

    Returns
    -------
    SplitIndices
        Disjoint ``calibration``/``test`` trajectory-index tensors, sorted
        ascending.

    Raises
    ------
    ValueError
        If any group has fewer than
        ``n_calibration_per_group + n_test_per_group`` trajectories.
    """
    values = constant_scalars.detach().cpu()
    row_keys = [tuple(row.tolist()) for row in values]
    unique_groups = sorted(set(row_keys))

    calibration_indices: list[int] = []
    test_indices: list[int] = []
    needed = n_calibration_per_group + n_test_per_group
    for group_idx, group_key in enumerate(unique_groups):
        group_members = [i for i, key in enumerate(row_keys) if key == group_key]
        if len(group_members) < needed:
            msg = (
                f"group {group_key} has {len(group_members)} trajectories, "
                f"need >= {needed} (n_calibration_per_group="
                f"{n_calibration_per_group} + n_test_per_group="
                f"{n_test_per_group})."
            )
            raise ValueError(msg)
        generator = torch.Generator().manual_seed(seed + 1000 * group_idx)
        perm = torch.randperm(len(group_members), generator=generator).tolist()
        ordered = [group_members[i] for i in perm]
        calibration_indices.extend(ordered[:n_calibration_per_group])
        test_indices.extend(ordered[n_calibration_per_group:needed])

    return SplitIndices(
        calibration=torch.tensor(sorted(calibration_indices), dtype=torch.long),
        test=torch.tensor(sorted(test_indices), dtype=torch.long),
    )


def draw_calibration_subset(
    pool_idx: torch.Tensor, k: int, draw_idx: int, seed0: int = DEFAULT_SPLIT_SEED
) -> torch.Tensor:
    """Seeded, reproducible K-sized draw from the calibration pool.

    Ported from July's ``sufficiency_common.draw_cal_subset``: a distinct
    :class:`torch.Generator` per ``(k, draw_idx)`` (``seed0 + 1000 * k +
    draw_idx``), so draws are reproducible without touching the global RNG
    state and different ``(K, draw)`` combinations never collide.

    Parameters
    ----------
    pool_idx
        Calibration-pool trajectory indices to draw from (without
        replacement).
    k
        Draw size.
    draw_idx
        Draw index within the sweep at this ``k`` (varies the seed).
    seed0
        Base seed.

    Returns
    -------
    torch.Tensor
        ``k`` trajectory indices drawn from ``pool_idx``.
    """
    pool_size = pool_idx.shape[0]
    if k > pool_size:
        msg = f"K={k} exceeds calibration-pool size {pool_size}."
        raise ValueError(msg)
    generator = torch.Generator().manual_seed(seed0 + 1000 * k + draw_idx)
    perm = torch.randperm(pool_size, generator=generator)
    return pool_idx[perm[:k]]


def git_commit() -> str:
    """Return ``git rev-parse HEAD`` for this repository (manifest provenance)."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def autouq_version() -> str:
    """Return the installed ``autouq`` package version (manifest provenance)."""
    return importlib.metadata.version("autouq")


def build_manifest(
    *,
    new_dump: PredictionDump,
    paper_valid_dump: PredictionDump,
    paper_test_dump: PredictionDump,
    new_split: SplitIndices,
    balanced_by_scalars: bool,
    split_seed: int,
    alpha: float,
    levels: list[float],
    windows: list[tuple[int, int]],
    relative_to: Path | None = None,
) -> dict[str, Any]:
    """Assemble the run manifest written as ``manifest.json``.

    Parameters
    ----------
    new_dump, paper_valid_dump, paper_test_dump
        The three loaded prediction dumps.
    new_split
        The fixed (or balanced) split of the new set into calibration/test.
    balanced_by_scalars
        Whether ``new_split`` came from :func:`balanced_split_by_scalars`
        rather than :func:`fixed_split`.
    split_seed
        Seed used for the split.
    alpha
        Headline miscoverage level (e.g. 0.1 for 90% intervals).
    levels
        Coverage-level grid used for reliability curves.
    windows
        Rollout windows used for ``rollout_metrics.csv``.
    relative_to
        If given, input paths are recorded relative to this directory (the
        output folder), so the manifest stays valid when the folder moves to
        another machine.

    Returns
    -------
    dict
        JSON-serializable manifest: input paths, md5 and trajectory counts,
        split indices, seeds,
        alpha, levels, windows, this repository's git commit, the installed
        autouq version, and a UTC timestamp.
    """

    def describe(dump: PredictionDump) -> dict[str, Any]:
        path = Path(dump.path)
        if relative_to is not None:
            path = Path(os.path.relpath(path.resolve(), Path(relative_to).resolve()))
        return {
            "path": str(path),
            "md5": dump.md5,
            "n_trajectories": dump.n_trajectories,
        }

    return {
        "inputs": {
            "new": describe(new_dump),
            "paper_valid": describe(paper_valid_dump),
            "paper_test": describe(paper_test_dump),
        },
        "split": {
            "balanced_by_scalars": balanced_by_scalars,
            "seed": split_seed,
            "new_calibration_idx": sorted(new_split.calibration.tolist()),
            "new_test_idx": sorted(new_split.test.tolist()),
        },
        "alpha": alpha,
        "levels": list(levels),
        "windows": [list(window) for window in windows],
        "git_commit": git_commit(),
        "autouq_version": autouq_version(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
