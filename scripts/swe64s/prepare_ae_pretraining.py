"""Plan CPU shards or assemble full-cadence AE data; never submit or train."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from prepare_ae_diversity_data import (
    _CHECKS,
    _check_payload,
    _contract,
    _load_config,
    _load_payload,
    _require,
    _sha256,
)

POOL_COUNT = 1000
POOL_SEED = 20280000
SHARD_COUNT = 10
SHARD_SIZE = 100
# Reserve all three CLI split namespaces, including retries, in every shard.
SHARD_SEED_STRIDE = 3 * (max(100, 20 * SHARD_SIZE) + 1)
CONFIG_NAME = "generate_data_swe_ae_pretraining_64"
MANIFEST_NAME = "ae_pretraining_manifest.json"


def generation_plan(output: Path, autosim_source: Path, workers: int = 4) -> dict:
    """Print reproducible shard commands without running them or creating files."""
    _require(1 <= workers <= SHARD_COUNT, "Worker count must be between 1 and 10")
    root = output.expanduser().absolute()
    shards = []
    for index in range(SHARD_COUNT):
        seed = POOL_SEED + index * SHARD_SEED_STRIDE
        destination = root / f"shard-{index:03d}"
        shards.append(
            {
                "index": index,
                "count": SHARD_SIZE,
                "seed": seed,
                "reserved_seed_range": [seed, seed + SHARD_SEED_STRIDE - 1],
                "output": str(destination),
                "cwd": str(autosim_source.expanduser().absolute()),
                "command": [
                    "env",
                    "CUDA_VISIBLE_DEVICES=",
                    "OMP_NUM_THREADS=1",
                    "MKL_NUM_THREADS=1",
                    "uv",
                    "run",
                    "--frozen",
                    "--no-sync",
                    "autosim",
                    f"--config-name={CONFIG_NAME}",
                    f"dataset.n_train={SHARD_SIZE}",
                    f"seed={seed}",
                    f"dataset.output_dir={destination}",
                    f"hydra.run.dir={root / 'hydra' / f'shard-{index:03d}'}",
                ],
            }
        )
    return {
        "config": CONFIG_NAME,
        "total_trajectories": POOL_COUNT,
        "concurrent_workers": workers,
        "shards": shards,
        "submitted": False,
        "identity_note": "Fixed shards/seeds; worker count changes scheduling only",
    }


def _check_sources(
    baseline: Path, sources: list[Path], output: Path, contract: dict[str, Any]
) -> dict:
    original_output = output.expanduser().absolute()
    if os.path.lexists(original_output):
        msg = f"Refusing existing output, including symlinks: {original_output}"
        raise FileExistsError(msg)
    baseline = baseline.expanduser().resolve()
    sources = [source.expanduser().resolve() for source in sources]
    output = original_output.resolve()
    _require(len(sources) in (1, SHARD_COUNT), "Supply one pool or ten fixed shards")
    paths = [baseline, *sources, output]
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            _require(
                first != second
                and first not in second.parents
                and second not in first.parents,
                "Baseline, sources and output must be disjoint, not nested",
            )
    _require(output.parent.is_dir(), "Output parent must already exist")
    baseline_config = _load_config(baseline / "resolved_config.yaml")
    hashes = {}
    for name, expected in contract["sha256"].items():
        path = baseline / name
        _require(_sha256(path) == expected, f"Baseline hash mismatch: {name}")
        hashes[str(path)] = expected
    _CHECKS["check_dataset"](baseline, contract, verify_hashes=False)
    seen_starts: dict[str, str] = {}
    starts = {}
    for split, count in contract["splits"].items():
        starts[f"baseline/{split}"] = _check_payload(
            baseline / split / "data.pt", count, contract, seen_starts
        )

    records = []
    # Input ordering cannot silently change the merged data or identity.
    for source in sources:
        config = _load_config(source / "resolved_config.yaml")
        seed = config.get("seed")
        _require(isinstance(seed, int) and not isinstance(seed, bool), "Missing seed")
        records.append({"path": source, "config": config, "seed": seed})
    records.sort(key=lambda record: record["seed"])
    expected_size = POOL_COUNT if len(records) == 1 else SHARD_SIZE
    for index, record in enumerate(records):
        source, config = record["path"], record["config"]
        _require(
            config.get("simulator") == baseline_config["simulator"],
            "Pretraining simulator settings must exactly match baseline",
        )
        _require(
            record["seed"] == POOL_SEED + index * SHARD_SEED_STRIDE,
            "Unexpected pretraining seed or shard identity",
        )
        counts = config.get("dataset", {})
        _require(
            [counts.get(f"n_{split}") for split in ("train", "valid", "test")]
            == [expected_size, 0, 0],
            "Unexpected pretraining split counts",
        )
        starts[str(source)] = _check_payload(
            source / "train/data.pt", expected_size, contract, seen_starts
        )
        for name in ("resolved_config.yaml", "train/data.pt"):
            path = source / name
            hashes[str(path)] = _sha256(path)
        for split in ("valid", "test"):
            path = source / split / "data.pt"
            if path.exists():
                _check_payload(path, 0, contract, seen_starts)
                hashes[str(path)] = _sha256(path)
    return {
        "baseline": str(baseline),
        "output": str(output),
        "baseline_config": baseline_config,
        "sources": [{**record, "path": str(record["path"])} for record in records],
        "input_sha256": hashes,
        "retained_start_state_sha256": starts,
    }


def _assemble(checked: dict, contract: dict) -> dict:
    output, baseline = (Path(checked[key]) for key in ("output", "baseline"))
    shape = (POOL_COUNT, contract["frames"], *contract["resolution"], 3)
    output.mkdir(exist_ok=False)
    if len(checked["sources"]) == 1:
        source = Path(checked["sources"][0]["path"])
        (output / "train").symlink_to(source / "train", target_is_directory=True)
    else:
        train = output / "train"
        train.mkdir()
        buffer = train / ".field-buffer.bin"
        elements = 1
        for dimension in shape:
            elements *= dimension
        fields = torch.from_file(
            str(buffer), shared=True, size=elements, dtype=torch.float32
        ).view(shape)
        scalars = []
        cursor = 0
        for record in checked["sources"]:
            payload = _load_payload(Path(record["path"]) / "train/data.pt")
            count = payload["data"].shape[0]
            fields[cursor : cursor + count].copy_(payload["data"])
            scalars.append(payload["constant_scalars"])
            cursor += count
        _require(cursor == POOL_COUNT, "Merged count differs from requested pool")
        with (train / "data.pt").open("xb") as stream:
            torch.save(
                {
                    "data": fields,
                    "constant_scalars": torch.cat(scalars),
                    "constant_fields": None,
                },
                stream,
            )
        del fields
        buffer.unlink()
    for split in ("valid", "test"):
        (output / split).symlink_to(baseline / split, target_is_directory=True)
    (output / "stats.yml").symlink_to(baseline / "stats.yml")
    config = {
        "ae_pretraining_only": True,
        "simulator": checked["baseline_config"]["simulator"],
        "dataset": {
            "n_train": POOL_COUNT,
            "n_valid": contract["splits"]["valid"],
            "n_test": contract["splits"]["test"],
            "output_dir": str(output),
        },
    }
    OmegaConf.save(OmegaConf.create(config), output / "resolved_config.yaml")
    manifest = checked | {
        "shape": list(shape),
        "field_names": contract["field_names"],
        "all_frames_retained": True,
        "saved_times": contract["saved_times"],
        "normalization": "Unchanged original training stats",
        "builder_sha256": _sha256(Path(__file__)),
        "output_sha256": {
            name: _sha256(output / name)
            for name in ("train/data.pt", "stats.yml", "resolved_config.yaml")
        },
    }
    for path, expected in checked["input_sha256"].items():
        _require(_sha256(Path(path)) == expected, f"Source changed: {path}")
    with (output / MANIFEST_NAME).open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def validate_pretraining_data(output: Path, contract: dict | None = None) -> dict:
    """Check the full pool, unchanged held-out data and original normalization."""
    contract = _contract() if contract is None else contract
    output = output.expanduser().resolve()
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    _require(manifest.get("output") == str(output), "Pool manifest location changed")
    _require(
        manifest.get("all_frames_retained") is True
        and manifest.get("shape")
        == [POOL_COUNT, contract["frames"], *contract["resolution"], 3]
        and manifest.get("field_names") == contract["field_names"]
        and manifest.get("saved_times") == contract["saved_times"],
        "Unexpected full-cadence pool manifest",
    )
    baseline = Path(manifest["baseline"])
    for name in ("valid", "test", "stats.yml"):
        link = output / name
        _require(
            link.is_symlink() and link.resolve() == (baseline / name).resolve(),
            f"Pool {name} must reference the unchanged baseline",
        )
    for name in ("valid/data.pt", "test/data.pt", "stats.yml", "resolved_config.yaml"):
        _require(
            _sha256(baseline / name) == contract["sha256"][name],
            f"Baseline hash mismatch: {name}",
        )
    baseline_config = _load_config(baseline / "resolved_config.yaml")
    _require(
        manifest.get("baseline_config") == baseline_config,
        "Baseline configuration differs from pool provenance",
    )
    sources = manifest.get("sources", [])
    _require(len(sources) in (1, SHARD_COUNT), "Missing source pool/shard identities")
    count = POOL_COUNT if len(sources) == 1 else SHARD_SIZE
    for index, record in enumerate(sources):
        config_path = Path(record["path"]) / "resolved_config.yaml"
        cfg = _load_config(config_path)
        _require(
            record.get("seed")
            == cfg.get("seed")
            == POOL_SEED + index * SHARD_SEED_STRIDE
            and record.get("config") == cfg
            and cfg.get("simulator") == baseline_config["simulator"]
            and [
                cfg.get("dataset", {}).get(f"n_{split}")
                for split in ("train", "valid", "test")
            ]
            == [count, 0, 0]
            and _sha256(config_path) == manifest["input_sha256"].get(str(config_path)),
            "Source identity differs from pool provenance",
        )
    _require(
        set(manifest["output_sha256"])
        == {"train/data.pt", "stats.yml", "resolved_config.yaml"},
        "Missing pool output hashes",
    )
    for name, expected in manifest["output_sha256"].items():
        _require(_sha256(output / name) == expected, f"Pool hash mismatch: {name}")
    cfg = _load_config(output / "resolved_config.yaml")
    _require(
        cfg.get("ae_pretraining_only") is True
        and cfg.get("simulator") == baseline_config["simulator"]
        and cfg.get("dataset")
        == {
            "n_train": POOL_COUNT,
            "n_valid": contract["splits"]["valid"],
            "n_test": contract["splits"]["test"],
            "output_dir": str(output),
        },
        "Pool configuration differs from approved full-cadence data",
    )
    payload = _load_payload(output / "train/data.pt")
    _require(
        list(payload["data"].shape) == manifest["shape"]
        and payload["data"].dtype == torch.float32,
        "Unexpected pool fields",
    )
    _require(
        tuple(payload["constant_scalars"].shape) == (POOL_COUNT, 1)
        and torch.isfinite(payload["constant_scalars"]).all().item()
        and torch.allclose(
            payload["constant_scalars"],
            torch.full_like(payload["constant_scalars"], 0.1),
        ),
        "Unexpected pool metadata",
    )
    return manifest


def main() -> None:
    """Plan, check or assemble local files without launching simulation or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    plan = modes.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--autosim-source", type=Path, required=True)
    plan.add_argument("--workers", type=int, default=4)
    build = modes.add_parser("prepare")
    build.add_argument("--baseline", type=Path, required=True)
    build.add_argument("--source", type=Path, action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--check-only", action="store_true")
    check = modes.add_parser("verify")
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        result = generation_plan(args.output, args.autosim_source, args.workers)
    elif args.mode == "verify":
        result = validate_pretraining_data(args.output)
    else:
        contract = _contract()
        result = _check_sources(args.baseline, args.source, args.output, contract)
        if not args.check_only:
            result = _assemble(result, contract)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
