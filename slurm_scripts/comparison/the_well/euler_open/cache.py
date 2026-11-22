"""Stage canonical Flatiron Euler trajectories in the MiniWell format."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import h5py
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from the_well.data import WellDataset
from the_well.data.datasets import BoundaryCondition

from autocast.external.lola.field_transform import (
    named_field_indices,
    standardize_fields,
)
from autocast.external.lola.lola_autoencoder import get_autoencoder

WellSplit = Literal["train", "valid", "test"]
SPLITS: tuple[WellSplit, ...] = ("train", "valid", "test")


def file_sha256(path: Path) -> str:
    """Hash the file contents without reading the entire file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_codec_config(spec: DictConfig) -> tuple[DictConfig, Path]:
    """Check the two codec files against the published archive's pilot receipt."""
    directory = Path(spec.codec.directory)
    for name, expected in (
        ("config.yaml", spec.codec.config_sha256),
        ("state.pth", spec.codec.state_sha256),
    ):
        path = directory / name
        if file_sha256(path) != expected:
            raise ValueError(f"Unexpected SHA-256 for {path}")
    cfg = OmegaConf.load(directory / "config.yaml")
    if (
        not isinstance(cfg, DictConfig)
        or cfg.name != spec.codec.name
        or cfg.dataset.physics != spec.codec.training_datasets
        or cfg.dataset.fields != spec.fields
        or dict(cfg.dataset.transform) != dict.fromkeys(spec.log1p_indices, "log1p")
        or cfg.dataset.stats != spec.codec.stats
        or cfg.ae.pix_channels != spec.physical_shape[-1]
        or cfg.ae.lat_channels != spec.latent_shape[0]
    ):
        raise ValueError(
            f"Codec {directory} does not match the Euler run configuration"
        )
    return cfg, directory


def shard_names(spec: DictConfig) -> list[str]:
    """Return the pinned inventory in the stock Well loader's file order."""
    return sorted(f"{spec.dataset}_{suffix}.hdf5" for suffix in spec.shards)


def planned_shards(spec: DictConfig) -> list[tuple[WellSplit, str]]:
    """Return the deterministic Slurm array mapping."""
    files = shard_names(spec)[: spec.cache.shards_per_split]
    return [(split, file) for split in SPLITS for file in files]


def _source_path(spec: DictConfig, split: str, name: str) -> Path:
    return Path(spec.well_base_path) / spec.dataset / "data" / split / name


def _provenance(spec: DictConfig, split: str, name: str) -> dict:
    path = _source_path(spec, split, name)
    stat = path.stat()
    count = spec.trajectories_per_shard[split]
    return {
        "format_version": 1,
        "dataset": spec.dataset,
        "flatiron_source": spec.flatiron_source,
        "split": split,
        "source_shard": str(path.resolve()),
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        # Cache row i is source trajectory i, starting at zero.
        "source_trajectory_range": [
            0,
            min(count, spec.cache.trajectory_limit or count),
        ],
        "autocast_revision": spec.autocast_revision,
        "lola_revision": spec.lola_revision,
        "cache_backend": spec.cache.backend,
        "codec": spec.codec.name,
        "codec_archive_sha256": spec.codec.archive_sha256,
        "codec_state_sha256": spec.codec.state_sha256,
        "codec_config_sha256": spec.codec.config_sha256,
        "fields": list(spec.fields),
        "log1p_indices": list(spec.log1p_indices),
        "stats": OmegaConf.to_container(spec.codec.stats),
        "conditioning": list(spec.conditioning),
        "frame_stride": spec.cache.frame_stride,
        "window_stride": spec.cache.window_stride,
        "physical_shape_hwc": list(spec.physical_shape),
        "latent_shape_tchw": [spec.frames, *spec.latent_shape],
        "augmentation": [],
    }


def _validate_shard(path: Path, provenance: dict) -> None:
    with h5py.File(path, "r") as handle:
        if json.loads(handle.attrs.get("provenance", "{}")) != provenance:
            raise ValueError(f"Cache provenance differs at {path}")
        count = provenance["source_trajectory_range"][1]
        for name, shape in (
            ("state", (count, *provenance["latent_shape_tchw"])),
            ("label", (count, len(provenance["conditioning"]))),
        ):
            dataset = handle[name]
            if (
                not isinstance(dataset, h5py.Dataset)
                or dataset.shape != shape
                or dataset.dtype != np.dtype("float32")
            ):
                raise ValueError(f"Invalid {name} shape or dtype in {path}")


def verify_cache(spec: DictConfig) -> None:
    """Require exactly the planned complete MiniWell shards before training."""
    cache = Path(spec.cache.directory)
    plan = planned_shards(spec)
    for split in SPLITS:
        expected = {name for part, name in plan if part == split}
        actual = {
            path.name
            for pattern in ("*.h5", "*.hdf5")
            for path in (cache / split).glob(pattern)
        }
        if actual != expected:
            raise ValueError(
                f"{split}: missing shards {sorted(expected - actual)}, "
                f"unexpected shards {sorted(actual - expected)}"
            )
    for split, name in plan:
        _validate_shard(cache / split / name, _provenance(spec, split, name))
    print(f"Verified {len(plan)} completed cache shards at {cache}", flush=True)


def _load_well_shard(spec: DictConfig, split: WellSplit, name: str) -> WellDataset:
    dataset = WellDataset(
        well_base_path=spec.well_base_path,
        well_dataset_name=spec.dataset,
        well_split_name=split,
        include_filters=[name],
        n_steps_input=spec.frames,
        n_steps_output=0,
        use_normalization=False,
        cache_small=False,
        return_grid=False,
        boundary_return_type="padding",
    )
    if (
        [Path(path).name for path in dataset.files_paths] != [name]
        or len(dataset) != spec.trajectories_per_shard[split]
        or dataset.n_steps_per_trajectory != [spec.frames]
        or tuple(dataset.metadata.spatial_resolution) != tuple(spec.physical_shape[:-1])
        or dataset.metadata.constant_scalar_names != ["gamma"]
        or {kind.upper() for kind in dataset.metadata.boundary_condition_types}
        != {spec.boundary_condition}
    ):
        raise ValueError(f"Unexpected canonical Euler metadata for {split}/{name}")
    return dataset


def _require_native_field_order(spec: DictConfig, dataset: WellDataset) -> None:
    order = named_field_indices(dataset.metadata.field_names.values(), spec.fields)
    if order != list(range(len(spec.fields))):
        msg = "Expected Well's native channel order to match the LoLA codec"
        raise ValueError(msg)


def verify_ambient_source(spec: DictConfig) -> None:
    """Verify native field order and boundary metadata before ambient training."""
    for split, name in planned_shards(spec):
        _require_native_field_order(spec, _load_well_shard(spec, split, name))


def _write_shard(
    handle: h5py.File,
    spec: DictConfig,
    provenance: dict,
    dataset: WellDataset,
    cfg: DictConfig,
    directory: Path,
    device: torch.device,
) -> None:
    """Write all source rows and frames with the verified frozen encoder."""
    latent_shape = tuple(spec.latent_shape)
    count = provenance["source_trajectory_range"][1]
    state = handle.create_dataset(
        "state",
        (count, spec.frames, *latent_shape),
        dtype="f4",
        chunks=(1, 1, *latent_shape),
        fletcher32=True,
    )
    labels = handle.create_dataset("label", (count, len(spec.conditioning)), dtype="f4")
    order = named_field_indices(dataset.metadata.field_names.values(), spec.fields)
    mean = torch.tensor(spec.codec.stats.mean, device=device)
    std = torch.tensor(spec.codec.stats.std, device=device)
    torch.set_float32_matmul_precision("high")
    codec_kwargs = cast("dict[str, Any]", OmegaConf.to_container(cfg.ae, resolve=True))
    model = get_autoencoder(**codec_kwargs)
    model.load_state_dict(
        torch.load(directory / "state.pth", map_location="cpu", weights_only=True)
    )
    model.eval().requires_grad_(False).to(device)
    boundary_code = BoundaryCondition[spec.boundary_condition].value
    with torch.inference_mode():
        for row in range(count):
            sample = dataset[row]
            raw = sample["input_fields"]
            bc = sample["boundary_conditions"]
            label = torch.cat((sample["constant_scalars"], bc.flatten()))
            if (
                raw.shape != (spec.frames, *spec.physical_shape)
                or label.shape != (len(spec.conditioning),)
                or not torch.isfinite(label).all()
                or not (bc == boundary_code).all()
            ):
                raise ValueError(
                    f"Invalid input at {provenance['source_shard']} row {row}"
                )
            labels[row] = label.numpy()
            for start in range(0, spec.frames, spec.cache.encode_chunk_size):
                chunk = raw[start : start + spec.cache.encode_chunk_size].to(device)
                x = standardize_fields(chunk[..., order], mean, std, spec.log1p_indices)
                z = model.encode(rearrange(x, "t h w c -> t c h w")).float().cpu()
                if (
                    z.shape != (len(chunk), *latent_shape)
                    or not torch.isfinite(z).all()
                ):
                    raise ValueError(f"Invalid latents at row {row}, frame {start}")
                state[row, start : start + len(chunk)] = z.numpy()
    handle.attrs["provenance"] = json.dumps(provenance, sort_keys=True)


def _cache_with_upstream(
    spec: DictConfig,
    dataset: WellDataset,
    temp: Path,
    codec: Path,
    provenance: dict,
) -> None:
    """Run the pinned LoLA shard callback unchanged in an isolated run directory."""
    if spec.cache.trajectory_limit is not None:
        msg = "The upstream cache callback processes an entire source shard"
        raise ValueError(msg)
    _require_native_field_order(spec, dataset)
    checkout = Path(os.environ["EULER_LOLA_CHECKOUT"]).resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != spec.lola_revision:
        raise ValueError(
            f"Expected LoLA checkout {spec.lola_revision}, found {revision}"
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "experiments/cache_latents.py",
            "lola",
        ],
        check=True,
    )
    for name in ("config.yaml", "state.pth"):
        (temp.parent / name).symlink_to(codec.resolve() / name)
    config = OmegaConf.create(
        {
            "run": str(temp.parent),
            "server": {"datasets": spec.well_base_path},
            "dataset": {
                "stats": OmegaConf.to_container(spec.codec.stats),
                "transform": dict.fromkeys(spec.log1p_indices, "log1p"),
                "augment": [],
            },
            "repeat": 1,
        }
    )
    launch_config = temp.parent / "upstream.yaml"
    OmegaConf.save(config, launch_config)
    split = provenance["split"]
    name = temp.name
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "--project",
            str(checkout),
            "python",
            "-I",
            "-B",
            str(Path(__file__).with_name("upstream_shard.py").resolve()),
            str(checkout),
            str(launch_config),
            spec.dataset,
            split,
            name,
        ],
        env={
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(
                Path(os.environ.get("EULER_LOLA_ENV", checkout / ".venv")).resolve()
            ),
        },
        check=True,
    )
    output = temp.parent / "cache" / spec.dataset / split / name
    with h5py.File(output, "r+") as handle:
        label_dataset = handle["label"]
        state = handle["state"]
        if not isinstance(label_dataset, h5py.Dataset) or not isinstance(
            state, h5py.Dataset
        ):
            raise ValueError(f"Missing upstream state or label dataset in {output}")
        labels = np.asarray(label_dataset[:])
        boundary_code = BoundaryCondition[spec.boundary_condition].value
        if not np.isfinite(labels).all() or not (labels[:, 1:] == boundary_code).all():
            raise ValueError(f"Invalid upstream labels in {output}")
        for row in range(state.shape[0]):
            if not np.isfinite(np.asarray(state[row])).all():
                raise ValueError(f"Invalid upstream latents in {output}, row {row}")
        handle.attrs["provenance"] = json.dumps(provenance, sort_keys=True)
    output.rename(temp)


def cache_shard(spec: DictConfig, index: int, device: torch.device) -> None:
    """Publish a complete shard atomically, or reuse a matching completed shard."""
    plan = planned_shards(spec)
    if not 0 <= index < len(plan):
        raise ValueError(f"Expected shard index in [0, {len(plan)}), got {index}")
    if spec.cache.frame_stride != 1:
        msg = "Euler cache must preserve every source frame"
        raise ValueError(msg)
    split, name = plan[index]
    dest = Path(spec.cache.directory) / split / name
    provenance = _provenance(spec, split, name)
    if dest.exists():
        _validate_shard(dest, provenance)
        print(f"Reusing {dest}", flush=True)
        return

    dataset = _load_well_shard(spec, split, name)
    cfg, directory = load_codec_config(spec)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Temporary staging shares the destination filesystem and is invisible to
    # the MiniWell glob. A hard link publishes without clobbering a racing job.
    with TemporaryDirectory(
        dir=dest.parent, prefix=f".{name}.", suffix=".partial"
    ) as staging:
        temp = Path(staging).resolve() / name
        if spec.cache.backend == "autocast":
            with h5py.File(temp, "x") as handle:
                _write_shard(handle, spec, provenance, dataset, cfg, directory, device)
        elif spec.cache.backend == "upstream":
            _cache_with_upstream(spec, dataset, temp, directory, provenance)
        else:
            raise ValueError(f"Unknown cache backend {spec.cache.backend}")
        if _provenance(spec, split, name) != provenance:
            raise ValueError(f"Source changed while caching {split}/{name}")
        _validate_shard(temp, provenance)
        try:
            os.link(temp, dest)
        except FileExistsError:
            _validate_shard(dest, provenance)
    print(f"Ready {dest}", flush=True)
