"""Read-only preflight for the stochastic SWE64 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS = REPO_ROOT / "local_hydra/local_experiment/swe64s"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_configs() -> dict[str, DictConfig]:
    """Compose presets without datasets, checkpoints or model allocation."""
    configs = {}
    for name, top_level in (
        ("afcrps", "encoder_processor_decoder"),
        ("autoencoder", "autoencoder"),
        ("cache_latents", "encoder_processor_decoder"),
        ("flow_matching", "processor"),
    ):
        overrides = [
            f"local_experiment=swe64s/{name}",
            f"hydra.searchpath=[file://{REPO_ROOT / 'local_hydra'}]",
        ]
        if name == "cache_latents":
            overrides += [
                "autoencoder_checkpoint=/preflight/autoencoder.ckpt",
                "cache_latents.output_dir=/preflight/cache",
            ]
        if name == "flow_matching":
            overrides += ["datamodule.data_path=/preflight/cache"]
        with initialize_config_dir(
            version_base=None, config_dir=str(REPO_ROOT / "src/autocast/configs")
        ):
            cfg = compose(config_name=top_level, overrides=overrides)
        windows = tuple(
            cfg.datamodule[key] for key in ("n_steps_input", "n_steps_output", "stride")
        )
        _require(windows == (1, 4, 1), f"{name}: inconsistent training windows")
        if name != "cache_latents":
            is_autoencoder = name == "autoencoder"
            expected_interval = "epoch" if is_autoencoder else "time"
            expected_cosine_epochs = 512 if is_autoencoder else None
            expected_max_epochs = 512 if is_autoencoder else 1000000
            _require(
                cfg.optimizer.scheduler == "cosine"
                and cfg.optimizer.scheduler_interval == expected_interval
                and cfg.optimizer.cosine_epochs == expected_cosine_epochs,
                f"{name}: unexpected cosine schedule",
            )
            _require(
                cfg.trainer.max_time == "00:23:30:00"
                and cfg.trainer.max_epochs == expected_max_epochs
                and cfg.trainer.max_steps == -1,
                f"{name}: training budget changed",
            )
        configs[name] = cfg

    afcrps = configs["afcrps"]
    _require(not afcrps.model.encoder.with_constants, "afCRPS includes constants")
    _require(
        not afcrps.model.processor.include_global_cond, "afCRPS includes parameters"
    )
    _require(
        afcrps.model.processor.n_noise_channels == 1024
        and afcrps.model.processor.n_noise_input_channels == 1024,
        "afCRPS global noise dimensions changed",
    )
    _require(afcrps.model.loss_func.alpha == 0.95, "afCRPS alpha changed")
    _require(not afcrps.model.train_in_latent_space, "afCRPS is not a full-field fit")
    for component in ("encoder", "decoder"):
        for key in (
            "periodic",
            "pixel_shuffle",
            "patch_size",
            "hid_channels",
            "hid_blocks",
            "stride",
        ):
            _require(
                configs["autoencoder"].model[component][key]
                == configs["cache_latents"].model[component][key],
                f"AE/cache mismatch: {component}.{key}",
            )
    fm = configs["flow_matching"].model.processor
    _require(not fm.backbone.include_global_cond, "FM includes simulator parameters")
    _require(fm.backbone.include_time_embedding, "FM lost flow-time conditioning")
    _require(fm.backbone.patch_size == 1, "FM latent patches changed")
    _require(fm.integrator == "euler" and fm.flow_ode_steps == 50, "FM sampler changed")
    return configs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dataset(
    dataset: Path, manifest: dict[str, Any], *, verify_hashes: bool
) -> dict[str, Any]:
    """Check the fixed-regime data contract using CPU memory-mapped tensors."""
    generated = OmegaConf.load(dataset / "resolved_config.yaml")
    simulator = OmegaConf.to_container(generated.simulator, resolve=True)
    if not isinstance(simulator, dict):
        msg = "Expected simulator settings in the saved generator config"
        raise TypeError(msg)
    for key, expected in manifest["simulator"].items():
        actual = simulator.get(key)
        _require(actual == expected, f"Unexpected simulator setting: {key}={actual!r}")
    stats = OmegaConf.load(dataset / "stats.yml")
    _require(
        list(stats.core_field_names) == manifest["field_names"], "Field order changed"
    )
    for field in manifest["field_names"]:
        scale = float(stats.stats.std[field])
        _require(math.isfinite(scale) and scale > 0, f"Invalid normalization: {field}")

    shapes = {}
    for split, count in manifest["splits"].items():
        path = dataset / split / "data.pt"
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        fields = payload.get("data")
        if not isinstance(fields, torch.Tensor):
            msg = f"Missing {split} field tensor"
            raise TypeError(msg)
        expected_shape = (count, manifest["frames"], *manifest["resolution"], 3)
        _require(tuple(fields.shape) == expected_shape, f"Unexpected {split} shape")
        _require(fields.dtype == torch.float32, f"Unexpected {split} dtype")
        scalars = payload.get("constant_scalars")
        if not isinstance(scalars, torch.Tensor):
            msg = f"Missing {split} amp metadata"
            raise TypeError(msg)
        _require(tuple(scalars.shape) == (count, 1), f"Unexpected {split} scalar shape")
        _require(
            bool(torch.allclose(scalars, torch.full_like(scalars, 0.1))), "Amp varies"
        )
        if verify_hashes:
            _require(
                _sha256(path) == manifest["sha256"][f"{split}/data.pt"],
                f"{split}: hash mismatch",
            )
        shapes[split] = list(fields.shape)
    return {"path": str(dataset), "shapes": shapes, "hashes_verified": verify_hashes}


def check_cache(
    cache: Path, manifest: dict[str, Any], expected: DictConfig
) -> dict[str, Any]:
    """Check full latent trajectories and the saved encoder/data configuration."""
    metadata = json.loads((cache / "metadata.json").read_text())
    ae_cfg = OmegaConf.load(cache / "autoencoder_config.yaml")
    _require(
        Path(ae_cfg.datamodule.data_path).name == manifest["directory"],
        "Cache dataset mismatch",
    )
    _require(bool(ae_cfg.datamodule.use_normalization), "Cache is not normalized")
    _require(
        list(OmegaConf.load(ae_cfg.datamodule.normalization_path).core_field_names)
        == manifest["field_names"],
        "Cache normalization field order changed",
    )
    _require(
        bool(ae_cfg.model.encoder.periodic) and bool(ae_cfg.model.decoder.periodic),
        "Cache autoencoder is not periodic",
    )
    for key in ("n_steps_input", "n_steps_output", "stride", "start_frame"):
        _require(
            ae_cfg.datamodule[key] == expected.datamodule[key],
            f"Cache window mismatch: {key}",
        )
    for component in ("encoder", "decoder"):
        for key in (
            "periodic",
            "pixel_shuffle",
            "patch_size",
            "hid_channels",
            "hid_blocks",
            "stride",
        ):
            _require(
                ae_cfg.model[component][key] == expected.model[component][key],
                f"Cache architecture mismatch: {component}.{key}",
            )
    encoder = expected.model.encoder
    compression = int(encoder.patch_size) * int(encoder.stride) ** (
        len(encoder.hid_channels) - 1
    )
    expected_shape = (
        manifest["frames"],
        *(size // compression for size in manifest["resolution"]),
        int(encoder.out_channels),
    )
    _require(metadata["latent_channels"] == 8, "Cache latent channel count changed")
    for split, count in manifest["splits"].items():
        files = sorted((cache / split).glob("traj_*.pt"))
        _require(len(files) == count, f"{split}: cache file count mismatch")
        _require(
            metadata["splits"][split]["num_trajectories"] == count,
            f"{split}: cache metadata count mismatch",
        )
        first = torch.load(files[0], map_location="cpu", mmap=True, weights_only=True)
        _require(
            tuple(first["encoded_fields"].shape) == expected_shape,
            f"{split}: expected full {expected_shape} latent trajectory",
        )
    return {
        "path": str(cache),
        "splits": manifest["splits"],
        "sample_shape": list(expected_shape),
    }


def main() -> None:
    """Print a preflight summary; never generate data, train or modify inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-only", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--verify-hashes", action="store_true", help="Read all raw split bytes."
    )
    parser.add_argument(
        "--cache", type=Path, help="Also verify an existing latent cache."
    )
    args = parser.parse_args()
    if args.configs_only and (args.verify_hashes or args.cache):
        parser.error("--configs-only cannot be combined with data/cache checks")
    configs = check_configs()
    manifest = json.loads((PRESETS / "dataset.json").read_text())
    for name in ("afcrps", "autoencoder", "cache_latents"):
        _require(
            Path(configs[name].datamodule.data_path).name == manifest["directory"],
            f"{name}: config/manifest dataset mismatch",
        )
    result: dict[str, Any] = {
        "presets": list(configs),
        "training_executed": False,
        "fit_budgets": {
            name: {
                "max_time": cfg.trainer.max_time,
                "max_epochs": cfg.trainer.max_epochs,
                "scheduler_interval": cfg.optimizer.scheduler_interval,
                "cosine_epochs": cfg.optimizer.cosine_epochs,
            }
            for name, cfg in configs.items()
            if name != "cache_latents"
        },
    }
    if not args.configs_only:
        dataset = (
            args.dataset
            or Path(os.environ.get("AUTOCAST_DATASETS", REPO_ROOT / "datasets"))
            / manifest["directory"]
        )
        result["dataset"] = check_dataset(
            dataset.expanduser().resolve(), manifest, verify_hashes=args.verify_hashes
        )
        if args.cache:
            result["cache"] = check_cache(
                args.cache.expanduser().resolve(), manifest, configs["cache_latents"]
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
