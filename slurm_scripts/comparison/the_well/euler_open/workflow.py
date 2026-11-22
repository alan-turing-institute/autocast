"""Inspect, stage and evaluate the Euler-open LoLA experiment."""

import argparse
import random
from pathlib import Path

import torch
import yaml
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from autocast.scripts.utils import get_default_config_path
from autocast.types import Batch
from slurm_scripts.comparison.the_well.euler_open.cache import (
    cache_shard,
    load_codec_config,
    planned_shards,
    shard_names,
    verify_ambient_source,
    verify_cache,
)
from slurm_scripts.comparison.the_well.rayleigh_benard.paired_test_windows import (
    make_selection,
    run,
)


def experiment_config(profile: str, *, joint: bool = False) -> DictConfig:
    """Compose the same run configuration used by Hydra processor training."""
    dataset = "euler_all" if joint else "euler_open"
    preset = f"the_well/{dataset}/crps_spatiotemporal_latent"
    if profile == "smoke":
        preset += "_smoke"
    with initialize_config_dir(get_default_config_path(), version_base=None):
        cfg = compose(config_name="processor", overrides=[f"local_experiment={preset}"])
    OmegaConf.resolve(cfg)
    return cfg


def source_spec(cfg: DictConfig, source: str) -> DictConfig:
    """Select a canonical source while sharing the published codec."""
    if source == "open":
        return cfg.euler_open_lola
    spec = OmegaConf.merge(cfg.euler_open_lola, cfg.euler_periodic_lola)
    assert isinstance(spec, DictConfig)
    return spec


def verify_joint_sources(*, ambient: bool) -> None:
    """Verify each original source before combining its split with the other."""
    cfg = experiment_config("production", joint=True)
    for source, root in zip(
        ("open", "periodic"), cfg.datamodule.data_paths, strict=True
    ):
        spec = source_spec(cfg, source).copy()
        spec.well_base_path = cfg.euler_all_lola.well_base_path
        if ambient:
            verify_ambient_source(spec)
        else:
            spec.cache.directory = root
            spec.cache.backend = "upstream"
            spec.codec.directory = str(Path(root).parent.parent)
            load_codec_config(spec)
            # Build revisions remain in the shard receipts. A newer reader
            # still checks the raw shard, codec, transforms and cache layout.
            verify_cache(spec, accept_recorded_revision=True)


def paired_selection(spec: DictConfig) -> dict:
    """Choose a boundary-specific test cohort from the pinned file order."""
    files = dict.fromkeys(shard_names(spec), int(spec.trajectories_per_shard["test"]))
    seed = spec.paired_test.seed
    indices = sorted(
        random.Random(seed).sample(
            range(sum(files.values())), spec.paired_test.trajectories
        )
    )
    return {
        "dataset": spec.dataset,
        "fields": list(spec.fields),
        **make_selection(
            {"files": files, "indices": indices},
            seed,
            frames=spec.frames,
            stride=spec.cache.window_stride,
            windows_per_trajectory=spec.paired_test.windows_per_trajectory,
        ),
    }


def configure_encode_once(cfg: DictConfig) -> DictConfig:
    """Resolve the verified all-Euler codec against the open-only raw loader."""
    from autocast.scripts.eval.encoder_processor_decoder import (  # noqa: PLC0415
        _build_lola_raw_datamodule_config,
        _inject_lola_autoencoder_components,
    )

    spec = cfg.euler_open_lola
    if (
        cfg.datamodule.n_steps_input,
        cfg.datamodule.n_steps_output,
        cfg.datamodule.stride,
    ) != (1, 4, spec.cache.window_stride):
        msg = "Checkpoint configuration differs from the Euler window protocol"
        raise ValueError(msg)
    codec, directory = load_codec_config(spec)
    raw = _build_lola_raw_datamodule_config(codec, directory, cfg.datamodule)
    raw.well_dataset_name = spec.dataset
    raw.well_base_path = spec.well_base_path
    raw.normalization_path = str(
        Path(spec.well_base_path) / spec.dataset / "stats.yaml"
    )
    raw.return_grid = False
    cfg = _inject_lola_autoencoder_components(cfg, codec, directory)
    cfg.model.freeze_encoder_decoder = True
    cfg.model.encoder.log1p_indices = spec.log1p_indices
    cfg.model.decoder.log1p_indices = spec.log1p_indices
    cfg.model.decoder.noisy = False
    cfg.datamodule = raw
    return cfg


def predict_window(model, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode each observed context once, then sample and decode the ensemble."""
    encoder, decoder = model.encoder_decoder.encoder, model.encoder_decoder.decoder
    latent, label = encoder.encode_with_cond(batch)
    if label is None:
        msg = "Euler forecasts require gamma and boundary labels"
        raise ValueError(msg)
    members = model.n_members
    z = model.processor.map(
        repeat(latent, "b ... -> (b m) ...", m=members),
        repeat(label, "b c -> (b m) c", m=members),
    )
    return (
        rearrange(decoder.decode(z), "(b m) ... -> b ... m", m=members),
        batch.output_fields,
    )


def _save_selection(spec: DictConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = yaml.safe_dump(paired_selection(spec), sort_keys=False)
    try:
        with path.open("x") as stream:
            stream.write(contents)
    except FileExistsError:
        if yaml.safe_load(path.read_text()) != paired_selection(spec):
            raise ValueError(
                f"Existing paired-window manifest differs: {path}"
            ) from None


def main() -> None:
    """Dispatch cache operations or run the shared paired evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("production", "smoke"), default="production"
    )
    parser.add_argument("--source", choices=("open", "periodic"), default="open")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("verify")
    commands.add_parser("verify-ambient")
    commands.add_parser("verify-joint-cache")
    commands.add_parser("verify-joint-ambient")
    cache = commands.add_parser("cache")
    cache.add_argument("index", type=int)
    select = commands.add_parser("windows")
    select.add_argument("manifest", type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--expected-step", type=int)
    args = parser.parse_args()

    if args.command == "evaluate":
        cfg = OmegaConf.load(args.config)
        if "euler_all_lola" in cfg or cfg.datamodule.get("data_paths"):
            parser.error(
                "Use the maintained evaluator with a per-boundary cache for joint Euler"
            )
        if args.source != "open":
            parser.error("Use the maintained evaluator for periodic Euler")
        if cfg.eval.mode not in {"encode_once", "ambient"}:
            msg = "Euler paired evaluation requires encode_once or ambient mode"
            raise ValueError(msg)
        _save_selection(cfg.euler_open_lola, args.manifest)
        args.members = cfg.eval.n_members
        run(
            args,
            configure_encode_once=configure_encode_once,
            predict_window=predict_window,
            max_windows=2 if args.profile == "smoke" else None,
            reference_diagnostics=True,
        )
        return

    if args.command in {"verify-joint-cache", "verify-joint-ambient"}:
        verify_joint_sources(ambient=args.command == "verify-joint-ambient")
        return

    spec = source_spec(experiment_config(args.profile), args.source)
    if args.command == "plan":
        for index, (split, name) in enumerate(planned_shards(spec)):
            print(f"{index}\t{split}\t{name}")
    elif args.command == "verify":
        verify_cache(spec)
    elif args.command == "verify-ambient":
        verify_ambient_source(spec)
    elif args.command == "cache":
        cache_shard(spec, args.index, torch.device("cuda"))
    else:
        _save_selection(spec, args.manifest)


if __name__ == "__main__":
    main()
