"""Read-only launch gate for fresh or resumed AE trajectory-diversity fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ae_diversity import SOURCE_SHA256, SOURCE_STEP, STOP_STEP, sha256
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from prepare_ae_diversity_data import validate_data_view
from prepare_ae_pretraining import validate_pretraining_data

REPO_ROOT = Path(__file__).resolve().parents[2]


def check_fresh_start(cfg: DictConfig) -> None:
    """Reject all supported AE checkpoint-loading shortcuts for a fresh fit."""
    forbidden = (
        "resume_from_checkpoint",
        "resume_weights_only",
        "reset_resume_time_budget",
        "autoencoder_checkpoint",
        "output.resume_from_checkpoint",
        "output.resume_weights_only",
        "output.reset_resume_time_budget",
    )
    for key in forbidden:
        if OmegaConf.select(cfg, key):
            msg = f"Fresh AE control forbids {key}"
            raise ValueError(msg)
    if cfg.trainer.callbacks[0].expected_resume_step != 0:
        msg = "Fresh AE control must start at global step zero"
        raise ValueError(msg)


def check_config(
    data: Path | None = None,
    checkpoint: Path | None = None,
    *,
    fresh: bool = False,
    pool_size: int = 400,
) -> DictConfig:
    """Compose the production configuration without loading models or real data."""
    if pool_size not in (400, 1000) or (pool_size == 1000 and not fresh):
        msg = "Select a 400-trajectory control or fresh 1000-trajectory pretraining"
        raise ValueError(msg)
    if fresh and checkpoint is not None:
        msg = "A fresh fit must not be supplied a checkpoint"
        raise ValueError(msg)
    preset = "autoencoder_diversity_400_fresh" if fresh else "autoencoder_diversity_400"
    if pool_size == 1000:
        preset = "autoencoder_pretraining_1000"
    data = data or Path(f"/preflight/ae{pool_size}")
    overrides = [
        f"local_experiment=swe64s/{preset}",
        f"hydra.searchpath=[file://{REPO_ROOT / 'local_hydra'}]",
        f"ae_diversity_data_path={data}",
        "+distributed=ddp_4gpu_slurm",
        f"trainer.max_time={'00:04:00:00' if fresh else '00:04:30:00'}",
    ]
    if not fresh:
        checkpoint = checkpoint or Path("/preflight/derived.ckpt")
        overrides.append(f"resume_from_checkpoint={checkpoint}")
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "src/autocast/configs")
    ):
        cfg = compose(
            config_name="autoencoder",
            overrides=overrides,
        )
    # Import/construct the study callback now rather than discovering a missing
    # PYTHONPATH after a GPU allocation has started.
    instantiate(cfg.trainer.callbacks[0])
    if fresh:
        check_fresh_start(cfg)
    return cfg


def check_checkpoint(path: Path) -> dict:
    """Check the derived checkpoint's hash and pinned source without unpickling."""
    path = path.expanduser().resolve()
    record = json.loads(path.with_suffix(".provenance.json").read_text())
    if (
        record.get("source_checkpoint_sha256") != SOURCE_SHA256
        or record.get("derived_checkpoint_sha256") != sha256(path)
        or record.get("weights_optimizer_and_ema_preserved") is not True
    ):
        msg = "Checkpoint provenance or checksum does not match the approved resume"
        raise ValueError(msg)
    return record


def main() -> None:
    """Verify inputs only; never generate data, train or submit a job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--pool-size", type=int, choices=(400, 1000), default=400)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.fresh and args.checkpoint is not None:
        parser.error("--fresh forbids --checkpoint")
    if args.pool_size == 1000 and not args.fresh:
        parser.error("--pool-size 1000 requires --fresh")
    if not args.configs_only and (
        args.data is None or (not args.fresh and args.checkpoint is None)
    ):
        parser.error("Supply --data and either --fresh or --checkpoint")
    cfg = check_config(
        args.data or Path(f"/preflight/ae{args.pool_size}"),
        args.checkpoint,
        fresh=args.fresh,
        pool_size=args.pool_size,
    )
    start_step = 0 if args.fresh else SOURCE_STEP
    result = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "start_mode": "fresh" if args.fresh else "continuation",
        "resume_step": start_step,
        "stop_step": STOP_STEP,
        "additional_updates": STOP_STEP - start_step,
        "pool_size": args.pool_size,
        "submitted": False,
    }
    if not args.configs_only:
        validate = (
            validate_pretraining_data if args.pool_size == 1000 else validate_data_view
        )
        result["data"] = validate(args.data)
        if not args.fresh:
            result["checkpoint"] = check_checkpoint(args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
