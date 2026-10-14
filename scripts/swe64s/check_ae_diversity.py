"""Read-only launch gate for the single AE trajectory-diversity continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ae_diversity import SOURCE_SHA256, SOURCE_STEP, STOP_STEP, sha256
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from prepare_ae_diversity_data import validate_data_view

REPO_ROOT = Path(__file__).resolve().parents[2]


def check_config(
    data: Path = Path("/preflight/ae400"),
    checkpoint: Path = Path("/preflight/derived.ckpt"),
) -> DictConfig:
    """Compose the production configuration without loading models or real data."""
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "src/autocast/configs")
    ):
        cfg = compose(
            config_name="autoencoder",
            overrides=[
                "local_experiment=swe64s/autoencoder_diversity_400",
                f"hydra.searchpath=[file://{REPO_ROOT / 'local_hydra'}]",
                f"ae_diversity_data_path={data}",
                f"resume_from_checkpoint={checkpoint}",
                "+distributed=ddp_4gpu_slurm",
                "trainer.max_time=00:04:30:00",
            ],
        )
    # Import/construct the study callback now rather than discovering a missing
    # PYTHONPATH after a GPU allocation has started.
    instantiate(cfg.trainer.callbacks[0])
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
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if not args.configs_only and (args.data is None or args.checkpoint is None):
        parser.error("Supply --data and --checkpoint, or use --configs-only")
    cfg = (
        check_config()
        if args.configs_only
        else check_config(args.data, args.checkpoint)
    )
    result = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "resume_step": SOURCE_STEP,
        "stop_step": STOP_STEP,
        "additional_updates": STOP_STEP - SOURCE_STEP,
        "submitted": False,
    }
    if not args.configs_only:
        result["data"] = validate_data_view(args.data)
        result["checkpoint"] = check_checkpoint(args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
