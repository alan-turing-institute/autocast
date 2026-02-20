"""Train-then-evaluate encoder-processor-decoder in a single Hydra job."""

import logging
from pathlib import Path
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from autocast.scripts.eval.encoder_processor_decoder import run_evaluation
from autocast.scripts.train.encoder_processor_decoder import run_epd_training
from autocast.scripts.utils import get_default_config_path

log = logging.getLogger(__name__)


def _resolve_train_checkpoint(cfg: DictConfig, work_dir: Path) -> Path:
    train_eval_cfg = cfg.get("train_eval", {})
    checkpoint_override = train_eval_cfg.get("checkpoint")
    if checkpoint_override is not None:
        return Path(checkpoint_override).expanduser().resolve()

    output_cfg = cfg.get("output", {})
    explicit_checkpoint = output_cfg.get("checkpoint_path")
    if explicit_checkpoint is not None:
        return Path(explicit_checkpoint).expanduser().resolve()

    checkpoint_name = output_cfg.get(
        "checkpoint_name", "encoder_processor_decoder.ckpt"
    )
    return (work_dir / checkpoint_name).resolve()


def _apply_eval_overrides(cfg: DictConfig) -> DictConfig:
    train_eval_cfg = cast(DictConfig, cfg.get("train_eval", {}))
    eval_overrides = list(train_eval_cfg.get("eval_overrides", []))
    if not eval_overrides:
        return cast(DictConfig, cfg)

    merged = OmegaConf.merge(cfg)
    for override in eval_overrides:
        force_add = override.startswith("+")
        normalized = override[1:] if force_add else override
        if "=" not in normalized:
            msg = f"Invalid eval override (expected key=value): {override}"
            raise ValueError(msg)

        key, raw_value = normalized.split("=", 1)
        parsed = OmegaConf.from_dotlist([f"v={raw_value}"])
        OmegaConf.update(
            merged,
            key,
            parsed.v,
            merge=True,
            force_add=force_add,
        )

    if not isinstance(merged, DictConfig):
        msg = "Expected DictConfig after applying eval overrides"
        raise TypeError(msg)
    return merged


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for single-job train→eval flow."""
    logging.basicConfig(level=logging.INFO)

    work_dir = Path.cwd()
    cfg = run_epd_training(cfg, work_dir=work_dir)

    checkpoint_path = _resolve_train_checkpoint(cfg, work_dir)

    train_eval_cfg = cast(DictConfig, cfg.get("train_eval", {}))
    eval_subdir = train_eval_cfg.get("eval_subdir", "eval")
    eval_work_dir = (work_dir / eval_subdir).resolve()
    eval_work_dir.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        eval_cfg_section = cfg.get("eval")
        if not isinstance(eval_cfg_section, DictConfig):
            eval_cfg_section = OmegaConf.create({})
            cfg["eval"] = eval_cfg_section
        eval_cfg_section["checkpoint"] = str(checkpoint_path)

        batch_indices = train_eval_cfg.get("batch_indices")
        if batch_indices is not None:
            eval_cfg_section["batch_indices"] = batch_indices

        video_dir = train_eval_cfg.get("video_dir")
        if video_dir is not None:
            eval_cfg_section["video_dir"] = video_dir

    eval_cfg = _apply_eval_overrides(cfg)
    run_evaluation(eval_cfg, work_dir=eval_work_dir)


if __name__ == "__main__":
    main()
