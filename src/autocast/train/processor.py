"""Train a ProcessorModel directly on encoded datasets."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import lightning as L
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch import nn

from autocast.logging import create_wandb_logger, maybe_watch_model
from autocast.models.processor import ProcessorModel
from autocast.types import EncodedBatch

log = logging.getLogger(__name__)


@dataclass
class ProcessorTrainingParams:
    """Resolved training knobs that may come from config or CLI overrides."""

    n_steps_input: int
    n_steps_output: int
    stride: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the processor training utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a ProcessorModel directly on encoded datasets with "
            "Hydra-configured processor components."
        )
    )
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--config-dir",
        "--config-path",
        dest="config_dir",
        type=Path,
        default=repo_root / "configs",
        help=(
            "Path to the Hydra config directory (accepts --config-path as an "
            "alias; defaults to <repo>/configs)."
        ),
    )
    parser.add_argument(
        "--config-name",
        default="processor",
        help="Hydra config name to compose (defaults to 'processor').",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra config overrides (e.g. trainer.max_epochs=5 "
            "logging.wandb.enabled=true)"
        ),
    )
    parser.add_argument(
        "--n-steps-input",
        type=int,
        default=None,
        help="Override training.n_steps_input (number of input time steps).",
    )
    parser.add_argument(
        "--n-steps-output",
        type=int,
        default=None,
        help="Override training.n_steps_output (number of target time steps).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override training stride (rollout interval between predictions).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Directory Lightning should treat as the default_root_dir "
            "(defaults to CWD)."
        ),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional explicit checkpoint filename; falls back to "
            "output.checkpoint_name from the config when omitted."
        ),
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip running trainer.test() after training completes.",
    )
    return parser.parse_args()


def compose_training_config(args: argparse.Namespace) -> DictConfig:
    """Compose the Hydra config prior to applying CLI-specific overrides."""
    config_dir = args.config_dir.resolve()
    overrides = args.overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=args.config_name, overrides=list(overrides))


def resolve_training_params(
    cfg: DictConfig, args: argparse.Namespace
) -> ProcessorTrainingParams:
    """Resolve training hyperparameters using the config plus CLI overrides."""
    training_cfg = cfg.get("training")
    n_steps_input_cfg = (
        training_cfg.get("n_steps_input", 1) if training_cfg is not None else 1
    )
    n_steps_output_cfg = (
        training_cfg.get("n_steps_output", 1) if training_cfg is not None else 1
    )
    stride_cfg = training_cfg.get("stride") if training_cfg is not None else None

    n_steps_input = args.n_steps_input or n_steps_input_cfg
    n_steps_output = args.n_steps_output or n_steps_output_cfg

    if stride_cfg in (None, "auto"):
        stride_cfg = n_steps_output
    stride_override = getattr(args, "stride", None)
    stride = stride_override or stride_cfg or n_steps_output
    if stride < 1:
        msg = "stride must be >= 1."
        raise ValueError(msg)

    if training_cfg is not None:
        with open_dict(training_cfg):
            training_cfg["stride"] = stride

    if n_steps_output < 1:
        msg = "n_steps_output must be >= 1 for processor training."
        raise ValueError(msg)

    return ProcessorTrainingParams(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        stride=stride,
    )


def update_data_cfg(cfg: DictConfig, n_steps_input: int, n_steps_output: int) -> None:
    """Update datamodule configuration to match resolved training step counts."""
    data_cfg = cfg.data
    with open_dict(data_cfg):
        data_cfg.n_steps_input = n_steps_input
        data_cfg.n_steps_output = n_steps_output


def _get_field(batch: EncodedBatch, primary: str, fallback: str):
    return (
        getattr(batch, primary) if hasattr(batch, primary) else getattr(batch, fallback)
    )


def prepare_encoded_datamodule(cfg: DictConfig):
    """Instantiate the encoded datamodule and inspect the first batch for sizing."""
    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="fit")
    batch = next(iter(datamodule.train_dataloader()))

    # For encoded datasets, use encoded_inputs and encoded_output_fields
    train_inputs = _get_field(batch, "encoded_inputs", "input_fields")
    train_outputs = _get_field(batch, "encoded_output_fields", "output_fields")
    global_cond = _get_field(batch, "encoded_inputs", "global_cond")

    # Shape is (B, T, *spatial, C) for channels-last encoded data
    in_channel_count = train_inputs.shape[-1]
    out_channel_count = train_outputs.shape[-1]
    inferred_n_steps_input = train_inputs.shape[1]
    inferred_n_steps_output = train_outputs.shape[1]
    global_cond_channels = global_cond.shape[-1] if global_cond is not None else None
    return (
        datamodule,
        in_channel_count,
        out_channel_count,
        global_cond_channels,
        inferred_n_steps_input,
        inferred_n_steps_output,
        train_inputs.shape,
        train_outputs.shape,
        batch,
    )


def normalize_processor_cfg(cfg: DictConfig) -> None:
    """Force config values into the shapes expected by processor classes."""
    processor_cfg = _model_cfg(cfg).get("processor")
    if processor_cfg is None:
        return
    tuple_fields = ("n_modes",)
    for field in tuple_fields:
        value = processor_cfg.get(field)
        if isinstance(value, ListConfig):
            with open_dict(processor_cfg):
                processor_cfg[field] = tuple(value)


def _model_cfg(cfg: DictConfig) -> DictConfig:
    """Return the nested model config when present, else the root config."""
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, DictConfig):
        return model_cfg
    return cfg


def _maybe_set(cfg_node: DictConfig | None, key: str, value: int) -> None:
    if cfg_node is None or key not in cfg_node:
        return
    current = cfg_node.get(key)
    if current not in (None, "auto"):
        return
    with open_dict(cfg_node):
        cfg_node[key] = value


def configure_processor_dimensions(
    cfg: DictConfig,
    in_channel_count: int,
    out_channel_count: int,
    global_cond_channels: int | None,
    n_steps_input: int,
    n_steps_output: int,
) -> None:
    """Populate missing dimension hints for processor modules."""
    model_cfg = _model_cfg(cfg)
    processor_cfg = model_cfg.get("processor")

    _maybe_set(processor_cfg, "in_channels", in_channel_count * n_steps_input)
    _maybe_set(processor_cfg, "out_channels", out_channel_count * n_steps_output)
    _maybe_set(processor_cfg, "n_steps_output", n_steps_output)
    _maybe_set(processor_cfg, "n_channels_out", out_channel_count)

    # Backbone expects per-timestep channel counts; it multiplies by n_steps internally
    backbone_cfg = processor_cfg.get("backbone") if processor_cfg else None
    _maybe_set(backbone_cfg, "in_channels", out_channel_count)
    _maybe_set(backbone_cfg, "out_channels", out_channel_count)
    _maybe_set(backbone_cfg, "cond_channels", in_channel_count)
    _maybe_set(backbone_cfg, "n_steps_input", n_steps_input)
    _maybe_set(backbone_cfg, "n_steps_output", n_steps_output)
    _maybe_set(
        backbone_cfg, "global_cond_channels", global_cond_channels
    ) if global_cond_channels is not None else None


def _ensure_output_path(path: Path, work_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (work_dir / path).resolve()


def instantiate_trainer(
    cfg: DictConfig,
    work_dir: Path,
    *,
    logger=None,
):
    """Instantiate the Lightning trainer with a concrete root directory."""
    return instantiate(
        cfg.trainer,
        default_root_dir=str(work_dir),
        logger=logger,
    )


def main() -> None:
    """CLI entrypoint for training the processor."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_training_config(args)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    model_cfg = cfg.get("model") or cfg

    wandb_logger, watch_cfg = create_wandb_logger(
        cfg.get("logging"),
        experiment_name=cfg.get("experiment_name", "processor"),
        job_type="train-processor",
        work_dir=work_dir,
        config={"hydra": resolved_cfg} if resolved_cfg is not None else None,
    )

    training_params = resolve_training_params(cfg, args)
    update_data_cfg(
        cfg,
        training_params.n_steps_input,
        training_params.n_steps_output,
    )

    L.seed_everything(cfg.get("seed", 42), workers=True)

    (
        datamodule,
        in_channel_count,
        out_channel_count,
        global_cond_channels,
        inferred_n_steps_input,
        inferred_n_steps_output,
        input_shape,
        output_shape,
        _example_batch,
    ) = prepare_encoded_datamodule(cfg)

    log.info("Detected input shape %s and output shape %s", input_shape, output_shape)
    if inferred_n_steps_input != training_params.n_steps_input:
        log.warning(
            "Datamodule produced %s input steps but training calls for %s; "
            "proceeding with inferred value.",
            inferred_n_steps_input,
            training_params.n_steps_input,
        )
    if inferred_n_steps_output != training_params.n_steps_output:
        log.warning(
            "Datamodule produced %s output steps but training calls for %s; "
            "proceeding with inferred value.",
            inferred_n_steps_output,
            training_params.n_steps_output,
        )

    configure_processor_dimensions(
        cfg,
        in_channel_count=in_channel_count,
        out_channel_count=out_channel_count,
        global_cond_channels=global_cond_channels,
        n_steps_input=inferred_n_steps_input,
        n_steps_output=inferred_n_steps_output,
    )
    normalize_processor_cfg(cfg)

    # Instantiate the processor
    processor = instantiate(model_cfg.processor)

    # Get model configuration
    learning_rate = model_cfg.get("learning_rate", 1e-3)
    loss_cfg = model_cfg.get("loss_func")
    loss_func = instantiate(loss_cfg) if loss_cfg is not None else nn.MSELoss()
    stride = training_params.stride

    # Create the ProcessorModel
    model = ProcessorModel(
        processor=processor,
        stride=stride,
        loss_func=loss_func,
        learning_rate=learning_rate,
    )

    maybe_watch_model(wandb_logger, model, watch_cfg)
    trainer = instantiate_trainer(cfg, work_dir, logger=wandb_logger)

    log.info("Starting training.")
    trainer.fit(model=model, datamodule=datamodule)

    if not args.skip_test:
        log.info("Running evaluation on the test split.")
        trainer.test(model=model, dataloaders=datamodule.test_dataloader())

    output_cfg = cfg.get("output") or {}
    checkpoint_name = output_cfg.get(
        "checkpoint_name",
        "processor.ckpt",
    )
    checkpoint_target = args.output_checkpoint or Path(checkpoint_name)
    checkpoint_path = _ensure_output_path(checkpoint_target, work_dir)
    trainer.save_checkpoint(checkpoint_path)
    log.info(
        "Saved processor checkpoint to %s",
        checkpoint_path,
    )

    if output_cfg.get("save_config", False):
        resolved_cfg_path = work_dir / "resolved_processor_config.yaml"
        OmegaConf.save(cfg, resolved_cfg_path)
        log.info("Wrote resolved config to %s", resolved_cfg_path)


if __name__ == "__main__":
    main()
