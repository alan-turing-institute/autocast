"""Tests that exercise real configs end-to-end."""

from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from autocast.scripts.setup import (
    setup_autoencoder_model,
    setup_epd_model,
    setup_processor_model,
)
from autocast.types import Batch, EncodedBatch


@pytest.fixture
def config_dir(REPO_ROOT: Path) -> str:
    return str(REPO_ROOT / "configs")


def _load_config(
    config_dir: str, config_name: str, overrides: list[str] | None = None
) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides or [])


def _wrap_model_config(model_cfg: DictConfig) -> DictConfig:
    if "model" in model_cfg:
        return model_cfg
    return OmegaConf.create({"model": model_cfg})


def _stats_from_batch(batch: Batch) -> dict:
    return {
        "channel_count": batch.input_fields.shape[-1],
        "n_steps_input": batch.input_fields.shape[1],
        "n_steps_output": batch.output_fields.shape[1],
        "input_shape": batch.input_fields.shape,
        "output_shape": batch.output_fields.shape,
        "example_batch": batch,
    }


def _stats_from_encoded_batch(batch: EncodedBatch) -> dict:
    return {
        "channel_count": batch.encoded_inputs.shape[-1],
        "n_steps_input": batch.encoded_inputs.shape[1],
        "n_steps_output": batch.encoded_output_fields.shape[1],
        "input_shape": batch.encoded_inputs.shape,
        "output_shape": batch.encoded_output_fields.shape,
        "example_batch": batch,
    }


def test_autoencoder_config_trainer_fit_smoke(
    config_dir: str, toy_batch: Batch, dummy_loader
):
    model_cfg = _load_config(config_dir, "model/autoencoder")
    cfg = _wrap_model_config(model_cfg)
    with open_dict(cfg):
        cfg.datamodule = {
            "n_steps_input": toy_batch.input_fields.shape[1],
            "n_steps_output": toy_batch.output_fields.shape[1],
        }
    stats = _stats_from_batch(toy_batch)
    model = setup_autoencoder_model(cfg, stats)

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=dummy_loader, val_dataloaders=dummy_loader)


def test_processor_config_training_step_smoke(config_dir: str):
    processor_cfg = _load_config(config_dir, "processor/flow_matching").processor
    with open_dict(processor_cfg):
        processor_cfg.backbone.include_global_cond = False
        processor_cfg.backbone.global_cond_channels = 0

    encoded_inputs = torch.randn(2, 2, 4, 4, 1)
    encoded_outputs = torch.randn(2, 2, 4, 4, 1)
    cfg = OmegaConf.create(
        {
            "model": {
                "processor": processor_cfg,
                "learning_rate": 1e-3,
                "loss_func": {"_target_": "torch.nn.MSELoss"},
            },
            "datamodule": {
                "stride": 1,
                "n_steps_input": encoded_inputs.shape[1],
                "n_steps_output": encoded_outputs.shape[1],
            },
        }
    )
    batch = EncodedBatch(
        encoded_inputs=encoded_inputs,
        encoded_output_fields=encoded_outputs,
        global_cond=None,
        encoded_info={},
    )
    stats = _stats_from_encoded_batch(batch)
    model = setup_processor_model(cfg, stats)

    loss = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0


def test_epd_config_forward_smoke(config_dir: str, toy_batch: Batch):
    model_cfg = _load_config(
        config_dir,
        "model/encoder_processor_decoder",
        overrides=[
            "encoder@model.encoder=dc",
            "decoder@model.decoder=dc",
            "processor@model.processor=flow_matching",
        ],
    )
    cfg = _wrap_model_config(model_cfg)
    with open_dict(cfg):
        cfg.datamodule = {
            "stride": 1,
            "n_steps_input": toy_batch.input_fields.shape[1],
            "n_steps_output": toy_batch.output_fields.shape[1],
        }
        cfg.model.processor.backbone.include_global_cond = False
        cfg.model.processor.backbone.global_cond_channels = 0
    stats = _stats_from_batch(toy_batch)
    model = setup_epd_model(cfg, stats)

    output = model(toy_batch)
    assert output.shape == toy_batch.output_fields.shape
