"""Check the continuation contract without real data, models or GPU allocation."""

import importlib
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationToMissingValueError

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def study_import_path(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts/swe64s"))


@pytest.fixture
def preflight(study_import_path):  # noqa: ARG001
    return importlib.import_module("check_ae_diversity")


def _config(name, *, distributed=False, provide_paths=True):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{REPO_ROOT / 'local_hydra'}]",
    ]
    if provide_paths and name == "autoencoder_diversity_400":
        overrides += [
            "ae_diversity_data_path=/preflight/ae-400x160",
            "resume_from_checkpoint=/preflight/derived.ckpt",
        ]
    if distributed:
        overrides += [
            "+distributed=ddp_4gpu_slurm",
            "trainer.max_time=00:04:30:00",
            "++hydra.launcher.nodes=1",
            "hydra.launcher.timeout_min=285",
        ]
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "src/autocast/configs")
    ):
        return compose(config_name="autoencoder", overrides=overrides)


@pytest.mark.parametrize("distributed", [False, True])
def test_continuation_matches_model_optimizer_and_update_budget(distributed):
    original = _config("autoencoder")
    cfg = _config("autoencoder_diversity_400", distributed=distributed)
    assert OmegaConf.to_container(cfg.model) == OmegaConf.to_container(original.model)
    optimizer = OmegaConf.to_container(original.optimizer)
    assert isinstance(optimizer, dict)
    optimizer["scheduler"] = None
    assert OmegaConf.to_container(cfg.optimizer) == optimizer
    assert cfg.trainer.max_steps - 29116 == 48192
    assert cfg.trainer.max_time == "00:04:30:00"
    assert cfg.trainer.max_epochs == 512
    assert cfg.trainer.devices == 4
    assert cfg.trainer.num_nodes == cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.strategy == "ddp"
    assert cfg.trainer.gradient_clip_val == original.trainer.gradient_clip_val == 1.0
    assert cfg.trainer.limit_val_batches == 1.0
    assert cfg.trainer.val_check_interval == 1004
    assert cfg.trainer.check_val_every_n_epoch is None
    assert cfg.datamodule.batch_size == original.datamodule.batch_size == 16
    assert cfg.datamodule.autoencoder_mode
    assert cfg.datamodule.normalization_path == "/preflight/ae-400x160/stats.yml"
    assert not cfg.resume_weights_only
    assert cfg.reset_resume_time_budget
    assert cfg.logging.wandb.enabled
    assert cfg.seed == original.seed == 42
    assert cfg.float32_matmul_precision == original.float32_matmul_precision
    lr = instantiate(cfg.trainer.callbacks[0])
    assert type(lr).__name__ == "HistoricalAELearningRate"
    ema = cfg.trainer.callbacks[-1]
    assert ema._target_ == "autocast.callbacks.ema.EMACallback"
    assert ema.decay == 0.999


def test_continuation_requires_explicit_new_data_and_checkpoint():
    cfg = _config("autoencoder_diversity_400", provide_paths=False)
    assert OmegaConf.is_missing(cfg, "ae_diversity_data_path")
    with pytest.raises(InterpolationToMissingValueError):
        _ = cfg.datamodule.data_path
    assert OmegaConf.is_missing(cfg, "resume_from_checkpoint")


def test_readonly_preflight_composes_without_dataset(preflight):
    cfg = preflight.check_config()
    assert cfg.trainer.max_steps == 77308
    assert cfg.optimizer.scheduler is None


def test_checkpoint_preflight_rejects_changed_file(tmp_path, preflight):
    checkpoint = tmp_path / "derived.ckpt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes, never unpickled")
    record = {
        "source_checkpoint_sha256": preflight.SOURCE_SHA256,
        "derived_checkpoint_sha256": preflight.sha256(checkpoint),
        "weights_optimizer_and_ema_preserved": True,
    }
    checkpoint.with_suffix(".provenance.json").write_text(json.dumps(record))
    assert preflight.check_checkpoint(checkpoint) == record
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance or checksum"):
        preflight.check_checkpoint(checkpoint)
