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
    fresh = name == "autoencoder_diversity_400_fresh"
    if provide_paths and name.startswith("autoencoder_diversity_400"):
        overrides.append("ae_diversity_data_path=/preflight/ae-400x160")
        if not fresh:
            overrides.append("resume_from_checkpoint=/preflight/derived.ckpt")
    if distributed:
        overrides += [
            "+distributed=ddp_4gpu_slurm",
            f"trainer.max_time={'00:04:00:00' if fresh else '00:04:30:00'}",
            "++hydra.launcher.nodes=1",
            f"hydra.launcher.timeout_min={255 if fresh else 285}",
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


@pytest.mark.parametrize("distributed", [False, True])
def test_fresh_fit_matches_protocol_without_restoring_state(distributed):
    resumed = _config("autoencoder_diversity_400", distributed=distributed)
    fresh = _config("autoencoder_diversity_400_fresh", distributed=distributed)
    for key in ("model", "optimizer", "datamodule", "logging"):
        assert OmegaConf.to_container(fresh[key]) == OmegaConf.to_container(
            resumed[key]
        )
    assert fresh.resume_from_checkpoint is None
    assert not fresh.resume_weights_only
    assert not fresh.reset_resume_time_budget
    assert fresh.ae_diversity_start_step == 0
    assert resumed.ae_diversity_start_step == 29116
    assert fresh.trainer.max_steps == 77308
    assert fresh.trainer.max_time == "00:04:00:00"
    assert fresh.seed == resumed.seed == 42
    assert fresh.float32_matmul_precision == resumed.float32_matmul_precision
    control = instantiate(fresh.trainer.callbacks[0])
    assert control.expected_resume_step == 0
    assert control.stop_step == 77308
    # The cap and initial counter are the only differences in trainer settings.
    expected = OmegaConf.to_container(resumed.trainer, resolve=True)
    assert isinstance(expected, dict)
    expected["max_time"] = "00:04:00:00"
    expected["callbacks"][0]["expected_resume_step"] = 0
    assert OmegaConf.to_container(fresh.trainer, resolve=True) == expected


def test_fresh_preflight_requires_only_data_and_forbids_checkpoint(preflight):
    cfg = preflight.check_config(fresh=True)
    assert cfg.resume_from_checkpoint is None
    assert cfg.trainer.max_time == "00:04:00:00"
    assert instantiate(cfg.trainer.callbacks[0]).expected_resume_step == 0
    with pytest.raises(ValueError, match="must not be supplied a checkpoint"):
        preflight.check_config(checkpoint=Path("/any/checkpoint.ckpt"), fresh=True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("resume_from_checkpoint", "/old.ckpt"),
        ("resume_weights_only", True),
        ("reset_resume_time_budget", True),
        ("autoencoder_checkpoint", "/old.ckpt"),
        ("output.resume_from_checkpoint", "/old.ckpt"),
        ("output.resume_weights_only", True),
        ("output.reset_resume_time_budget", True),
    ],
)
def test_fresh_preflight_rejects_alternate_restore_options(preflight, key, value):
    cfg = preflight.check_config(fresh=True)
    OmegaConf.update(cfg, key, value, force_add=True)
    with pytest.raises(ValueError, match="Fresh AE control forbids"):
        preflight.check_fresh_start(cfg)


def test_fresh_preflight_rejects_nonzero_start_step(preflight):
    cfg = preflight.check_config(fresh=True)
    cfg.ae_diversity_start_step = 29116
    with pytest.raises(ValueError, match="global step zero"):
        preflight.check_fresh_start(cfg)


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
