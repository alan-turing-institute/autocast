"""Check that the production wrapper cannot submit during ordinary previews."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "slurm_scripts/ablations/submit_diffusion_extensions.py"
    )
    spec = importlib.util.spec_from_file_location("diffusion_extension_launcher", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    launches = [
        (
            tmp_path / "outputs/group" / f"diff_{dataset}64_diffusion_vit_abc1234_new",
            [
                "uv",
                "run",
                "autocast",
                "processor",
                "--mode",
                "slurm",
                f"dataset={dataset}",
            ],
        )
        for dataset in ["ad", "gs", "gpe"]
    ]
    monkeypatch.setattr(module, "_prepare_launches", lambda _group: launches)
    calls = []

    def record_command(command, *, check):
        assert check is True
        calls.append(command)

    def git_output(command, *, text):
        assert text is True
        return "" if command[1] == "status" else "abc1234\n"

    monkeypatch.setattr(module.subprocess, "run", record_command)
    monkeypatch.setattr(module.subprocess, "check_output", git_output)
    return SimpleNamespace(module=module, launches=launches, calls=calls)


def test_default_only_previews(launcher):
    launcher.module.main([])

    assert launcher.calls == [
        [*command, "--dry-run"] for _, command in launcher.launches
    ]


def test_submit_previews_all_runs_before_submitting(launcher):
    launcher.module.main(["--submit"])

    commands = [command for _, command in launcher.launches]
    assert launcher.calls == [
        *[[*command, "--dry-run"] for command in commands],
        *commands,
    ]


def test_existing_second_run_prevents_all_submissions(launcher):
    workdir, _ = launcher.launches[1]
    workdir.with_name("diff_gs64_diffusion_vit_oldhash_previous").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        launcher.module.main(["--submit"])

    assert launcher.calls == []


def test_dirty_checkout_prevents_submission(launcher, monkeypatch):
    def dirty_status(_command, *, text):
        assert text is True
        return " M config.yaml\n"

    monkeypatch.setattr(launcher.module.subprocess, "check_output", dirty_status)
    with pytest.raises(SystemExit, match="2"):
        launcher.module.main(["--submit"])

    assert launcher.calls == []


@pytest.fixture
def measured_config(launcher, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    alias = tmp_path / "original-cache"
    alias.symlink_to(cache, target_is_directory=True)
    checkpoint = tmp_path / "timing.ckpt"
    checkpoint.touch()
    (tmp_path / ".hydra").mkdir()
    timed = OmegaConf.create(
        {
            "model": {"val_metrics": []},
            "datamodule": {"data_path": str(alias), "batch_size": 256},
            "optimizer": {"cosine_epochs": 5},
            "trainer": {
                "max_epochs": 5,
                "max_time": None,
                "callbacks": [
                    {
                        "_target_": (
                            "autocast.callbacks.checkpoint.ProgressModelCheckpoint"
                        ),
                        "every_n_train_steps_fraction": 0.05,
                    }
                ],
            },
            "seed": 42,
            "float32_matmul_precision": "high",
        }
    )
    OmegaConf.save(timed, tmp_path / ".hydra/config.yaml")
    cfg = OmegaConf.merge(
        timed,
        {
            "datamodule": {"data_path": str(cache)},
            "optimizer": {"cosine_epochs": 2601},
            "trainer": {"max_epochs": 2601, "max_time": "00:23:59:00"},
            "logging": {"wandb": {"enabled": True}},
            "output": {"skip_test": True, "checkpoint_path": "processor.ckpt"},
        },
    )
    cfg.trainer.callbacks[0].every_n_epochs = 0
    monkeypatch.setattr(launcher.module, "compose", lambda **_kwargs: cfg)
    run = {
        "run_id": "latent_diffusion_ad",
        "epoch_budget": 2601,
        "production_budget_ready": True,
        "timing_checkpoint": str(checkpoint),
        "overrides": [],
    }
    return SimpleNamespace(run=run, cfg=cfg)


def test_equivalent_cache_alias_and_snapshot_default_are_accepted(
    launcher, measured_config
):
    launcher.module._validate_config(measured_config.run)


def test_different_existing_cache_is_rejected(launcher, measured_config, tmp_path):
    other_cache = tmp_path / "other-cache"
    other_cache.mkdir()
    measured_config.cfg.datamodule.data_path = str(other_cache)

    with pytest.raises(ValueError, match="datamodule differs"):
        launcher.module._validate_config(measured_config.run)


def test_changed_snapshot_cadence_is_rejected(launcher, measured_config):
    measured_config.cfg.trainer.callbacks[0].every_n_train_steps_fraction = 0.1

    with pytest.raises(ValueError, match="trainer differs"):
        launcher.module._validate_config(measured_config.run)
