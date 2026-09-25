"""Tests for the conformal-paper prediction driver's compose-only check.

``scripts/run_conformal_paper.py`` is not part of the ``autocast`` package, so it is
loaded from its file path rather than imported by module name (same pattern as
``tests/scripts/test_diffusion_extension_launcher.py``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_conformal_paper.py"
    spec = importlib.util.spec_from_file_location("run_conformal_paper", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses.dataclass() looks the defining module up in sys.modules while
    # the class body executes, so the module must be registered before running it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r = _load_module()


def _write_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content))


@pytest.fixture
def fm_layout(tmp_path, monkeypatch):
    """A minimal FM model + system, wired into MODELS/SYSTEMS, with every file on disk.

    Mirrors the real layout ``preflight_predict`` walks: a processor run with a
    resolved eval config, a companion autoencoder run, and paper/new dataset
    directories each with three splits plus stats.
    """
    root = tmp_path
    run = "2000-01-01/fm_test_run"
    ae_run = "2000-01-01/ae_test_run"
    paper_dataset = "paper_system_abc"
    new_dataset = "fresh_oos/test_sys"

    _write_yaml(
        root / "outputs" / run / "eval" / "resolved_eval_config.yaml",
        {
            "eval": {
                "benchmark": {"enabled": True},
                "benchmark_rollout": {"enabled": True},
            },
            "logging": {"wandb": {"enabled": True}},
            "trainer": {"devices": 4},
        },
    )
    (root / "outputs" / run / "processor.ckpt").parent.mkdir(
        parents=True, exist_ok=True
    )
    (root / "outputs" / run / "processor.ckpt").write_bytes(b"fake-ckpt")

    _write_yaml(
        root / "outputs" / ae_run / "resolved_autoencoder_config.yaml",
        {
            "datamodule": {
                "_target_": "autocast.data.datamodule.SpatioTemporalDataModule",
                "batch_size": 16,
            }
        },
    )
    (root / "outputs" / ae_run / "autoencoder.ckpt").write_bytes(b"fake-ae-ckpt")

    for dataset in (paper_dataset, new_dataset):
        for split in ("train", "valid", "test"):
            (root / "datasets" / dataset / split / "data.pt").parent.mkdir(
                parents=True, exist_ok=True
            )
            (root / "datasets" / dataset / split / "data.pt").write_bytes(b"data")
        (root / "datasets" / dataset / "stats.yml").write_text("mean: 0\nstd: 1\n")

    monkeypatch.setitem(r.SYSTEMS, "test_sys", r.System(paper_dataset, new_dataset))
    monkeypatch.setitem(
        r.MODELS,
        "fm_test",
        r.Model("test_sys", run, "processor.ckpt", "eval", ae_run),
    )
    return root


def test_preflight_predict_composes_and_reports_no_missing_files(fm_layout):
    cfg = r.preflight_predict(fm_layout, "fm_test", [r.PredictionSet.PAPER_TEST])

    assert isinstance(cfg, DictConfig)
    run = r.MODELS["fm_test"].run
    assert cfg.eval.checkpoint == str(fm_layout / "outputs" / run / "processor.ckpt")
    assert cfg.eval.dump_split == "test"
    assert cfg.eval.devices == 1
    # eval.checkpoint composing does not launch anything: no subprocess module used.
    assert cfg.logging.wandb.enabled is False


def test_preflight_predict_covers_every_requested_set(fm_layout):
    # Should not raise for any of the three sets, individually or together.
    cfg = r.preflight_predict(
        fm_layout,
        "fm_test",
        [r.PredictionSet.NEW, r.PredictionSet.PAPER_VALID, r.PredictionSet.PAPER_TEST],
    )
    # The last set processed (paper_test) is what's returned.
    assert cfg.eval.dump_split == "test"


def test_preflight_predict_missing_checkpoint_raises_with_path(fm_layout):
    run = r.MODELS["fm_test"].run
    ckpt = fm_layout / "outputs" / run / "processor.ckpt"
    ckpt.unlink()

    with pytest.raises(FileNotFoundError, match=str(ckpt)):
        r.preflight_predict(fm_layout, "fm_test", [r.PredictionSet.PAPER_TEST])


def test_preflight_predict_missing_dataset_split_raises_with_path(fm_layout):
    paper_dataset = r.SYSTEMS["test_sys"].paper_dataset
    missing = fm_layout / "datasets" / paper_dataset / "test" / "data.pt"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match=str(missing)):
        r.preflight_predict(fm_layout, "fm_test", [r.PredictionSet.PAPER_TEST])


def test_preflight_predict_missing_autoencoder_checkpoint_raises(fm_layout):
    ae_run = r.MODELS["fm_test"].autoencoder_run
    ae_ckpt = fm_layout / "outputs" / ae_run / "autoencoder.ckpt"
    ae_ckpt.unlink()

    with pytest.raises(FileNotFoundError, match=str(ae_ckpt)):
        r.preflight_predict(fm_layout, "fm_test", [r.PredictionSet.PAPER_TEST])


def test_cfg_job_prints_composed_config_without_running_predict(
    fm_layout, monkeypatch, capsys
):
    """``--cfg job`` must not call ``predict``, which launches the real subprocess."""

    def _forbidden(*args, **kwargs):
        msg = "predict() must not run under --cfg job"
        raise AssertionError(msg)

    monkeypatch.setattr(r, "predict", _forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_conformal_paper.py",
            "--root",
            str(fm_layout),
            "predict",
            "--model",
            "fm_test",
            "--set",
            "paper_test",
            "--cfg",
            "job",
        ],
    )

    r.main()

    out = capsys.readouterr().out
    assert "checkpoint:" in out
    assert "dump_split: test" in out


def test_cfg_job_writes_to_preflight_cfg_out_env_var(fm_layout, monkeypatch, tmp_path):
    cfg_out = tmp_path / "composed.yaml"
    monkeypatch.setenv("AUTOCAST_PREFLIGHT_CFG_OUT", str(cfg_out))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_conformal_paper.py",
            "--root",
            str(fm_layout),
            "predict",
            "--model",
            "fm_test",
            "--set",
            "paper_test",
            "--cfg",
            "job",
        ],
    )

    r.main()

    assert cfg_out.is_file()
    assert "dump_split: test" in cfg_out.read_text()


def test_cfg_job_refuses_non_predict_stage(fm_layout, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_conformal_paper.py",
            "--root",
            str(fm_layout),
            "calibrate",
            "--model",
            "fm_test",
            "--cfg",
            "job",
        ],
    )

    with pytest.raises(ValueError, match="predict stage"):
        r.main()


def test_max_traj_refuses_calibration_stages(fm_layout, monkeypatch):
    for stage in ("calibrate", "sufficiency"):
        argv = ["run_conformal_paper.py", "--root", str(fm_layout), stage]
        argv += ["--model", "fm_test", "--max-traj", "2"]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(ValueError, match="smoke-test predictions"):
            r.main()
