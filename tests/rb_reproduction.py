"""Test fixtures for repository-only RB reproduction tools."""

import argparse
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "slurm_scripts.comparison.the_well.rayleigh_benard"


def load_rb_tool(name):
    """Import a checkout tool without adding the root to the whole test session."""
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT))
        return import_module(f"{PACKAGE}.{name}")


def diagnostic_case(tmp_path, monkeypatch, *, latent=False):
    """Prepare small external artifacts for the real diagnostic preparation path."""
    runner = load_rb_tool("complete_diagnostics")
    paired = load_rb_tool("paired_test_windows")
    archive = tmp_path / "outputs/archived"
    archive.mkdir(parents=True)
    checkpoint = archive / "frozen.ckpt"
    checkpoint.write_bytes(b"frozen test weights")
    stats = archive / "stats.yaml"
    stats.write_text("mean: 1\nstd: 2\n")
    cohort = yaml.safe_load((runner.HERE / "test64_seed0.yaml").read_text())
    (archive / "window_selection.yaml").write_text(
        yaml.safe_dump(paired.make_selection(cohort))
    )
    entry = {
        "label": "Test",
        "mode": "encode_once" if latent else "ambient",
        "expected_step": 262144,
        "config": "archived/resolved_test_config.yaml",
        "checkpoint_sha256": runner.sha256(checkpoint),
    }
    model = {}
    codecs = {}
    if latent:
        entry["codec_sha256"] = {}
        for part in ("encoder", "decoder"):
            folder = archive / part
            folder.mkdir()
            codecs[part] = folder / "state.pth"
            codecs[part].write_bytes(part.encode())
            model[part] = {"runpath": str(folder)}
            entry["codec_sha256"][part] = runner.sha256(codecs[part])
        model["processor"] = {
            "_target_": "FlowMatchingProcessor",
            "integrator": "ab3",
            "flow_ode_steps": 16,
        }
    else:
        entry["normalization_sha256"] = runner.sha256(stats)
    config = {
        "eval": {
            "mode": entry["mode"],
            "use_ema": False,
            "checkpoint": str(checkpoint),
        },
        "datamodule": {
            "well_dataset_name": "rayleigh_benard",
            "use_normalization": not latent,
            "normalization_path": str(stats),
            "n_steps_input": 1,
            "n_steps_output": 4,
            "min_dt_stride": 1,
            "max_dt_stride": 1,
        },
        "model": model,
        "logging": {"wandb": {"enabled": True}},
    }
    (archive / "resolved_test_config.yaml").write_text(yaml.safe_dump(config))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"models": {"test_model": entry}}))
    monkeypatch.setattr(
        runner,
        "source_record",
        lambda root: {"source_commit": "abc123", "source_root": str(root)},
    )
    args = argparse.Namespace(
        recipe=recipe,
        outputs_root=tmp_path / "outputs",
        output=tmp_path / "campaign",
        raw_data_root=None,
        autoencoder_root=None,
        workers=2,
        horizon=180,
        account="test.account",
    )
    return SimpleNamespace(
        args=args, checkpoint=checkpoint, stats=stats, codecs=codecs, entry=entry
    )
