"""Tests for the manifest-driven seed-43 main-comparison reruns."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import yaml
from hydra import compose, initialize_config_dir


@pytest.fixture
def pipeline(REPO_ROOT: Path) -> ModuleType:
    """Load the campaign script as a testable module."""
    path = REPO_ROOT / "slurm_scripts/comparison/main_comparison_seed43/pipeline.py"
    spec = importlib.util.spec_from_file_location("seed43_pipeline", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cast(Any, module)._ACTIVE_MANIFEST = path.with_name("campaign.yaml")
    return module


@pytest.fixture
def campaign(pipeline: ModuleType) -> dict:
    """Load and validate the checked-in campaign manifest."""
    manifest = pipeline._load_yaml(pipeline._manifest_path())
    pipeline._validate_manifest(manifest)
    return manifest


def _fake_state(campaign: dict, tmp_path: Path) -> dict:
    runs = {}
    for dataset in campaign["datasets"]:
        runs[dataset] = {
            "crps_id": f"crps_{dataset}_abcdef0_1234567",
            "crps_dir": str(tmp_path / f"crps_{dataset}_abcdef0_1234567"),
            "cache_id": f"cache_{dataset}_abcdef0_1234567",
            "cache_dir": str(tmp_path / f"cache_{dataset}_abcdef0_1234567"),
            "fm_id": f"diff_{dataset}_abcdef0_1234567",
            "fm_dir": str(tmp_path / f"diff_{dataset}_abcdef0_1234567"),
        }
    return {"source_commit": "a" * 40, "runs": runs}


def _overrides(path: Path) -> list[str]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, list)
    return value


def _override_value(overrides: list[str], key: str) -> str:
    prefix = f"{key}="
    matches = [
        value.removeprefix(prefix) for value in overrides if value.startswith(prefix)
    ]
    assert len(matches) == 1
    return matches[0]


def test_state_uses_date_hash_uuid_names(
    pipeline: ModuleType, campaign: dict, REPO_ROOT: Path
) -> None:
    """All prepared artifacts retain the standard postprocessing identity."""
    source_commit = pipeline._source_commit()
    state = pipeline._build_state(campaign, source_commit)
    dated_root = REPO_ROOT / "outputs" / campaign["campaign"]["run_group"]
    suffix = re.compile(rf"_{source_commit[:7]}_[0-9a-f]{{7}}$")
    assert Path(state["campaign_dir"]).parent == dated_root
    assert suffix.search(Path(state["campaign_dir"]).name)
    for run in state["runs"].values():
        for key in ("crps_dir", "cache_dir", "fm_dir"):
            path = Path(run[key])
            assert path.parent == dated_root
            assert suffix.search(path.name)
        assert suffix.search(run["eval_crps_subdir"])
        assert suffix.search(run["eval_fm_subdir"])


def test_shared_crps_callbacks_equal_cns_rerun(
    pipeline: ModuleType, REPO_ROOT: Path
) -> None:
    """The reusable callback config is byte-for-byte equivalent as YAML data."""
    shared = pipeline._load_yaml(
        REPO_ROOT / "src/autocast/configs/trainer/crps_main_comparison_rerun.yaml"
    )
    cns = pipeline._load_yaml(
        REPO_ROOT
        / "local_hydra/local_experiment/reruns/main_comparison_cns_seed43"
        / "crps_vit_azula_large.yaml"
    )
    assert shared["callbacks"] == cns["trainer"]["callbacks"]


@pytest.mark.parametrize("dataset", ["ad", "gpe", "gs"])
def test_generator_overrides_match_published_dataset(
    campaign: dict, dataset: str
) -> None:
    """Only the deliberate top-level seed change is absent from generator args."""
    spec = campaign["datasets"][dataset]
    published = _overrides(
        Path(spec["published_dataset_dir"]) / ".hydra/overrides.yaml"
    )
    historical_seed = [item for item in published if item.startswith("seed=")]
    historical_generator = [item for item in published if not item.startswith("seed=")]
    assert spec["generator"]["overrides"] == historical_generator
    if dataset == "gpe":
        assert historical_seed == ["seed=42"]
    else:
        assert historical_seed == []


@pytest.mark.parametrize("dataset", ["ad", "gpe", "gs"])
def test_epoch_budgets_match_published_runs(campaign: dict, dataset: str) -> None:
    """The manifest copies both cosine schedules from the published references."""
    spec = campaign["datasets"][dataset]
    crps_overrides = _overrides(
        Path(spec["reference_crps_run"]) / ".hydra/overrides.yaml"
    )
    fm_overrides = _overrides(Path(spec["reference_fm_run"]) / ".hydra/overrides.yaml")
    assert (
        int(_override_value(crps_overrides, "optimizer.cosine_epochs"))
        == spec["crps_epochs"]
    )
    assert (
        int(_override_value(fm_overrides, "optimizer.cosine_epochs"))
        == spec["fm_epochs"]
    )
    expected_quarter = spec["fm_epochs"] // 4
    assert int(_override_value(fm_overrides, "trainer.callbacks.0.every_n_epochs")) == (
        expected_quarter
    )


@pytest.mark.parametrize("dataset", ["ad", "gpe", "gs"])
def test_crps_composition_preserves_cns_callbacks(
    pipeline: ModuleType,
    campaign: dict,
    tmp_path: Path,
    REPO_ROOT: Path,
    dataset: str,
) -> None:
    """Each CRPS repeat composes the exact successful CNS callback policy."""
    state = _fake_state(campaign, tmp_path)
    _, overrides = pipeline._training_overrides(campaign, state, dataset, "crps")
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "src/autocast/configs"),
    ):
        cfg = compose(config_name="encoder_processor_decoder", overrides=overrides)

    callbacks = cfg.trainer.callbacks
    targets = [callback._target_ for callback in callbacks]
    assert len(callbacks) == 13
    assert targets[-2:] == [
        "autocast.callbacks.metrics.ValidationMetricPlotCallback",
        "autocast.callbacks.ema.EMACallback",
    ]
    assert callbacks[0].every_n_train_steps_fraction == 0.05
    assert callbacks[0].save_top_k == -1
    assert callbacks[10].start_after_fraction == 0.05
    assert "best-multiwinkler-overall" in callbacks[10].filename
    assert callbacks[-1].decay == 0.999
    assert cfg.trainer.devices == 4
    assert cfg.trainer.strategy == "ddp"
    assert cfg.trainer.max_epochs == campaign["datasets"][dataset]["crps_epochs"]


@pytest.mark.parametrize("dataset", ["ad", "gpe", "gs"])
def test_fm_composition_preserves_reference_callbacks(
    pipeline: ModuleType,
    campaign: dict,
    tmp_path: Path,
    REPO_ROOT: Path,
    dataset: str,
) -> None:
    """Each FM repeat composes the established main-comparison callback stack."""
    state = _fake_state(campaign, tmp_path)
    _, overrides = pipeline._training_overrides(campaign, state, dataset, "fm")
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "src/autocast/configs"),
    ):
        cfg = compose(config_name="processor", overrides=overrides)

    callbacks = cfg.trainer.callbacks
    assert len(callbacks) == 3
    expected_quarter = campaign["datasets"][dataset]["fm_epochs"] // 4
    assert callbacks[0].every_n_epochs == expected_quarter
    assert callbacks[0].save_top_k == -1
    assert callbacks[1].monitor == "val_loss"
    assert callbacks[2].decay == 0.999
    assert cfg.trainer.devices == 4
    assert cfg.trainer.strategy == "ddp"
    assert cfg.trainer.max_epochs == campaign["datasets"][dataset]["fm_epochs"]


def test_gpe_schema_matches_published_ae_channels(
    pipeline: ModuleType, campaign: dict
) -> None:
    """GPE fixed-AE caching retains the published real/imag channel subset."""
    spec = campaign["datasets"]["gpe"]
    ae_run = pipeline._absolute_repo_path(spec["published_ae_run"])
    config = pipeline._load_yaml(ae_run / "resolved_autoencoder_config.yaml")
    assert spec["schema"]["channel_idxs"] == [1, 2]
    assert config["datamodule"]["channel_idxs"] == [1, 2]
