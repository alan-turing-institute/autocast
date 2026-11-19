"""Exercise trajectory outputs through the complete evaluation orchestration."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from einops import repeat
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import autocast
from autocast.data.datamodule import SpatioTemporalDataModule
from autocast.data.zarr_dataset import ZarrDataModule
from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.identity import IdentityEncoder
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.processors.base import Processor
from autocast.scripts.eval.encoder_processor_decoder import run_evaluation
from autocast.types import EncodedBatch


class _IncrementProcessor(Processor[EncodedBatch]):
    def __init__(self):
        super().__init__()
        self.increment = torch.nn.Parameter(torch.tensor(1.0))

    def map(self, x, global_cond=None):  # noqa: ARG002
        return x + self.increment

    def loss(self, batch):
        return (
            (self.map(batch.encoded_inputs) - batch.encoded_output_fields)
            .square()
            .mean()
        )


class _DoubleDecoder(IdentityDecoder):
    def decode(self, z):
        return 2 * z


EVAL_MODULE = "autocast.scripts.eval.encoder_processor_decoder"
DEFAULT_EVAL_CONFIG = (
    Path(autocast.__file__).parent / "configs" / "eval" / "default.yaml"
)


def _install_ensemble_eval_fixture(tmp_path, monkeypatch):
    """Patch the eval setup to a tiny 3-member model; return its checkpoint path."""
    data = {"data": repeat(torch.arange(7.0), "t -> b t h w c", b=2, h=2, w=2, c=1)}
    datamodule = SpatioTemporalDataModule(
        data_path=None,
        data=dict.fromkeys(("train", "valid", "test"), data),
        start_frame=1,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=EncoderDecoder(
            encoder=IdentityEncoder(1), decoder=_DoubleDecoder(1)
        ),
        processor=_IncrementProcessor(),
        n_members=3,
    )
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    monkeypatch.setattr(
        f"{EVAL_MODULE}.setup_datamodule",
        lambda config: (
            datamodule,
            config,
            {
                "example_batch": next(iter(datamodule.test_dataloader())),
                "n_steps_output": 1,
            },
        ),
    )
    monkeypatch.setattr(
        f"{EVAL_MODULE}.setup_epd_model", lambda *_args, **_kwargs: model
    )
    return checkpoint


@pytest.mark.parametrize("mode", ["ambient", "encode_once"])
@pytest.mark.parametrize(
    ("trajectory_statistics", "compute_metrics"),
    [(False, False), (False, True), (True, False)],
)
def test_evaluation_keeps_trajectory_and_diagnostic_outputs_separate(
    tmp_path, monkeypatch, mode, trajectory_statistics, compute_metrics
):
    checkpoint = _install_ensemble_eval_fixture(tmp_path, monkeypatch)
    cfg = OmegaConf.create(
        {
            "model": {"encoder": {}, "decoder": {}, "n_members": 3},
            "datamodule": {"start_frame": 1},
            "eval": {
                "checkpoint": str(checkpoint),
                "mode": mode,
                "accelerator": "cpu",
                "batch_size": 2,
                "metrics": ["mse", "crps"],
                "deterministic_metric_member_indices": [0],
                "deterministic_metric_member_average": True,
                "compute_test_metrics": compute_metrics,
                "compute_rollout_metrics": compute_metrics,
                "compute_rollout_autoencoded_target_metrics": True,
                "rollout_start": 1,
                "max_rollout_steps": 3,
                "metric_windows_rollout": [[0, 3]],
                "trajectory_statistics": {"enabled": trajectory_statistics},
            },
        }
    )
    eval_module = "autocast.scripts.eval.encoder_processor_decoder"
    if mode == "encode_once":
        # Model construction is supplied below; only processor weights are loaded.
        # The autoencoder marker keeps the real encode-once dispatch active.
        cfg.autoencoder_checkpoint = str(tmp_path / "autoencoder.ckpt")
        monkeypatch.setattr(
            f"{eval_module}._maybe_inject_encoder_decoder_from_autoencoder_checkpoint",
            lambda config: config,
        )
    legacy_csv = tmp_path / "evaluation_metrics.csv"
    if trajectory_statistics:
        legacy_csv.write_text("Existing evaluation\n")

    run_evaluation(cfg, work_dir=tmp_path)

    output_dir = (
        tmp_path / "trajectory_statistics" if trajectory_statistics else tmp_path
    )
    if not trajectory_statistics and not compute_metrics:
        assert not legacy_csv.exists()
        assert not (output_dir / "rollout_metrics.csv").exists()
        return

    rollout = pd.read_csv(output_dir / "rollout_metrics.csv")
    if mode == "encode_once":
        # After the dataset crop and rollout start, raw targets are [3, 4, 5].
        # Predictions are [6, 8, 10]; autoencoded targets match those predictions.
        assert rollout["mse"].tolist() == pytest.approx([50 / 3] * len(rollout))
        diagnostic = pd.read_csv(output_dir / "rollout_metrics_autoencoded_target.csv")
        assert diagnostic["mse"].tolist() == pytest.approx([0] * len(diagnostic))
    else:
        assert not (output_dir / "rollout_metrics_autoencoded_target.csv").exists()

    if trajectory_statistics:
        assert legacy_csv.read_text() == "Existing evaluation\n"
        expected_rows = {
            "single_step_metrics_per_trajectory.csv": 2,
            "rollout_metrics_per_trajectory.csv": 2,
            "rollout_metrics_per_timestep_per_trajectory.csv": 6,
        }
        for name, count in expected_rows.items():
            rows = pd.read_csv(output_dir / name)
            assert len(rows) == count
            assert {"mse", "mse_member_0", "mse_member_avg", "coverage_mae"} <= set(
                rows
            )
            assert rows["mse_member_avg"].tolist() == pytest.approx(
                rows["mse"].tolist()
            )
        trajectories = pd.read_csv(output_dir / "rollout_metrics_per_trajectory.csv")
        assert trajectories["mse"].tolist() == pytest.approx(
            [rollout["mse"].iloc[0]] * 2
        )
        leads = pd.read_csv(
            output_dir / "rollout_metrics_per_timestep_per_trajectory.csv"
        )
        if mode == "encode_once":
            assert sorted(leads["mse"].tolist()) == pytest.approx(
                [9, 9, 16, 16, 25, 25]
            )
    else:
        assert not (output_dir / "rollout_metrics_per_trajectory.csv").exists()


@pytest.mark.parametrize("include_per_timestep", [True, False])
def test_windowed_trajectory_statistics_validate_before_inference(
    tmp_path, monkeypatch, include_per_timestep
):
    zarr = pytest.importorskip("zarr")
    group = zarr.open_group(str(tmp_path / "test" / "data.zarr"), mode="w")
    values = repeat(torch.arange(7.0), "t -> b t h w c", b=2, h=2, w=2, c=1)
    group.create_array("data", data=values.numpy(), chunks=(1, 1, 2, 2, 1))
    group.attrs.update(
        schema="autosim.spatiotemporal",
        schema_version=1,
        field_names=["field_0"],
        complete=True,
    )
    datamodule = ZarrDataModule(
        tmp_path, rollout_steps=3, use_normalization=False, pin_memory=False
    )
    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=EncoderDecoder(
            encoder=IdentityEncoder(1), decoder=IdentityDecoder(1)
        ),
        processor=_IncrementProcessor(),
        n_members=3,
    )
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    cfg = OmegaConf.create(
        {
            "model": {"encoder": {}, "decoder": {}, "n_members": 3},
            "datamodule": {},
            "eval": {
                "checkpoint": str(checkpoint),
                "mode": "ambient",
                "accelerator": "cpu",
                "batch_size": 2,
                "metrics": ["mse"],
                "max_rollout_steps": 3,
                "metric_windows_rollout": [[0, 3]],
                "trajectory_statistics": {
                    "enabled": True,
                    "include_per_timestep": include_per_timestep,
                },
            },
        }
    )
    eval_module = "autocast.scripts.eval.encoder_processor_decoder"
    monkeypatch.setattr(
        f"{eval_module}.setup_datamodule",
        lambda config: (
            datamodule,
            config,
            {
                "example_batch": next(iter(datamodule.test_dataloader())),
                "n_steps_output": 1,
            },
        ),
    )
    monkeypatch.setattr(
        f"{eval_module}.setup_epd_model", lambda *_args, **_kwargs: model
    )
    if include_per_timestep:

        def unexpected_inference(*_args, **_kwargs):
            pytest.fail("Trajectory metadata must be checked before inference.")

        monkeypatch.setattr(model, "forward", unexpected_inference)
        with pytest.raises(ValueError, match="include_per_timestep=false"):
            run_evaluation(cfg, work_dir=tmp_path)
        assert not list((tmp_path / "trajectory_statistics").glob("*.csv"))
    else:
        run_evaluation(cfg, work_dir=tmp_path)
        output_dir = tmp_path / "trajectory_statistics"
        rows = pd.read_csv(output_dir / "rollout_metrics_per_trajectory.csv")
        assert rows["trajectory_idx"].tolist() == [0, 1]
        assert rows["n_windows"].tolist() == [4, 4]
        assert rows["mse"].tolist() == pytest.approx([0, 0])
        assert not (
            output_dir / "rollout_metrics_per_timestep_per_trajectory.csv"
        ).exists()


@pytest.mark.parametrize("mode", ["ambient", "encode_once"])
@pytest.mark.parametrize("rollout_start", [3, 16])
def test_zarr_rejects_absolute_rollout_start_before_model_setup(
    tmp_path, monkeypatch, mode, rollout_start
):
    zarr = pytest.importorskip("zarr")
    group = zarr.open_group(str(tmp_path / "test" / "data.zarr"), mode="w")
    values = repeat(torch.arange(32.0), "t -> b t h w c", b=1, h=2, w=2, c=1)
    group.create_array("data", data=values.numpy(), chunks=(1, 1, 2, 2, 1))
    group.attrs.update(
        schema="autosim.spatiotemporal",
        schema_version=1,
        field_names=["field_0"],
        complete=True,
    )
    datamodule = ZarrDataModule(tmp_path, use_normalization=False, pin_memory=False)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    cfg = OmegaConf.create(
        {
            "model": {"encoder": {}, "decoder": {}},
            "eval": {
                "checkpoint": str(checkpoint),
                "mode": mode,
                "rollout_start": rollout_start,
                "compute_rollout_metrics": True,
            },
        }
    )
    eval_module = "autocast.scripts.eval.encoder_processor_decoder"
    monkeypatch.setattr(
        f"{eval_module}.setup_datamodule",
        lambda config: (
            datamodule,
            config,
            {"example_batch": next(iter(datamodule.test_dataloader()))},
        ),
    )

    def unexpected_model_setup(*_args, **_kwargs):
        pytest.fail("Invalid rollout offsets must be rejected before model setup.")

    monkeypatch.setattr(f"{eval_module}.setup_epd_model", unexpected_model_setup)
    with pytest.raises(ValueError, match=r"datamodule\.start_frame"):
        run_evaluation(cfg, work_dir=tmp_path)


def _default_eval_cfg(checkpoint, shipped=None, **overrides):
    """A shipped eval config (default: `configs/eval/default.yaml`) on the fixture."""
    if shipped is None:
        shipped = OmegaConf.load(DEFAULT_EVAL_CONFIG)
    fixture = {
        "checkpoint": str(checkpoint),
        "mode": "ambient",
        "accelerator": "cpu",
        "batch_size": 2,
        "rollout_start": 1,
        "max_rollout_steps": 3,
        "compute_rollout_metrics": True,
        "metric_windows_rollout": [[0, 3]],
        **overrides,
    }
    return OmegaConf.create(
        {
            "model": {"encoder": {}, "decoder": {}, "n_members": 3},
            "datamodule": {"start_frame": 1},
            "eval": OmegaConf.merge(shipped, fixture),
        }
    )


@pytest.mark.parametrize("include_memory_intensive", [False, True])
def test_default_eval_config_scores_cheap_new_metrics_and_guards_the_costly_ones(
    tmp_path, monkeypatch, include_memory_intensive
):
    """Defaults write `ssr_pooled` and `collection_coverage`, and no variogram.

    A variogram is selectable by name but sits behind
    `eval.include_memory_intensive_metrics`: it must be absent from the shipped
    defaults, and present once that switch is on. `autocorr_match` needs at least
    two lead steps, so it is scored on the 3-step rollout window and left out of
    the single-step pass instead of failing the run.
    """
    default_metrics = list(OmegaConf.load(DEFAULT_EVAL_CONFIG).metrics)
    assert not {"variogram", "dss"} & set(default_metrics)
    checkpoint = _install_ensemble_eval_fixture(tmp_path, monkeypatch)
    cfg = _default_eval_cfg(
        checkpoint,
        include_memory_intensive_metrics=include_memory_intensive,
        metrics=[*default_metrics, "variogram", "dss", "autocorr_match"],
    )

    run_evaluation(cfg, work_dir=tmp_path)

    for csv_name in ("evaluation_metrics.csv", "rollout_metrics.csv"):
        columns = set(pd.read_csv(tmp_path / csv_name))
        assert {"ssr_pooled", "collection_coverage"} <= columns
        if include_memory_intensive:
            assert {"variogram", "dss"} <= columns
        else:
            assert not columns & {"variogram", "dss"}
    assert list(tmp_path.glob("test_collection_coverage_window_*.csv"))
    assert "autocorr_match" not in set(pd.read_csv(tmp_path / "evaluation_metrics.csv"))
    assert "autocorr_match" in set(pd.read_csv(tmp_path / "rollout_metrics.csv"))


def test_reference_eval_config_composes_and_scores_every_metric_it_lists(
    tmp_path, monkeypatch
):
    """`eval=reference` composes through its defaults and scores every listed metric.

    The evaluator skips an unknown metric name with only a log warning, so a
    misspelt name in the reference list would drop that column from every
    comparison without failing a run.
    """
    with initialize_config_dir(
        version_base=None, config_dir=str(DEFAULT_EVAL_CONFIG.parent.parent)
    ):
        reference = compose(
            config_name="encoder_processor_decoder", overrides=["eval=reference"]
        ).eval
    checkpoint = _install_ensemble_eval_fixture(tmp_path, monkeypatch)
    # The reference config inherits video batches; videos need ffmpeg, which CI
    # runners lack, and this test checks metric columns only.
    cfg = _default_eval_cfg(checkpoint, shipped=reference, batch_indices=[])

    run_evaluation(cfg, work_dir=tmp_path)

    # The reference config turns trajectory statistics on, which writes the run's
    # CSVs under trajectory_statistics/.
    assert reference.trajectory_statistics.enabled
    for csv_name in ("evaluation_metrics.csv", "rollout_metrics.csv"):
        columns = set(pd.read_csv(tmp_path / "trajectory_statistics" / csv_name))
        assert set(reference.metrics) <= columns, set(reference.metrics) - columns
