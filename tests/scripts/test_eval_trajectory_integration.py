"""Exercise trajectory outputs through the complete evaluation orchestration."""

import pandas as pd
import pytest
import torch
from einops import repeat
from omegaconf import OmegaConf

from autocast.data.datamodule import SpatioTemporalDataModule
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


@pytest.mark.parametrize("mode", ["ambient", "encode_once"])
@pytest.mark.parametrize(
    ("trajectory_statistics", "compute_metrics"),
    [(False, False), (False, True), (True, False)],
)
def test_evaluation_keeps_trajectory_and_diagnostic_outputs_separate(
    tmp_path, monkeypatch, mode, trajectory_statistics, compute_metrics
):
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
