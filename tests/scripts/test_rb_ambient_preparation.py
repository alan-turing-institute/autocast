"""Scientific contracts for the RB training and evaluation presets."""

from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rb_reproduction import load_rb_tool

from autocast.scripts.setup import _build_processor

prep = load_rb_tool("prepare_ambient_crps")
make_selection = load_rb_tool("paired_test_windows").make_selection


@pytest.fixture(autouse=True)
def committed_source(monkeypatch):
    monkeypatch.setattr(prep, "_source_commit", lambda: "review-source")


@pytest.mark.parametrize(
    ("nodes", "microbatch"), [(1, None), (1, 1), (2, None), (4, None), (8, None)]
)
@pytest.mark.parametrize("phase", ["smoke", "timing", "train"])
def test_training_preserves_reference_exposure_and_probe_horizon(
    tmp_path, phase, nodes, microbatch
):
    output = tmp_path / "new preparation"
    state = prep.prepare_training(
        output, tmp_path / "data", phase, microbatch, nodes=nodes
    )
    cfg = OmegaConf.load(state["jobs"][phase]["config"])
    accumulation = cfg.trainer.accumulate_grad_batches
    microbatch = cfg.datamodule.batch_size
    assert cfg.trainer.num_nodes == nodes
    assert microbatch * accumulation * 4 * nodes == 32
    assert cfg.model.n_members == cfg.model.processor.n_members == 8
    assert cfg.datamodule._target_.endswith("TheWellDataModule")
    assert cfg.model.encoder._target_.endswith("IdentityEncoder")
    assert cfg.autoencoder_checkpoint is None
    assert cfg.model.processor.backbone.hid_channels == 1024
    assert cfg.model.processor.backbone.hid_blocks == 16
    assert cfg.model.processor.backbone.attention_heads == 8
    assert cfg.model.processor.backbone.checkpointing
    assert list(cfg.model.processor.backbone.patch_size) == [1, 16, 16]
    assert cfg.model.processor.context_mask.lmbda == 1
    assert cfg.optimizer.cosine_epochs == 4096
    assert cfg.trainer.max_epochs == {"smoke": 1, "timing": 16, "train": 4096}[phase]
    updates = cfg.trainer.limit_train_batches // accumulation
    assert updates == (1 if phase == "smoke" else 64)
    if phase != "smoke":
        assert cfg.trainer.limit_val_batches * microbatch * 4 * nodes == 512
    if nodes == 8:
        assert microbatch == accumulation == 1
    assert cfg.logging.wandb.enabled is (phase == "train")
    callbacks = instantiate(cfg.trainer.callbacks)
    assert len({c.state_key for c in callbacks}) == len(callbacks)
    if phase == "train":
        assert cfg.trainer.max_time is None
        assert cfg.trainer.callbacks[-1].interval == "epoch"


@pytest.mark.parametrize(
    ("phase", "batch", "epochs", "nodes"),
    [
        ("train", None, 128, 8),
        ("timing", None, 9, 8),
        ("smoke", 3, None, 8),
        ("smoke", 2, None, 8),
        ("smoke", None, None, 16),
    ],
)
def test_invalid_budget_or_microbatch_does_not_create_outputs(
    tmp_path, phase, batch, epochs, nodes
):
    output = tmp_path / "new"
    with pytest.raises(ValueError, match=r"Matched|Use"):
        prep.prepare_training(
            output, tmp_path / "data", phase, batch, epochs, nodes=nodes
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("size", "expected_parameters", "width", "heads"),
    [("matched", 225_969_408, 1024, 8), ("lola", 859_499_776, 2048, 16)],
)
def test_prepared_sizes_have_the_measured_processors(
    tmp_path, size, expected_parameters, width, heads
):
    state = prep.prepare_training(
        tmp_path / "train", tmp_path / "data", "train", size=size
    )
    cfg = OmegaConf.load(state["jobs"]["train"]["config"])
    assert state["size"] == size
    assert cfg.model.processor.backbone.hid_channels == width
    assert cfg.model.processor.backbone.attention_heads == heads
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 4096
    assert cfg.trainer.num_nodes == 8
    assert cfg.trainer.devices == 4
    assert cfg.datamodule.batch_size == cfg.trainer.accumulate_grad_batches == 1
    with torch.device("meta"):
        processor = _build_processor(
            cfg.model,
            {
                "in_channels": 4,
                "out_channels": 4,
                "n_channels_out": 4,
                "n_steps_input": 1,
                "n_steps_output": 4,
                "spatial_resolution": (512, 128),
            },
            global_cond_channels=6,
        )
    assert all(p.is_meta for p in processor.parameters())
    parameters = sum(p.numel() for p in processor.parameters())
    assert parameters == expected_parameters
    if size == "matched":
        assert abs(parameters / 223_019_328 - 1) < 0.02


def test_unknown_size_fails_before_creating_outputs(tmp_path):
    output = tmp_path / "unknown"
    with pytest.raises(ValueError, match="model size"):
        prep.prepare_training(output, tmp_path / "data", "train", size="unknown")
    assert not output.exists()


def test_evaluation_reuses_both_saved_cohorts_and_fitted_model(tmp_path):
    training = prep.prepare_training(tmp_path / "training", tmp_path / "data", "train")
    source = Path(training["jobs"]["train"]["config"])
    original = OmegaConf.load(source)
    # A training/debug override must not truncate the controlled evaluation.
    original.eval.max_test_batches = 1
    OmegaConf.save(original, source)
    output = tmp_path / "evaluation"
    checkpoint = tmp_path / "selected.ckpt"
    state = prep.prepare_evaluation(output, source, checkpoint, 8192)
    trajectories = OmegaConf.to_container(OmegaConf.load(output / "test64_seed0.yaml"))
    windows = OmegaConf.to_container(
        OmegaConf.load(output / "paired_test512_seed0.yaml")
    )
    assert isinstance(trajectories, dict)
    assert isinstance(windows, dict)
    assert windows == make_selection(trajectories)
    assert len(trajectories["indices"]) == 64
    assert len(windows["windows"]) == 512
    for name in ("rollout", "paired"):
        cfg = OmegaConf.load(state["jobs"][name]["config"])
        assert cfg.model == original.model
        assert (
            cfg.datamodule.normalization_path == original.datamodule.normalization_path
        )
        assert cfg.eval.checkpoint == str(checkpoint)
        assert cfg.eval.devices == cfg.eval.num_nodes == 1
        assert cfg.eval.rollout_start == cfg.eval.n_members == 16
        assert cfg.eval.max_rollout_steps == 46
        assert cfg.datamodule.max_rollout_steps == 200
        assert not cfg.eval.compute_test_metrics
        assert not cfg.eval.compute_rollout_autoencoded_target_metrics
        assert cfg.eval.max_rollout_batches is None
        assert cfg.eval.max_test_batches is None
