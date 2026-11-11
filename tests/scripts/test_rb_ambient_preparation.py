"""Check ambient preparation without loading data, allocating GPUs or fitting."""

import ast
import shlex
import subprocess
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autocast.scripts.eval.encoder_processor_decoder import (
    _resolve_rollout_batch_limit,
)
from autocast.scripts.setup import _build_processor
from slurm_scripts.comparison.the_well.rayleigh_benard import (
    prepare_ambient_crps as prep,
)
from slurm_scripts.comparison.the_well.rayleigh_benard.paired_test_windows import (
    make_selection,
)


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
    script = output / f"{phase}.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    body = script.read_text()
    assert f"#SBATCH --nodes={nodes}" in body
    assert "#SBATCH --gpus-per-node=4" in body
    assert "#SBATCH --ntasks-per-node=4" in body
    assert "#SBATCH --no-requeue" in body
    assert "${SLURM_JOB_ID:?" in body
    assert ("module load brics/nccl/2.26.6-1" in body) == (nodes > 1)
    assert "srun --kill-on-bad-exit=1" in body
    # Validate the checksum line itself, including paths containing spaces.
    checksum_line = next(
        line for line in body.splitlines() if line.startswith("printf")
    )
    printed = subprocess.check_output(
        ["bash", "-c", checksum_line.split(" | ")[0]], text=True
    )
    job = state["jobs"][phase]
    assert job["gpus"] == 4 * nodes
    assert printed == f"{job['config_sha256']}  {job['config']}\n"


@pytest.mark.parametrize("existing", ["absent", "directory", "dangling_link"])
def test_initial_job_rejects_partial_run_outputs(tmp_path, existing):
    output = tmp_path / "new preparation"
    prep.prepare_training(output, tmp_path / "data", "smoke")
    run_dir = output / "run"
    if existing == "directory":
        run_dir.mkdir()
    elif existing == "dangling_link":
        run_dir.symlink_to(tmp_path / "missing", target_is_directory=True)
    # Exercise only the rendered guard, never the training command.
    body = (output / "smoke.sh").read_text()
    guard = "if [[" + body.split("if [[", 1)[1].split("exec srun", 1)[0]
    result = subprocess.run(
        ["bash", "-c", guard], check=False, capture_output=True, text=True
    )
    assert result.returncode == (0 if existing == "absent" else 1)
    if existing != "absent":
        assert "Run directory exists" in result.stderr


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
        assert _resolve_rollout_batch_limit(cfg.eval) is None
        script = output / f"{name}.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        assert "#SBATCH --no-requeue" in script.read_text()
        args = shlex.split(script.read_text().split("exec srun ", 1)[1])
        if name == "rollout":
            code = args[args.index("-c") + 1]
            ast.parse(code)
            assert "L.seed_everything(42" in code
            assert "global_step" in code
        else:
            assert args[args.index("--expected-step") + 1] == "8192"
        (output / name).mkdir()
        guard = (
            "if [[" + script.read_text().split("if [[", 1)[1].split("exec srun", 1)[0]
        )
        result = subprocess.run(
            ["bash", "-c", guard], check=False, capture_output=True, text=True
        )
        assert result.returncode == 1
        assert "Run directory exists" in result.stderr
