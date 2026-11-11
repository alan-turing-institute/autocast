"""Check ambient preparation without loading data, allocating GPUs or fitting."""

import ast
import shlex
import subprocess
from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from slurm_scripts.comparison.the_well.rayleigh_benard import (
    prepare_ambient_crps as prep,
)
from slurm_scripts.comparison.the_well.rayleigh_benard.paired_test_windows import (
    make_selection,
)


@pytest.fixture(autouse=True)
def committed_source(monkeypatch):
    monkeypatch.setattr(prep, "_source_commit", lambda: "review-source")


@pytest.mark.parametrize("microbatch", [1, 2, 4, 8])
@pytest.mark.parametrize("phase", ["smoke", "timing", "train"])
def test_training_preserves_reference_exposure_and_probe_horizon(
    tmp_path, phase, microbatch
):
    output = tmp_path / "new preparation"
    epochs = 128 if phase == "train" else None
    state = prep.prepare_training(output, tmp_path / "data", phase, microbatch, epochs)
    cfg = OmegaConf.load(state["jobs"][phase]["config"])
    accumulation = cfg.trainer.accumulate_grad_batches
    assert microbatch * accumulation * 4 == 32
    assert cfg.model.n_members == cfg.model.processor.n_members == 8
    assert cfg.datamodule._target_.endswith("TheWellDataModule")
    assert cfg.model.encoder._target_.endswith("IdentityEncoder")
    assert cfg.autoencoder_checkpoint is None
    assert cfg.model.processor.backbone.hid_channels == 2048
    assert cfg.model.processor.backbone.checkpointing
    assert list(cfg.model.processor.backbone.patch_size) == [1, 16, 16]
    assert cfg.model.processor.context_mask.lmbda == 1
    assert cfg.optimizer.cosine_epochs == (128 if phase == "train" else 4096)
    assert cfg.trainer.max_epochs == {"smoke": 1, "timing": 16, "train": 128}[phase]
    updates = cfg.trainer.limit_train_batches // accumulation
    assert updates == (1 if phase == "smoke" else 64)
    if phase != "smoke":
        assert cfg.trainer.limit_val_batches * microbatch * 4 == 512
    assert cfg.logging.wandb.enabled is (phase == "train")
    callbacks = instantiate(cfg.trainer.callbacks)
    assert len({c.state_key for c in callbacks}) == len(callbacks)
    if phase == "train":
        assert cfg.trainer.max_time is None
        assert cfg.trainer.callbacks[-1].interval == "epoch"
    script = output / f"{phase}.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    body = script.read_text()
    assert "#SBATCH --nodes=1" in body
    assert "#SBATCH --gpus=4" in body
    assert "#SBATCH --ntasks-per-node=4" in body
    assert "${SLURM_JOB_ID:?" in body
    # Validate the checksum line itself, including paths containing spaces.
    checksum_line = next(
        line for line in body.splitlines() if line.startswith("printf")
    )
    printed = subprocess.check_output(
        ["bash", "-c", checksum_line.split(" | ")[0]], text=True
    )
    job = state["jobs"][phase]
    assert printed == f"{job['config_sha256']}  {job['config']}\n"


@pytest.mark.parametrize(
    ("phase", "batch", "epochs"),
    [("train", 1, None), ("timing", 1, 9), ("smoke", 3, None)],
)
def test_invalid_budget_or_microbatch_does_not_create_outputs(
    tmp_path, phase, batch, epochs
):
    output = tmp_path / "new"
    with pytest.raises(ValueError, match=r"Production|Use"):
        prep.prepare_training(output, tmp_path / "data", phase, batch, epochs)
    assert not output.exists()


def test_evaluation_reuses_both_saved_cohorts_and_fitted_model(tmp_path):
    training = prep.prepare_training(
        tmp_path / "training", tmp_path / "data", "train", epochs=128
    )
    source = Path(training["jobs"]["train"]["config"])
    original = OmegaConf.load(source)
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
        assert cfg.eval.rollout_start == cfg.eval.n_members == 16
        assert cfg.eval.max_rollout_steps == 46
        assert cfg.datamodule.max_rollout_steps == 200
        assert not cfg.eval.compute_test_metrics
        assert not cfg.eval.compute_rollout_autoencoded_target_metrics
        script = output / f"{name}.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        args = shlex.split(script.read_text().split("exec srun ", 1)[1])
        if name == "rollout":
            code = args[args.index("-c") + 1]
            ast.parse(code)
            assert "L.seed_everything(42" in code
            assert "global_step" in code
        else:
            assert args[args.index("--expected-step") + 1] == "8192"
