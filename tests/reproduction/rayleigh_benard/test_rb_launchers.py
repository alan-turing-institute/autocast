"""Opt-in checks for the archived RB launch scripts; no jobs are submitted."""

import ast
import os
import shlex
import subprocess
from pathlib import Path

import pytest
from rb_reproduction import diagnostic_case, load_rb_tool

prep = load_rb_tool("prepare_ambient_crps")
runner = load_rb_tool("complete_diagnostics")


@pytest.fixture(autouse=True)
def committed_source(monkeypatch, REPO_ROOT):
    monkeypatch.setattr(prep, "ROOT", REPO_ROOT)
    monkeypatch.setattr(prep, "_source_commit", lambda: "review-source")


@pytest.mark.skipif(os.name != "posix", reason="Slurm scripts require POSIX Bash")
@pytest.mark.parametrize(
    ("nodes", "microbatch"), [(1, None), (1, 1), (2, None), (4, None), (8, None)]
)
@pytest.mark.parametrize("phase", ["smoke", "timing", "train"])
def test_generated_training_script(tmp_path, nodes, microbatch, phase):
    output = tmp_path / "new preparation"
    state = prep.prepare_training(
        output, tmp_path / "data", phase, microbatch, nodes=nodes
    )
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
    checksum_line = next(
        line for line in body.splitlines() if line.startswith("printf")
    )
    printed = subprocess.check_output(
        ["bash", "-c", checksum_line.split(" | ")[0]], text=True
    )
    job = state["jobs"][phase]
    assert job["gpus"] == 4 * nodes
    assert printed == f"{job['config_sha256']}  {job['config']}\n"


@pytest.mark.skipif(os.name != "posix", reason="Slurm run guards require POSIX Bash")
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


@pytest.mark.skipif(os.name != "posix", reason="Slurm scripts require POSIX Bash")
def test_generated_evaluation_scripts(tmp_path):
    training = prep.prepare_training(tmp_path / "training", tmp_path / "data", "train")
    output = tmp_path / "evaluation"
    prep.prepare_evaluation(
        output,
        Path(training["jobs"]["train"]["config"]),
        tmp_path / "selected.ckpt",
        8192,
    )
    for name in ("rollout", "paired"):
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
        body = script.read_text()
        guard = "if [[" + body.split("if [[", 1)[1].split("exec srun", 1)[0]
        result = subprocess.run(
            ["bash", "-c", guard], check=False, capture_output=True, text=True
        )
        assert result.returncode == 1
        assert "Run directory exists" in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="Slurm scripts require POSIX Bash")
def test_generated_diagnostic_scripts(tmp_path, monkeypatch):
    case = diagnostic_case(tmp_path, monkeypatch)
    runner.prepare(case.args)
    for phase in ("pilot", "production"):
        script = case.args.output / f"test_model-{phase}.sbatch"
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text()
        assert "#SBATCH --gpus=1" in text
        assert "--time=" in text
        assert "sbatch " not in text
        assert " train " not in text
        assert ("--pilot" in text) == (phase == "pilot")
