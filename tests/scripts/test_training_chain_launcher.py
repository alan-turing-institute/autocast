import subprocess
import sys

from autocast.scripts.workflow.cli import main


def test_chain_reuses_launcher_with_afterok_dependency(tmp_path, monkeypatch, capsys):
    work_dir = tmp_path / "chain run"
    args = [
        "autocast",
        "processor",
        "--mode",
        "slurm",
        "--chain",
        "2",
        "--workdir",
        str(work_dir),
        "datamodule=cached_latents",
        "datamodule.data_path=/data/gray-scott",
        "++hydra.launcher.timeout_min=90",
    ]
    monkeypatch.setattr(sys, "argv", [*args, "--dry-run"])
    main()
    assert not work_dir.exists()
    assert "afterok:<segment-0001-job-id>" in capsys.readouterr().out

    commands = []

    def fake_sbatch(command, **_kwargs):
        assert command[0] == "sbatch"
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, f"{90 + len(commands)};cluster\n"
        )

    with monkeypatch.context() as mock:
        mock.setattr("autocast.scripts.workflow.slurm.subprocess.run", fake_sbatch)
        mock.setattr(sys, "argv", args)
        main()
    assert "--dependency=afterok:91" in commands[1]
    chain = work_dir / ".chain"
    script = chain / "segment-0002.sh"
    assert "$((allocation_start + 4860))" in script.read_text()
    assert "segment-0001.ckpt" in script.read_text()
    assert (chain / "segment-0002.job-id").read_text() == "92\n"

    monkeypatch.setenv("SLURM_JOB_ID", "92")
    endpoint = chain / "segment-0001.ckpt"
    (chain / "finished").symlink_to(endpoint.name)
    command = ["bash", str(script)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "Chain completion checkpoint is missing" in result.stderr
    endpoint.touch()
    subprocess.run(command, check=True)
    assert not (chain / "segment-0002.started").exists()
