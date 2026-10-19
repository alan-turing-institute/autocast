"""Check that the production wrapper cannot submit during ordinary previews."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "slurm_scripts/ablations/submit_unet_extensions.py"
    )
    spec = importlib.util.spec_from_file_location("unet_extension_launcher", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    launches = [
        (
            tmp_path
            / "outputs/group"
            / f"crps_{dataset}64_unet_azula_large_abc1234_new",
            ["uv", "run", "autocast", "epd", "--mode", "slurm", f"dataset={dataset}"],
        )
        for dataset in ["ad", "gs", "gpe"]
    ]
    monkeypatch.setattr(module, "_prepare_launches", lambda _group: launches)
    calls = []

    def record_command(command, *, check):
        assert check is True
        calls.append(command)

    def git_output(command, *, text):
        assert text is True
        return "" if command[1] == "status" else "abc1234\n"

    monkeypatch.setattr(module.subprocess, "run", record_command)
    monkeypatch.setattr(module.subprocess, "check_output", git_output)
    return SimpleNamespace(module=module, launches=launches, calls=calls)


def test_default_only_previews(launcher):
    launcher.module.main([])

    assert launcher.calls == [
        [*command, "--dry-run"] for _, command in launcher.launches
    ]


def test_submit_previews_all_runs_before_submitting(launcher):
    launcher.module.main(["--submit"])

    commands = [command for _, command in launcher.launches]
    assert launcher.calls == [
        *[[*command, "--dry-run"] for command in commands],
        *commands,
    ]


def test_existing_second_run_prevents_all_submissions(launcher):
    workdir, _ = launcher.launches[1]
    workdir.with_name("crps_gs64_unet_azula_large_oldhash_previous").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        launcher.module.main(["--submit"])

    assert launcher.calls == []


def test_dirty_checkout_prevents_submission(launcher, monkeypatch):
    def dirty_status(_command, *, text):
        assert text is True
        return " M config.yaml\n"

    monkeypatch.setattr(launcher.module.subprocess, "check_output", dirty_status)
    with pytest.raises(SystemExit, match="2"):
        launcher.module.main(["--submit"])

    assert launcher.calls == []
