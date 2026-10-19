"""SWE64 saved-task evaluation guards; no real data, training or submission."""

import runpy
from argparse import Namespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
HELPER = runpy.run_path(str(ROOT / "scripts/swe64s/evaluate.py"))


@pytest.mark.parametrize("outputs", [1, 4])
def test_rollout_horizon_is_one_hundred_leads(outputs):
    commands = HELPER["build_commands"](
        run_dir=Path("/saved/run"),
        checkpoint=Path("/saved/best.ckpt"),
        output_root=Path("/new/eval"),
        autoencoder_checkpoint=None,
        n_steps_output=outputs,
    )
    for index, command in enumerate(commands):
        assert f"++eval.max_rollout_steps={100 // outputs}" in command
        assert f"++eval.teacher_forcing_ratio={float(index)}" in command
        assert (
            f"++eval.free_running_only={'true' if index == 0 else 'false'}" in command
        )
        assert "++eval.coverage_levels=[0.9]" in command
        assert not any(arg.startswith("++model.") for arg in command)


@pytest.mark.parametrize("outputs", [0, 2, 3])
def test_unknown_output_windows_rejected(outputs):
    with pytest.raises(ValueError, match="one or four"):
        HELPER["build_commands"](
            run_dir=Path("/saved/run"),
            checkpoint=Path("/saved/best.ckpt"),
            output_root=Path("/new/eval"),
            autoencoder_checkpoint=None,
            n_steps_output=outputs,
        )


@pytest.mark.parametrize("kind", ["crps", "ambient_fm", "latent_fm"])
@pytest.mark.parametrize("outputs", [1, 4])
def test_saved_task_and_codec_validation(tmp_path, kind, outputs):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "datamodule": {"n_steps_input": 1, "n_steps_output": outputs},
            "model": {
                "encoder": {
                    "_target_": "autocast.encoders.IdentityEncoder"
                    if kind == "ambient_fm"
                    else "autocast.encoders.DCEncoder"
                },
                "processor": {
                    "_target_": "autocast.processors.ResidualFlowMatchingProcessor"
                    if "fm" in kind
                    else "autocast.processors.AzulaViTProcessor"
                },
            },
        }
    )
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.touch()
    args = Namespace(
        run_dir=run_dir,
        checkpoint=checkpoint,
        output_root=tmp_path / "eval",
        autoencoder_checkpoint=None,
    )
    assert HELPER["_saved_output_steps"](run_dir) == outputs
    if kind == "latent_fm":
        with pytest.raises(FileNotFoundError, match="Latent FM"):
            HELPER["_validate_execution"](args)
        args.autoencoder_checkpoint = tmp_path / "ae.ckpt"
        args.autoencoder_checkpoint.touch()
    HELPER["_validate_execution"](args)
    (args.output_root / "free_running").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        HELPER["_validate_execution"](args)
