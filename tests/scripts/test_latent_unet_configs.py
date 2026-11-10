"""Check latent U-Net overrides without data, training or scheduler calls."""

import os
import subprocess
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from autocast.scripts.workflow.slurm import (
    _build_sbatch_command,
    _resolve_launcher_submission_context,
)

DATASETS = {
    "advection_diffusion": "AD",
    "conditioned_navier_stokes": "CNS",
    "gray_scott": "GS",
    "gpe_laser_wake_only": "GPE",
}
# Original raw fields (after channel selection) and scalar conditioning.
# CRPS's PermuteConcat broadcasts the scalar channels into its spatial input.
LAYOUTS = {
    "advection_diffusion": (1, 2),
    "conditioned_navier_stokes": (3, 4),
    "gray_scott": (2, 4),
    "gpe_laser_wake_only": (2, 24),
}
ROOT = Path(__file__).resolve().parents[2]


def _compose(config_name, experiment, overrides=()):
    with initialize_config_dir(
        config_dir=str(ROOT / "src/autocast/configs"), version_base=None
    ):
        return compose(
            config_name=config_name,
            overrides=[
                f"local_experiment={experiment}",
                f"hydra.searchpath=[file://{ROOT}/local_hydra]",
                *overrides,
            ],
        )


def _config(dataset, method):
    return _compose(
        "processor", f"ablations/latent_unet/{dataset}/{method}_unet_matched"
    )


@pytest.mark.parametrize(("dataset", "key"), DATASETS.items())
@pytest.mark.parametrize("method", ["fm", "diffusion"])
def test_controls_keep_latent_data_and_reference_formulation(
    dataset, key, method, monkeypatch
):
    monkeypatch.setenv(f"AUTOCAST_UNET_{key}_CACHE", f"/reference/{key}/cached_latents")
    monkeypatch.setenv(f"AUTOCAST_UNET_{method.upper()}_{key}_EPOCHS", "123")
    cfg = _config(dataset, method)
    OmegaConf.resolve(cfg)
    assert cfg.datamodule._target_.endswith("EncodedDataModule")
    assert cfg.datamodule.data_path == f"/reference/{key}/cached_latents"
    assert cfg.datamodule.batch_size == 256
    assert cfg.trainer.devices == 4
    assert cfg.trainer.num_nodes == 1
    assert cfg.trainer.strategy == "ddp"
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 123
    assert cfg.trainer.max_time == "00:23:59:00"
    assert cfg.optimizer.learning_rate == 1e-4
    assert cfg.optimizer.warmup == 0
    assert cfg.model.processor.backbone._target_.endswith("TemporalUNetBackbone")
    if method == "fm":
        assert cfg.model.processor.flow_ode_steps == 50
        assert len(cfg.trainer.callbacks) == 3
    else:
        assert cfg.model.processor.sampler_steps == 50
        assert cfg.model.processor.denoiser_type == "karras"
        reference = OmegaConf.load(
            ROOT / "local_hydra/trainer/original_diffusion_ablation.yaml"
        )
        assert cfg.trainer.callbacks == reference.callbacks
    callbacks = instantiate(cfg.trainer.callbacks)
    assert len({callback.state_key for callback in callbacks}) == len(callbacks)


@pytest.mark.parametrize(("dataset", "key"), DATASETS.items())
@pytest.mark.parametrize("method", ["fm", "diffusion"])
def test_missing_epoch_budget_is_not_a_silent_vit_default(
    dataset, key, method, monkeypatch
):
    monkeypatch.setenv(f"AUTOCAST_UNET_{key}_CACHE", f"/reference/{key}/cache")
    monkeypatch.delenv(f"AUTOCAST_UNET_{method.upper()}_{key}_EPOCHS", raising=False)
    with pytest.raises(InterpolationResolutionError):
        OmegaConf.resolve(_config(dataset, method))


@pytest.mark.parametrize(("dataset", "layout"), LAYOUTS.items())
def test_matched_backbone_runs_on_original_layouts(dataset, layout):
    raw_channels, global_channels = layout
    cfg = _config(dataset, "fm")
    reference = _compose(
        "encoder_processor_decoder",
        f"ablations/arch_unet_fno_vit/{dataset}/crps_unet_azula_80m",
    )
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = instantiate(
            cfg.model.processor.backbone,
            in_channels=8,
            out_channels=8,
            cond_channels=8,
            n_steps_input=1,
            n_steps_output=4,
            global_cond_channels=global_channels,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        # Count the actual per-system ambient processor, including its 1024-wide
        # noise modulation. Meta tensors avoid allocating a second large model.
        with torch.device("meta"):
            crps = instantiate(
                reference.model.processor,
                in_channels=raw_channels + global_channels,
                out_channels=4 * raw_channels,
            )
        crps_count = sum(parameter.numel() for parameter in crps.parameters())
        assert abs(parameter_count / crps_count - 1) < 0.01
        with torch.no_grad():
            output = model(
                torch.randn(1, 4, 16, 16, 8),
                t=torch.rand(1),
                cond=torch.randn(1, 1, 16, 16, 8),
                global_cond=torch.randn(1, global_channels),
            )
        assert output.shape == (1, 4, 16, 16, 8)
        assert torch.isfinite(output).all()
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("phase", ["timing", "train"])
def test_previews_use_processor_workflow_and_never_submit(phase, tmp_path, monkeypatch):
    capture = tmp_path / "commands"
    stub = tmp_path / "uv"
    stub.write_text(
        '#!/bin/bash\nprintf "%s\\0" "$@" >> "$CAPTURE"\nprintf "\\0" >> "$CAPTURE"\n'
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("CAPTURE", str(capture))
    monkeypatch.setenv("RUN_GROUP", "test/latent_unet")
    for key in DATASETS.values():
        monkeypatch.setenv(f"AUTOCAST_UNET_{key}_CACHE", f"/reference/{key}/cache")
        for method in ("FM", "DIFFUSION"):
            variable = f"AUTOCAST_UNET_{method}_{key}_EPOCHS"
            if phase == "timing":
                monkeypatch.delenv(variable, raising=False)
            else:
                monkeypatch.setenv(variable, "3" if key == "AD" else "123")
    subprocess.run(
        ["bash", str(ROOT / "slurm_scripts/ablations/preview_latent_unet.sh"), phase],
        check=True,
        cwd=ROOT,
    )
    commands = capture.read_bytes().rstrip(b"\0").split(b"\0\0")
    assert len(commands) == 8
    for command in commands:
        args = command.decode().split("\0")
        assert "--dry-run" in args
        if phase == "timing":
            assert args[3:6] == ["autocast", "time-epochs", "--kind"]
            assert args[args.index("--kind") + 1] == "processor"
        else:
            assert args[3:5] == ["autocast", "processor"]
        overrides = [arg for arg in args if "=" in arg]
        launcher, module_overrides = _resolve_launcher_submission_context(overrides)
        sbatch = _build_sbatch_command(
            job_name="preview",
            log_dir=tmp_path,
            launcher_cfg=launcher,
            batch_script_path=tmp_path / "submit.sh",
        )
        assert "--nodes=1" in sbatch
        assert "--gpus-per-node=4" in sbatch
        assert "--ntasks-per-node=4" in sbatch
        assert f"--time={'04:00:00' if phase == 'timing' else '24:00:00'}" in sbatch
        if phase == "timing":
            module_overrides += ["++trainer.max_epochs=5", "++trainer.max_time=null"]
        experiment = module_overrides.pop(0).split("=", 1)[1]
        cfg = _compose("processor", experiment, module_overrides)
        OmegaConf.resolve(cfg)
        if phase == "train" and "/fm_unet_" in experiment:
            assert cfg.trainer.callbacks[0].every_n_epochs == max(
                1, cfg.trainer.max_epochs // 4
            )
            assert cfg.trainer.callbacks[0].save_top_k == -1
            assert cfg.trainer.callbacks[0].filename == "quarter-{epoch:04d}"
            instantiate(cfg.trainer.callbacks)
