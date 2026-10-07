"""Small CPU interface checks for the SWE64 presets; no fitting or real data."""

import json
import runpy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from autocast.types import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPERS = REPO_ROOT / "scripts/swe64s"


@pytest.fixture(scope="module")
def swe64_configs() -> dict[str, DictConfig]:
    return runpy.run_path(str(HELPERS / "check_inputs.py"))["check_configs"]()


def test_swe64_afcrps_uses_standard_global_processor(swe64_configs):
    cfg = swe64_configs["afcrps"]
    processor = instantiate(
        cfg.model.processor,
        in_channels=3,
        out_channels=12,
        spatial_resolution=[64, 64],
        hidden_dim=16,
        num_heads=4,
        n_layers=1,
    ).eval()
    encoder = instantiate(cfg.model.encoder, in_channels=3, n_steps_input=1)
    decoder = instantiate(cfg.model.decoder, output_channels=3, time_steps=4)
    batch = Batch(
        input_fields=torch.randn(1, 1, 64, 64, 3),
        output_fields=torch.randn(1, 4, 64, 64, 3),
        constant_scalars=torch.full((1, 1), 0.1),
        constant_fields=None,
    )
    inputs, _ = encoder.encode_with_cond(batch)
    assert inputs.shape == (1, 3, 64, 64)
    with torch.no_grad():
        prediction = decoder.decode(processor.map(inputs, global_cond=None))
    assert prediction.shape == batch.output_fields.shape
    assert torch.isfinite(prediction).all()


def test_swe64_ae_cache_and_fm_shapes(swe64_configs):
    cfg = swe64_configs["autoencoder"]
    # Reduce capacity only; retain the real compression, patching and channels.
    encoder = instantiate(
        cfg.model.encoder, in_channels=3, hid_channels=[8, 16, 32], hid_blocks=[1, 1, 1]
    ).eval()
    decoder = instantiate(
        cfg.model.decoder,
        out_channels=3,
        hid_channels=[32, 16, 8],
        hid_blocks=[1, 1, 1],
    ).eval()
    batch = Batch(
        input_fields=torch.randn(1, 1, 64, 64, 3),
        output_fields=torch.randn(1, 4, 64, 64, 3),
        constant_scalars=None,
        constant_fields=None,
    )
    with torch.no_grad():
        encoded = encoder.encode_batch(batch)
        reconstruction = decoder.decode(encoded.encoded_output_fields)
    assert encoded.encoded_inputs.shape == (1, 1, 16, 16, 8)
    assert encoded.encoded_output_fields.shape == (1, 4, 16, 16, 8)
    assert reconstruction.shape == batch.output_fields.shape

    fm_cfg = OmegaConf.merge(
        swe64_configs["flow_matching"].model.processor,
        {
            "n_steps_output": 4,
            "n_channels_out": 8,
            "flow_ode_steps": 2,  # Sampler smoke check, not the 50-step benchmark.
            "backbone": {
                "in_channels": 8,
                "out_channels": 8,
                "cond_channels": 8,
                "n_steps_input": 1,
                "n_steps_output": 4,
                "global_cond_channels": None,
                "hid_channels": 16,
                "hid_blocks": 1,
                "attention_heads": 4,
                "mod_features": 16,
            },
        },
    )
    processor = instantiate(fm_cfg).eval()
    with torch.no_grad():
        prediction = processor.map(encoded.encoded_inputs, global_cond=None)
    assert prediction.shape == encoded.encoded_output_fields.shape
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(processor.loss(encoded))


@pytest.mark.parametrize("grid_size", [16, 8])
def test_swe64_cache_contract_uses_actual_compression(
    swe64_configs, tmp_path, grid_size
):
    manifest = json.loads(
        (REPO_ROOT / "local_hydra/local_experiment/swe64s/dataset.json").read_text()
    )
    manifest.update(frames=5, splits=dict.fromkeys(("train", "valid", "test"), 1))
    dataset = tmp_path / manifest["directory"]
    dataset.mkdir()
    OmegaConf.save(
        OmegaConf.create({"core_field_names": ["h", "u", "v"]}), dataset / "stats.yml"
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    cfg = OmegaConf.merge(
        swe64_configs["cache_latents"],
        {"datamodule": {"data_path": str(dataset)}},
    )
    OmegaConf.save(cfg, cache / "autoencoder_config.yaml")
    metadata = {
        "latent_channels": 8,
        "splits": {split: {"num_trajectories": 1} for split in manifest["splits"]},
    }
    (cache / "metadata.json").write_text(json.dumps(metadata))
    for split in manifest["splits"]:
        (cache / split).mkdir()
        torch.save(
            {"encoded_fields": torch.zeros(5, grid_size, grid_size, 8)},
            cache / split / "traj_000000.pt",
        )
    check_cache = runpy.run_path(str(HELPERS / "check_inputs.py"))["check_cache"]
    if grid_size == 16:
        result = check_cache(cache, manifest, swe64_configs["cache_latents"])
        assert result["sample_shape"] == [5, 16, 16, 8]
    else:
        with pytest.raises(ValueError, match="expected full"):
            check_cache(cache, manifest, swe64_configs["cache_latents"])


def test_swe64_evaluation_commands_preserve_saved_model(swe64_configs):
    build_commands = runpy.run_path(str(HELPERS / "evaluate.py"))["build_commands"]
    commands = build_commands(
        run_dir=Path("/preview/run"),
        checkpoint=Path("/preview/model.ckpt"),
        output_root=Path("/preview/eval"),
        autoencoder_checkpoint=None,
    )
    for index, command in enumerate(commands):
        with initialize_config_dir(
            version_base=None, config_dir=str(REPO_ROOT / "src/autocast/configs")
        ):
            cfg = compose(
                config_name="encoder_processor_decoder",
                overrides=[
                    "local_experiment=swe64s/afcrps",
                    "trainer.max_epochs=1",
                    f"hydra.searchpath=[file://{REPO_ROOT / 'local_hydra'}]",
                    *[arg for arg in command if arg.startswith("++")],
                ],
            )
        assert cfg.model == swe64_configs["afcrps"].model
        assert cfg.eval.teacher_forcing_ratio == float(index)
        assert cfg.eval.free_running_only == (index == 0)
        assert list(cfg.eval.coverage_levels) == [0.9]
        assert "--execute" not in command
