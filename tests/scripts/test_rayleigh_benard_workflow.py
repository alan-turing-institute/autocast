"""Exercise both RB cache routes with tiny Well files and a LoLA autoencoder."""

from pathlib import Path

import h5py
import lightning as L
import pandas as pd
import pytest
import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from the_well.utils.dummy_data import write_dummy_data

from autocast.external.lola.lola_autoencoder import get_autoencoder
from autocast.scripts.cache_latents import cache_latents
from autocast.scripts.eval.encoder_processor_decoder import run_evaluation
from autocast.scripts.setup import setup_datamodule, setup_processor_model


@pytest.fixture
def rb_cache_config(tmp_path, REPO_ROOT):
    dataset_dir = tmp_path / "datasets" / "rayleigh_benard"
    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / "data" / split
        split_dir.mkdir(parents=True)
        path = split_dir / "tiny.hdf5"
        write_dummy_data(str(path))
        with h5py.File(path, "r+") as data:
            data.attrs["dataset_name"] = "rayleigh_benard"
            scalars = data["scalars"]
            assert isinstance(scalars, h5py.Group)
            scalars.move("a", "Rayleigh")
            scalars.move("b", "Prandtl")
            scalars.attrs["field_names"] = [
                "Rayleigh",
                "Prandtl",
                "time_varying_scalar",
            ]

    config_dir = REPO_ROOT / "src" / "autocast" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"hydra.searchpath=[file://{REPO_ROOT / 'local_hydra'}]",
                "local_experiment=cache_latents/the_well/rayleigh_benard/lola_f32c64",
                f"lola_autoencoder_path={dataset_dir / 'tiny_autoencoder'}",
                f"datamodule.well_base_path={tmp_path / 'datasets'}",
            ],
        )
    assert cfg.datamodule.max_rollout_steps == float("inf")
    assert cfg.model.encoder.pix_channels == 4
    # Keep the checkpoint normalization/conditioning, but shrink the network
    # and use the dummy file's two variable channels for this CPU fixture.
    for component in (cfg.model.encoder, cfg.model.decoder):
        component.pix_channels = 2
        component.mean = [0.38, 0.01]
        component.std = [0.25, 0.15]
        component.hid_channels = [4, 8]
        component.hid_blocks = [1, 1]
        component.lat_channels = 2
    run_dir = Path(cfg.lola_autoencoder_path)
    run_dir.mkdir()
    ae_config = OmegaConf.to_container(cfg.model.encoder, resolve=True)
    assert isinstance(ae_config, dict)
    ae_kwargs = {str(key): value for key, value in ae_config.items()}
    for key in (
        "_target_",
        "mean",
        "std",
        "log_scalars",
        "device",
        "runpath",
        "chunk_size",
    ):
        ae_kwargs.pop(key)
    torch.save(get_autoencoder(**ae_kwargs).state_dict(), run_dir / "state.pth")
    OmegaConf.save(
        OmegaConf.create(
            {
                "ae": ae_kwargs,
                "dataset": {
                    "name": "rayleigh_benard",
                    "augment": ["log_scalars"],
                    "stats": {"mean": [0.38, 0.01], "std": [0.25, 0.15]},
                },
            }
        ),
        run_dir / "config.yaml",
    )
    return cfg


@pytest.mark.parametrize("cache_format", ["native", "lola"])
@pytest.mark.parametrize("processor", ["flow_matching_vit", "diffusion_vit"])
def test_rb_cache_train_and_physical_eval(
    rb_cache_config, tmp_path, REPO_ROOT, cache_format, processor
):
    cfg = rb_cache_config
    native_dir = cache_latents(cfg, tmp_path / "native", device="cpu")
    saved = (native_dir / "autoencoder_config.yaml").read_text()
    assert "${" not in saved
    trajectory = torch.load(native_dir / "train" / "traj_000000.pt", weights_only=True)
    assert trajectory["encoded_fields"].shape[0] == 10
    assert trajectory["global_cond"].dtype == trajectory["encoded_fields"].dtype
    assert torch.allclose(
        trajectory["global_cond"][:2], torch.tensor([0.25, 0.75]).log()
    )

    data_path = native_dir
    if cache_format == "lola":
        # Match LoLA's on-disk state/label schema using the same encoded values.
        data_path = Path(cfg.lola_autoencoder_path) / "cache" / "rayleigh_benard"
        for split in ("train", "valid", "test"):
            split_dir = data_path / split
            split_dir.mkdir(parents=True)
            trajectories = [
                torch.load(path, weights_only=True)
                for path in sorted((native_dir / split).glob("traj_*.pt"))
            ]
            with h5py.File(split_dir / "tiny.hdf5", "w") as data:
                fields = torch.stack([item["encoded_fields"] for item in trajectories])
                data["state"] = rearrange(fields, "b t h w c -> b t c h w").numpy()
                data["label"] = torch.stack(
                    [item["global_cond"] for item in trajectories]
                ).numpy()

    datamodule = "cached_latents" if cache_format == "native" else "miniwell"
    sampling_steps = (
        "flow_ode_steps" if processor == "flow_matching_vit" else "sampler_steps"
    )
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "src" / "autocast" / "configs")
    ):
        processor_cfg = compose(
            config_name="processor",
            overrides=[
                f"datamodule={datamodule}",
                f"datamodule.data_path={data_path}",
                "datamodule.batch_size=1",
                "datamodule.num_workers=0",
                f"processor@model.processor={processor}",
                f"model.processor.{sampling_steps}=1",
                "model.processor.backbone.hid_channels=8",
                "model.processor.backbone.hid_blocks=1",
                "model.processor.backbone.attention_heads=1",
            ],
        )
    dm, processor_cfg, stats = setup_datamodule(processor_cfg)
    model = setup_processor_model(processor_cfg, stats, datamodule=dm)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        max_epochs=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    checkpoint = tmp_path / "processor.ckpt"
    trainer.save_checkpoint(checkpoint)
    processor_cfg.eval = {
        "mode": "auto",
        "accelerator": "cpu",
        "chunk_size": 1,
        "checkpoint": str(checkpoint),
        "max_test_batches": 1,
        "max_rollout_batches": 1,
        "max_rollout_steps": 4,
        "metrics": ["mse"],
        "batch_indices": [],
    }
    (tmp_path / "eval").mkdir()
    run_evaluation(processor_cfg, work_dir=tmp_path / "eval")
    assert processor_cfg.datamodule.well_dataset_name == "rayleigh_benard"
    assert processor_cfg.datamodule.use_normalization is False
    assert processor_cfg.model.encoder.runpath == cfg.lola_autoencoder_path
    metrics = pd.read_csv(tmp_path / "eval" / "evaluation_metrics.csv")
    assert not metrics.empty
    assert torch.isfinite(torch.as_tensor(metrics["mse"].to_numpy())).all()
