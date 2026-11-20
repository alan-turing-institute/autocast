"""Model normalization follows the available split during Zarr evaluation."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from autocast.scripts.setup import (
    setup_autoencoder_model,
    setup_datamodule,
    setup_epd_model,
    setup_processor_model,
)

zarr = pytest.importorskip("zarr")


@pytest.mark.parametrize(
    "builder", [setup_autoencoder_model, setup_processor_model, setup_epd_model]
)
@pytest.mark.parametrize("splits", [("test",), ("train", "valid", "test")])
@pytest.mark.parametrize("use_normalization", [True, False])
def test_model_denormalizes_available_zarr_split(
    tmp_path, builder, splits, use_normalization
):
    for split in splits:
        group = zarr.open_group(str(tmp_path / split / "data.zarr"), mode="w")
        group.create_array(
            "data", data=np.full((1, 4, 2, 2, 1), 12.0, dtype=np.float32)
        )
        group.attrs.update(
            schema="autosim.spatiotemporal",
            schema_version=1,
            complete=True,
            field_names=["field_0"],
        )
    config = OmegaConf.create(
        {
            "datamodule": {
                "_target_": "autocast.data.zarr_dataset.ZarrDataModule",
                "data_path": str(tmp_path),
                "batch_size": 1,
                "pin_memory": False,
                "use_normalization": use_normalization,
                "normalization_stats": {
                    "stats": {
                        name: {"field_0": value}
                        for name, value in {
                            "mean": 10.0,
                            "std": 2.0,
                            "mean_delta": 0.0,
                            "std_delta": 1.0,
                        }.items()
                    }
                },
            },
            "model": {
                "encoder": {
                    "_target_": "autocast.encoders.identity.IdentityEncoder",
                    "in_channels": "auto",
                },
                "decoder": {
                    "_target_": "autocast.decoders.identity.IdentityDecoder",
                    "in_channels": "auto",
                },
                "processor": {
                    "_target_": "autocast.processors.vit_latent.AViTLatentProcessor",
                    "global_cond_channels": 0,
                    "include_global_cond": False,
                    "hidden_dim": 8,
                    "num_heads": 2,
                    "n_layers": 1,
                },
            },
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
        }
    )
    datamodule, config, stats = setup_datamodule(config)
    model = builder(config, stats, datamodule)

    targets = stats["example_batch"].output_fields
    torch.testing.assert_close(
        targets, torch.full_like(targets, 1.0 if use_normalization else 12.0)
    )
    torch.testing.assert_close(
        model.denormalize_tensor(targets), torch.full_like(targets, 12.0)
    )


@pytest.mark.parametrize("splits", [("test",), ("train", "valid", "test")])
@pytest.mark.parametrize("channel_idxs", [[1, 0], [1]])
@pytest.mark.parametrize("use_normalization", [True, False])
def test_residual_stats_follow_available_zarr_channels(
    tmp_path, epd_config, splits, channel_idxs, use_normalization
):
    fields = ["field_0", "field_1"]
    for split in splits:
        group = zarr.open_group(str(tmp_path / split / "data.zarr"), mode="w")
        group.create_array("data", data=np.ones((1, 4, 2, 2, 2), dtype=np.float32))
        group.attrs.update(
            schema="autosim.spatiotemporal",
            schema_version=1,
            complete=True,
            field_names=fields,
        )
    stats_path = tmp_path / "stats.yml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "core_field_names": fields,
                "constant_field_names": [],
                "stats": {
                    "mean": dict.fromkeys(fields, 0.0),
                    "std": dict.fromkeys(fields, 1.0),
                    "mean_delta": dict.fromkeys(fields, 0.0),
                    "std_delta": dict(zip(fields, [1.0, 10.0], strict=True)),
                },
            }
        ),
        stats_path,
    )
    epd_config.datamodule = {
        "_target_": "autocast.data.zarr_dataset.ZarrDataModule",
        "data_path": str(tmp_path),
        "channel_idxs": channel_idxs,
        "pin_memory": False,
        "use_normalization": use_normalization,
    }
    epd_config.model.update(
        residual_prediction=True, residual_delta_stats_path=str(stats_path)
    )
    datamodule, config, stats = setup_datamodule(epd_config)
    model = setup_epd_model(config, stats, datamodule)
    batch = stats["example_batch"]
    expected = torch.tensor([2.0, 11.0])[channel_idxs]
    torch.testing.assert_close(model(batch), expected.expand_as(batch.output_fields))
