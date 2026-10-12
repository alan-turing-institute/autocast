from unittest.mock import Mock

import fsspec
import h5py
import pytest
import torch
import yaml
from einops import repeat
from hydra.utils import instantiate
from omegaconf import OmegaConf
from the_well.data.normalization import RMSNormalization, ZScoreNormalization

from autocast.data.datamodule import TheWellDataModule
from autocast.data.dataset import TheWell
from autocast.scripts.setup import setup_epd_model


@pytest.mark.parametrize("use_normalization", [True, False])
@pytest.mark.parametrize("delta_source", [None, "state", "local", "filesystem"])
@pytest.mark.parametrize("n_members", [1, 3])
@pytest.mark.parametrize("normalization_type", [ZScoreNormalization, RMSNormalization])
def test_well_model_setup_preserves_output_units(
    tmp_path,
    monkeypatch,
    make_toy_batch,
    use_normalization,
    delta_source,
    n_members,
    normalization_type,
):
    """Use native statistics, filesystem and field order for either state mode."""
    # Native Well statistics have no wrapper or field-name list. Metadata orders
    # the vector before the scalar, unlike the YAML mapping's alphabetical order.
    stats = {
        "mean": {"a": 10.0, "velocity": [20.0, -5.0]},
        "std": {"a": 2.0, "velocity": [3.0, 4.0]},
        "mean_delta": {"a": 0.5, "velocity": [-1.0, 2.0]},
        "std_delta": {"a": 0.25, "velocity": [2.0, 3.0]},
        "rms": {"a": 12.0, "velocity": [20.0, 6.0]},
        "rms_delta": {"a": 0.75, "velocity": [4.0, 5.0]},
    }
    field_names = ["velocity", "a"]
    normalizer = normalization_type(stats, field_names, [])
    fs = fsspec.filesystem("memory")
    stats_path = f"memory://{tmp_path.name}/stats.yaml"
    fs.pipe(stats_path, yaml.safe_dump(stats).encode())
    if delta_source == "local":
        local_stats = tmp_path / "stats.yaml"
        local_stats.write_text(yaml.safe_dump(stats))
        stats_path = str(local_stats)
    residual_prediction = delta_source is not None
    delta_file = delta_source in {"local", "filesystem"}
    # Replace external HDF5 discovery, retaining real filesystem/normalizers.
    loader = Mock(
        norm=normalizer if use_normalization else None,
        metadata=None,
        fs=fs,
        core_field_names=field_names,
        core_constant_field_names=[],
        normalization_type=normalization_type,
    )
    monkeypatch.setattr("autocast.data.dataset.WellDataset", Mock(return_value=loader))
    dataset = TheWell(path="unused", use_normalization=use_normalization)
    batch = make_toy_batch(batch_size=1, t_in=1, w=2, h=2, c=3)
    config = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "_target_": "autocast.encoders.identity.IdentityEncoder",
                    "in_channels": 3,
                },
                "decoder": {
                    "_target_": "autocast.decoders.identity.IdentityDecoder",
                    "in_channels": 3,
                },
                "processor": {"_target_": "conftest.CondCaptureProcessor"},
                "n_members": n_members,
                "residual_prediction": residual_prediction,
                "residual_delta_stats_path": stats_path if delta_file else None,
            },
            "optimizer": {"optimizer": "adamw", "learning_rate": 1e-4},
        }
    )
    model = setup_epd_model(
        config,
        {"example_batch": batch, "n_steps_input": 1, "n_steps_output": 1},
        Mock(spec=TheWellDataModule, train_dataset=dataset),
    )

    # Training stays in model units; predict_step restores physical units once.
    baseline = batch.input_fields
    if normalization_type is ZScoreNormalization:
        state_mean = torch.tensor([20.0, -5.0, 10.0])
        state_scale = torch.tensor([3.0, 4.0, 2.0])
        increment = baseline * torch.tensor([2.0, 3.0, 0.25])
        increment = increment + torch.tensor([-1.0, 2.0, 0.5])
    else:
        state_mean = torch.zeros(3)
        state_scale = torch.tensor([20.0, 6.0, 12.0])
        increment = baseline * torch.tensor([4.0, 5.0, 0.75])
    expected = baseline
    if residual_prediction:
        if delta_file or use_normalization:
            expected = baseline + (
                increment / state_scale if use_normalization else increment
            )
        else:
            expected = baseline + baseline
    physical = expected
    if use_normalization:
        physical = expected * state_scale + state_mean
    if n_members > 1:
        expected = repeat(expected, "... -> ... m", m=n_members)
        physical = repeat(physical, "... -> ... m", m=n_members)
    torch.testing.assert_close(model(batch), expected)
    torch.testing.assert_close(model.predict_step(batch, 0), physical)


@pytest.fixture
def native_well_files(tmp_path):
    """Minimal native files read by the actual WellDataset, without mocks."""
    root = tmp_path / "well"
    dataset_path = root / "rayleigh_benard"
    dataset_path.mkdir(parents=True)
    stats = {
        "mean": {"z": 10.0, "velocity": [20.0, -5.0]},
        "std": {"z": 2.0, "velocity": [3.0, 4.0]},
        "mean_delta": {"z": 0.5, "velocity": [-1.0, 2.0]},
        "std_delta": {"z": 0.25, "velocity": [2.0, 3.0]},
        "rms": {"z": 12.0, "velocity": [20.0, 6.0]},
        "rms_delta": {"z": 0.75, "velocity": [4.0, 5.0]},
    }
    (dataset_path / "stats.yaml").write_text(yaml.safe_dump(stats))
    physical = torch.tensor([14.0, 26.0, 3.0]).expand(4, 2, 2, 3)
    for split in ("train", "valid", "test"):
        split_path = dataset_path / "data" / split
        split_path.mkdir(parents=True)
        with h5py.File(split_path / "sample.h5", "w") as handle:
            handle.attrs.update(
                grid_type="cartesian",
                n_trajectories=1,
                n_spatial_dims=2,
                dataset_name="rayleigh_benard",
            )
            dimensions = handle.create_group("dimensions")
            dimensions.attrs["spatial_dims"] = ["x", "y"]
            for name, size in (("time", 4), ("x", 2), ("y", 2)):
                dimension = dimensions.create_dataset(name, data=list(range(size)))
                dimension.attrs["sample_varying"] = False
            handle.create_group("scalars").attrs["field_names"] = []
            handle.create_group("boundary_conditions")
            for order, names in enumerate((["z"], ["velocity"], [])):
                fields = handle.create_group(f"t{order}_fields")
                fields.attrs["field_names"] = names
                for name in names:
                    data = physical[..., 0] if order == 0 else physical[..., 1:]
                    field = fields.create_dataset(name, data=data.numpy())
                    field.attrs.update(
                        time_varying=True,
                        sample_varying=False,
                        dim_varying=[True, True],
                    )
    return root


@pytest.mark.parametrize("use_normalization", [False, True])
@pytest.mark.parametrize("normalization_type", [ZScoreNormalization, RMSNormalization])
@pytest.mark.parametrize("filesystem", [False, True])
def test_native_well_base_path_override_loads_model_statistics(
    native_well_files,
    monkeypatch,
    REPO_ROOT,
    use_normalization,
    normalization_type,
    filesystem,
):
    monkeypatch.setenv("AUTOCAST_DATASETS", str(native_well_files / "wrong"))
    root = native_well_files.as_uri() if filesystem else str(native_well_files)
    config = OmegaConf.create(
        {
            "datamodule": OmegaConf.load(
                REPO_ROOT / "src/autocast/configs/datamodule/the_well.yaml"
            ),
            "model": {
                "encoder": {
                    "_target_": "autocast.encoders.identity.IdentityEncoder",
                    "in_channels": 3,
                },
                "decoder": {
                    "_target_": "autocast.decoders.identity.IdentityDecoder",
                    "in_channels": 3,
                },
                "processor": {"_target_": "conftest.CondCaptureProcessor"},
                "residual_prediction": True,
                "residual_delta_stats_path": "${datamodule.normalization_path}",
            },
            "optimizer": {"optimizer": "adamw", "learning_rate": 1e-4},
        }
    )
    normalizer_target = f"the_well.data.normalization.{normalization_type.__name__}"
    config.datamodule.update(
        {
            "well_base_path": root,
            "well_dataset_name": "rayleigh_benard",
            "n_steps_output": 1,
            "batch_size": 1,
            "use_normalization": use_normalization,
            "normalization_type": {
                "_target_": normalizer_target,
                "_partial_": True,
            },
        }
    )
    datamodule = instantiate(config.datamodule)
    batch = next(iter(datamodule.train_dataloader()))
    loader = datamodule.train_dataset.well_dataset
    assert loader.core_field_names == ["z", "velocity"]
    assert (
        loader.fs.info(config.model.residual_delta_stats_path)["name"]
        == loader.fs.info(loader.normalization_path)["name"]
    )
    model = setup_epd_model(
        config,
        {"example_batch": batch, "n_steps_input": 1, "n_steps_output": 1},
        datamodule,
    )
    if normalization_type is ZScoreNormalization:
        increment = batch.input_fields * torch.tensor([0.25, 2.0, 3.0])
        increment = increment + torch.tensor([0.5, -1.0, 2.0])
    else:
        increment = batch.input_fields * torch.tensor([0.75, 4.0, 5.0])
    expected = torch.tensor([14.0, 26.0, 3.0]) + increment
    torch.testing.assert_close(model.predict_step(batch, 0), expected)
    # An explicit delta file is required even when state stats are available.
    config.model.residual_delta_stats_path = f"{root}/rayleigh_benard/missing.yaml"
    with pytest.raises(FileNotFoundError):
        setup_epd_model(
            config,
            {"example_batch": batch, "n_steps_input": 1, "n_steps_output": 1},
            datamodule,
        )


@pytest.mark.parametrize("invalid_stats", [None, "", "mean: [", "mean: {}"])
def test_native_well_requires_requested_normalization_stats(
    native_well_files, invalid_stats
):
    stats_path = native_well_files / "rayleigh_benard/stats.yaml"
    if invalid_stats is None:
        stats_path.unlink()
    else:
        stats_path.write_text(invalid_stats)
    with (
        pytest.warns(UserWarning, match=r"[Nn]ormalization file"),
        pytest.raises(ValueError, match=r"normalization.*statistics"),
    ):
        TheWell(
            well_base_path=str(native_well_files),
            well_dataset_name="rayleigh_benard",
            well_split_name="train",
            use_normalization=True,
        )


@pytest.mark.parametrize(
    ("use_normalization", "normalization_type"),
    [(False, ZScoreNormalization), (True, None)],
)
def test_native_well_allows_explicitly_unnormalized_data(
    native_well_files, use_normalization, normalization_type
):
    (native_well_files / "rayleigh_benard/stats.yaml").unlink()
    dataset = TheWell(
        well_base_path=str(native_well_files),
        well_dataset_name="rayleigh_benard",
        well_split_name="train",
        use_normalization=use_normalization,
        normalization_type=normalization_type,
    )
    assert dataset.norm is None
    expected = torch.tensor([14.0, 26.0, 3.0]).expand(1, 2, 2, 3)
    torch.testing.assert_close(dataset[0].input_fields, expected)
