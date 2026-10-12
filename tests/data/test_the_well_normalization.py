import fsspec
import h5py
import pytest
import torch
import yaml
from einops import repeat
from hydra.utils import instantiate
from omegaconf import OmegaConf
from the_well.data.normalization import RMSNormalization, ZScoreNormalization

from autocast.data.dataset import TheWell
from autocast.scripts.setup import _load_residual_delta_normalizer, setup_epd_model


@pytest.mark.parametrize("normalization_type", [ZScoreNormalization, RMSNormalization])
@pytest.mark.parametrize("local_override", [False, True])
def test_well_delta_stats_use_metadata_and_existing_filesystem(
    native_well_files, normalization_type, local_override
):
    dataset = TheWell(
        well_base_path=str(native_well_files),
        well_dataset_name="rayleigh_benard",
        well_split_name="train",
        use_normalization=False,
        normalization_type=normalization_type,
    )
    stats_path = native_well_files / "rayleigh_benard/stats.yaml"
    remote_path = f"memory://{native_well_files.parent.name}/stats.yaml"
    fs = fsspec.filesystem("memory")
    fs.pipe(remote_path, stats_path.read_bytes())
    dataset.well_dataset.fs = fs
    normalizer = _load_residual_delta_normalizer(
        stats_path if local_override else remote_path, dataset
    )
    expected = (
        [0.75, 1.0, 5.0]
        if normalization_type is ZScoreNormalization
        else [0.75, 4.0, 5.0]
    )
    torch.testing.assert_close(
        normalizer.delta_denormalize_flattened(torch.ones(1, 3), "variable"),
        torch.tensor([expected]),
    )


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
@pytest.mark.parametrize("use_delta_stats", [False, True])
@pytest.mark.parametrize("n_members", [1, 3])
@pytest.mark.parametrize("normalization_type", [ZScoreNormalization, RMSNormalization])
@pytest.mark.parametrize("filesystem", [False, True])
def test_native_well_base_path_override_loads_model_statistics(
    native_well_files,
    monkeypatch,
    REPO_ROOT,
    epd_config,
    use_normalization,
    use_delta_stats,
    n_members,
    normalization_type,
    filesystem,
):
    monkeypatch.setenv("AUTOCAST_DATASETS", str(native_well_files / "wrong"))
    root = native_well_files.as_uri() if filesystem else str(native_well_files)
    config = epd_config
    config.datamodule = OmegaConf.load(
        REPO_ROOT / "src/autocast/configs/datamodule/the_well.yaml"
    )
    config.model.update(
        residual_prediction=True,
        residual_use_delta_stats=use_delta_stats,
        n_members=n_members,
        residual_delta_stats_path="${datamodule.normalization_path}",
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
        {
            "channel_count": 3,
            "example_batch": batch,
            "n_steps_input": 1,
            "n_steps_output": 1,
        },
        datamodule,
    )
    increment = batch.input_fields
    if use_delta_stats:
        if normalization_type is ZScoreNormalization:
            increment = increment * torch.tensor([0.25, 2.0, 3.0]) + torch.tensor(
                [0.5, -1.0, 2.0]
            )
        else:
            increment = increment * torch.tensor([0.75, 4.0, 5.0])
    elif use_normalization:
        increment = increment * torch.tensor(
            [2.0, 3.0, 4.0]
            if normalization_type is ZScoreNormalization
            else [12.0, 20.0, 6.0]
        )
    expected = torch.tensor([14.0, 26.0, 3.0]) + increment
    if n_members > 1:
        expected = repeat(expected, "... -> ... m", m=n_members)
    torch.testing.assert_close(model.predict_step(batch, 0), expected)
    model.residual_prediction = False
    direct = torch.tensor([14.0, 26.0, 3.0]).expand_as(batch.input_fields)
    torch.testing.assert_close(
        model.predict_step(batch, 0),
        repeat(direct, "... -> ... m", m=n_members) if n_members > 1 else direct,
    )
    config.model.residual_delta_stats_path = f"{root}/rayleigh_benard/missing.yaml"
    if use_delta_stats:
        with pytest.raises(FileNotFoundError):
            setup_epd_model(
                config,
                {
                    "channel_count": 3,
                    "example_batch": batch,
                    "n_steps_input": 1,
                    "n_steps_output": 1,
                },
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
