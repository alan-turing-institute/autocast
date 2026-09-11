import torch

from autocast.data.dataset import ReactionDiffusionDataset
from autocast.data.metadata import Metadata
from autocast.types import Sample
from autocast.types.collation import collate_batches


def _data_with_forcing(n_traj: int = 2, n_t: int = 8, w: int = 4, h: int = 4):
    return {
        "data": torch.randn(n_traj, n_t, w, h, 2),
        "constant_scalars": None,
        "constant_fields": None,
        "forcing_fields": torch.randn(n_traj, n_t, w, h, 1),
    }


def test_dataset_without_forcing_fields_has_none():
    data = _data_with_forcing()
    del data["forcing_fields"]
    dataset = ReactionDiffusionDataset(
        data_path=None, data=data, n_steps_input=2, n_steps_output=1
    )
    assert dataset[0].forcing_fields is None


def test_dataset_windows_forcing_fields_alongside_data():
    data = _data_with_forcing()
    dataset = ReactionDiffusionDataset(
        data_path=None, data=data, n_steps_input=2, n_steps_output=1
    )

    sample = dataset[0]
    assert sample.forcing_fields is not None
    # Forcing fields span the full input+output window (T_in + T_out = 3),
    # not just the input window, since they are never a prediction target.
    assert sample.forcing_fields.shape == (3, 4, 4, 1)
    assert torch.equal(sample.forcing_fields, data["forcing_fields"][0, :3])


def test_dataset_forcing_fields_windowing_matches_second_trajectory():
    data = _data_with_forcing()
    dataset = ReactionDiffusionDataset(
        data_path=None, data=data, n_steps_input=2, n_steps_output=1
    )
    # 8 timesteps, window size 3, stride 1 -> 6 windows per trajectory.
    sample = dataset[6]
    assert sample.forcing_fields is not None
    assert torch.equal(sample.forcing_fields, data["forcing_fields"][1, :3])


def test_to_preloaded_data_roundtrips_forcing_fields():
    data = _data_with_forcing()
    dataset = ReactionDiffusionDataset(
        data_path=None, data=data, n_steps_input=2, n_steps_output=1
    )
    payload = dataset.to_preloaded_data()
    assert payload["forcing_fields"] is dataset.forcing_fields

    rebuilt = ReactionDiffusionDataset(
        data_path=None, data=payload, n_steps_input=2, n_steps_output=1
    )
    assert torch.equal(rebuilt[0].forcing_fields, dataset[0].forcing_fields)


def test_collate_batches_stacks_forcing_fields():
    samples = [
        Sample(
            input_fields=torch.randn(2, 4, 4, 2),
            output_fields=torch.randn(1, 4, 4, 2),
            constant_scalars=None,
            constant_fields=None,
            boundary_conditions=None,
            forcing_fields=torch.full((3, 4, 4, 1), float(i)),
        )
        for i in range(3)
    ]

    batch = collate_batches(samples)

    assert batch.forcing_fields is not None
    assert batch.forcing_fields.shape == (3, 3, 4, 4, 1)
    for i in range(3):
        assert torch.equal(batch.forcing_fields[i], samples[i].forcing_fields)


def test_collate_batches_forcing_fields_none_when_all_none():
    samples = [
        Sample(
            input_fields=torch.randn(2, 4, 4, 2),
            output_fields=torch.randn(1, 4, 4, 2),
            constant_scalars=None,
            constant_fields=None,
            boundary_conditions=None,
        )
        for _ in range(2)
    ]
    batch = collate_batches(samples)
    assert batch.forcing_fields is None


def test_metadata_forcing_field_names_defaults_to_none():
    metadata = Metadata(
        dataset_name="test",
        n_spatial_dims=2,
        spatial_resolution=(4, 4),
        scalar_names=[],
        constant_scalar_names=[],
        constant_field_names={},
        boundary_condition_types=[],
        field_names={0: ["U"]},
        n_steps_per_trajectory=[8],
    )
    assert metadata.forcing_field_names is None

    metadata_with_forcing = Metadata(
        dataset_name="test",
        n_spatial_dims=2,
        spatial_resolution=(4, 4),
        scalar_names=[],
        constant_scalar_names=[],
        constant_field_names={},
        boundary_condition_types=[],
        field_names={0: ["U"]},
        n_steps_per_trajectory=[8],
        forcing_field_names={0: ["era5_wind"]},
    )
    assert metadata_with_forcing.forcing_field_names == {0: ["era5_wind"]}
