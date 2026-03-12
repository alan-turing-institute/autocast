import pytest

from autocast.data.utils import get_datamodule


def test_get_datamodule_well_rejects_normalization_stats():
    with pytest.raises(ValueError, match=r"normalization_stats.*the_well=True"):
        get_datamodule(
            the_well=True,
            simulation_name="dummy_well_dataset",
            n_steps_input=1,
            n_steps_output=1,
            stride=1,
            normalization_stats={
                "stats": {},
                "core_field_names": [],
                "constant_field_names": [],
            },
        )
