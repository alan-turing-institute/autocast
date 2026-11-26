"""Spherical batch types keep their grid through every batch operation."""

import pytest
import torch

from autocast.types import (
    Batch,
    EncodedBatch,
    EncodedSphericalBatch,
    GridType,
    SphericalBatch,
    SphericalSample,
)

SPHERE_GRIDS = [grid for grid in GridType if grid is not GridType.CARTESIAN]


def _fields(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "input_fields": torch.randn(batch_size, 1, 4, 8, 3),
        "output_fields": torch.randn(batch_size, 2, 4, 8, 3),
        "constant_scalars": torch.randn(batch_size, 2),
        "constant_fields": torch.randn(batch_size, 4, 8, 2),
    }


def test_grid_names_are_the_torch_harmonics_names():
    torch_harmonics = pytest.importorskip("torch_harmonics")

    assert {grid.value for grid in SPHERE_GRIDS} == set(torch_harmonics.grid_types())


def test_grid_type_is_its_string():
    assert GridType.LEGENDRE_GAUSS == "legendre-gauss"
    assert str(GridType.LEGENDRE_GAUSS) == "legendre-gauss"
    assert GridType("equiangular") is GridType.EQUIANGULAR


@pytest.mark.parametrize("grid", SPHERE_GRIDS)
def test_batch_operations_keep_the_grid(grid, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch = SphericalBatch(**_fields(), grid_type=grid)

    for result in (batch.to("cpu"), batch.pin_memory(), batch.repeat(3)):
        assert type(result) is SphericalBatch
        assert result.grid_type is grid
    assert batch.repeat(3).input_fields.shape[0] == 6


def test_encoded_batch_operations_keep_the_grid(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    encoded = EncodedSphericalBatch(
        encoded_inputs=torch.randn(2, 1, 4, 8, 3),
        encoded_output_fields=torch.randn(2, 2, 4, 8, 3),
        global_cond=torch.randn(2, 2),
        encoded_info={},
        grid_type=GridType.HEALPIX,
    )

    for result in (encoded.to("cpu"), encoded.pin_memory(), encoded.repeat(2)):
        assert type(result) is EncodedSphericalBatch
        assert result.grid_type is GridType.HEALPIX


def test_plain_batches_stay_plain(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch = Batch(**_fields())
    encoded = EncodedBatch(
        encoded_inputs=torch.randn(2, 1, 4, 8, 3),
        encoded_output_fields=torch.randn(2, 2, 4, 8, 3),
        global_cond=None,
        encoded_info={},
    )

    for result in (batch.to("cpu"), batch.pin_memory(), batch.repeat(2)):
        assert type(result) is Batch
    for result in (encoded.to("cpu"), encoded.pin_memory(), encoded.repeat(2)):
        assert type(result) is EncodedBatch


def test_spherical_types_refuse_the_cartesian_grid():
    sample_fields = {
        "input_fields": torch.randn(1, 4, 8, 3),
        "output_fields": torch.randn(1, 4, 8, 3),
        "constant_scalars": None,
        "constant_fields": None,
        "boundary_conditions": None,
    }

    with pytest.raises(ValueError, match="grid on the sphere"):
        SphericalSample(**sample_fields, grid_type=GridType.CARTESIAN)
    with pytest.raises(ValueError, match="grid on the sphere"):
        SphericalBatch(**_fields(), grid_type=GridType.CARTESIAN)
