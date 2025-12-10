from dataclasses import dataclass
from typing import Literal


@dataclass
class Metadata:
    """Metadata for spatiotemporal datasets."""

    dataset_name: str
    n_spatial_dims: int
    spatial_resolution: tuple[int, ...]
    scalar_names: list[str]
    constant_scalar_names: list[str]
    constant_field_names: dict[str, list[str]]
    boundary_condition_types: list[str]
    field_names: dict[int, list[str]]
    n_steps_per_trajectory: list[int]
    n_files: int | None = None
    n_trajectories_per_file: list[int] | None = None
    grid_type: Literal["cartesian"] = "cartesian"
