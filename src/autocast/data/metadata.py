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
    # Names of time-varying external drivers, keyed by tensor order like
    # `field_names`. These are inputs the model is conditioned on but never
    # predicts, so they are listed separately from the state `field_names`.
    forcing_field_names: dict[int, list[str]] | None = None
    n_files: int | None = None
    n_trajectories_per_file: list[int] | None = None
    grid_type: Literal["cartesian"] = "cartesian"
