"""Module for downsampling TheWellDataset to create lower resolution datasets.

This module provides utilities to downsample TheWell datasets both spatially
and temporally, creating new TheWell-compatible datasets at reduced resolution.
This is useful for:
- Creating smaller datasets for faster experimentation
- Training on lower resolution before fine-tuning on full resolution
- Memory-efficient debugging and development

Usage:
    python -m autocast.data.downsample \
        --input-path /path/to/dataset \
        --output-path /path/to/output \
        --spatial-factor 2 \
        --temporal-factor 2

Or programmatically:
    from autocast.data.downsample import downsample_dataset
    downsample_dataset(
        input_path="/path/to/dataset",
        output_path="/path/to/output",
        spatial_downsample_factor=2,
        temporal_downsample_factor=2,
    )
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def downsample_field(
    data: np.ndarray,
    *,
    time_varying: bool | np.bool_,
    spatial_filtering: bool | np.bool_,
    n_batch_dims: int,
    n_tensor_dims: int,
    spatial_downsample_factor: int,
    temporal_downsample_factor: int,
    time_fraction: float = 1.0,
) -> np.ndarray:
    """Downsample a field array with optional spatial filtering.

    Args:
        data: Input array to downsample.
        time_varying: Whether the field varies in time.
        spatial_filtering: Whether to apply Gaussian filtering before downsampling.
        n_batch_dims: Number of batch dimensions (e.g., sample dimension).
        n_tensor_dims: Number of tensor dimensions (0 for scalars, 1 for vectors, 2 for tensors).
        spatial_downsample_factor: Factor by which to downsample spatial dimensions.
        temporal_downsample_factor: Factor by which to downsample time dimension.
        time_fraction: Fraction of time steps to keep (applied before downsampling).

    Returns:
        Downsampled array.
    """
    n_time_dims = 1 if time_varying else 0
    n_spatial_dims = len(data.shape) - n_batch_dims - n_tensor_dims - n_time_dims

    # Compute the new time length before downsampling
    new_time_length = (
        int(data.shape[n_batch_dims] * time_fraction) if time_varying else None
    )

    # First, do time downsampling to save compute
    time_slices = (
        [slice(None)] * n_batch_dims
        + [slice(None, new_time_length, temporal_downsample_factor)] * n_time_dims
        + [slice(None)] * n_spatial_dims
        + [slice(None)] * n_tensor_dims
    )
    data = data[tuple(time_slices)]

    # Apply Gaussian filtering if requested (anti-aliasing for spatial downsampling)
    if spatial_filtering and spatial_downsample_factor > 1:
        # Sigma should be proportional to downsample factor for proper anti-aliasing
        sigma = spatial_downsample_factor / 2.0

        # Build sigma array: 0 for batch/time/tensor dims, sigma for spatial dims
        sigma_array = (
            [0] * n_batch_dims
            + [0] * n_time_dims
            + [sigma] * n_spatial_dims
            + [0] * n_tensor_dims
        )

        data = gaussian_filter(data.astype(np.float32), sigma=sigma_array)

    # Now do spatial downsampling
    spatial_slices = (
        [slice(None)] * n_batch_dims
        + [slice(None)] * n_time_dims
        + [slice(None, None, spatial_downsample_factor)] * n_spatial_dims
        + [slice(None)] * n_tensor_dims
    )
    data = data[tuple(spatial_slices)]

    return data


def process_dataset_item(
    src_dataset: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    full_name: str,
    spatial_downsample_factor: int,
    temporal_downsample_factor: int,
    time_fraction: float,
    trajectories_to_process: int | None,
) -> None:
    """Process a single HDF5 dataset (array) and write to destination.

    Args:
        src_dataset: Source HDF5 dataset.
        dst_group: Destination HDF5 group to write to.
        name: Name of the dataset within the group.
        full_name: Full path name of the dataset.
        spatial_downsample_factor: Spatial downsampling factor.
        temporal_downsample_factor: Temporal downsampling factor.
        time_fraction: Fraction of time to keep.
        trajectories_to_process: Number of trajectories to process (None for all).
    """
    attrs = dict(src_dataset.attrs)

    # Handle scalar datasets
    if src_dataset.shape == ():
        data = src_dataset[()]
    else:
        data = src_dataset[:]

        downsample_kws = dict(
            spatial_downsample_factor=spatial_downsample_factor,
            temporal_downsample_factor=temporal_downsample_factor,
            time_fraction=time_fraction,
        )

        # Handle different dataset types
        if re.match(r"t[012]_fields.*", full_name) or full_name == "additional_information/g_contravariant":
            # Field data - apply full downsampling
            if attrs.get("sample_varying", False) and trajectories_to_process is not None:
                data = data[:trajectories_to_process, ...]

            if full_name.startswith("t0_fields"):
                n_tensor_dims = 0
            elif full_name.startswith("t1_fields"):
                n_tensor_dims = 1
            elif full_name.startswith("t2_fields"):
                n_tensor_dims = 2
            elif full_name == "additional_information/g_contravariant":
                n_tensor_dims = 2
            else:
                n_tensor_dims = 0

            data = downsample_field(
                data,
                time_varying=attrs.get("time_varying", False),
                spatial_filtering=True,
                n_batch_dims=int(attrs.get("sample_varying", False)),
                n_tensor_dims=n_tensor_dims,
                **downsample_kws,
            )

        elif re.match(r"dimensions/time", full_name):
            # Time dimension - only temporal downsampling
            data = downsample_field(
                data,
                time_varying=True,
                spatial_filtering=False,
                n_batch_dims=int(attrs.get("sample_varying", False)),
                n_tensor_dims=0,
                **downsample_kws,
            )

        elif re.match(r"dimensions/([xyz]|phi|theta|log_r|r)", full_name) and len(data.shape) == 1:
            # Spatial dimension arrays - only spatial downsampling
            data = downsample_field(
                data,
                time_varying=False,
                spatial_filtering=False,
                n_batch_dims=0,
                n_tensor_dims=0,
                **downsample_kws,
            )

        elif re.match(r"scalars/.*", full_name):
            # Scalar data - trajectory limiting and time downsampling only
            if attrs.get("sample_varying", False) and trajectories_to_process is not None:
                data = data[:trajectories_to_process, ...]
            if attrs.get("time_varying", False):
                new_time_length = int(data.shape[-1] * time_fraction)
                time_slice = slice(None, new_time_length, temporal_downsample_factor)
                if attrs.get("sample_varying", False):
                    data = data[:, time_slice]
                else:
                    data = data[time_slice]

        elif re.match(r"boundary_conditions/.*/mask", full_name):
            # Boundary condition masks - handle specially
            if len(data.shape) == 1:
                num_elements = data.shape[0] // spatial_downsample_factor
                # Preserve first and last elements for boundary info
                data = np.array(
                    [data[0]] + [False] * (num_elements - 2) + [data[-1]],
                    dtype=bool
                )
            elif len(data.shape) == 2:
                data = downsample_field(
                    data,
                    time_varying=False,
                    spatial_filtering=False,
                    n_batch_dims=0,
                    n_tensor_dims=0,
                    **downsample_kws,
                )

    # Create the dataset in destination
    dst_group.create_dataset(name, data=data)

    # Copy attributes
    for key, value in attrs.items():
        dst_group[name].attrs[key] = value

    # Update spatial resolution if present
    if "spatial_resolution" in attrs:
        old_resolution = attrs["spatial_resolution"]
        new_resolution = tuple(
            dim // spatial_downsample_factor for dim in old_resolution
        )
        dst_group[name].attrs["spatial_resolution"] = new_resolution


def process_group(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    spatial_downsample_factor: int,
    temporal_downsample_factor: int,
    time_fraction: float,
    trajectories_to_process: int | None,
    full_name: str,
) -> None:
    """Recursively process an HDF5 group.

    Args:
        src_group: Source HDF5 group.
        dst_group: Destination HDF5 group.
        spatial_downsample_factor: Spatial downsampling factor.
        temporal_downsample_factor: Temporal downsampling factor.
        time_fraction: Fraction of time to keep.
        trajectories_to_process: Number of trajectories to process.
        full_name: Full path name of the group.
    """
    # Copy group attributes
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value

    # Process items in group
    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            process_group(
                item,
                dst_group.create_group(name),
                spatial_downsample_factor,
                temporal_downsample_factor,
                time_fraction,
                trajectories_to_process,
                full_name=full_name + "/" + name,
            )
        elif isinstance(item, h5py.Dataset):
            process_dataset_item(
                item,
                dst_group,
                name=name,
                full_name=full_name + "/" + name,
                spatial_downsample_factor=spatial_downsample_factor,
                temporal_downsample_factor=temporal_downsample_factor,
                time_fraction=time_fraction,
                trajectories_to_process=trajectories_to_process,
            )


def process_file(
    src_file: h5py.File,
    dst_file: h5py.File,
    spatial_downsample_factor: int,
    temporal_downsample_factor: int,
    time_fraction: float,
    trajectories_to_process: int | None,
) -> None:
    """Process a single HDF5 file.

    Args:
        src_file: Source HDF5 file.
        dst_file: Destination HDF5 file.
        spatial_downsample_factor: Spatial downsampling factor.
        temporal_downsample_factor: Temporal downsampling factor.
        time_fraction: Fraction of time to keep.
        trajectories_to_process: Number of trajectories to process.
    """
    # Copy and update file-level attributes
    for key, value in src_file.attrs.items():
        dst_file.attrs[key] = value

    # Update spatial resolution
    if "spatial_resolution" in dst_file.attrs:
        old_resolution = dst_file.attrs["spatial_resolution"]
        dst_file.attrs["spatial_resolution"] = tuple(
            dim // spatial_downsample_factor for dim in old_resolution
        )

    # Update spatial grid size if present
    if "spatial_grid_size" in dst_file.attrs:
        old_grid_size = dst_file.attrs["spatial_grid_size"]
        dst_file.attrs["spatial_grid_size"] = tuple(
            dim // spatial_downsample_factor for dim in old_grid_size
        )

    # Update n_trajectories if limiting
    if trajectories_to_process is not None and "n_trajectories" in dst_file.attrs:
        dst_file.attrs["n_trajectories"] = min(
            trajectories_to_process, dst_file.attrs["n_trajectories"]
        )

    # Process all top-level groups
    for group_name in src_file.keys():
        process_group(
            src_file[group_name],
            dst_file.create_group(group_name),
            spatial_downsample_factor,
            temporal_downsample_factor,
            time_fraction,
            trajectories_to_process,
            full_name=group_name,
        )


def update_stats_file(
    stats_path: Path,
    output_stats_path: Path,
) -> None:
    """Copy and potentially update the stats.yaml file.

    Note: For z-score normalization, mean and std are resolution-independent,
    so we just copy the file. For other normalization schemes, this may need
    to be recomputed.

    Args:
        stats_path: Path to source stats.yaml.
        output_stats_path: Path to destination stats.yaml.
    """
    if stats_path.exists():
        shutil.copy2(stats_path, output_stats_path)


def update_metadata_file(
    metadata_path: Path,
    output_metadata_path: Path,
    spatial_downsample_factor: int,
    temporal_downsample_factor: int,
    time_fraction: float,
) -> None:
    """Update the dataset metadata YAML file with new resolution info.

    Args:
        metadata_path: Path to source metadata YAML.
        output_metadata_path: Path to destination metadata YAML.
        spatial_downsample_factor: Spatial downsampling factor.
        temporal_downsample_factor: Temporal downsampling factor.
        time_fraction: Fraction of time to keep.
    """
    if not metadata_path.exists():
        return

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    if metadata is None:
        return

    # Update spatial resolution
    if "spatial_resolution" in metadata:
        metadata["spatial_resolution"] = [
            dim // spatial_downsample_factor
            for dim in metadata["spatial_resolution"]
        ]

    # Update n_steps_per_simulation
    if "n_steps_per_simulation" in metadata:
        metadata["n_steps_per_simulation"] = [
            int(steps * time_fraction) // temporal_downsample_factor
            for steps in metadata["n_steps_per_simulation"]
        ]

    # Update sample_shapes
    if "sample_shapes" in metadata:
        for key in ["input_fields", "output_fields", "constant_fields", "space_grid"]:
            if key in metadata["sample_shapes"]:
                shape = metadata["sample_shapes"][key]
                # Update spatial dimensions (assuming they come first)
                n_spatial = len(metadata.get("spatial_resolution", []))
                for i in range(min(n_spatial, len(shape))):
                    shape[i] = shape[i] // spatial_downsample_factor
                metadata["sample_shapes"][key] = shape

    with open(output_metadata_path, "w") as f:
        yaml.safe_dump(metadata, f, default_flow_style=False)


def downsample_dataset(
    input_path: str | Path,
    output_path: str | Path,
    spatial_downsample_factor: int = 2,
    temporal_downsample_factor: int = 1,
    time_fraction: float = 1.0,
    max_trajectories: int | None = None,
    splits: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Downsample a TheWell dataset to create a new lower-resolution dataset.

    Args:
        input_path: Path to the input TheWell dataset directory.
        output_path: Path to the output directory for the downsampled dataset.
        spatial_downsample_factor: Factor by which to downsample spatial dimensions.
            Must divide evenly into the original resolution.
        temporal_downsample_factor: Factor by which to downsample the time dimension.
        time_fraction: Fraction of the total time to keep (applied before downsampling).
        max_trajectories: Maximum number of trajectories to include per file.
            None means include all trajectories.
        splits: List of splits to process (e.g., ["train", "valid", "test"]).
            None means process all available splits.
        overwrite: If True, overwrite existing output directory.

    Returns:
        Path to the output directory.

    Raises:
        ValueError: If input path doesn't exist or parameters are invalid.
        FileExistsError: If output path exists and overwrite is False.

    Example:
        >>> downsample_dataset(
        ...     input_path="/data/the_well/turbulent_radiative_layer_2D",
        ...     output_path="/data/the_well_downsampled/turbulent_radiative_layer_2D",
        ...     spatial_downsample_factor=4,
        ...     temporal_downsample_factor=2,
        ... )
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate inputs
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if spatial_downsample_factor < 1:
        raise ValueError("spatial_downsample_factor must be >= 1")

    if temporal_downsample_factor < 1:
        raise ValueError("temporal_downsample_factor must be >= 1")

    if not 0 < time_fraction <= 1:
        raise ValueError("time_fraction must be in (0, 1]")

    # Handle output directory
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output path already exists: {output_path}. "
                "Use overwrite=True to replace it."
            )

    output_path.mkdir(parents=True, exist_ok=True)

    # Determine splits to process
    data_dir = input_path / "data"
    if not data_dir.exists():
        # Input might be a single split directory or have different structure
        # Try to find HDF5 files directly
        hdf5_files = list(input_path.glob("*.hdf5")) + list(input_path.glob("*.h5"))
        if hdf5_files:
            splits = [""]
            data_dir = input_path
        else:
            raise ValueError(f"No data directory or HDF5 files found in {input_path}")
    else:
        if splits is None:
            splits = [d.name for d in data_dir.iterdir() if d.is_dir()]

    # Create output data directory structure
    output_data_dir = output_path / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in splits:
        if split:
            split_input_dir = data_dir / split
            split_output_dir = output_data_dir / split
        else:
            split_input_dir = data_dir
            split_output_dir = output_data_dir

        if not split_input_dir.exists():
            print(f"Warning: Split directory not found: {split_input_dir}")
            continue

        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Find all HDF5 files in this split
        hdf5_files = sorted(
            list(split_input_dir.glob("*.hdf5")) + list(split_input_dir.glob("*.h5"))
        )

        if not hdf5_files:
            print(f"Warning: No HDF5 files found in {split_input_dir}")
            continue

        print(f"\nProcessing {split or 'data'} split ({len(hdf5_files)} files)...")

        for src_file_path in tqdm(hdf5_files, desc=f"Processing {split or 'files'}"):
            dst_file_path = split_output_dir / src_file_path.name

            with h5py.File(src_file_path, "r") as src_file:
                # Determine trajectories to process
                n_traj = src_file.attrs.get("n_trajectories", None)
                trajectories_to_process = max_trajectories
                if n_traj is not None and max_trajectories is not None:
                    trajectories_to_process = min(max_trajectories, n_traj)

                with h5py.File(dst_file_path, "w") as dst_file:
                    process_file(
                        src_file,
                        dst_file,
                        spatial_downsample_factor,
                        temporal_downsample_factor,
                        time_fraction,
                        trajectories_to_process,
                    )

    # Copy/update stats file
    stats_path = input_path / "stats.yaml"
    output_stats_path = output_path / "stats.yaml"
    update_stats_file(stats_path, output_stats_path)

    # Update metadata file
    dataset_name = input_path.name
    metadata_path = input_path / f"{dataset_name}.yaml"
    output_metadata_path = output_path / f"{dataset_name}.yaml"
    update_metadata_file(
        metadata_path,
        output_metadata_path,
        spatial_downsample_factor,
        temporal_downsample_factor,
        time_fraction,
    )

    # Copy README if present
    readme_path = input_path / "README.md"
    if readme_path.exists():
        shutil.copy2(readme_path, output_path / "README.md")

    print(f"\nDownsampled dataset created at: {output_path}")
    print(f"  Spatial factor: {spatial_downsample_factor}x")
    print(f"  Temporal factor: {temporal_downsample_factor}x")
    if time_fraction < 1.0:
        print(f"  Time fraction: {time_fraction}")
    if max_trajectories is not None:
        print(f"  Max trajectories per file: {max_trajectories}")

    return output_path


def downsample_from_welldataset(
    well_base_path: str | Path,
    well_dataset_name: str,
    output_base_path: str | Path,
    spatial_downsample_factor: int = 2,
    temporal_downsample_factor: int = 1,
    time_fraction: float = 1.0,
    max_trajectories: int | None = None,
    splits: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Downsample a TheWell dataset using the standard Well directory structure.

    This is a convenience wrapper around downsample_dataset that uses the
    standard TheWell directory structure: {well_base_path}/{well_dataset_name}/

    Args:
        well_base_path: Base path to TheWell datasets.
        well_dataset_name: Name of the dataset (subdirectory name).
        output_base_path: Base path for output datasets.
        spatial_downsample_factor: Spatial downsampling factor.
        temporal_downsample_factor: Temporal downsampling factor.
        time_fraction: Fraction of time to keep.
        max_trajectories: Maximum trajectories per file.
        splits: List of splits to process.
        overwrite: Whether to overwrite existing output.

    Returns:
        Path to the output directory.

    Example:
        >>> downsample_from_welldataset(
        ...     well_base_path="/data/the_well",
        ...     well_dataset_name="turbulent_radiative_layer_2D",
        ...     output_base_path="/data/the_well_downsampled",
        ...     spatial_downsample_factor=4,
        ... )
    """
    input_path = Path(well_base_path) / well_dataset_name
    output_path = Path(output_base_path) / well_dataset_name

    return downsample_dataset(
        input_path=input_path,
        output_path=output_path,
        spatial_downsample_factor=spatial_downsample_factor,
        temporal_downsample_factor=temporal_downsample_factor,
        time_fraction=time_fraction,
        max_trajectories=max_trajectories,
        splits=splits,
        overwrite=overwrite,
    )


def main():
    """Command-line interface for dataset downsampling."""
    parser = argparse.ArgumentParser(
        description="Downsample a TheWell dataset to create a lower resolution version.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Downsample spatially by 2x
  python -m autocast.data.downsample --input-path /data/dataset --output-path /data/dataset_2x --spatial-factor 2

  # Downsample both spatially (4x) and temporally (2x)
  python -m autocast.data.downsample --input-path /data/dataset --output-path /data/dataset_4x2x \\
      --spatial-factor 4 --temporal-factor 2

  # Create a small subset for debugging (100 trajectories, 50% of time)
  python -m autocast.data.downsample --input-path /data/dataset --output-path /data/dataset_mini \\
      --spatial-factor 4 --max-trajectories 100 --time-fraction 0.5

  # Process only train split
  python -m autocast.data.downsample --input-path /data/dataset --output-path /data/dataset_2x \\
      --spatial-factor 2 --splits train
        """,
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input TheWell dataset directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the output directory for the downsampled dataset",
    )
    parser.add_argument(
        "--spatial-factor",
        type=int,
        default=2,
        help="Spatial downsampling factor (default: 2)",
    )
    parser.add_argument(
        "--temporal-factor",
        type=int,
        default=1,
        help="Temporal downsampling factor (default: 1, no temporal downsampling)",
    )
    parser.add_argument(
        "--time-fraction",
        type=float,
        default=1.0,
        help="Fraction of total time to keep, applied before downsampling (default: 1.0)",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to include per file (default: all)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Splits to process (e.g., train valid test). Default: all available splits",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists",
    )

    args = parser.parse_args()

    downsample_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        spatial_downsample_factor=args.spatial_factor,
        temporal_downsample_factor=args.temporal_factor,
        time_fraction=args.time_fraction,
        max_trajectories=args.max_trajectories,
        splits=args.splits,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
