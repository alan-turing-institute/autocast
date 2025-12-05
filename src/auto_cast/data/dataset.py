from collections.abc import Callable
from typing import Any, Literal

import h5py
import torch
from the_well.data import Augmentation, WellDataset
from the_well.data.normalization import ZScoreNormalization
from torch.utils.data import Dataset

from auto_cast.data.metadata import Metadata
from auto_cast.types import Sample


class BatchMixin:
    """A mixin class to provide Batch conversion functionality."""

    @staticmethod
    def to_sample(data: dict) -> Sample:
        """Convert a dictionary of tensors to a Sample object."""
        return Sample(
            input_fields=data["input_fields"],
            output_fields=data["output_fields"],
            constant_scalars=data.get("constant_scalars"),
            constant_fields=data.get("constant_fields"),
        )


class SpatioTemporalDataset(Dataset, BatchMixin):
    """A class for spatio-temporal datasets."""

    def __init__(
        self,
        data_path: str | None,
        data: dict | None = None,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        # TODO: support for passing data from dict
        input_channel_idxs: tuple[int, ...] | None = None,
        output_channel_idxs: tuple[int, ...] | None = None,
        full_trajectory_mode: bool = False,
        autoencoder_mode: bool = False,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        use_normalization: bool = False,
        norm: type[ZScoreNormalization] | None = None,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        data_path: str
            Path to the HDF5 file containing the dataset.
        n_steps_input: int
            Number of input time steps.
        n_steps_output: int
            Number of output time steps.
        stride: int
            Stride for sampling the data.
        data: dict | None
            Preloaded data. Defaults to None.
        input_channel_idxs: tuple[int, ...] | None
            Indices of input channels to use. Defaults to None.
        output_channel_idxs: tuple[int, ...] | None
            Indices of output channels to use. Defaults to None.
        full_trajectory_mode: bool
            If True, use full trajectories without creating subtrajectories.
        autoencoder_mode: bool
            If True, return (input, input) pairs for autoencoder training.
            Defaults to False.
        dtype: torch.dtype
            Data type for tensors. Defaults to torch.float32.
        verbose: bool
            If True, print dataset information.
        use_normalization: bool
            Whether to apply Z-score normalization. Defaults to False.
        norm: type[Standardizer] | None
            Normalization object (computed from training data). Defaults to None.
        """
        self.dtype = dtype
        self.verbose = verbose
        self.use_normalization = use_normalization
        self.norm = norm
        self.autoencoder_mode = autoencoder_mode

        if data_path is not None:
            self.read_data(data_path)
        # TODO: consider ensuring only one passed and not overridden
        if data is not None:
            self.parse_data(data)

        if autoencoder_mode and full_trajectory_mode:
            msg = "autoencoder_mode and full_trajectory_mode cannot both be True."
            raise ValueError(msg)
        if autoencoder_mode:
            # In autoencoder mode, input and output steps are overridde
            n_steps_input = 1
            n_steps_output = 0
        if full_trajectory_mode:
            # In full trajectory mode, we want:
            # - input: first n_steps_input timesteps
            # - output: all remaining timesteps for rollout comparison
            n_steps_output = self.data.shape[1] - n_steps_input

        self.full_trajectory_mode = full_trajectory_mode
        self.autoencoder_mode = autoencoder_mode
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.stride = stride
        self.input_channel_idxs = input_channel_idxs
        self.output_channel_idxs = output_channel_idxs

        # Destructured here
        (
            self.n_trajectories,
            self.n_timesteps,
            self.width,
            self.height,
            self.n_channels,
        ) = self.data.shape

        # Pre-compute all subtrajectories for efficient indexing
        self.all_input_fields = []
        self.all_output_fields = []
        self.all_constant_scalars = []
        self.all_constant_fields = []

        # Create input-output pairs
        for traj_idx in range(self.n_trajectories):
            # Create subtrajectories for this trajectory
            fields = (
                self.data[traj_idx]
                .unfold(0, self.n_steps_input + self.n_steps_output, self.stride)
                # [num_subtrajectories, T_in + T_out, W, H, C]
                .permute(0, -1, 1, 2, 3)
            )

            # Split into input and output
            input_fields = fields[:, : self.n_steps_input, ...]
            output_fields = (
                fields[:, self.n_steps_input :, ...]
                if not self.autoencoder_mode
                else input_fields
            )

            # Store each subtrajectory separately
            for sub_idx in range(input_fields.shape[0]):
                self.all_input_fields.append(
                    input_fields[sub_idx].to(self.dtype)
                )  # [T_in, W, H, C]
                self.all_output_fields.append(
                    output_fields[sub_idx].to(self.dtype)
                )  # [T_out, W, H, C]

                # Handle constant scalars
                if self.constant_scalars is not None:
                    self.all_constant_scalars.append(
                        self.constant_scalars[traj_idx].to(self.dtype)
                    )

                # Handle constant fields
                if self.constant_fields is not None:
                    self.all_constant_fields.append(
                        self.constant_fields[traj_idx].to(self.dtype)
                    )

        if self.verbose:
            print(f"Created {len(self.all_input_fields)} subtrajectory samples")
            print(f"Each input sample shape: {self.all_input_fields[0].shape}")
            print(f"Each output sample shape: {self.all_output_fields[0].shape}")
            print(f"Data type: {self.all_input_fields[0].dtype}")

    def _from_f(self, f):
        assert "data" in f, "HDF5 file must contain 'data' dataset"
        self.data = torch.Tensor(f["data"][:]).to(self.dtype)  # type: ignore # [N, T, W, H, C]  # noqa: PGH003
        if self.verbose:
            print(f"Loaded data shape: {self.data.shape}")
        # TODO: add the constant scalars
        self.constant_scalars = (
            torch.Tensor(f["constant_scalars"][:]).to(self.dtype)  # type: ignore  # noqa: PGH003
            if "constant_scalars" in f
            else None
        )  # [N, C]

        # Constant fields
        self.constant_fields = (
            torch.Tensor(f["constant_fields"][:]).to(  # type: ignore # noqa: PGH003
                self.dtype
            )  # [N, W, H, C]
            if "constant_fields" in f and f["constant_fields"] != {}
            else None
        )

    def read_data(self, data_path: str):
        """Read data.

        By default assumes HDF5 format in `data_path` with correct shape and fields.
        """
        self.data_path = data_path
        if self.data_path.endswith(".h5") or self.data_path.endswith(".hdf5"):
            with h5py.File(self.data_path, "r") as f:
                self._from_f(f)
        if self.data_path.endswith(".pt"):
            self._from_f(torch.load(self.data_path))

    def parse_data(self, data: dict | None):
        """Parse data from a dictionary."""
        if data is not None:
            self.data = (
                data["data"].to(self.dtype)
                if torch.is_tensor(data["data"])
                else torch.tensor(data["data"], dtype=self.dtype)
            )
            self.constant_scalars = data.get("constant_scalars", None)
            self.constant_fields = data.get("constant_fields", None)
            return
        msg = "No data provided to parse."
        raise ValueError(msg)

    def __len__(self):  # noqa: D105
        return len(self.all_input_fields)

    def __getitem__(self, idx):
        """Get item at index."""
        input_fields = self.all_input_fields[idx]
        output_fields = (
            input_fields if self.autoencoder_mode else self.all_output_fields[idx]
        )

        item = {
            "input_fields": input_fields,
            "output_fields": output_fields,
        }
        if len(self.all_constant_scalars) > 0:
            item["constant_scalars"] = self.all_constant_scalars[idx]
        if len(self.all_constant_fields) > 0:
            item["constant_fields"] = self.all_constant_fields[idx]

        return self.to_sample(item)


class ReactionDiffusionDataset(SpatioTemporalDataset):
    """Reaction-Diffusion dataset."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metadata = Metadata(
            dataset_name="ReactionDiffusion",
            n_spatial_dims=2,
            spatial_resolution=self.data.shape[-3:-1],
            scalar_names=[],
            constant_scalar_names=["beta", "d"],
            field_names={0: ["U", "V"]},
            constant_field_names={},
            boundary_condition_types=["periodic", "periodic"],
            n_files=0,
            n_trajectories_per_file=[],
            n_steps_per_trajectory=[self.data.shape[1]] * self.data.shape[0],
            grid_type="cartesian",
        )


class AdvectionDiffusionDataset(SpatioTemporalDataset):
    """Advection-Diffusion dataset."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metadata = Metadata(
            dataset_name="AdvectionDiffusion",
            n_spatial_dims=2,
            spatial_resolution=self.data.shape[-3:-1],
            scalar_names=[],
            constant_scalar_names=["nu", "mu"],
            field_names={0: ["vorticity"]},
            constant_field_names={},
            boundary_condition_types=["periodic", "periodic"],
            n_files=0,
            n_trajectories_per_file=[],
            n_steps_per_trajectory=[self.data.shape[1]] * self.data.shape[0],
            grid_type="cartesian",
        )


class BOUTDataset(SpatioTemporalDataset):
    """BOUT++ dataset."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metadata = Metadata(
            dataset_name="BOUT++",
            n_spatial_dims=2,
            spatial_resolution=self.data.shape[-3:-1],
            scalar_names=[],
            constant_scalar_names=[
                f"const{i}"
                for i in range(self.constant_scalars.shape[-1])  # type: ignore  # noqa: PGH003
            ],
            field_names={0: ["vorticity"]},
            constant_field_names={},
            boundary_condition_types=["periodic", "periodic"],
            n_files=0,
            n_trajectories_per_file=[],
            n_steps_per_trajectory=[self.data.shape[1]] * self.data.shape[0],
            grid_type="cartesian",
        )


class TheWell(SpatioTemporalDataset):
    """A wrapper around The Well's WellDataset to provide Batch objects."""

    well_dataset: WellDataset

    def __init__(
        self,
        path: None | str = None,
        normalization_path: str = "../stats.yaml",
        well_base_path: None | str = None,
        well_dataset_name: None | str = None,
        well_split_name: Literal["train", "valid", "test", None] = None,
        include_filters: list[str] | None = None,
        exclude_filters: list[str] | None = None,
        use_normalization: bool = False,
        normalization_type: None | Callable[..., Any] = None,
        max_rollout_steps=100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        flatten_tensors: bool = True,
        cache_small: bool = True,
        max_cache_size: float = 1e9,
        return_grid: bool = True,
        boundary_return_type: str = "padding",
        full_trajectory_mode: bool = False,
        autoencoder_mode: bool = False,
        name_override: None | str = None,
        transform: None | Augmentation = None,
        min_std: float = 1e-4,
        storage_options: None | dict = None,
    ):
        super().__init__(
            data_path=None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            full_trajectory_mode=full_trajectory_mode,
            autoencoder_mode=autoencoder_mode,
            use_normalization=use_normalization,
        )
        exclude_filters = exclude_filters or []
        include_filters = include_filters or []
        self.autoencoder_mode = autoencoder_mode
        self.well_dataset = WellDataset(
            path=path,
            normalization_path=normalization_path,
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name=well_split_name,
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            max_rollout_steps=max_rollout_steps,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output if not autoencoder_mode else 0,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            flatten_tensors=flatten_tensors,
            cache_small=cache_small,
            max_cache_size=max_cache_size,
            return_grid=return_grid,
            boundary_return_type=boundary_return_type,
            full_trajectory_mode=full_trajectory_mode,
            name_override=name_override,
            transform=transform,
            min_std=min_std,
            storage_options=storage_options,
        )
        self.well_metadata = self.well_dataset.metadata

    def __getitem__(self, index) -> Sample:  # noqa: D105
        data = self.well_dataset.__getitem__(index)
        if self.autoencoder_mode:
            # Replace output_fields with input_fields for autoencoder training
            data["input_fields"] = data["input_fields"]
            data["output_fields"] = data["input_fields"]
        return self.to_sample(data)
