from pathlib import Path

import torch
from autoemulate.simulations.reaction_diffusion import ReactionDiffusion

from autocast.data.advection_diffusion import AdvectionDiffusion
from autocast.data.datamodule import SpatioTemporalDataModule, TheWellDataModule


def get_datamodule(
    the_well: bool,
    simulation_name: str,
    n_steps_input: int,
    n_steps_output: int,
    stride: int,
    autoencoder_mode: bool = False,
    n_train: int = 20,
    n_valid: int = 4,
    n_test: int = 4,
    simulation_datasets_path: str = "../datasets/tmp",
    the_well_dataset_path: str = "../datasets/",
    overwrite_tmp: bool = False,
    num_workers: int = 8,
    batch_size: int = 16,
    use_normalization: bool = True,
    normalization_path: str = "../stats.yaml",  # TODO: choose better default
):
    """Get the configured datamodule.

    Parameters
    ----------
    the_well: bool
        Whether to use The Well dataset.
    simulation_name: str
        Name of the simulation to use (either "advection_diffusion" or
        "reaction_diffusion", or "advection_diffusion_multichannel") or the name of
        The Well dataset.
    n_steps_input: int
        Number of input time steps.
    n_steps_output: int
        Number of output time steps.
    stride: int
        Stride between time steps.
    autoencoder_mode: bool
        Whether to use autoencoder mode.
    n_train: int
        Number of training samples to generate (if not using The Well).
    n_valid: int
        Number of validation samples to generate (if not using The Well).
    n_test: int
        Number of test samples to generate (if not using The Well).
    simulation_datasets_path: str
        Base path to store and load temporary datasets from running simulations.
    the_well_dataset_path: str
        Base path to The Well datasets.
    overwrite_tmp: bool
        Whether to overwrite existing temporary datasets.
    num_workers: int
        Number of workers for data loading.
    batch_size: int = 16,
        Batch size for the datamodule.
    """

    def generate_split(simulator):
        """Generate training, validation, and test splits from the simulator."""
        train = simulator.forward_samples_spatiotemporal(n_train)
        valid = simulator.forward_samples_spatiotemporal(n_valid)
        test = simulator.forward_samples_spatiotemporal(n_test)
        return {"train": train, "valid": valid, "test": test}

    if not the_well:
        if simulation_name.startswith("advection_diffusion"):
            Sim = AdvectionDiffusion
        elif simulation_name == "reaction_diffusion":
            Sim = ReactionDiffusion
        else:
            raise ValueError(f"Unknown simulation name: {simulation_name}")

        # Initialize simulator
        sim = Sim(return_timeseries=True, log_level="error")

        # Cache file path
        cache_path = Path(f"{simulation_datasets_path}/{simulation_name}")

        # Load from cache if it exists, otherwise generate and save
        if cache_path.exists() and not overwrite_tmp:
            print(f"Loading cached simulation data from {cache_path}")
        else:
            print("Generating simulation data...")
            combined_data = generate_split(sim)
            print(f"Saving simulation data to {cache_path}")
            for split in ["train", "valid", "test"]:
                split_path = Path(cache_path, split)
                split_path.mkdir(parents=True, exist_ok=True)
                combined_data[split]["data"] = (
                    combined_data[split]["data"][..., :1]
                    if simulation_name == "advection_diffusion"
                    else combined_data[split]["data"]
                )
                torch.save(combined_data[split], Path(split_path, "data.pt"))

        return SpatioTemporalDataModule(
            data=None,
            data_path=str(cache_path),
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=n_steps_output,
            autoencoder_mode=autoencoder_mode,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    # If the well dataset
    return TheWellDataModule(
        well_base_path=the_well_dataset_path,
        well_dataset_name=simulation_name,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        min_dt_stride=stride,
        max_dt_stride=stride,
        use_normalization=use_normalization,
        autoencoder_mode=autoencoder_mode,
        num_workers=num_workers,
        batch_size=batch_size,
        normalization_path=normalization_path,
    )
