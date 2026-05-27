from pathlib import Path
from typing import Any

import torch
from autoemulate.simulations.reaction_diffusion import ReactionDiffusion
from autosim.cli import (
    build_simulator,
    compute_normalization_stats,
    generate_dataset_splits,
    save_dataset_splits,
    save_normalization_stats,
    save_resolved_config,
)
from omegaconf import OmegaConf

from autocast.data.advection_diffusion import AdvectionDiffusion
from autocast.data.datamodule import SpatioTemporalDataModule, TheWellDataModule
from autocast.data.dataset import SpatioTemporalDataset

_AUTOSIM_SIMULATOR_CONFIGS: dict[str, dict[str, Any]] = {
    "advection_diffusion": {
        "_target_": "autosim.simulations.AdvectionDiffusionMultichannel",
        "output_indices": [0],
        "return_timeseries": True,
        "log_level": "warning",
        "n": 32,
        "L": 10.0,
        "T": 80.0,
        "dt": 0.25,
        "parameters_range": {
            "nu": (0.0001, 0.01),
            "mu": (0.5, 2.0),
        },
    },
    "advection_diffusion_multichannel": {
        "_target_": "autosim.simulations.AdvectionDiffusionMultichannel",
        "output_indices": [0, 1, 2, 3],
        "return_timeseries": True,
        "log_level": "warning",
        "n": 32,
        "L": 10.0,
        "T": 80.0,
        "dt": 0.25,
        "parameters_range": {
            "nu": (0.0001, 0.01),
            "mu": (0.5, 2.0),
        },
    },
    "reaction_diffusion": {
        "_target_": "autosim.experimental.simulations.ReactionDiffusion",
        "return_timeseries": True,
        "log_level": "warning",
        "n": 32,
        "L": 20,
        "T": 10.0,
        "dt": 0.1,
        "parameters_range": {
            "beta": (1.0, 2.0),
            "d": (0.05, 0.3),
        },
    },
}


def get_autosim_datamodule(
    simulation_name: str,
    n_steps_input: int,
    n_steps_output: int,
    stride: int,
    autoencoder_mode: bool = False,
    n_train: int = 20,
    n_valid: int = 4,
    n_test: int = 4,
    simulation_datasets_path: str = "../datasets/tmp",
    data_path: str | None = None,
    overwrite: bool = False,
    seed: int | None = None,
    ensure_exact_n: bool = True,
    simulator_kwargs: dict[str, Any] | None = None,
    num_workers: int = 8,
    batch_size: int = 16,
    use_normalization: bool = True,
):
    """Generate/load an autosim dataset and return an AutoCast DataModule.

    The generated cache follows the same split layout as the autosim CLI:
    ``train/data.pt``, ``valid/data.pt``, ``test/data.pt``, and ``stats.yml``.
    """
    if simulation_name not in _AUTOSIM_SIMULATOR_CONFIGS:
        msg = (
            f"Unknown autosim simulation name: {simulation_name}. "
            f"Known names: {sorted(_AUTOSIM_SIMULATOR_CONFIGS)}"
        )
        raise ValueError(msg)

    simulator_config = {
        **_AUTOSIM_SIMULATOR_CONFIGS[simulation_name],
        **(simulator_kwargs or {}),
    }
    sim_cfg = OmegaConf.create(simulator_config)
    simulator = build_simulator(sim_cfg)

    cache_path = (
        Path(data_path)
        if data_path is not None
        else Path(simulation_datasets_path) / simulation_name
    )
    expected_split_files = [
        cache_path / split / "data.pt" for split in ("train", "valid", "test")
    ]
    needs_generation = overwrite or not all(
        path.exists() for path in expected_split_files
    )

    splits = None
    if needs_generation:
        print(f"Generating autosim data in {cache_path}")
        splits = generate_dataset_splits(
            sim=simulator,
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test,
            base_seed=seed,
            ensure_exact_n=ensure_exact_n,
        )
        save_dataset_splits(splits=splits, output_dir=cache_path, overwrite=overwrite)
        resolved_config = OmegaConf.create(
            {
                "simulator": simulator_config,
                "dataset": {
                    "output_dir": str(cache_path),
                    "n_train": n_train,
                    "n_valid": n_valid,
                    "n_test": n_test,
                    "ensure_exact_n": ensure_exact_n,
                },
                "seed": seed,
            }
        )
        save_resolved_config(cfg=resolved_config, output_dir=cache_path)
    else:
        print(f"Loading cached autosim data from {cache_path}")

    normalization_path = cache_path / "stats.yml"
    if use_normalization and (needs_generation or not normalization_path.exists()):
        train_payload = (
            splits["train"]
            if splits is not None
            else torch.load(cache_path / "train" / "data.pt", map_location="cpu")
        )
        stats_payload = compute_normalization_stats(
            split_payload=train_payload,
            core_field_names=list(simulator.output_names),
        )
        save_normalization_stats(
            stats_payload=stats_payload,
            output_path=normalization_path,
        )

    return SpatioTemporalDataModule(
        data=None,
        data_path=str(cache_path),
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        stride=stride,
        autoencoder_mode=autoencoder_mode,
        batch_size=batch_size,
        use_normalization=use_normalization,
        normalization_path=str(normalization_path) if use_normalization else None,
        dataset_cls=SpatioTemporalDataset,
        num_workers=num_workers,
    )


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
    normalization_stats: dict[str, Any] | None = None,
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
    use_normalization: bool
        Whether to use normalization.
    normalization_path: str
        Path to normalization statistics.
    normalization_stats: dict | None
        Preloaded normalization statistics (e.g. from Hydra config). Only
        supported for non-The Well datasets; when provided, used instead of
        normalization_path.
    """
    if the_well and normalization_stats is not None:
        msg = (
            "normalization_stats is not supported when the_well=True. "
            "The Well normalization is configured via normalization_path "
            "(handled by the underlying WellDataset)."
        )
        raise ValueError(msg)

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
            use_normalization=use_normalization,
            normalization_path=None if normalization_stats else normalization_path,
            normalization_stats=normalization_stats,
            dataset_cls=SpatioTemporalDataset,
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
        normalization_path=normalization_path,
        autoencoder_mode=autoencoder_mode,
        num_workers=num_workers,
        batch_size=batch_size,
    )
