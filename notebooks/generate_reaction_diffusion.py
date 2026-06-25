import os

import numpy as np
import torch
from autosim.experimental.simulations import ReactionDiffusion
from tqdm import tqdm


def generate_multi_fidelity_dataset(n_samples, seed_offset=0):
    """
    Generate High-Fidelity (d=0.05) and Low-Fidelity (d=0.1) explicitly by cycling
    the beta parameter over individual trajectories. Then extract High-Fidelity
    sensors and apply LogNormal(0, 0.8) noise to represent measurement imperfection.
    """
    hf_data_list = []
    lf_data_list = []
    scalars_list = []

    # Draw parameters natively
    torch.manual_seed(42 + seed_offset)
    # Uniform sample beta from [1.0, 2.0]
    betas = torch.rand(n_samples) * (2.0 - 1.0) + 1.0

    print(f"Generating {n_samples} physics sequences...")
    for i, beta in enumerate(tqdm(betas)):
        # HF uses d=0.05 and n=64
        sim_hf = ReactionDiffusion(
            return_timeseries=True,
            log_level="error",
            n=64,
            L=20,
            T=32.2,
            dt=0.1,
            parameters_range={"beta": (beta.item(), beta.item()), "d": (0.05, 0.05)},
        )

        # LF uses d=0.10 and n=8
        sim_lf = ReactionDiffusion(
            return_timeseries=True,
            log_level="error",
            n=8,
            L=20,
            T=32.2,
            dt=0.1,
            parameters_range={"beta": (beta.item(), beta.item()), "d": (0.10, 0.10)},
        )

        target_seed = 1234 + i + seed_offset
        batch_hf = sim_hf.forward_samples_spatiotemporal(n=1, random_seed=target_seed)
        batch_lf = sim_lf.forward_samples_spatiotemporal(n=1, random_seed=target_seed)

        hf_data_list.append(batch_hf["data"])
        lf_data_list.append(batch_lf["data"])

        # Track beta and target model target fluid state
        scalars_list.append(torch.tensor([beta.item(), 0.05]))

    hf_data = torch.cat(hf_data_list, dim=0).float()  # Shape: (B, T, 64, 64, 2)
    lf_data = torch.cat(lf_data_list, dim=0).float()  # Shape: (B, T, 8, 8, 2)
    constant_scalars = torch.stack(scalars_list, dim=0)

    # -------------------------------------------------------------
    # Extract Level 1: 4 Vertices from the HIGH FIDELITY mesh
    # -------------------------------------------------------------
    c1 = hf_data[:, :, 0, 0, :]  # Top-Left
    c2 = hf_data[:, :, 0, -1, :]  # Top-Right
    c3 = hf_data[:, :, -1, 0, :]  # Bottom-Left
    c4 = hf_data[:, :, -1, -1, :]  # Bottom-Right

    row1 = torch.stack([c1, c2], dim=-2)
    row2 = torch.stack([c3, c4], dim=-2)

    level_1 = torch.stack([row1, row2], dim=-3)  # Shape: (B, T, 2, 2, 2)

    # -------------------------------------------------------------
    # Add Lognormal Noise to simulate observational imperfection
    # LogNormal parameters: mu=0.0, sigma=0.8
    # -------------------------------------------------------------
    noise_dist = torch.distributions.LogNormal(0.0, 0.8)
    sensor_noise = noise_dist.sample(level_1.shape)

    # Inject additive noise to the sensor streams
    level_1_noisy = level_1 + sensor_noise

    # Map to MultiFidelityDataset format
    features_dict = {"level_0": level_1_noisy, "level_1": lf_data}

    return {
        "features_by_level": features_dict,
        "targets": hf_data,
        "constant_scalars": constant_scalars,
    }


def main():
    # Desired trajectory blocks
    splits = {"train": 30, "valid": 10, "test": 10}

    seed_offset = 0
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../datasets/multi_reaction_diffusion")
    )

    # Build each split iteratively
    for split_name, n_samples in splits.items():
        print(f"\nGenerating {split_name.upper()} split...")
        split_data = generate_multi_fidelity_dataset(n_samples, seed_offset)
        seed_offset += 100

        # Output saving
        save_dir = os.path.join(BASE_DIR, split_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "data.pt")

        torch.save(split_data, save_path)
        print(f"Saved to: {save_path}")


if __name__ == "__main__":
    print("Starting Multifidelity Reaction-Diffusion Generation Process...")
    main()
