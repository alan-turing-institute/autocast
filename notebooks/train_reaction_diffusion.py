import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

import wandb
from autocast.decoders.dc import DCDecoder
from autocast.encoders.dc import DCEncoder
from autocast.models.encoder_decoder import EncoderDecoder, MultiEncoder
from autocast.types.batch import Batch, ListBatch
from autocast.utils.plots import plot_spatiotemporal_video

SCENARIO_LABELS = [
    "lf1_only",
    "lf2_only",
    "both_available",
]


def build_combinatorial_mask(batch_size: int, device: torch.device | None = None):
    """Build TensorDBM mask (Dataset, Batch, Ensemble) for 3 scenarios.

    Mask semantics: True means missing level.
    Scenario order:
    - lf1_only           -> [False, True]
    - lf2_only           -> [True, False]
    - both_available     -> [False, False]

    Notes
    -----
    Dataset index mapping in this script is:
    - dataset 0 -> LF1
    - dataset 1 -> LF2
    """
    scenario_mask = torch.tensor(
        [
            [False, True, False],
            [True, False, False],
        ],
        dtype=torch.bool,
        device=device,
    )  # (Dataset=2, Ensemble=3)

    return scenario_mask.unsqueeze(1).expand(-1, batch_size, -1).contiguous()


class MultiFidelityLocalDataset(Dataset):
    """
    Dynamically loads the saved .pt tensor dictionary into a
    fully iterable PyTorch Dataset formatted exactly to what PyTorch Lightning expects.
    """

    def __init__(self, data_path, mask_mode="combinatorial"):
        data = torch.load(data_path, weights_only=False)
        self.level_0 = data["features_by_level"][
            "level_0"
        ]  # (B, T, 2, 2, 2) -> corners
        self.level_1 = data["features_by_level"][
            "level_1"
        ]  # (B, T, 8, 8, 2) -> lf grid
        self.targets = data["targets"]  # (B, T, 64, 64, 2) -> hf grid target
        self.scalars = data["constant_scalars"]  # Shape: (B, 2)

        self.mask_mode = mask_mode

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        # We package the indexed data identically to how MultiSpatioTemporalDataset works
        b0 = Batch(
            input_fields=self.level_0[idx],
            output_fields=None,
            constant_scalars=self.scalars[idx],
            constant_fields=None,
        )
        b1 = Batch(
            input_fields=self.level_1[idx],
            output_fields=None,
            constant_scalars=self.scalars[idx],
            constant_fields=None,
        )

        return ListBatch(inner=[b0, b1], mask=None, output_fields=self.targets[idx])


def custom_collate(batch):
    # Organizes dynamically generated ListBatch tuples into fully collated Batches.
    l0_inputs = torch.stack([b.inner[0].input_fields for b in batch])
    l1_inputs = torch.stack([b.inner[1].input_fields for b in batch])
    targets = torch.stack([b.output_fields for b in batch])

    # Build full combinatorial TensorDBM mask (Dataset, Batch, Ensemble=3).
    # True means missing level.
    batch_size = l0_inputs.shape[0]
    all_masks = build_combinatorial_mask(batch_size=batch_size, device=l0_inputs.device)
    masks = all_masks

    # MultiEncoder folds the mask ensemble axis into the batch axis before
    # decoding, so targets must follow the same layout: (B, T, ...) ->
    # (B, M, T, ...) -> (B*M, T, ...).
    n_scenarios = masks.shape[-1]
    targets = targets.unsqueeze(1).expand(-1, n_scenarios, -1, -1, -1, -1)
    targets = targets.flatten(0, 1).contiguous()

    return ListBatch(
        inner=[
            Batch(
                input_fields=l0_inputs,
                output_fields=None,
                constant_scalars=None,
                constant_fields=None,
            ),
            Batch(
                input_fields=l1_inputs,
                output_fields=None,
                constant_scalars=None,
                constant_fields=None,
            ),
        ],
        mask=masks,
        output_fields=targets,
    )


def main():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../datasets/multi_reaction_diffusion")
    )

    # -------------------------------------------------------------
    # 1. Prepare Datasets & Dataloaders
    # -------------------------------------------------------------
    print("Loading synthetic Reaction-Diffusion datasets...")
    train_ds = MultiFidelityLocalDataset(
        os.path.join(base_dir, "train", "data.pt"),
        mask_mode="combinatorial",
    )
    val_ds = MultiFidelityLocalDataset(
        os.path.join(base_dir, "valid", "data.pt"),
        mask_mode="combinatorial",
    )

    # Use standard dataloaders providing our collated structural chunks
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=custom_collate
    )
    # Using batch size of val size so we can fetch all instances easily for visualization if needed
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=custom_collate
    )

    # -------------------------------------------------------------
    # 2. Build the High-Fidelity Encoders Pipeline
    # -------------------------------------------------------------
    print("Initializing MultiEncoder layout...")
    # Encoder for the 2x2 Boundary grid
    enc_vertices = DCEncoder(
        in_channels=2,
        out_channels=16,
        spatial=2,
        hid_channels=(16, 32),
        hid_blocks=(1, 1),
    )

    # Encoder for the 8x8 Low-Fidelity mesh
    enc_lf = DCEncoder(
        in_channels=2,
        out_channels=16,
        spatial=2,
        hid_channels=(16, 32),
        hid_blocks=(1, 1),
    )

    # Unify context across resolutions and combinatorial omissions through MultiHeadAttention
    multi_encoder = MultiEncoder(
        encoders=[enc_vertices, enc_lf],
        attention=True,
        transformer_dim=64,
        n_heads=2,
        n_transformer_blocks=1,
        dropout=0.1,
    )

    # -------------------------------------------------------------
    # 3. Build the Target Decoder Pipeline
    # -------------------------------------------------------------
    print("Initializing DCDecoder projection...")
    # Scale back up to 64x64 explicitly from the pooled tokens block!
    # Because there are 2 channels in L0, and 2 in L1 -> attention stack produces 4 parallel channel latents.
    decoder_hf = DCDecoder(
        in_channels=32,
        out_channels=2,
        spatial=2,
        kernel_size=3,
        hid_channels=(64, 32, 16, 8, 4),  # Scaled down for MPS Memory safety
        hid_blocks=(1, 1, 1, 1, 1),  # Upsamples x4 (4x4 -> 8 -> 16 -> 32 -> 64x64)
    )

    # -------------------------------------------------------------
    # 4. Integrate into Autocast & PyTorch Lightning
    # -------------------------------------------------------------
    autoencoder_model = EncoderDecoder(
        encoder=multi_encoder,
        decoder=decoder_hf,
        loss_func=nn.MSELoss(),
        optimizer_config={"optimizer": "AdamW", "learning_rate": 1e-4},
    )

    # -------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------
    print("\nExecuting PyTorch Lightning training loop...")
    wandb_logger = WandbLogger(project="autocast-reaction-diffusion")

    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        enable_checkpointing=False,
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(
        autoencoder_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # -------------------------------------------------------------
    # 6. Evaluation & Visualizations
    # -------------------------------------------------------------
    print("\nGenerating evaluation predictions for the validation split...")
    autoencoder_model.eval()

    # Grab the first available set
    batch = next(iter(val_loader))
    b0_target = batch.output_fields[0].cpu().numpy()
    batch_size = batch.inner[0].input_fields.shape[0]
    eval_masks = build_combinatorial_mask(
        batch_size=batch_size,
        device=batch.inner[0].input_fields.device,
    )
    scenario_preds: dict[str, np.ndarray] = {}

    scenario_labels = SCENARIO_LABELS[: eval_masks.shape[-1]]

    scenario_metrics = {}
    for m_idx, scenario_name in enumerate(scenario_labels):
        scenario_batch = ListBatch(
            inner=batch.inner,
            mask=eval_masks[:, :, m_idx : m_idx + 1],
            output_fields=batch.output_fields,
        )

        with torch.no_grad():
            pred_m = autoencoder_model(scenario_batch).detach().cpu().numpy()[0]

        scenario_preds[scenario_name] = pred_m
        true_m = b0_target

        print(f"Prediction raw shape [{scenario_name}]: {pred_m.shape}")
        print(
            f"Prediction diagnostics [{scenario_name}] "
            f"(finite={np.isfinite(pred_m).all()}, "
            f"min={float(np.nanmin(pred_m)):.6f}, "
            f"max={float(np.nanmax(pred_m)):.6f})"
        )

        mse_m = float(np.mean((pred_m - true_m) ** 2))
        mae_m = float(np.mean(np.abs(pred_m - true_m)))
        rmse_m = float(np.sqrt(mse_m))
        scenario_metrics[scenario_name] = {
            "mse": mse_m,
            "mae": mae_m,
            "rmse": rmse_m,
        }

        pred_tensor = torch.tensor(pred_m).unsqueeze(0)
        true_tensor = torch.tensor(true_m).unsqueeze(0)
        video_path = os.path.join(base_dir, f"validation_{scenario_name}.mp4")

        plot_spatiotemporal_video(
            true=true_tensor,
            pred=pred_tensor,
            batch_idx=0,
            save_path=video_path,
            colorbar_mode="column",
            channel_names=["U", "V"],
            title=f"Validation: {scenario_name}",
        )

        wandb.log(
            {
                f"Validation/{scenario_name}_video": wandb.Video(
                    video_path, fps=5, format="mp4"
                ),
                f"Validation/{scenario_name}_mse": mse_m,
                f"Validation/{scenario_name}_mae": mae_m,
                f"Validation/{scenario_name}_rmse": rmse_m,
            }
        )

    # Comparison plot across scenarios for quick error inspection.
    comparison_fig_path = os.path.join(
        base_dir, "validation_scenarios_error_comparison.png"
    )
    metric_names = ["mse", "mae", "rmse"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric_name in zip(axes, metric_names, strict=False):
        values = [scenario_metrics[name][metric_name] for name in scenario_labels]
        ax.bar(scenario_labels, values)
        ax.set_title(metric_name.upper())
        ax.set_ylabel(metric_name.upper())
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(comparison_fig_path, dpi=150)
    plt.close(fig)

    # Snapshot comparison: final time step predictions and absolute error for each scenario.
    snapshot_path = os.path.join(
        base_dir, "validation_scenarios_snapshot_comparison.png"
    )
    last_t = -1
    ncols = len(scenario_labels)
    fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 10))
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    for c_idx, scenario_name in enumerate(scenario_labels):
        pred_frame = scenario_preds[scenario_name][last_t, :, :, 0]
        true_frame = b0_target[last_t, :, :, 0]
        err_frame = np.abs(pred_frame - true_frame)

        im0 = axes[0, c_idx].imshow(true_frame, cmap="viridis", aspect="auto")
        axes[0, c_idx].set_title(f"{scenario_name}: true (U)")
        fig.colorbar(im0, ax=axes[0, c_idx], fraction=0.046, pad=0.04)

        im1 = axes[1, c_idx].imshow(pred_frame, cmap="viridis", aspect="auto")
        axes[1, c_idx].set_title(f"{scenario_name}: pred (U)")
        fig.colorbar(im1, ax=axes[1, c_idx], fraction=0.046, pad=0.04)

        im2 = axes[2, c_idx].imshow(err_frame, cmap="inferno", aspect="auto")
        axes[2, c_idx].set_title(f"{scenario_name}: |error| (U)")
        fig.colorbar(im2, ax=axes[2, c_idx], fraction=0.046, pad=0.04)

        for r_idx in range(3):
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])

    fig.tight_layout()
    fig.savefig(snapshot_path, dpi=150)
    plt.close(fig)

    wandb.log(
        {
            "Validation/scenarios_error_comparison": wandb.Image(comparison_fig_path),
            "Validation/scenarios_snapshot_comparison": wandb.Image(snapshot_path),
        }
    )

    print("Per-scenario validation metrics:")
    for scenario_name, metric_values in scenario_metrics.items():
        print(
            f"  {scenario_name}: "
            f"MSE={metric_values['mse']:.6e}, "
            f"MAE={metric_values['mae']:.6e}, "
            f"RMSE={metric_values['rmse']:.6e}"
        )

    wandb.finish()
    print("Done! View scenario-wise videos, plots, and metrics on the WANDB dashboard.")


if __name__ == "__main__":
    main()
