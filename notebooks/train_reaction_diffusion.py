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

        # Combinatorial mask across the 2 available fidelity levels.
        # (D, M) -> 2 Datasets, 3 Combos ([T,F], [F,T], [F,F])
        if mask_mode == "combinatorial":
            # MultiEncoder expects TensorDBM: (Dataset, Batch, Ensemble).
            # In AttentionMixer, True means "missing level".
            # Here all fidelity levels are present, so mask must be False.
            self.masks = torch.zeros(2, self.level_0.shape[0], 1, dtype=torch.bool)
        else:
            self.masks = None

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

        return ListBatch(
            inner=[b0, b1], mask=self.masks, output_fields=self.targets[idx]
        )


def custom_collate(batch):
    # Organizes dynamically generated ListBatch tuples into fully collated Batches.
    l0_inputs = torch.stack([b.inner[0].input_fields for b in batch])
    l1_inputs = torch.stack([b.inner[1].input_fields for b in batch])
    targets = torch.stack([b.output_fields for b in batch])

    # Expand to TensorDBM layout (Dataset, Batch, Ensemble).
    # True means missing; use all False when all levels are available.
    masks = torch.zeros(2, l0_inputs.shape[0], 1, dtype=torch.bool)

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
    train_ds = MultiFidelityLocalDataset(os.path.join(base_dir, "train", "data.pt"))
    val_ds = MultiFidelityLocalDataset(os.path.join(base_dir, "valid", "data.pt"))

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
        max_epochs=300,
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
    b0_hf = None
    b0_target = batch.output_fields[0].numpy()

    # Move data sequentially through prediction
    with torch.no_grad():
        preds = autoencoder_model(batch).detach().cpu().numpy()
        b0_hf = preds[0]
        # B0 predicted data has shape (T, 64, 64, 2) if attention mask has no M dimension
        # Wait, if MultiEncoder masks exist, the shape is (T, M, 64, 64, 2).

    from autocast.utils.plots import plot_spatiotemporal_video

    print(f"Prediction raw shape: {b0_hf.shape}")
    print(
        "Prediction diagnostics "
        f"(finite={np.isfinite(b0_hf).all()}, "
        f"min={float(np.nanmin(b0_hf)):.6f}, "
        f"max={float(np.nanmax(b0_hf)):.6f})"
    )

    # If the network predicted multiple models for the M dimensional ensemble, we select M=0.
    if len(b0_hf.shape) == 5:
        b0_hf = b0_hf[:, 0, :, :, :]  # Pick the first ensemble mask

    # We add a dummy batch dimension for plot_spatiotemporal_video since it expects TensorBTSC
    pred_tensor = torch.tensor(b0_hf).unsqueeze(0)
    true_tensor = torch.tensor(b0_target).unsqueeze(0)

    video_path = os.path.join(base_dir, "validation_reaction_diffusion.mp4")

    anim = plot_spatiotemporal_video(
        true=true_tensor,
        pred=pred_tensor,
        batch_idx=0,
        save_path=video_path,
        colorbar_mode="column",
        channel_names=["U", "V"],
    )

    wandb.log(
        {
            "Validation/Target_vs_Prediction_Video": wandb.Video(
                video_path, fps=5, format="mp4"
            )
        }
    )
    wandb.finish()
    print("Done! View your visualizations on the WANDB dashboard.")


if __name__ == "__main__":
    main()
