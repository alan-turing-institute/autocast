# Scripts and Configuration Guide

This guide explains the script structure and configuration system used in AutoCast.

## Entry Points

AutoCast uses a set of Python scripts located in `src/autocast/scripts/` as entry points for training and evaluation. These scripts are exposed as CLI commands via `pyproject.toml`.

### Main Scripts

1.  **`train_autoencoder`** (`src/autocast/scripts/train/autoencoder.py`)
    *   **Purpose**: Trains an Autoencoder (Encoder + Decoder) on a given dataset.
    *   **Config Group**: `autoencoder` (defaults to `configs/autoencoder.yaml`).
    *   **Key Output**: `autoencoder.ckpt` (Lightning checkpoint).

2.  **`train_encoder_processor_decoder`** (`src/autocast/scripts/train/encoder_processor_decoder.py`)
    *   **Purpose**: Trains a Processor model in the latent space of a pre-trained Autoencoder (or trains end-to-end).
    *   **Config Group**: `encoder_processor_decoder` (defaults to `configs/encoder_processor_decoder.yaml`).
    *   **Key Dependencies**: Takes a pre-trained Autoencoder checkpoint (optional, but recommended for latent training).

3.  **`evaluate_encoder_processor_decoder`** (`src/autocast/scripts/eval/encoder_processor_decoder.py`)
    *   **Purpose**: Evaluates a trained Encoder-Processor-Decoder stack.
    *   **Config Group**: `encoder_processor_decoder` (uses `eval` sub-config).
    *   **Key Inputs**: A checkpoint file (`.ckpt`) and a dataset.
    *   **Outputs**: Metrics CSV, rollout videos.

## Configuration System (Hydra)

AutoCast uses [Hydra](https://hydra.cc/) for configuration management. All configurations are YAML files located in the `configs/` directory.

### Directory Structure

```text
configs/
├── autoencoder.yaml             # default config for train_autoencoder
├── encoder_processor_decoder.yaml # default config for train_epd
├── backbone/                    # Architectures (UNet, ViT)
├── datamodule/                  # Datasets (ReactionDiffusion, The Well)
├── encoder/                     # Encoder components (DC, PermuteConcat)
├── decoder/                     # Decoder components (DC, ChannelsLast)
├── processor/                   # Latent processors (FlowMatching, Diffusion)
├── model/                       # Model assembly configs
├── optimizer/                   # Optimizer settings (Adam, AdamW)
├── trainer/                     # Lightning Trainer settings
├── logging/                     # WandB configuration
└── eval/                        # Evaluation-specific settings
```

### Composition and Overrides

Hydra allows you to compose configurations dynamically and override values from the command line.

#### 1. Basic Overrides (Dot Notation)
You can change any value in the config tree using dot notation.

```bash
uv run train_autoencoder \
    optimizer.learning_rate=0.001 \
    trainer.max_epochs=50 \
    datamodule.batch_size=32
```

#### 2. Swapping Components (Config Groups)
You can swap entire components (like the backbone or encoder) by selecting a different file from the config group.

**Example: Use a Vision Transformer (ViT) processor instead of the default**
```bash
uv run train_encoder_processor_decoder \
    model.processor=vit
```

**Example: Change the encoder architecture**
```bash
uv run train_encoder_processor_decoder \
    encoder@model.encoder=permute_concat
```
*Note: The `@` syntax specifies where in the config tree to mount the selected config file. The format is `group@destination`.*

### Important Config Parameters

#### `hydra.run.dir`
Controls where the output of the run is saved. By default, Hydra creates a hierarchy based on date/time. We recommend overriding this to a meaningful path.

```bash
uv run train_autoencoder hydra.run.dir=outputs/my_experiment/version_1
```

#### `model`
Defines the neural network architecture.
*   **`_target_`**: The Python class to instantiate.
*   **`encoder`**, **`decoder`**, **`processor`**: Sub-configs for specific modules.

#### `datamodule`
Defines the data source.
*   **`data_path`**: Path to the dataset on disk.
*   **`n_steps_input`**: Number of context frames.
*   **`n_steps_output`**: Number of frames to predict.

#### `eval` (Evaluation Script Only)
*   **`checkpoint`**: Path to the trained model checkpoint to load.
*   **`metrics`**: List of metrics to compute (e.g., `["mse", "rmse"]`).
*   **`video_dir`**: Where to save rollout visualizations.

## Workflow Examples

### 1. Train an Autoencoder
```bash
uv run train_autoencoder \
    hydra.run.dir=outputs/autoencoder_v1 \
    datamodule=reaction_diffusion \
    model.encoder=dc \
    model.decoder=dc
```

### 2. Train a Processor (Latent Space)
Uses the autoencoder from step 1.

```bash
uv run train_encoder_processor_decoder \
    hydra.run.dir=outputs/processor_v1 \
    datamodule=reaction_diffusion \
    autoencoder_checkpoint=outputs/autoencoder_v1/autoencoder.ckpt \
    model.processor=flow_matching
```

### 3. Hyperparameter Sweep (SLURM)
See `slurm_templates/encoder-processor-decoder-parameter_sweep.sh` for an example of how to run sweeps using SLURM arrays and Hydra overrides.

## Shell Scripts
For convenience, we provide shell scripts in `scripts/` that wrap the python commands. These are useful for quickly running experiments without typing the full `uv run ...` command.

*   `scripts/ae.sh`: Wraps `train_autoencoder`
*   `scripts/epd.sh`: Wraps `train_encoder_processor_decoder`
*   `scripts/eval.sh`: Wraps `evaluate_encoder_processor_decoder`

Example usage:
```bash
./scripts/ae.sh <run_label> <run_id> <dataset_name> [overrides...]
```
This will create a work directory at `outputs/<run_label>/<run_id>`.
