# Scripts and Configuration Guide

This guide explains the script structure and configuration system used in AutoCast.

## Entry Points

AutoCast uses a set of Python scripts located in `src/autocast/scripts/` as entry points for training and evaluation. These scripts are exposed as CLI commands via `pyproject.toml`.

### Main Scripts

1.  **`train_autoencoder`** (`src/autocast/scripts/train/autoencoder.py`)
    *   **Purpose**: Trains an Autoencoder (Encoder + Decoder) on a given dataset.
    *   **Config Group**: `autoencoder` (defaults to `src/autocast/configs/autoencoder.yaml`).
    *   **Key Output**: `autoencoder.ckpt` (Lightning checkpoint).

2.  **`train_encoder_processor_decoder`** (`src/autocast/scripts/train/encoder_processor_decoder.py`)
    *   **Purpose**: Trains a Processor model in the latent space of a pre-trained Autoencoder (or trains end-to-end).
    *   **Config Group**: `encoder_processor_decoder` (defaults to `src/autocast/configs/encoder_processor_decoder.yaml`).
    *   **Key Dependencies**: Takes a pre-trained Autoencoder checkpoint (optional, but recommended for latent training).

3.  **`evaluate_encoder_processor_decoder`** (`src/autocast/scripts/eval/encoder_processor_decoder.py`)
    *   **Purpose**: Evaluates a trained Encoder-Processor-Decoder stack.
    *   **Config Group**: `encoder_processor_decoder` (uses `eval` sub-config).
    *   **Key Inputs**: A checkpoint file (`.ckpt`) and a dataset.
    *   **Outputs**: Metrics CSV, rollout videos.

## Configuration System (Hydra)

AutoCast uses [Hydra](https://hydra.cc/) for configuration management. All configurations are YAML files located in `src/autocast/configs/`.

### Directory Structure

```text
src/autocast/configs/
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
Use Hydra multi-run directly (or the manifest launcher) for sweeps, e.g. `uv run autocast epd --mode slurm --dataset reaction_diffusion trainer.max_epochs=5,10`.

## Workflow CLI
Use the unified Python workflow command `autocast` instead of bash wrappers.

Example usage:
```bash
# Train autoencoder locally
uv run autocast ae \
    --dataset reaction_diffusion \
    --run-label rd

# Train EPD on SLURM
uv run autocast epd \
    --mode slurm \
    --dataset reaction_diffusion \
    --run-label rd \
    trainer.max_epochs=10

# Re-run evaluation from an existing workdir
uv run autocast eval \
    --dataset reaction_diffusion \
    --workdir outputs/rd/00
```

For restart training, pass:
```bash
uv run autocast epd \
    --dataset reaction_diffusion \
    --workdir outputs/rd/00 \
    --resume-from outputs/rd/00/encoder_processor_decoder.ckpt
```

For `train-eval`, direct overrides are applied to training by default. Pass eval
settings with `--eval-overrides`, e.g.:
```bash
uv run autocast train-eval \
    --dataset reaction_diffusion \
    --run-label rd \
    trainer.max_epochs=1 \
    --eval-overrides eval.batch_indices=[0,1]
```

For SLURM train+eval submission:
```bash
uv run autocast train-eval \
    --mode slurm \
    --dataset reaction_diffusion \
    --run-label rd
```
This runs train and eval inside a single Hydra/Submitit SLURM job.

### Config-to-CLI mapping (to avoid override confusion)

- Hydra launcher config path: `src/autocast/configs/hydra/launcher/slurm.yaml`
- Cluster preset available: `local_hydra/hydra/launcher/slurm_baskerville.yaml` (repo-level)
- Mapping rule: config key `X` maps to CLI override `hydra.launcher.X=<value>`
    - `timeout_min` -> `hydra.launcher.timeout_min=...`
    - `cpus_per_task` -> `hydra.launcher.cpus_per_task=...`
    - `gpus_per_node` -> `hydra.launcher.gpus_per_node=...`
    - `tasks_per_node` -> `hydra.launcher.tasks_per_node=...`
    - `additional_parameters.mem` -> `hydra.launcher.additional_parameters.mem=...`

- For `autocast train-eval` specifically:
    - Positional overrides apply to **train**.
    - `--eval-overrides` applies to **eval**.
    - If the same key appears in both, eval uses the eval value.

File permissions / group-write:
- Submitit path reads config key `umask` (default `0002` in
    `src/autocast/configs/encoder_processor_decoder.yaml`).

To avoid long CLI override lists, put experiment defaults in a preset config
under `src/autocast/configs/experiment/` and enable it with `experiment=<name>`.

Example preset: `src/autocast/configs/experiment/epd_flow_matching_64_fast.yaml`

```bash
uv run autocast train-eval --mode slurm \
    --dataset advection_diffusion_multichannel_64_64 \
    experiment=epd_flow_matching_64_fast \
    autoencoder_checkpoint=/path/to/autoencoder.ckpt \
    hydra.launcher.timeout_min=30 \
    --eval-overrides +model.n_members=10
```

To use Baskerville module setup + scheduler defaults:

```bash
uv run autocast epd --mode slurm --dataset reaction_diffusion \
    hydra/launcher=slurm_baskerville
```

`--run-label` controls the top-level output folder (defaults to current date).
`--date` remains available as a backward-compatible alias.
If `--run-name` is omitted, `autocast` auto-generates a legacy-style run id and
uses it for both output folder naming and default `logging.wandb.name`.

Private/local experiment presets can be placed under repo-level
`local_hydra/local_experiment/` and enabled with `local_experiment=<name>`.
YAML files in this folder are git-ignored by default.

If you keep configs outside this repository (or when running from an installed
package), set:

```bash
export AUTOCAST_CONFIG_PATH=/absolute/path/to/configs
```

This directory should contain the same Hydra group layout (e.g.
`datamodule/`, `model/`, `experiment/`) expected by AutoCast.

Use `--dry-run` with any command to print resolved commands/scripts without
executing them.

CLI equivalents of removed `slurm_scripts/*.sh` examples are provided in:
```bash
bash scripts/cli_equivalents.sh
```

For launching many prewritten runs from a manifest list:
```bash
bash scripts/launch_from_manifest.sh run_manifests/example_runs.txt
```
```
