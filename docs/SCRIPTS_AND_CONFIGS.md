# Scripts and Configuration Guide

This guide explains the script structure and configuration system used in AutoCast.

## Workflow CLI (recommended)

Use the unified Python workflow command `autocast` as the primary interface.

Example usage:
```bash
# Train autoencoder locally
uv run autocast ae \
    datamodule=reaction_diffusion \
    --run-group rd

# Train EPD on SLURM
uv run autocast epd \
    --mode slurm \
    datamodule=reaction_diffusion \
    --run-group rd \
    trainer.max_epochs=10

# Re-run evaluation from an existing workdir
uv run autocast eval \
    datamodule=reaction_diffusion \
    --workdir outputs/rd/00
```

To restart training, pass:
```bash
uv run autocast epd \
    datamodule=reaction_diffusion \
    --workdir outputs/rd/00 \
    --resume-from outputs/rd/00/encoder_processor_decoder.ckpt
```

When running training and evaluation in a single command (`train-eval`), the provided arguments override training settings by default. Pass eval
settings with `--eval-overrides`, e.g.:
```bash
uv run autocast train-eval \
    datamodule=reaction_diffusion \
    --run-group rd \
    trainer.max_epochs=1 \
    --eval-overrides eval.batch_indices=[0,1]
```

For SLURM train+eval submission:
```bash
uv run autocast train-eval \
    --mode slurm \
    datamodule=reaction_diffusion \
    --run-group rd
```
This submits one SLURM job via `sbatch`; the CLI exits immediately after
submission.

### Config-to-CLI mapping (to avoid override confusion)

- Hydra launcher config path: `src/autocast/configs/hydra/launcher/slurm.yaml`
- Cluster preset available: `local_hydra/hydra/launcher/slurm_baskerville.yaml` (repo-level)
- Mapping rule: config key `X` maps to CLI override `hydra.launcher.X=<value>`
    - `timeout_min` -> `hydra.launcher.timeout_min=...`
    - `cpus_per_task` -> `hydra.launcher.cpus_per_task=...`
    - `gpus_per_node` -> `hydra.launcher.gpus_per_node=...`
    - `tasks_per_node` -> `hydra.launcher.tasks_per_node=...`
    - `use_srun` -> `hydra.launcher.use_srun=<true|false>`
    - `additional_parameters.mem` -> `hydra.launcher.additional_parameters.mem=...`

- SLURM launch behavior:
    - Default is auto: batch script uses `srun` when `tasks_per_node > 1` or `gpus_per_node > 1`.
    - Override explicitly with `hydra.launcher.use_srun=true` or `hydra.launcher.use_srun=false`.

- For `autocast train-eval` specifically:
    - Positional overrides apply to **train**.
    - `--eval-overrides` applies to **eval**.
        - `--eval-overrides` acts as a separator: put train overrides before it and
            eval overrides after it.
    - If the same key appears in both, eval uses the eval value.

File permissions / group-write:
- Training/eval scripts read config key `umask` (default `0002` in
    `src/autocast/configs/encoder_processor_decoder.yaml`).

To avoid long CLI override lists, put experiment defaults in a preset config
under `src/autocast/configs/experiment/` and enable it with `experiment=<name>`.

Example preset: `src/autocast/configs/experiment/epd_flow_matching_64_fast.yaml`

```bash
uv run autocast train-eval --mode slurm \
    datamodule=advection_diffusion_multichannel_64_64 \
    experiment=epd_flow_matching_64_fast \
    autoencoder_checkpoint=/path/to/autoencoder.ckpt \
    hydra.launcher.timeout_min=30 \
    --eval-overrides +model.n_members=10
```

To use Baskerville module setup + scheduler defaults:

```bash
uv run autocast epd --mode slurm datamodule=reaction_diffusion \
    hydra/launcher=slurm_baskerville
```

`--run-group` controls the top-level output folder (defaults to current date).
Use `--run-group` to set the top-level output folder label.
If `--run-id` is omitted, `autocast` auto-generates a legacy-style run id and
uses it for both output folder naming and default `logging.wandb.name`.
Backward-compatible aliases remain available: `--run-label` and `--run-name`.

W&B naming behavior:
- `--run-group` does not set W&B naming.
- `--run-id` sets the run folder name and default `logging.wandb.name`.
- Set `logging.wandb.name=...` directly as a Hydra override to explicitly name
    the W&B run.

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

### Timing epochs and computing `max_epochs` for cosine schedules

When using the `adamw_half` optimizer (half-period cosine LR schedule), the
learning rate decays from its initial value to zero over exactly
`trainer.max_epochs` epochs.  If training is cut short by `trainer.max_time`
before all epochs complete, the schedule will not have reached zero.

The `time-epochs` subcommand solves this by running a short timing run (a few
epochs), measuring per-epoch wall-clock duration, and computing the
`max_epochs` that fits within a given budget:

```bash
# Time 3 EPD epochs (default) and compute max_epochs for a 24h budget
uv run autocast time-epochs datamodule=advection_diffusion_multichannel

# Time an autoencoder run
uv run autocast time-epochs --kind ae datamodule=reaction_diffusion

# Time a processor run
uv run autocast time-epochs --kind processor datamodule=reaction_diffusion

# Custom: 5 timing epochs, 12h budget, 2% safety margin
uv run autocast time-epochs -n 5 -b 12 -m 0.02 \
    datamodule=shallow_water2d

# With experiment overrides
uv run autocast time-epochs experiment=epd_crps_vit_large_ps4_64

# Dry-run to inspect the generated command
uv run autocast time-epochs --dry-run datamodule=reaction_diffusion
```

`--kind` selects the training type to time: `ae`, `epd` (default), or
`processor`.  Use the same kind you intend to train so that the per-epoch
measurement reflects the actual model and data pipeline.

#### Batch timing via SLURM

With `--mode slurm` the timing run is submitted as a SLURM job and the CLI
exits immediately, printing a follow-up command to retrieve results once the
job completes:

```bash
# Submit timing jobs for several configs at once
uv run autocast time-epochs --mode slurm --kind ae \
    datamodule=reaction_diffusion --run-group timing
uv run autocast time-epochs --mode slurm --kind epd \
    datamodule=shallow_water2d --run-group timing \
    experiment=epd_crps_vit_large_ps4_64

# Once the SLURM jobs finish, compute results from the checkpoints
uv run autocast time-epochs --from-checkpoint outputs/timing/ae_.../timing.ckpt
uv run autocast time-epochs --from-checkpoint outputs/timing/epd_.../timing.ckpt
```

`--from-checkpoint` reads an existing checkpoint, extracts the per-epoch
times, and prints the recommendation — no training is run.  You can also
use it to recompute with a different budget or margin:

```bash
uv run autocast time-epochs --from-checkpoint outputs/timing/epd_.../timing.ckpt \
    -b 12 -m 0.05
```

The output includes recommended Hydra overrides ready to copy-paste:

```
============================================================
  Seconds/epoch:  150.0s
  Budget:         24.0h (margin: 2%)
  max_epochs:     564
  Expected time:  23.5h
  Headroom:       0.5h
============================================================

Recommended overrides:
  trainer.max_epochs=564 trainer.max_time=24:00:00:00 optimizer=adamw_half
```

The calculation is conservative:
- A 2% safety margin (configurable with `-m`) is subtracted from the budget.
- The result is rounded **down** to a whole epoch (`floor`), so the cosine
  schedule always completes its full half-period.
- `trainer.max_time` is set to the full (un-margined) budget as a hard stop.

Per-epoch times are extracted from the `TrainingTimerCallback` saved in the
checkpoint, which excludes model setup and data loading overhead.

#### How `max_epochs` and `max_time` interact at runtime

The recommended overrides set **two** stopping conditions:

| Condition | Controlled by | What happens |
|---|---|---|
| Epoch limit | `trainer.max_epochs` | Training stops cleanly after completing this many epochs. |
| Wall-clock limit | `trainer.max_time` | Lightning hard-stops training when the clock runs out. |

Lightning stops at whichever fires first.

**Faster than expected** (each epoch takes less time than the timing run
measured): `max_epochs` fires first.  All epochs complete, and the cosine LR
schedule reaches exactly zero.  `max_time` is never triggered.  This is the
ideal outcome.

**Slower than expected** (each epoch takes more time): `max_time` fires first,
cutting training short before all `max_epochs` have completed.  The cosine
schedule has *not* reached zero — the final LR is positive.

The 2% default margin tolerates up to ~2% slower epochs before `max_time`
intervenes.  The `floor()` rounding adds a small additional buffer (up to
one epoch's worth).  For workloads where epoch duration is stable
(compute-bound, data in memory), 2% is sufficient.  For I/O-bound workloads
that stream from a shared parallel filesystem, consider `--margin 0.05` or
higher.

**The cosine cannot overshoot and start increasing.**
`cosine_lambda(t) = 0.5 * (1 + cos(pi * t / max_epochs))` is monotonically
decreasing over `[0, max_epochs]`.  Training terminates at `max_epochs`, so
the second half of the cosine period is never entered.  If `max_time`
intervenes earlier, the LR is still on the decreasing branch — it simply
hasn't reached zero yet.

#### Choosing a margin

| Scenario | Recommended `--margin` |
|---|---|
| Data in memory, single GPU (very stable epoch times) | 0.02 (default) |
| Local NVMe data loading | 0.02 – 0.03 |
| Streaming from Lustre / GPFS | 0.05 – 0.10 |

To empirically check variance, run `time-epochs` twice at different cluster
load levels.  If the two per-epoch estimates agree within 3%, 2% margin is
safe.  If they diverge more, match the margin to the observed variance.

## Lower-level script entry points (advanced)

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
├── encoder_processor_decoder.yaml # default config for train_encoder_processor_decoder / autocast epd
├── processor.yaml               # default config for autocast processor
├── backbone/                    # Architectures (UNet, ViT)
├── datamodule/                  # Datasets (ReactionDiffusion, The Well)
├── encoder/                     # Encoder components (DC, PermuteConcat)
├── decoder/                     # Decoder components (DC, ChannelsLast)
├── processor/                   # Latent processors (FlowMatching, Diffusion)
├── model/                       # Model assembly configs
├── optimizer/                   # Optimizer settings (Adam, AdamW)
├── trainer/                     # Lightning Trainer settings
├── logging/                     # Weights & Biases configuration
├── eval/                        # Evaluation-specific settings
├── experiment/                  # Reusable experiment presets
├── hydra/                       # Hydra launcher/runtime configs
├── input_noise_injector/        # Input perturbation configs
├── simulator/                   # Simulator-related configs
└── external/                    # External model/config integrations
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
Use Hydra multi-run directly (or the manifest launcher) for sweeps, e.g. `uv run autocast epd --mode slurm datamodule=reaction_diffusion trainer.max_epochs=5,10`.

```
