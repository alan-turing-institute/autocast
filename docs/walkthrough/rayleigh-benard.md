# Rayleigh–Bénard with The Well

Download [Rayleigh–Bénard convection from The Well](https://polymathic-ai.org/the_well/datasets/rayleigh_benard/), reuse a pretrained autoencoder, train a latent processor, and evaluate decoded forecasts.
Choose either an AutoCast-native cache or an existing LoLA HDF5 cache; you do not need both.

:::{warning}
The dataset and pretrained autoencoder are large, and processor training is best run on a GPU.
The commands below are not executed while building these docs.
Add `--first-only` to the data download for a smaller workflow check.
:::

Run the commands from the AutoCast repository root unless noted otherwise.

## Download the data

The Well downloader creates `datasets/rayleigh_benard` with train, validation, and test splits.
Raw data is needed for caching and for evaluation against the original fields, even when reusing a latent cache.

```bash
uv run the-well-download \
    --dataset rayleigh_benard \
    --base-path .
```

## Download the pretrained autoencoder

[LoLA provides the autoencoder weights](https://github.com/francois-rozet/lola#pre-trained-models) used for its Rayleigh–Bénard latent experiments.
Download and unpack the 64× compression model:

```bash
curl --fail --location \
    --output datasets/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large.zip \
    https://users.flatironinstitute.org/~polymathic/data/lola/ae/1e3z5x2c_rayleigh_benard_dcae_f32c64_large.zip

unzip \
    datasets/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large.zip \
    -d datasets/rayleigh_benard
```

The archive contains only the model configuration and `state.pth` weights; it does not include cached latents.
AutoCast's checkpoint preset supplies the matching architecture, field normalization, and Rayleigh/Prandtl conditioning.

## Choose a latent cache

### Generate with AutoCast

Encode each trajectory once with the downloaded checkpoint:

```bash
uv run autocast cache-latents \
    --workdir outputs/rayleigh_benard/cache \
    --output-dir datasets/rayleigh_benard/cached_latents \
    local_experiment=cache_latents/the_well/rayleigh_benard/lola_f32c64
```

The preset encodes one frame at a time to limit activation memory and retains the full trajectories.
AutoCast saves `.pt` trajectories and the configuration needed to reload the autoencoder:

```text
datasets/rayleigh_benard/cached_latents
├── autoencoder_config.yaml
├── metadata.json
├── train
├── valid
└── test
```

### Reuse a LoLA cache

The maintained RB experiment presets use HDF5 caches produced by [LoLA's `cache_latents.py`](https://github.com/francois-rozet/lola/blob/main/experiments/cache_latents.py).
Keep those caches beside their matching `config.yaml` and `state.pth`:

```text
datasets/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large
├── config.yaml
├── state.pth
└── cache/rayleigh_benard
    ├── train
    ├── valid
    └── test
```

Each split contains HDF5 files with `state` (latent trajectories) and `label` (conditioning).
These use AutoCast's `miniwell` loader, not `cached_latents`.

:::{dropdown} Generate this format with LoLA instead

Follow [LoLA's setup instructions](https://github.com/francois-rozet/lola#code), then run from its `experiments` directory in that environment:

```bash
uv run python cache_latents.py \
    dataset=rayleigh_benard \
    run=/absolute/path/to/autocast/datasets/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large \
    server.datasets=/absolute/path/to/autocast/datasets \
    split=train repeat=4
```

Adapt `server.*` and `compute.*` to your cluster first: this script submits SLURM GPU jobs.
Repeat for `split=valid` and `split=test` with `repeat=1`, and wait for all jobs to finish.
LoLA's training recipe caches repeated random augmentations; the AutoCast preset above encodes each original trajectory once.
:::

## Train a processor

Train a small ViT-backed flow-matching processor on those cached latents:

```bash
uv run autocast processor \
    --workdir outputs/rayleigh_benard/processor \
    datamodule=cached_latents \
    ++datamodule.data_path="$PWD/datasets/rayleigh_benard/cached_latents" \
    ++datamodule.in_memory=false \
    processor@model.processor=flow_matching_vit \
    ++trainer.max_epochs=10
```

For a LoLA cache, replace the three `datamodule` arguments with:

```bash
datamodule=miniwell \
++datamodule.data_path="$PWD/datasets/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
```

To train diffusion with the same backbone, replace that processor override with:

```text
processor@model.processor=diffusion_vit
```

## Evaluate in physical space

Both cache routes use the same evaluation command:

```bash
uv run autocast eval \
    --workdir outputs/rayleigh_benard/processor \
    ++eval.chunk_size=1 \
    ++eval.max_test_batches=2 \
    ++eval.max_rollout_batches=2 \
    '++eval.batch_indices=[0]'
```

AutoCast finds the processor checkpoint and reconstructs the autoencoder from native-cache metadata or the adjacent LoLA checkpoint, then selects `encode_once` evaluation.
The processor rolls out in latent space, while predictions are decoded and scored against the original physical fields.
Results appear under `outputs/rayleigh_benard/processor/eval`, including aggregate CSV files, lead-time metrics, and rollout visualizations.
The batch limits make this an initial check; remove them for full evaluation.

## Scale up

The commands above expose the complete workflow with a short training budget.
For paper-scale runs, use the maintained [Rayleigh–Bénard experiment presets](https://github.com/alan-turing-institute/autocast/tree/main/local_hydra/local_experiment/the_well/rayleigh_benard) and [SLURM submitters](https://github.com/alan-turing-institute/autocast/tree/main/slurm_scripts/comparison/the_well/rayleigh_benard).
Those presets use the LoLA HDF5 cache and a larger ViT configuration.
They cover ambient and latent CRPS, flow matching, diffusion, and longer evaluations.
