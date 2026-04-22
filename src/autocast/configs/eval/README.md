# Evaluation Configurations

This directory contains composable evaluation configurations for different model types.

## Available Configs

- **`default.yaml`**: Base evaluation config with sensible defaults
- **`encoder_processor_decoder.yaml`**: EPD-specific eval settings (extends default)
- **`processor.yaml`**: Processor model eval settings (extends default)

## Usage

### Using Default Eval Config

Eval configs are automatically included via the `defaults` section in main configs:

```yaml
defaults:
  - optional eval: encoder_processor_decoder
```

### Override Eval Config

You can switch to a different eval config:

```bash
python -m autocast.scripts.eval.encoder_processor_decoder \
  eval=default \
  eval.checkpoint=path/to/model.ckpt
```

### Override Specific Eval Parameters

```bash
python -m autocast.scripts.eval.encoder_processor_decoder \
  eval.checkpoint=path/to/model.ckpt \
  eval.metrics=[mse,rmse,mae] \
  eval.max_rollout_steps=50 \
  eval.batch_indices=[0,1,2,3]
```

## Config Structure

All eval configs support these parameters:

- `checkpoint`: Path to model checkpoint (required for evaluation)
- `mode`: Evaluation regime (`auto` (default) | `encode_once` | `ambient` |
  `latent`). Controls the **rollout space**, not just the metrics space.
  `auto` dispatches to a concrete mode at run time based on the checkpoint
  and datamodule, so omitting the flag gives the fair default for every
  run. See [Evaluation modes](#evaluation-modes) below.
- `metrics`: List of metrics to compute (default includes mse/mae/rmse/vrmse,
  power spectrum scores `psrmse*`, cross-correlation spectrum scores `pscc*`,
  and ensemble scores `crps`, `fcrps`, `afcrps`, `energy`, `spread`, `skill`,
  `ssr`; `variogram` remains available via explicit opt-in. Note that for
  ensemble predictions, `skill` is the RMSE of the ensemble mean, so it matches
  `rmse` numerically and is included for explicit spread/skill reporting.)
- `csv_path`: Custom path for metrics CSV (default: work_dir/evaluation_metrics.csv)
- `video_dir`: Custom directory for rollout videos (default: work_dir/videos)
- `batch_indices`: List of rollout sample indices to visualize (resolved across
  batched rollout dataloader samples)
- `video_format`: Video format (mp4 or gif)
- `video_sample_index`: Sample index within batch to visualize
- `fps`: Frames per second for videos
- `accelerator`: Accelerator for evaluation (auto, cpu, cuda, mps)
- `devices`: Number of GPUs for DDP evaluation (default: 1; set explicitly,
  e.g. 4, for multi-GPU runs)
- Ensemble-only metrics (`crps`, `fcrps`, `afcrps`, `energy`, `variogram`,
  `ssr`) are skipped automatically when `model.n_members <= 1`

## Multi-GPU Evaluation

Multi-GPU evaluation uses PyTorch Lightning Fabric. By default, eval uses
`eval.devices=1` for reproducibility and easier comparisons. Override to
multi-GPU explicitly (or via a distributed preset):

```bash
# 4-GPU evaluation via the distributed config
autocast eval +distributed=ddp_4gpu_slurm eval.checkpoint=path/to/model.ckpt

# Explicit GPU count override
autocast eval eval.devices=4 eval.checkpoint=path/to/model.ckpt
```

On SLURM, `srun` propagates `LOCAL_RANK` / `WORLD_SIZE` into the
process so Fabric DDP initialises automatically — no extra flags needed.
- `max_rollout_steps`: Maximum number of rollout steps
- `free_running_only`: Whether to disable teacher forcing

## Evaluation modes

The `eval.mode` knob controls the **rollout space** and what the metrics
compare against. The three concrete modes give the same answer on single-
step (windowed test) metrics; they only diverge during free-running
rollout. `auto` is a dispatcher that picks one of the concrete modes at
run time.

| mode          | encoder runs        | processor rolls out in | decoder runs | ground truth used                          | when to use                                                                                                        |
| ------------- | ------------------- | ---------------------- | ------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `encode_once` | **once** (step 0)   | latent space           | per step     | raw `batch.output_fields` (denormalized)   | fair processor-only eval that avoids decode/encode drift but still scores against real ground truth.               |
| `ambient`     | per rollout step    | data space (re-encoded each step) | per step     | raw `batch.output_fields` (denormalized)   | apples-to-apples comparisons with pure-ambient baselines (e.g. CRPS vs. a non-autoencoder model).                  |
| `latent`      | once (step 0)       | latent space           | only for metrics (or skipped via `latent_space_metrics=true`) | **decoded cached latents** (autoencoder reconstruction of ground truth) | measure the processor against what the autoencoder sees -- isolates processor error but hides AE reconstruction error. |

### `auto` (default)

`eval.mode=auto` dispatches to the faithful concrete mode for the current
run:

- **Full EPD checkpoints** (including processor runs with stateless
  encoder/decoder baked in, e.g. `permute_concat` + `identity`) -> `ambient`.
  `encode_once` and `ambient` are numerically identical here; `auto`
  picks `ambient` to keep logs quiet. Passing `eval.mode=encode_once`
  explicitly on such a run still works but emits a warning.
- **Processor trained on cached latents + autoencoder available**
  (either via `autoencoder_checkpoint=<ae.ckpt>` or via
  `<cache_dir>/autoencoder_config.yaml`) -> `encode_once`. Strictly fairer
  than `ambient` (no drift penalty) **and** than `latent` (AE
  reconstruction error is visible against raw ground truth).
- **Processor trained on cached latents, autoencoder not reachable**
  -> `latent`. The only faithful option when you can decode but not
  re-encode. If no decoder can be built either, `auto` does not silently
  fall through to latent-only metrics -- it fails fast so you either fix
  the autoencoder path or opt in explicitly via
  `eval.mode=latent eval.latent_space_metrics=true`.

The resolved mode is logged at INFO as `eval.mode=auto resolved to <X>`.

### Explicit modes

#### Ambient: apples-to-apples with pure-ambient baselines

`eval.mode=ambient` forces full `encoder -> processor -> decoder` at every
rollout step. The decoded field is re-encoded as the next step's input, so
autoencoder decode/encode drift compounds into the metrics. This is the
right regime when the baseline model operates natively in data space and you
want to charge the autoencoder for any error it introduces. Requires
`autoencoder_checkpoint=<ae.ckpt>` and a raw-Batch datamodule. When the
current datamodule yields `EncodedBatch` (cached latents), eval
auto-substitutes the datamodule from `<cache_dir>/autoencoder_config.yaml`
saved by `autocast cache-latents` (pass `datamodule=...` explicitly to
override).

#### Latent: measure the processor against the AE's view of the world

`eval.mode=latent` forces latent-space rollout: the processor's predicted
latent is fed back as the next latent input and the encoder is never
invoked past step 0. Metrics are decoded to data space via the decoder
saved alongside the cached latents and **compared against decoded cached
latents** -- i.e. an autoencoder reconstruction of ground truth, not the
raw fields. Use this when you want to isolate the processor's rollout
quality in its own training distribution and explicitly accept that AE
reconstruction error is hidden from the metric.

A reachable decoder is required; if the cache directory's
`autoencoder_config.yaml` or checkpoint is missing the run fails fast
rather than silently falling back to computing metrics in raw latent
space (those numbers were never comparable across runs).

##### Dev sense-check: latent-only metrics

Sometimes you want to iterate on a small processor paired with a large /
expensive autoencoder and skip the decoder entirely. Pass
`eval.mode=latent eval.latent_space_metrics=true` to opt in:

```bash
autocast eval --workdir <processor_workdir> \
  eval.mode=latent \
  eval.latent_space_metrics=true \
  eval.checkpoint=<processor.ckpt>
```

This skips the decoder lookup and compares processor predictions against
cached latents directly in the autoencoder's raw latent space. Treat the
numbers as a cheap sanity check only: they are **not comparable across
runs** (latent space is basis-dependent) and physics-aware metrics
(`psrmse*`, `pscc*`, `variogram`) are not meaningful. The flag is
rejected for any other `eval.mode` because the raw-space modes (`auto`,
`ambient`, `encode_once`) require a decoder by definition.

### Running the ablations

Given an autoencoder checkpoint and a processor checkpoint trained on its
cached latents:

```bash
# Default: auto -> encode_once here (fair processor-only eval, raw ground truth).
autocast eval --workdir <processor_workdir> \
  eval.checkpoint=<processor.ckpt> \
  autoencoder_checkpoint=<autoencoder.ckpt>

# Apples-to-apples with pure-ambient baselines (charges AE drift).
autocast eval --workdir <processor_workdir> \
  eval.mode=ambient \
  eval.checkpoint=<processor.ckpt> \
  autoencoder_checkpoint=<autoencoder.ckpt>

# Processor-only latent view; no raw ground truth, hides AE reconstruction error.
autocast eval --workdir <processor_workdir> \
  eval.mode=latent \
  eval.checkpoint=<processor.ckpt>
```

The three runs differ on rollout metrics as follows:

- `ambient - encode_once` = decode/encode drift accumulated over rollout
  steps (charged to the autoencoder).
- `encode_once - latent` = visibility of AE reconstruction error against the
  raw field (absent from `latent`, included in `encode_once`).
