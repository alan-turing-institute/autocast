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
- `mode`: Evaluation regime (`auto` | `ambient` | `latent`). Controls the
  **rollout space**, not just the metrics space. See
  [Ambient vs latent rollout](#ambient-vs-latent-rollout) below.
- `metrics`: List of metrics to compute (default includes mse/mae/rmse/vrmse,
  power spectrum scores `psrmse*`, cross-correlation spectrum scores `pscc*`,
  and ensemble scores `crps`, `fcrps`, `afcrps`, `energy`, `ssr`; `variogram`
  remains available via explicit opt-in)
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

## Ambient vs latent rollout

Processor checkpoints trained on cached latents can be evaluated in two
qualitatively different regimes. The `eval.mode` knob makes the choice
explicit and surfaces clear errors when the rest of the config is
inconsistent with the request.

- `eval.mode=auto` (default) preserves historical behavior: the script picks
  a path based on `(checkpoint type, datamodule batch type,
  autoencoder_checkpoint)`.
- `eval.mode=ambient` forces full `encoder -> processor -> decoder` rollout.
  Each rollout step decodes to ambient fields and re-encodes on the next
  step, so decode/encode drift is included in the metrics. **This is the
  apples-to-apples regime for comparing against baselines that natively roll
  out in data space (e.g. a CRPS comparison against a non-autoencoder
  model).** Requires `autoencoder_checkpoint=<ae.ckpt>` and a raw-Batch
  datamodule. When the current datamodule yields `EncodedBatch` (cached
  latents), eval auto-substitutes the datamodule from
  `<cache_dir>/autoencoder_config.yaml` saved by `autocast cache-latents`.
  Pass `datamodule=...` explicitly to override the default.
- `eval.mode=latent` forces latent-space rollout: the processor's predicted
  latent is fed back as the next latent input; the encoder is invoked only
  once. Metrics are decoded to data space via the decoder saved alongside
  the cached latents when available, otherwise they are reported in latent
  space. Requires an `EncodedBatch` / cached-latents datamodule.

### Running the ambient ablation

Given an autoencoder checkpoint and a processor checkpoint trained on its
cached latents, a minimal invocation is:

```bash
# Ambient (encoder -> processor -> decoder at every rollout step)
autocast eval --workdir <processor_workdir> \
  eval.mode=ambient \
  eval.checkpoint=<processor.ckpt> \
  autoencoder_checkpoint=<autoencoder.ckpt>

# Latent (processor rollout stays in latent space; decoded only for metrics)
autocast eval --workdir <processor_workdir> \
  eval.mode=latent \
  eval.checkpoint=<processor.ckpt>
```

The ambient run will differ from the latent run by exactly the
decode/encode drift accumulated over rollout steps, which is the relevant
delta when comparing against purely-ambient baselines.
