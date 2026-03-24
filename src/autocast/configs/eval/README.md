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
- `metrics`: List of metrics to compute (default includes mse/mae/rmse/vrmse,
  power spectrum scores `psrmse*`, cross-correlation spectrum scores `pscc*`,
  and ensemble scores `crps`, `fcrps`, `afcrps`, `energy`, `variogram`, `ssr`)
- `csv_path`: Custom path for metrics CSV (default: work_dir/evaluation_metrics.csv)
- `video_dir`: Custom directory for rollout videos (default: work_dir/videos)
- `batch_indices`: List of batch indices to visualize
- `video_format`: Video format (mp4 or gif)
- `video_sample_index`: Sample index within batch to visualize
- `fps`: Frames per second for videos
- `accelerator`: Accelerator for evaluation (auto, cpu, cuda, mps)
- `devices`: Number of GPUs for DDP evaluation (auto, or int e.g. 1, 4)
- Ensemble-only metrics (`crps`, `fcrps`, `afcrps`, `energy`, `variogram`,
  `ssr`) are skipped automatically when `model.n_members <= 1`

## Multi-GPU Evaluation

Multi-GPU evaluation uses PyTorch Lightning Fabric and automatically
scales when the `distributed` config is composed. The distributed
configs already set `eval.devices` to match `gpus_per_node`:

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
