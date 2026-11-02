# Simulating data

To begin with, we're going to need some spatiotemporal data to work with.

You can get large-scale datasets from sources such as [The Well](https://polymathic-ai.org/the_well/), but these are too large to work with in this tutorial!
Instead of that, we'll generate some synthetic data by forward simulation of a partial differential equation (PDE).
Our sister tool [AutoSim](https://alan-turing-institute.github.io/autosim/) provides this functionality, with several built-in PDEs.

For this example we'll choose the advection–diffusion (AD) equation, which models the transport of a substance due to both diffusion (movement from high to low concentration) and advection (movement due to flow of the medium that the substance is in).
Details of the actual equation can be found on, e.g., [Wikipedia](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation).

First, install AutoSim:

```bash
git clone https://github.com/alan-turing-institute/autosim.git
cd autosim
uv sync
```

For the purposes of this tutorial, we'll assume that you've already [installed AutoCast](../installation.md), and that AutoSim and AutoCast have been cloned to the same parent folder.
In other words, your directory structure looks like the following.
This isn't mandatory: you can have a different directory structure, but you will need to adjust the paths in the later commands accordingly.

```
parent_folder
├── autocast
│   ├── pyproject.toml
│   ├── src
│   └── ...
└── autosim
    ├── pyproject.toml
    ├── src
    └── ...
```

:::{note}
This is also a good time to install `ffmpeg` if you haven't already: it will let you visualise the generated data as a video. See the [installation instructions](../installation.md#ffmpeg) for details.
:::

Then, from the `autosim` directory, run:

```bash
uv run autosim \
    simulator=spatiotemporal/advection_diffusion \
    simulator.n=16 \
    seed=42 \
    simulator.T=1.0 simulator.dt=0.1 \
    dataset.n_train=10 dataset.n_valid=3 dataset.n_test=3 \
    dataset.output_dir=../ad_data
```

This will generate a dataset with a spatial grid of size 16×16, with 10 trajectories (i.e. different sets of forward simulations) in the test set, 3 in the validation set, and 3 in the test set.

`T` and `dt` respectively refer to the total time of the simulation and the time step size.
In this case, we should expect each trajectory to contain 11 frames (including the starting point at time 0).

We won't go into more detail about `autosim` here, but you can find more information in the [documentation](https://alan-turing-institute.github.io/autosim/).

Your directory structure should now look similar to this:

```
parent_folder
├── ad_data
│   ├── cli.log
│   ├── examples
│   │   └── train
│   │       ├── batch_0.mp4
│   │       ├── batch_1.mp4
│   │       ├── batch_2.mp4
│   │       └── batch_3.mp4
│   ├── resolved_config.yaml
│   ├── stats.yml
│   ├── test
│   │   └── data.pt
│   ├── train
│   │   └── data.pt
│   └── valid
│       └── data.pt
├── autocast
└── autosim
```

There are a few files of interest here.
You won't need to handle any of them manually, but it's good to know what they are:

- `stats.yml` contains normalisation statistics for the dataset, calculated on the training set.
- `batch_*.mp4` are example videos of the generated trajectories.
  If you open these videos you will likely see not much going on: this is just because of the very short simulation we did!
  In a more realistic scenario the videos would be more interesting.
- `data.pt` files contain the actual data, stored as PyTorch tensors.
- `resolved_config.yaml` contains the configuration used to generate the dataset.
  This includes both the custom settings we used in the command-line invocation above, as well as other defaults that we didn't override.
  This is useful if you want to later see what settings were used to generate the dataset, or if you want to reproduce the dataset exactly.

We can load the data with PyTorch to see what it looks like.
You can run `uv run python` from the `autocast` directory to get a Python shell with the correct environment set up, and then run:

```python
>>> import torch

>>> torch.load("../ad_data/train/data.pt")["data"].shape
torch.Size([10, 11, 16, 16, 1])
```

The dimensions here are:

- 10 trajectories
- 11 time steps (including the initial state)
- 16×16 spatial grid
- 1 input channel (the concentration)

Now that we have some data in hand, let's move on to training a model on it!

## Larger datasets with Zarr

AutoSim can also export synthetic fields as sharded Zarr v3 arrays. Install the
optional reader with `uv sync --extra zarr`. Select `datamodule=zarr` and set
`datamodule.data_path` to a dataset root containing `train/data.zarr`,
`valid/data.zarr`, `test/data.zarr` and training-derived `stats.yml`.

The reader expects the `autosim.spatiotemporal` schema, version 1: each split is
a Zarr group with a `data` array in `[trajectory, time, height, width, channel]`
order, field names, provenance and a completion attribute. External Zarr stores
need preprocessing or an adapter to this schema; the file format alone does
not define variable selection or channel order.

Initialization reads metadata, and each sample reads only its requested window.
Choose `n_steps_input`, `n_steps_output` and `stride` in the datamodule; `stride`
advances the window start, while frames within the window remain consecutive.
Storage shards do not reset trajectories, and windows can cross shard boundaries.
Distributed training partitions these window indices with its normal sampler.

`datamodule.rollout_steps` bounds the target horizon of evaluation windows.
Shorter trajectories use the available frames after the context and `start_frame`;
the horizon must still contain at least one `n_steps_output` prediction block.
`datamodule.rollout_window_stride` optionally changes the spacing between their
start frames, defaulting to the training window stride.
Keep `eval.rollout_start=0` for Zarr: it assumes full trajectories. To start window
sampling later, set `datamodule.start_frame` (use `+datamodule.start_frame=N` on the
command line); this offsets the first context frame without shortening each window.
For trajectory statistics over multiple rollout windows per trajectory, set
`eval.trajectory_statistics.include_per_timestep=false`. Per-timestep trajectory
statistics currently require one sample per trajectory; evaluation checks this
before computing metrics.
The full-trajectory latent-cache workflow is not supported by this reader: long
trajectories require a streaming encoder, rather than loading an entire sequence
onto the device. Existing `.pt` and HDF5 configurations retain their own loaders.

### Measuring the storage layout

Create otherwise identical datasets using different AutoSim `dataset.chunk_steps`
and `dataset.shard_steps` values, then compare the actual lazy DataLoader:

```bash
uv run --extra zarr python -m autocast.scripts.benchmark_zarr_loader \
  ../data/layout_a/train/data.zarr ../data/layout_b/train/data.zarr \
  --input-steps 2 --output-steps 12 --batch-size 2 --workers 2 \
  --warmup-batches 5 --batches 100 --output outputs/loader-comparison.json
```

The benchmark uses the same random window indices for each layout. It reports
metadata initialization, first-batch startup, steady-state samples per second,
effective returned bytes per second and batch-wait latency. Throughput includes
the full measured consumption loop, not just time spent waiting for batches.
The batch size is **per process**, not the global training batch. File-cache
state is uncontrolled; these are CPU loader measurements, not physical disk
bandwidth or GPU training throughput. Repeat representative cases on the target
filesystem, including multiple training ranks, before selecting a layout.

Read chunks control how much data each small window must fetch. Shards group
chunks into larger files to reduce file count. Neither size fixes the training
window length. Compare short and longer windows, several worker counts and
prefetch depths; more workers are not necessarily faster. Worker-prefetch memory
scales roughly with `workers × prefetch_factor × batch_size × window_bytes`,
in addition to codec buffers and filesystem caches. The datamodule uses spawned
workers and process-local array handles.

### Native-grid training preset

`experiment=benchmark_synthetic_swin` uses the existing noise-conditioned
Swin encoder-decoder with a stride-one input projection. It processes every
input location before hierarchical merging, retaining fine-grid skip connections.
For 121×240 fields with 69 channels and two input frames, it has
**199,205,445 trainable parameters**:

| Stage | Grid | Spatial tokens | Hidden channels | Encoder / decoder blocks |
| --- | --- | ---: | ---: | ---: |
| 1 | 121×240 | 29,040 | 384 | 4 / 2 |
| 2 | 61×120 | 7,320 | 768 | 4 / 2 |
| 3 | 31×60 | 1,860 | 1,536 | 2 / 2 |

Shifted attention uses 7×7 windows. Existing merging and splitting handle
odd dimensions and restore the exact output grid. The preset uses two context
frames, one target frame, two training members, alpha-fair CRPS (alpha 0.95),
AdamW, and FP32/TF32. Batch size two is per GPU; keep the member count and
local batch fixed when comparing GPU counts.

Inspect the normal training API command without launching a job:

```bash
uv run autocast time-epochs --dry-run --kind epd --num-epochs 3 \
  experiment=benchmark_synthetic_swin \
  datamodule.data_path=/path/to/synthetic-fields \
  ++trainer.limit_train_batches=50 ++trainer.limit_val_batches=0 \
  ++trainer.num_sanity_val_steps=0 output.save_config=true
```

Before timing, check a few optimizer updates on one GPU, first with
`datamodule.batch_size=1`, then with the target batch size. Record actual CUDA
allocation peaks; retained activations are much larger than in a coarsely
patched model with the same parameter count. An eight-member training case
is a separate workload and requires its own memory check. Save the resolved
configuration, code revision and software versions alongside timing results.

This is a native-grid AutoCast workload, not a computational reproduction of
[Mosaic](https://arxiv.org/html/2604.16429v3): Mosaic uses HEALPix block-sparse
attention, 82 dynamic channels and mixed precision. Synthetic data measure
training cost; real atmospheric data are needed to establish forecast quality.

### Comparing synthetic training workloads

Four presets share the same training settings: two context frames, one target,
two ensemble members, alpha-fair CRPS (alpha 0.95), AdamW without a scheduler,
FP32 with TF32 allowed, and two persistent loader workers. Batch size defaults
to one per GPU. Grid names describe the intended dataset; select a matching
dataset explicitly. Parameter counts below assume 69 channels.

| `experiment=benchmark_synthetic_...` | Grid | Patch | Parameters | Finest spatial tokens |
| --- | --- | --- | ---: | ---: |
| `121x240_vit` | 121×240 | 4×4 | 193,741,040 | 1,860 |
| `121x240_swin` | 121×240 | 1×1 | 199,205,445 | 29,040 |
| `721x1440_vit` | 721×1440 | 16×16 | 244,629,920 | 4,140 |
| `721x1440_swin` | 721×1440 | 4×4 | 200,397,765 | 65,160 |

ViT uses global attention, width 1,024, 14 blocks and eight heads. Swin uses the
three-stage model above. Both pad incomplete patches and crop predictions back
to the input grid. Larger ViT patches bound attention cost at the larger grid;
these are distinct compute workloads, not matched-quality model comparisons.

An existing compatible dataset can be reused. Otherwise, generate a small shared
sample for both backbones from an AutoSim checkout:

```bash
uv run --extra zarr autosim --config-name=generate_data_synthetic_zarr \
  simulator.height=721 simulator.width=1440 simulator.n_channels=69 \
  dataset.n_train=1 dataset.n_valid=1 dataset.n_test=1 \
  dataset.steps_per_split.train=82 dataset.steps_per_split.valid=4 \
  dataset.steps_per_split.test=4 dataset.chunk_steps=1 dataset.shard_steps=1 \
  dataset.output_dir=/path/to/synthetic-721x1440x69
```

This is 25.79 GB of raw FP32 data across all splits. Its 80 training windows
support five complete batches per rank at eight GPUs and local batch two.
Use `height=121 width=240` for the smaller grid. Generate on a compute host
before timing; repeated access can warm filesystem caches, so this measures
training throughput with repeated data rather than full-archive storage speed.

From the AutoCast checkout, first run three optimizer updates on one GPU:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
experiment=benchmark_synthetic_721x1440_vit
dataset=/path/to/synthetic-721x1440x69
results=/path/to/benchmark-results

uv run --extra zarr autocast epd --mode slurm --workdir "$results/fit-b1" \
  experiment="$experiment" datamodule.data_path="$dataset" \
  +distributed=single_gpu_slurm hydra.launcher.nodes=1 \
  hydra.launcher.cpus_per_task=8 hydra.launcher.timeout_min=5 \
  hydra.launcher.additional_parameters.mem=40G \
  trainer.max_epochs=1 trainer.max_steps=3 trainer.limit_train_batches=3 \
  trainer.max_time=00:00:03:00 trainer.log_every_n_steps=1 \
  datamodule.batch_size=1
```

Require three completed updates and finite losses in `metrics.csv`. Try batch
two separately with a fresh work directory after batch one passes; an OOM at
batch two does not invalidate batch one. Keep the selected local batch fixed
for that workload's scaling curve. Record CUDA allocated/reserved peaks with
the fit check if needed; `timing.json` does not contain memory measurements.

Then time 150 updates, with a 15-minute Slurm limit:

```bash
uv run --extra zarr autocast time-epochs --mode slurm -n 30 \
  --workdir "$results/timing-1gpu-b1" \
  experiment="$experiment" datamodule.data_path="$dataset" \
  +distributed=single_gpu_slurm hydra.launcher.nodes=1 \
  hydra.launcher.cpus_per_task=8 hydra.launcher.timeout_min=15 \
  hydra.launcher.additional_parameters.mem=40G \
  datamodule.batch_size=1 ++trainer.max_time=00:00:12:00 \
  output.save_config=true
```

Use these substitutions for the other GPU counts and a fresh work directory
for each run. The existing presets set both Lightning and Slurm resources:

| GPUs | `+distributed=` | `hydra.launcher.nodes=` | `hydra.launcher.additional_parameters.mem=` |
| ---: | --- | ---: | --- |
| 1 | `single_gpu_slurm` | 1 | `40G` |
| 4 | `ddp_4gpu_slurm` | 1 | `160G` |
| 8 | `ddp_4gpu_2node_slurm` | 2 | `160G` |

Add site-specific module/NCCL setup through `hydra.launcher.setup`. Add
`--dry-run` to preview the training command without submission. On an allocated
single-GPU compute host, omit `--mode slurm`, the distributed preset and the
`hydra.launcher.*` overrides. Keep `--workdir` so local timing outputs persist.
The explicit `-n 30`, `++trainer.max_time` and `output.save_config=true` override
the timing command's defaults; they are required even though the preset defines
these settings.

Each short epoch contains five updates. Require `global_step=150`, 30 entries
in `timing.json`'s `epoch_times_s`, the expected `world_size`, finite losses and
no checkpoints. Discard the first ten epochs (50 warmup updates); for the
remaining 100 updates, compute:

```text
seconds_per_update = sum(epoch_times_s[10:30]) / 100
windows_per_second = world_size * batch_per_gpu / seconds_per_update
```

Compare the totals for epochs 10–19 and 20–29; investigate differences above
10% before treating the result as stable. Timings include normal epoch-reset
overhead. The CLI's printed average includes warmup, and its epoch-budget
recommendation refers to these five-update epochs, not full dataset passes.
Retain the resolved config, code revision, dataset identity, software versions
and per-rank GPU-utilization logs with the JSON. Synthetic timing establishes
compute cost, not convergence or predictive quality.
