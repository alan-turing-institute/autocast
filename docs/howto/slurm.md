# Running AutoCast on SLURM

AutoCast supports running experiments on SLURM clusters by adding the `--mode slurm` flag.
This automatically generates a submission Bash script and submits it to the cluster, so you don't have to worry about writing your own submission scripts.

There are also a few [`distributed` configurations](https://github.com/alan-turing-institute/autocast/tree/main/src/autocast/configs/distributed) which allow you to specify the number of nodes/GPUs to use.

In particular, you can add the following command-line arguments:

- `distributed: single_gpu_slurm` to only request one GPU, even if a compute node has more than one GPU
- `distributed: ddp_4gpu_slurm` to request 4 GPUs on a single node
- `distributed: ddp_4gpu_2node_slurm` to request 4 GPUs per node across 2 nodes (8 GPUs in total)

If you want to use more than 2 nodes, you can use `ddp_4gpu_2node_slurm` and additionally specify the following overrides:

```
++trainer.num_nodes=3 ++eval.num_nodes=3 ++hydra.launcher.nodes=3
```

(replace `3` with the number of nodes you want to use).

## Training across allocations

Add `--chain N` to `epd` or `processor` to train across up to `N` Slurm allocations.
Use a new work directory and the usual dataset and launcher overrides.

```bash
uv run autocast epd --mode slurm --chain 3 --dry-run \
  --workdir outputs/rb-ambient-chain \
  local_experiment=the_well/rayleigh_benard/crps_spatiotemporal_ambient_matched \
  hydra.launcher.timeout_min=1440

uv run autocast processor --mode slurm --chain 2 --dry-run \
  datamodule=cached_latents datamodule.data_path=/path/to/gray-scott/cached \
  +distributed=single_gpu_slurm ++trainer.max_epochs=500
```

`--dry-run` prints the scripts and dependencies. For epoch- or step-based
schedules, set `trainer.max_epochs` and any explicit cosine horizon for the whole run.
Leave `trainer.min_steps` unset and keep `trainer.min_epochs` at most one so
training can hand off after any completed epoch.

For time-based cosine scheduling, set `++optimizer.scheduler_interval=time`
and set `trainer.max_time` explicitly for the total training budget. Use
`++trainer.max_epochs=-1` to remove the epoch cap. The clock resumes from the
checkpoint and excludes queue time. Training finishes at an epoch boundary.

The per-allocation budget (`hydra.launcher.timeout_min`) starts with the batch
script. The chain uses the longest observed epoch to decide whether another can
fit, reserving 10% of each allocation (1–30 minutes) for checkpointing. Allow
enough walltime for setup and at least one full epoch, including any validation.
A 48-hour training budget (`trainer.max_time=02:00:00:00`) needs at least three
24-hour allocations to allow for checkpointing, setup, and epoch boundaries.

Each continuation restores its predecessor's full checkpoint. The usual
checkpoint path points to the latest endpoint for evaluation. The run keeps its
initial resolved config, and `.chain/` keeps each checkpoint, script, and job ID.
Slurm logs record epoch, step, and handoff status. Later jobs skip training
once `.chain/finished` points to the completed checkpoint. If a time-based chain
runs out of allocations, the final job saves a full checkpoint and exits with an error.
Run evaluation separately with `autocast eval --workdir <run-directory>`.
