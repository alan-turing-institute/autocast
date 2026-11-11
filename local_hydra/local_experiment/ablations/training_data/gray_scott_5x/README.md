# Gray–Scott data and training-budget ablations

These configurations compare a fivefold increase in training trajectories at
approximately the reference update budget, with separate doubled-budget
processor controls to examine sensitivity to additional optimization.

## Experiment matrix

All runs use training seed 42 and the original model sizes and per-GPU batches.
The training and validation pools grow fivefold. Training batches are sampled
at 20%, and validation window starts use stride 5 across every trajectory.
The primary test cohort remains the original 24 cases. The 2x controls also
double the processor training budget.

| Experiment suffix | Entry point | Sampling epochs / cosine horizon | Time cap | Batch/GPU |
| --- | --- | ---: | ---: | ---: |
| `ae_dc_large` | `ae` | 512 | 24 h | 16 |
| `fm_vit_large` | `processor` | 2830 | 24 h | 256 |
| `crps_vit_azula_large` | `epd` | 399 | 24 h | 32, M=8 |
| `fm_vit_large_2x` | `processor` | 5660 | 48 h | 256 |
| `crps_vit_azula_large_2x` | `epd` | 798 | 48 h | 32, M=8 |

All experiment overrides use the prefix
`ablations/training_data/gray_scott_5x/`. The separate `cache_latents` config
encodes all 1,200 training, 120 validation and 24 original test trajectories
using the newly trained AE. Both FM budgets share this one frozen AE and
latent cache. CRPS trains directly on the ambient fields.

## Sampling and budgets

`trainer.limit_train_batches: 0.2` consumes the first fifth of batches from a
fresh shuffled loader each epoch. In DDP, Lightning's distributed sampler
reshuffles at each epoch. No permanent subset of trajectories or windows is
selected. The complete 1,200-trajectory training pool remains eligible.

Validation uses `datamodule.validation_stride: 5`, retaining window starts
0, 5, 10, ... within every one of the 120 validation trajectories. The input
and output frames within each forecast window remain consecutive. This yields
a fixed validation cohort at each epoch without taking a regime-biased prefix
or introducing fresh sampling noise into checkpoint scores. Processor validation
uses 7,680 windows versus 7,608 in the original 24-trajectory benchmark; AE
validation uses 7,800 frames versus 7,704. This is approximately the original
validation compute spread over five times as many independent trajectories.
`trainer.limit_val_batches` stays at 1.0: do not thin the sparse loader again.
Final testing uses the original 24 trajectories with the original window stride.

These are **sampling epochs**, not complete passes over the enlarged dataset.
For example, the AE's 512 sampling epochs correspond to approximately 102.4 full
passes over the 5x training set. Training sample presentations remain
approximately equal to the original budget while the pool of independent
trajectories is five times larger. Batch rounding can change updates per epoch
slightly: with four GPUs, the baseline/full-pool/limited batch counts are
1204/6019/1203 for AE, 75/372/74 for FM, and 595/2972/594 for CRPS. Record actual
updates and sample counts rather than claiming exact equality.

The reference epoch counts are defined in:

- `slurm_scripts/comparison/ae/submit_ae_large.sh`
- `slurm_scripts/comparison/cached_latents/submit_fm_large.sh`
- `slurm_scripts/comparison/epd/submit_crps_large.sh`

Use these reference epoch horizons without new timing calibration. The larger
pool may change initialization, I/O and epoch timings, so record actual runtime
and updates. A time cap is a ceiling, not a guarantee that all configured epochs
finish. Keep the AE at its declared 512-sampling-epoch target, and report any
timer-truncated schedule explicitly.

The 2x processor experiments start from scratch with doubled epoch and cosine
horizons. They do not resume or fine-tune a completed 24-hour run. The midpoint
of a 48-hour schedule does not replace the separately annealed 24-hour control.

## Dataset and checkpoint preparation

The AutoSim config `generate_data_gray_scott_5x` generates 1,200/120/120 splits.
Leave these files intact. The training draw is new and is not nested around
the original 240 trajectories. `test_data_path` independently
points at the published test file, retaining exactly the same cases for the
primary comparison without overwriting the generated test pool. Keep the newly
generated training-only normalization statistics. Record both repositories'
commits, the resolved generation configuration, reference test hash and AE/cache
identities.

The 120 validation trajectories are new; checkpoint selection is therefore not
identical to the original 24-case protocol. State that change explicitly. The
published validation files remain available for a selection sensitivity check.

Both CRPS budgets select the best validation MultiWinkler checkpoint from the
first validation, following the September GPE/GS/AD U-Net ablations. The
`gs5x_crps` trainer replaces the separate pre-25% and after-25% MultiWinkler
trackers with one overall tracker starting at 0%. It retains the after-50% and
after-75% trackers, coverage trackers, 5%-interval snapshots, best validation
loss and EMA. This differs from the published GS ViT selection after 25%;
report the selection rule explicitly. Existing prepared campaigns retain their
saved callback list and source commit until deliberately refreshed.

Supply dataset and artifact paths through environment variables:

```bash
export AUTOCAST_GS_5X_DATA=/path/to/expanded_dataset
export AUTOCAST_GS_REFERENCE_DATA=/path/to/reference_dataset
export AUTOCAST_GS_5X_CACHE=/path/to/latent_cache
# Set after AE training; use the same selected final AE for both FM budgets.
export AUTOCAST_GS_5X_AE_CHECKPOINT=/path/to/autoencoder.ckpt
```

The required environment variables prevent accidental fallback to the original
240-trajectory dataset or latent cache. The cache config intentionally inherits
the encoder/decoder settings rather than the AE training config: caching must
visit every trajectory, without the training batch limit or AE frame mode.
Its validation stride is explicitly 1; FM subsequently applies stride 5 when
loading the complete cached trajectories. The cache's test split is the original
24 cases encoded with the new AE, not the old AE's cached representations.

The cache writer saves a fully resolved `autoencoder_config.yaml`, pinning the
AE checkpoint, raw-data directory and normalization path used for this cache.
Changing or unsetting the environment variables later cannot redirect decoding
to a different AE or dataset.

## Inspect and run the configurations

Run commands from the repository root so Hydra finds its `local_hydra` directory.
Use the normal uv project environment. For example, inspect each config without
loading data, training, or contacting W&B:

```bash
uv run python -m autocast.scripts.train.autoencoder \
  local_experiment=ablations/training_data/gray_scott_5x/ae_dc_large \
  --cfg job --resolve
uv run python -m autocast.scripts.cache_latents \
  local_experiment=ablations/training_data/gray_scott_5x/cache_latents \
  --cfg job --resolve
uv run python -m autocast.scripts.train.processor \
  local_experiment=ablations/training_data/gray_scott_5x/fm_vit_large \
  --cfg job --resolve
uv run python -m autocast.scripts.train.encoder_processor_decoder \
  local_experiment=ablations/training_data/gray_scott_5x/crps_vit_azula_large \
  --cfg job --resolve
```

Inspect the `_2x` names in the same way. Preview training commands with
`--dry-run`:

```bash
uv run autocast ae --mode local --dry-run datamodule=gray_scott_5x \
  local_experiment=ablations/training_data/gray_scott_5x/ae_dc_large
uv run autocast processor --mode local --dry-run datamodule=cached_latents \
  local_experiment=ablations/training_data/gray_scott_5x/fm_vit_large
uv run autocast epd --mode local --dry-run datamodule=gray_scott_5x \
  local_experiment=ablations/training_data/gray_scott_5x/crps_vit_azula_large
```

The training configurations use four GPUs with DDP and the reference per-GPU
batch sizes. Account for global batch size and update count when adapting to
other device counts.

## Prepare jobs and manual continuations

`slurm_scripts/ablations/gray_scott_data_scaling.py` writes resolved configs and
batch scripts without submitting jobs. Run preparation from the committed
checkout on the machine that will execute the jobs:

```bash
uv run python -m slurm_scripts.ablations.gray_scott_data_scaling prepare \
  --run-root /path/to/new_campaign \
  --data "$AUTOCAST_GS_5X_DATA" \
  --reference-data "$AUTOCAST_GS_REFERENCE_DATA"
```

This prepares AE, CRPS 24-hour, CRPS 48-hour and FM 24-hour training, caching,
and evaluations on both test cohorts. Add `--include-fm48` for the optional
longer FM control. Inspect an individual job with:

```bash
uv run python -m slurm_scripts.ablations.gray_scott_data_scaling preview \
  --campaign /path/to/new_campaign/campaign.json --stage crps48
```

Each training allocation stops at a completed epoch with checkpoint headroom.
The experiment keeps its full epoch target, cosine horizon and cumulative time
budget. A clean allocation exit can therefore mean `continuation_required` in
the run's `status.json`; it must not be interpreted as a completed experiment.
Only completed dependencies unlock caching, FM and evaluation.

After inspecting a stopped run, prepare its next allocation with:

```bash
uv run python -m slurm_scripts.ablations.gray_scott_data_scaling continue \
  --campaign /path/to/new_campaign/campaign.json --stage crps48
```

This writes a new batch script naming the latest recorded checkpoint. It does
not submit it. Each stopped segment records the checkpoint's SHA-256 alongside
its step count. Continuation requires that exact content, including when an
explicit checkpoint path names a byte-identical copy. A checkpoint from another
run at the same step is rejected. Execution checks the code/config identity
and verifies restored
model, optimizer, scheduler, EMA and timer state before the first update. Keep
the campaign directory and tracking ID fixed across continuations. Automatic
requeue is disabled. The preparation tools never call `sbatch`.

When ready, run AE, then cache latents, then the processor experiments. Keep the
same checkpoint-selection conventions as the main comparison: final FM weights
and validation-selected MultiWinkler CRPS weights. Evaluate both single-step
and rollout VRMSE, coverage MAE and interval scores on the same test cases.
Report actual GPU-hours and training curves, separating AE and cache costs
from processor training. The longer processor runs probe additional optimization;
they do not establish AE convergence or universal sufficiency of the data.

The generated 120-trajectory test set is an optional second final evaluation.
For raw-data evaluation, override `datamodule.test_data_path` with
`$AUTOCAST_GS_5X_DATA/test/data.pt`; retain each model's own training-time
normalization and frozen AE. Evaluate original and new checkpoints on this same
expanded cohort and report it separately from the historical 24-case test.
Prepared evaluations preserve the reference ten-member, 25-rollout-step
protocol and metric windows. FM uses raw-field encode-once evaluation with its
selected AE for either cohort; the old-test latent cache is not reused for the
new test cases. CRPS selects the validation-best overall MultiWinkler checkpoint;
FM uses final weights. Both retain their training normalization statistics.

To prepare the original FM and CRPS baselines on both cohorts, copy
`reference_baselines.example.yaml`, supply the exact saved configs and selected
checkpoints used for the reported baseline results, and add
`--baseline-manifest /path/to/reference_baselines.yaml` to `prepare`.
Use the original final FM processor checkpoint and matching AE. For CRPS, retain
the validation-selected checkpoint used in the reported comparison, including
its selection window. Do not select weights using either test cohort.

The manifest's configs must be readable during preparation; checkpoint paths
must be absolute and available when evaluation eventually runs. A local copy
of the saved metadata can be used for preparation. Configs are resolved and
pinned in the campaign; checkpoint files are never loaded during preparation.
The four `eval_baseline_<method>_<cohort>` jobs have no dependency on new
training. They retain each original model's normalization and AE while changing
the test-file path, and use the same metrics, trajectory statistics and sampling
seed as the newly trained models. This enables the expanded-test comparison
without substituting the enlarged training statistics into the old models.
