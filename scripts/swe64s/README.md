# Stochastic SWE 64x64 baselines

This directory is the entry point for the larger stochastic SWE baselines:
paper-scale global-noise afCRPS and latent flow matching (FM). The `s` suffix
means stochastic; a future deterministic counterpart can use `swe64d`.
Keep simulator-parameter variation and model conditioning as separate config
choices within each study.

The baseline workflow uses the standard AutoCast model, data and metric APIs.
Historical 32x32 runs and follow-on loss, noise and physics variants are separate
comparisons; they are not dependencies of these initial fits.

## Data and scope

The [dataset manifest](../../local_hydra/local_experiment/swe64s/dataset.json)
records the generator identity, physical settings and split checksums. The
existing dataset has 200/20/20 trajectories, 321 frames, channels `h,u,v`,
saved every 0.25 from t=40 through t=120. Physical parameters are fixed across
trajectories; initial states and forcing realizations vary. Training
normalization comes from the existing `stats.yml`, not recomputed test data.

The new prediction task uses one input frame, four output frames and window
stride one, matching the paper-style setup. These are direct full-field targets,
not unrolled two-step or physics-residual training. The larger dataset, model
and output window mean this is not a resolution-only ablation of the earlier
32x32 fits.

## Presets and shared settings

All presets live in
[`local_hydra/local_experiment/swe64s`](../../local_hydra/local_experiment/swe64s).
`common.yaml` holds windowing and portable launch defaults; `data.yaml` adds the
raw dataset and normalization paths. `afcrps.yaml` pins width 568, 12 blocks,
8 heads, patch 4, 1024D global AdaLN noise, eight training members and
alpha=0.95. The fixed scalar metadata is excluded by `with_constants=false`
and `include_global_cond=false`; the current fields and latent noise still
condition the forecast.

The processor's standard global-noise sampling and gated transformer blocks
are used without extension-specific options. FM likewise uses its standard
single-draw training objective. This preserves the baseline settings without
requiring spatial-noise, sublayer-conditioning or multistep-training APIs.

The architectural basis is
[`epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`](../../local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml).
Published run and checkpoint selections are indexed in
`autocast_paper_02/provenance/2026-09-25_submission_outputs/SUMMARY.md`.
Use that inventory to distinguish the main afCRPS run from ordinary/fair-CRPS
and architecture ablations; preset names alone do not identify those runs.

`autoencoder.yaml` uses the paper DC autoencoder capacity and factor-four
compression: 64x64x3 fields become 16x16x8 latents. Its three levels have two
stride-two transitions; input patching is explicitly one. Circular padding is
a deliberate SWE adaptation to the periodic domain; the published CNS AE used nonperiodic
padding. `cache_latents.yaml` mirrors the encoder/decoder settings and saves
all 321 frames, one trajectory per encoding call. Do not reuse another dataset's
AE or silently change normalization between fitting and caching.

`flow_matching.yaml` follows the
[`processor/conditioned_navier_stokes/fm_vit_large.yaml`](../../local_hydra/local_experiment/processor/conditioned_navier_stokes/fm_vit_large.yaml)
capacity: width 704, 12 blocks, 8 heads, patch 1 on the latent grid and 50 Euler
steps at inference. FM retains field conditioning and the flow-time embedding,
but excludes simulator-parameter conditioning. Its flow-matching objective is
not comparable numerically with afCRPS. Batch 256 matches 32x8 afCRPS model
evaluations per GPU, not the number of independent training inputs; record
optimizer updates and sample counts when comparing training budgets.

## Local preparation and launch gate

Run commands from the AutoCast repository root with `uv`. Set
`AUTOCAST_DATASETS` to the existing dataset parent and `SWE64S_OUTPUTS` to an
approved scratch/output directory before training. With no output override,
Hydra uses the ignored `outputs/swe64s` directory. W&B is disabled by default.
The presets do not select a cluster launcher or submit jobs.

The read-only helper composes all four job presets without loading the dataset
or allocating model weights. Without `--configs-only`, it checks the raw split
shapes, fixed parameters and normalization on CPU using memory-mapped tensors.
`--verify-hashes` additionally reads every split byte; `--cache /path/to/cache`
checks an existing cache's counts, first trajectory shapes and saved AE settings.

```sh
uv run --frozen python scripts/swe64s/check_inputs.py --configs-only
uv run --frozen python scripts/swe64s/check_inputs.py --verify-hashes
```

```sh
export AUTOCAST_DATASETS=/projects/u6eo/autocast/datasets
export SWE64S_OUTPUTS=/path/to/approved/scratch/swe64s

# Compose only: no dataset load or model fit.
uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps --cfg job

uv run --frozen train_autoencoder \
  local_experiment=swe64s/autoencoder --cfg job

uv run --frozen cache_latents \
  local_experiment=swe64s/cache_latents \
  autoencoder_checkpoint=/path/to/new-swe64/autoencoder.ckpt \
  cache_latents.output_dir=/path/to/new-empty-cache --cfg job

uv run --frozen python -m autocast.scripts.train.processor \
  local_experiment=swe64s/flow_matching \
  datamodule.data_path=/path/to/new-cache --cfg job
```

afCRPS and FM use the existing wall-clock cosine scheduler:
`scheduler=cosine`, `scheduler_interval=time` and zero warmup. LR decay follows
Lightning's resume-aware Timer, with a 23h30m training cap and no timing runs or
estimated epoch counts. Their finite ceiling of one million epochs only keeps
progress callbacks well-defined; elapsed time is the binding budget.

The AE retains the historical fixed schedule: `max_epochs=512`,
`scheduler_interval=epoch`, `cosine_epochs=512` and zero warmup. Its 23h30m
`max_time` is a safety cap, not the cosine horizon; if time expires early, record
the completed epochs rather than claiming the full 512. The original setting is
in [`submit_ae_large.sh`](../../slurm_scripts/comparison/ae/submit_ae_large.sh).
Learning rates remain 2e-4 for afCRPS, 1e-4 for FM and 1e-5 for the PSGD AE.

The allocation is nominally 24 hours on four GPUs per fit, leaving a 30-minute
buffer for startup and finalization. afCRPS/FM do not reuse CNS's 473/3223 epoch
estimates. The portable presets retain one-device defaults; use the four-GPU
launch settings below to match the intended compute allocation.

The study-local `trainer=swe64s_time` policy saves hourly snapshots and
`last.ckpt`, best validation loss and, when logged, overall/post-25% MultiWinkler
checkpoints. It uses existing callbacks, not a new scheduler implementation.
The post-25% window follows elapsed time. EMA remains stored separately;
the main evaluation uses raw weights.

Record batch/device count, updates, wall time, overrides, code/lock hashes,
dataset checksums and the selected checkpoint hash with each run. Evaluate raw
weights for the main comparison, even if the inherited callback stores EMA.

Fit and review the AE before caching and launching FM. Check
reconstruction error and each channel's spectrum, plus divergence/vorticity
balance: decoder artifacts must not be mistaken for FM uncertainty artifacts.
The cache must retain the raw split counts and `(321,16,16,8)` trajectory shape.
The existing cacher writes into its output directory, so use a fresh directory;
do not reuse a completed cache destination.

## Four GPU launch previews

Run these from the repository root on the target machine after environment
and data preflight. They are previews: `--dry-run` does not submit a job.
Use fresh work directories. Only remove `--dry-run` after approving launch.
The explicit time override prevents the distributed preset's 12-hour default
from replacing the study budget. The allocation is one node, four GPUs and
four tasks for up to 24 hours; the fit cap leaves a 30-minute allocation buffer
for startup and finalization.

```bash
swe64_launch_overrides=(
  '+distributed=ddp_4gpu_slurm'
  'trainer.max_time=00:23:30:00'
  '++hydra.launcher.nodes=1'
  '++hydra.launcher.cpus_per_task=72'
  'hydra.launcher.timeout_min=1440'
)

uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/afcrps" \
  local_experiment=swe64s/afcrps "${swe64_launch_overrides[@]}"

uv run --frozen autocast ae --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/autoencoder" \
  local_experiment=swe64s/autoencoder "${swe64_launch_overrides[@]}"

# Only after the matching AE has been reviewed and its fresh cache verified.
uv run --frozen autocast processor --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/flow_matching" \
  local_experiment=swe64s/flow_matching \
  datamodule.data_path="$SWE64S_OUTPUTS/cached_latents" \
  "${swe64_launch_overrides[@]}"
```

## Training pilots

Before the full allocations, run a short target-GPU smoke test of afCRPS and
the AE with the intended four-GPU model, batch, precision and data settings.
The `trainer=swe64s_pilot` profile uses a 30-minute fit cap inside a 60-minute
allocation, at most 200 optimizer updates, validation every 50 updates and four
validation batches. It keeps the model, optimizer, training data and batch size
unchanged, and enables anomaly detection. Step checkpoints are saved every 50
updates, alongside best validation loss and `last.ckpt`. Existing callbacks log
gradient norms/LR and each rank's GPU utilization; local metric plots and saved
callback histories include training/validation loss, gradients and LR. This is
a training check, not a timing calibration.

```bash
swe64_pilot_overrides=(
  'trainer=swe64s_pilot'
  '+distributed=ddp_4gpu_slurm'
  'trainer.max_time=00:00:30:00'
  'trainer.max_steps=200'
  '++hydra.launcher.nodes=1'
  '++hydra.launcher.cpus_per_task=72'
  'hydra.launcher.timeout_min=60'
)

uv run --frozen autocast epd --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/pilot_afcrps" \
  local_experiment=swe64s/afcrps "${swe64_pilot_overrides[@]}"

uv run --frozen autocast ae --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/pilot_autoencoder" \
  local_experiment=swe64s/autoencoder "${swe64_pilot_overrides[@]}"
```

Use fresh directories and remove `--dry-run` only for an approved submission.
The explicit cap/step overrides prevent the distributed/common presets from
restoring their full-run limits. Record the code/lock/dataset identity and Slurm
job ID, and check CUDA/driver visibility inside the allocated compute node, not
by trying GPU training on a login node.

Check finite losses/gradients, nonzero parameter updates, functioning DDP,
memory headroom, LR evolution, at least two validation passes and successful
checkpoint save/reload. Inspect initial versus final validation loss and a few
predictions/reconstructions, without treating a short pilot as evidence of
convergence or requiring calibrated uncertainty already. If the time cap is
reached before enough updates/validation, the pilot is inconclusive, not passed.

The shorter time cap compresses afCRPS's cosine schedule; the AE keeps its
512-epoch horizon. Start both full fits fresh rather than continuing these
diagnostic runs. FM needs its own pilot once a reviewed SWE AE and matching
cache are available; an afCRPS/AE pilot does not validate latent FM.

## Bounded capacity and AE refinement comparison

The separate `afcrps_large_2h` and `afcrps_small_2h` presets use a fresh
two-hour time-cosine fit at LR 2e-4. The large model retains width 568 and
12 blocks (80.85M parameters); the small model uses width 256 and four
blocks (10.69M). Both retain patch 4, eight heads, 1024D global AdaLN noise,
eight training members, zero dropout/weight decay and the original data and
one-input/four-output task. Full validation, half-hour checkpoints and
gradient/LR logging use `trainer=swe64s_short`. The original full-run presets
are unchanged. Equal wall time is a compute-budget comparison, not a pure
capacity ablation: compare curves against updates as well as elapsed time.

`autoencoder_refine` is a separate 30-minute, constant-LR 3e-6 continuation
from a selected best AE checkpoint. It preserves the architecture, optimizer
moments/preconditioners and epoch/step counters. Prepare a new full-state
checkpoint first; simply overriding the config LR would restore the old LR.
Only use trusted checkpoints. The helper refuses existing outputs, checks an
optional source SHA256 and writes source/derived hashes beside the result.
AE checkpoints need the original resolved training config supplied explicitly;
the helper checks its base LR against the checkpoint and records its hash.
Old checkpoint paths and diagnostic histories are removed; the existing timer
reset callback gives the continuation its own budget.

```sh
uv run --frozen python scripts/swe64s/prepare_ae_refinement.py \
  --source /path/to/best-ae.ckpt --target /path/to/new/resume.ckpt \
  --source-config /path/to/original/resolved_autoencoder_config.yaml \
  --expected-sha256 SELECTED_SOURCE_SHA256

uv run --frozen autocast ae --mode slurm --dry-run \
  --workdir "$SWE64S_OUTPUTS/autoencoder_refine" \
  local_experiment=swe64s/autoencoder_refine \
  resume_from_checkpoint=/path/to/new/resume.ckpt \
  +distributed=ddp_4gpu_slurm trainer.max_time=00:00:30:00 \
  ++hydra.launcher.nodes=1 ++hydra.launcher.cpus_per_task=72 \
  hydra.launcher.timeout_min=45
```

For each fresh afCRPS fit, use the same distributed overrides but
`trainer.max_time=00:02:00:00` and `hydra.launcher.timeout_min=135`.
The two 2h15 allocations, one 45-minute AE allocation and a reserved
45-minute matched evaluation allowance total at most six node-hours on
one four-GPU node per allocation. The short evaluation should be labelled
as a bounded preview, use identical test subsets/seeds and raw best-validation
checkpoints, and not be presented as the full test protocol below.

## Common evaluation

`evaluation.yaml` defines the common physical-field metric settings: 16 members,
raw weights, central 90 percent coverage through the existing `MultiCoverage`,
VRMSE and SSR with per-channel/per-lead output. Aggregate coverage-error scores
are distinct from empirical coverage; use the `coverage_0.90` trajectory columns
for the 90 percent coverage plots. The wrapper reuses the saved training config,
not a reconstructed generic architecture.

```sh
# Preview only; neither command is executed without --execute.
uv run --frozen python scripts/swe64s/evaluate.py \
  --run-dir /path/to/afcrps-run \
  --checkpoint /path/to/selected-checkpoint.ckpt \
  --output-root /path/to/new-evaluation
```

For latent FM also pass `--autoencoder-checkpoint /path/to/new-swe64/autoencoder.ckpt`.
The evaluator then restores the raw dataset and normalization from the saved
cache config and follows the existing encode-once path. Outputs use separate
`free_running` and `teacher_forced` directories; existing destinations are
rejected before execution. Teacher forcing supplies truth at each four-frame
block boundary, not at every frame inside a jointly predicted block. The
25-call cap covers 100 saved-time leads, or 25 simulator time units, on all test
trajectories. Review evaluation cost on the target GPU before executing.

The shared metric runner is not yet the complete structural evaluation.
Exports of mean member power, mean member-anomaly power and divergence/vorticity
balance, plus a matching 64x64 conditional-redraw reference, are follow-on
evaluation work. They are not dependencies of the afCRPS/FM fits or common
metric evaluations, but are required before drawing uncertainty-structure
conclusions from those runs. The generic diagnostic port is deferred.

## Environment before deployment

Use the repository's `uv.lock` and CUDA 12.6 source selection; the baseline
does not change dependencies. Check GPU driver/backend compatibility on the
target machine before syncing or launching. Record the installed AutoSim
revision separately from the dataset generator identity. Do not silently
regenerate this dataset or its redraw reference with whichever AutoSim
revision happens to import.

## Implementation sequence

1. Pin the data identity and prepare the afCRPS preset.
2. Add a new SWE autoencoder, matching latent cache and FM presets. Review
   reconstruction error, per-channel spectra and physical balance before FM.
3. Prepare common teacher-forced/free-running evaluation using the existing
   evaluator. Include VRMSE, central 90 percent coverage and SSR by lead time;
   mean member power and mean member-anomaly power remain distinct diagnostics.
4. Prepare a 64x64 conditional simulator-redraw reference and compare anomaly
   spectra and divergence/vorticity balance against it. Do not substitute the
   full state spectrum or the existing 32x32 reference.
5. Review the portable GPU environment and launch commands locally. Push and
   launch only with approval; datasets, caches, checkpoints and media stay
   outside Git.

Longer runs, spectral losses, parameter sweeps, structured-noise controls and
differentiable-physics variants are follow-on comparisons. They should not
change these baseline presets in place.
