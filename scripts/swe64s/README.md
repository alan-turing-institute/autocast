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

# Compose only: no dataset load or model fit. The one-epoch value is for inspection.
uv run --frozen train_encoder_processor_decoder \
  local_experiment=swe64s/afcrps trainer.max_epochs=1 --cfg job

uv run --frozen train_autoencoder \
  local_experiment=swe64s/autoencoder trainer.max_epochs=1 --cfg job

uv run --frozen cache_latents \
  local_experiment=swe64s/cache_latents \
  autoencoder_checkpoint=/path/to/new-swe64/autoencoder.ckpt \
  cache_latents.output_dir=/path/to/new-empty-cache --cfg job

uv run --frozen python -m autocast.scripts.train.processor \
  local_experiment=swe64s/flow_matching \
  datamodule.data_path=/path/to/new-cache trainer.max_epochs=1 --cfg job
```

The real epoch budget is deliberately required (`trainer.max_epochs=???`).
Choose it after target-GPU timing and review; the cosine schedule follows the
explicit epoch budget. Learning rates are 2e-4 for afCRPS and 1e-4 for FM.
Record batch/device count, updates, wall time, overrides, code/lock hashes,
dataset checksums and the selected checkpoint hash with each run. Evaluate raw
weights for the main comparison, even if the inherited callback stores EMA.

For a real fit, remove `--cfg job` and replace the inspection budget with the
reviewed one. Fit and review the AE before caching and launching FM. Check
reconstruction error and each channel's spectrum, plus divergence/vorticity
balance: decoder artifacts must not be mistaken for FM uncertainty artifacts.
The cache must retain the raw split counts and `(321,16,16,8)` trajectory shape.
The existing cacher writes into its output directory, so use a fresh directory;
do not reuse a completed cache destination.

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

The shared metric runner is not yet the complete structural evaluation. Before
launch, add exports of mean member power, mean member-anomaly power and
divergence/vorticity balance using the existing SWE helpers, with the same
channels, leads and checkpoints. Prepare the matching 64x64 conditional-redraw
reference separately. These remain launch gates, not completed measurements.

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
