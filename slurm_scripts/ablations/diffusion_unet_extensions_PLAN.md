# Extend the original diffusion and U-Net ablations

Prepared for `/home/u6eo/ltcx7228.u6eo/autocast-02` at commit
`c53d2b2c4314769e78b0cf7a4a01bb50cbeecde9`. No jobs are submitted by this plan
or its YAML manifest. Scope: six new runs on the original AD, GS and GPE
datasets, with training seed 42. The three five-epoch U-Net timing jobs
completed on 2026-09-18 and their measured production budgets are now set.
Diffusion uses its existing dataset-specific timing measurements. No CNS
repeats, data regeneration, new autoencoders, or changes to the training
implementation are included. U-Net production was submitted on 2026-09-18
from commit `138dd8cf`: AD job 6675374, GS job 6675381, GPE job 6675383.

## Reference runs and preserved settings

- Latent diffusion: `diff_cns64_diffusion_vit_0c75022_80967c4`, trained on
  cached AE latents with Karras preconditioning, `VPSchedule`, Euler/50,
  ViT width 704, 12 blocks, 8 attention heads, batch size 256/GPU,
  AdamW at 1e-4, no warmup, four-GPU DDP.
- Ambient U-Net CRPS: `crps_cns64_unet_azula_large_9c98db0_65f8f71`,
  eight training members, batch size 32/GPU, normalized raw fields,
  `permute_concat`/`channels_last`, AlphaFairCRPS, AdamW at 2e-4,
  no warmup, four-GPU DDP. The U-Net uses widths [62,124,248,496],
  three blocks at each scale, FFN factor 1, layer norm, 1024 noise
  channels, no dropout or gradient checkpointing. Preserve periodic=false
  from the original ablation; dataset-dependent channel counts are inferred.
- Both retain one input/four output frames, stride 1, high float32 matmul
  precision, the original data normalization, and their original metrics.
- Use the published April 17 dataset-specific AEs and their existing caches.
  GPE retains the original real/imaginary channel subset [1,2].

## Timing and callback policy

The current default trainer removed validation plotting and added an overall
Winkler checkpoint relative to the April U-Net reference. Do not inherit that
default for this extension. Two explicit trainer configs start from the saved
reference trainers, excluding their dataset-specific max_epochs:

- `local_hydra/trainer/original_unet_ablation.yaml`
- `local_hydra/trainer/original_diffusion_ablation.yaml`

Both preserve 5% optimizer-step snapshots, best val_loss, early coverage
selection through 25%, later coverage selections from 25/50/75%, validation
metric plotting, and EMA decay 0.999. At the user's request, U-Net replaces
the pre0p25 and from0p25 MultiWinkler callbacks with one overall callback
starting at 0% progress. The original pre0p25 did run from the first eligible
validation through 25%, so starting overall at zero avoids losing the first
5% after removing that early tracker. The from0p50/from0p75 callbacks remain.
This removes one tracker in total, limiting added checkpoint I/O, but save
counts depend on metric improvements so identical overhead is not guaranteed.
The timing jobs and subsequent production runs use this same callback list.
It also deliberately changes primary checkpoint selection versus the CNS
reference, which used from0p25. Diffusion preserves its original optional
overall-Winkler and windowed callbacks exactly. Because diffusion
has val_metrics=[], the optional coverage/Winkler callbacks remain dormant.
Do not remove them or add prediction-based validation to that run.

The existing run_training function adds TrainingTimerCallback automatically;
do not add a second timer. It records the full epoch cycle, including
validation and callback overhead. Its implementation and the step-progress
checkpoint implementation match the originals on this checkout.
Existing fixes to collective DDP final checkpoint saving and EMA checkpoint
serialization (deepcopy instead of clone) are retained. The per-step EMA
update, U-Net implementation and optimizer schedule are unchanged; checkpoint
serialization overhead need not be identical to the historical environment.

Progress thresholds are fractions of estimated optimizer steps, NOT fractions
of elapsed wall-clock time. Overall MultiWinkler is eligible from the first
validation checkpoint; the later windows still start at 50% and 75%. Keep
trainer.max_epochs equal to optimizer.cosine_epochs so the cosine schedule,
5% snapshots, and checkpoint-selection windows share the same budget.

Preserve the original budget calculation:

`epochs = floor(24 * 3600 * (1 - 0.02) / mean_epoch_seconds)`

The original operational limits remain trainer.max_time=00:23:59:00 and Slurm
timeout_min=1439. These are the same limit, not a final-save grace interval.
The 2% estimate margin is intended to leave headroom; throughput variation can
still cause early epoch completion or a wall-clock cutoff. Historical timing
never guarantees an exact 24-hour runtime, even with identical configs.

| Dataset | Diffusion measured mean epoch | Diffusion epochs | U-Net measured mean epoch | U-Net production epochs |
| --- | ---: | ---: | ---: | ---: |
| AD | 32.5424 s | 2601 | 131.3 s | 644 |
| GS | 37.6415 s | 2249 | 156.3 s | 541 |
| GPE | 31.6558 s | 2674 | 133.8 s | 632 |

Diffusion budgets are measured: read from the existing April 27 timing
checkpoints and identical to the pinned values in submit_planned_03_large.sh.
The source is `outputs/2026-04-27/timing_planned_03/latent_diffusion_{ad,gs,gpe}/timing.ckpt`.
The mean includes all five recorded TrainingTimerCallback.epoch_times_s values,
matching the original time-epochs calculation (no warmup epoch excluded).
The source checkpoints retain the raw durations. Timing extracts and one-off
validation reports are supporting records kept outside version control; the
YAML manifest and trainer configs define the intended run settings.

U-Net budgets come from the completed five-epoch timing jobs 6673314 (AD),
6673315 (GS), and 6673316 (GPE), all with successful Slurm exits. Their source
checkpoints are
`outputs/2026-09-18/timing_diffusion_unet_extensions/unet_m8_crps_{ad,gs,gpe}/timing.ckpt`.
All five stored epoch durations are averaged, using full precision before
applying the same 24h/2% formula. The table rounds means for display only.
Both trainer.max_epochs and optimizer.cosine_epochs use the resulting count:
644 for AD, 541 for GS, and 632 for GPE. The manifest records each source
checkpoint. Timing specifications are retained for provenance, not for reruns.

The timing workflow preserves the original exceptions: W&B logging and
testing are disabled, max_time is null, the final checkpoint is timing.ckpt,
and the cosine horizon follows the five timing epochs. Validation plotting
to disk, EMA and all checkpoint callbacks stay enabled. Four GPUs, model,
batch size, dataset, validation metrics and precision match production.
The Slurm timing allocation is capped at 240 minutes, as in the old default.
Callback configuration matches production exactly; actual save counts and
cost cannot be made identical by matching windows alone. In particular, 5%
snapshots scale to a much shorter run during timing. This is the original
measurement procedure and should be reported as such, not as proof of exact
production I/O or wall-clock parity.

The original U-Net timing behavior was checked against the actual saved
submit_job, .hydra/config.yaml and encoder_processor_decoder.log under
`outputs/2026-04-25/timing_planned_cns/unet_m8_crps_cns/`. They confirm five
epochs, four GPUs, W&B/testing disabled and the complete callback list. The
snapshot callback resolved to every 124 of 2480 optimizer steps. With one
validation per epoch, pre0p25 covered the first validation (20% progress)
and from0p25 was eligible from the second (40% progress). The five original
epoch times averaged 138.3811 seconds, yielding the original 611-epoch budget.

## Ordered implementation and validation plan

1. **Pin the scientific settings and callback policies.** Prepare the two
   trainer files above from the saved resolved reference configs. Verify
   diffusion callback-list equality and that U-Net differs only by the
   requested replacement of two MultiWinkler trackers with the overall one.
2. **Prepare dataset configs.** Reuse the existing
   `local_hydra/local_experiment/processor/{advection_diffusion,gray_scott,gpe_laser_wake_only}/diffusion_vit_large.yaml`.
   Add `crps_unet_azula_80m.yaml` under the corresponding three
   `local_hydra/local_experiment/ablations/arch_unet_fno_vit/` directories.
   Copy the CNS U-Net settings, change dataset/name, and pin its original
   trainer policy. Infer channel counts with the existing setup code.
3. **Record production and timing override lists.** The planning manifest
   `diffusion_unet_extensions.yaml` pins each config, original data/cache path,
   trainer policy, seed, measured horizons, timeout, logging and output
   behavior. U-Net horizons use its completed timing checkpoints;
   timing_runs retains the corresponding five-epoch specifications.
   CNS is excluded. Preserve the original caches and run
   the existing validate_cached_latents_against_ae helper for AD/GS/GPE.
4. **Validate without fitting.** Compose all six Hydra configs with their
   override lists, compare model/optimizer/callback fields to the CNS
   reference compositions, instantiate callbacks to check constructor
   compatibility, and check data, normalization and cache paths. Confirm
   four GPUs/tasks, batch sizes, validation metrics and equal measured/timing
   horizons. Require complete resolution of all U-Net production configs.
   Compare each against its saved timing config, allowing only the production
   horizon, max_time, logging, and output changes.
   This does not call Trainer.fit, time-epochs training, sbatch or srun.
5. **Timing procedure (completed).** The forced dry-run command
   `bash slurm_scripts/ablations/preview_unet_extension_timings.sh`
   preceded the three five-epoch timing jobs. Read their saved
   timers with `uv run --frozen --no-sync autocast time-epochs --from-checkpoint
   <timing.ckpt> -b 24 -m 0.02`. Their measured counts are now recorded in both
   production horizons. No new timing jobs are needed for these configs.
6. **Launch production in two stages.** The three diffusion and three U-Net
   production jobs were submitted on 2026-09-18. The committed
   `submit_unet_extensions.py` launcher reads the U-Net override lists from
   the manifest and validates them against the saved timing configs. It
   previews by default; `--submit` explicitly enables submission. Existing
   U-Net output directories in the requested group block duplicate submission.
   Use `outputs/YYYY-MM-DD/diffusion_unet_extensions/` for production
   and `outputs/YYYY-MM-DD/timing_diffusion_unet_extensions/` for timing.
   Do not execute the old batch launchers for these extensions:
   they loop over preview and submission and would also include CNS.
7. **After training, preserve original checkpoint/eval choices.** Diffusion
   uses processor.ckpt with its matching published AE, encode_once, ten eval
   members, Euler/50, and the original rollout/metric windows. U-Net selects
   best-multiwinkler-overall as requested and uses ambient evaluation with
   ten members. This selection differs from the original CNS from0p25 rule;
   keep that explicit in comparisons. Use raw weights, as in the references.
   Record observed TrainingTimerCallback metadata with final results.

For a composition-only inspection, use the manifest override list with:

```bash
uv run --frozen --no-sync python -m autocast.scripts.train.processor \
  --cfg job --resolve <diffusion overrides from manifest>
uv run --frozen --no-sync python -m autocast.scripts.train.encoder_processor_decoder \
  --cfg job --resolve <U-Net overrides from manifest>
```

Hydra's --cfg exits before the training function. The placeholders above are
explanatory; the manifest contains the override strings. Full U-Net production
resolution now succeeds with the measured horizons. No changes
to shared model, timer, scheduler or callback code are required.

## Prepared-state checks

The original published AE/cache settings and train/valid/test cache directories
were checked for all three datasets. Timing metadata was read from saved
checkpoints using CPU mmap. Initial preparation started no runs; the later
authorized launch submitted the three U-Net timing and three diffusion jobs.
All six configs were composed and their callbacks instantiated. Diffusion
budgets were recomputed from the original five-epoch timing checkpoints.
The original timing configs use the same effective snapshot trigger: their
omitted every_n_epochs defaults to zero when a fractional step cadence is set.
The new trainer files make that value explicit, as in the saved production
configs. Model and optimizer differences are limited to inferred data shapes
and the dataset-specific measured horizons; U-Net retains the documented
MultiWinkler selection change.

After timing completion, each U-Net production config was fully resolved and
compared against its actual timing config. Model, datamodule, validation
metrics, callback policy, batch size, seed and precision are identical.
Only the intended production horizon, time cap, logging and output settings
differ. All three production commands passed a dry-run preview before the
authorized submission. The original one-off wrapper and submission log are
preserved under `outputs/2026-09-18/diffusion_unet_extensions_review/`.

## Reproducing the U-Net submission

Run from the checkout root with the saved timing records accessible:

```bash
uv run --frozen --no-sync python slurm_scripts/ablations/submit_unet_extensions.py
```

This prints the current source commit and previews all three U-Net production
commands using the committed manifest. For an intentional new launch, append
`--submit`; use `--run-group YYYY-MM-DD/diffusion_unet_extensions_repeat` to
select a fresh output group. Submission requires a clean checkout and checks
all three destinations before submitting any job. The default output group
is `YYYY-MM-DD/diffusion_unet_extensions`, using the launch date. The launcher
does not submit diffusion or timing jobs.

The existing workflow CLI writes `submit_job_*.sh` into each run directory
and submits it with `sbatch`. Each generated script invokes the training
module via `srun`; the run directory also keeps its Slurm logs and Hydra
configuration snapshots. The three original production jobs already have
these records, independently of this reusable launcher.

## Reproducing the diffusion submission

The diffusion counterpart reads the same manifest and the original 2026-04-27
timing records; the manifest records each timing checkpoint explicitly.
From the checkout root, preview with:

```bash
uv run --frozen --no-sync python slurm_scripts/ablations/submit_diffusion_extensions.py
```

It retains the AD/GS/GPE horizons of 2601/2249/2674 epochs and validates model,
data, optimizer, callbacks, seed and precision against those timing configs.
Comparison resolves the old checkout's output symlink to the shared cache
directory and accounts for the original implicit `every_n_epochs=0` snapshot
default. Neither normalization changes the submitted training config.
All three commands use four GPUs, four tasks and a 23h59m Slurm limit.

The default group is `YYYY-MM-DD/diffusion_unet_extensions`. For an intentional
repeat, select a fresh `--run-group` and append `--submit`. Submission requires
a clean checkout, validates all three destinations and previews every command
before the first submission. This launcher selects diffusion production only.
The original diffusion jobs 6673317, 6673318 and 6673319 were submitted from
`d80b5b3a`; adding this reusable launcher does not restart those jobs. Their
exact commands and submission scripts remain in the original output folders
and the review records cited above.

The post-fit NCCL fix from upstream PR #386 is present as backport
`f5ee48356934e0e80f9da77c0922edcb8a4cf4b7` on this branch. Collective saves
run on every rank and checkpoint decisions are broadcast. The PR merge commit
itself is not an ancestor, so ancestry alone would miss this fix.
