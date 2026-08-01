# CNS seed-43 main-comparison repeat

This pipeline repeats the conditioned Navier--Stokes main comparison on a
new, explicit data draw while holding the published model and training seeds
fixed. Every output path is new and every launcher refuses an existing target.

## Published reference

`outputs/2026-05-06_submission_outputs/SUMMARY_v1.md` identifies this pair:

- ambient CRPS ViT:
  `outputs/2026-04-24/crps_cns64_vit_azula_large_bed4611_c99f534`
- latent flow-matching ViT:
  `outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3`

The FM run used the AE and cache under
`outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8`. The derived configs in this
directory inherit those exact authoritative main-comparison configs rather
than copying their model blocks.

The run names encode the source commits of the published references:

- CRPS: `bed4611609d224bb3497e858ba278d028e7430d2`
- AE: `3a7999b733254d6a9e572644be3c694744c07305`
- FM: `09490dad1093b304a69c0b2d14695887c536e67f`

Those commits are provenance references only. The rerun executes an immutable
snapshot of the checked-out `2026-07-24/updates` branch that was recorded when
stage 0 generated the dataset. The pinned source must contain the multi-GPU
checkpoint teardown fix `f5ee48356934e0e80f9da77c0922edcb8a4cf4b7`.
CRPS pins the callback list from the successful 2026-07-24 training-seed repeat
(`crps_cns64_vit_azula_large_103985e_6360e51`): the published progress/window
schedule plus its overall-Winkler checkpoint. AE pins its published callback
list. FM pins the published three-callback stack through the current
`fm_main_comparison` trainer config, added in commit `103985ef` for the
successful 2026-07-24 repeats.

| Stage | Inherited config | Published schedule |
| --- | --- | --- |
| CRPS | `epd/conditioned_navier_stokes/crps_vit_azula_large` | 473 epochs, 24 h |
| AE | `ae/conditioned_navier_stokes/ae_dc_large` | 512 epochs |
| cache | `cache_latents/conditioned_navier_stokes/cache_latents` | exact final AE checkpoint |
| FM | `processor/conditioned_navier_stokes/fm_vit_large` | 3223 epochs, 24 h |

CRPS retains 8 members with batch size 32 per GPU; FM retains batch size 256
per GPU. Thus both retain the published effective batch of 256 per GPU. The
training seed remains 42 so the only intentional experimental change is the
data draw.

The epoch budgets do not need to be retimed. The completed 2026-07-24 CNS
training-seed repeats used the same 473-epoch CRPS and 3223-epoch FM schedules.
Their CRPS 5%-progress snapshots landed on the same epochs and optimizer steps
as the published run, and their FM snapshots used the same 805-epoch quarter
cadence. AE retains the published fixed 512-epoch schedule.

## Data identity

- new data seed: 43 (historical split seeds 43, 44, and 45)
- splits: 200 train, 20 validation, 20 test
- samples: 321 frames, 64 x 64, 3 channels (`smoke`, `u`, `v`)
- target:
  `/projects/u6eo/autocast/datasets/conditioned_navier_stokes_2d_seed43_20260801`
- published generator commit:
  `bd5ac317c3533a18a52fb07e05abe3a0247b437d`
- published statistics commit:
  `0bc366adf92fac25228da7550f70d58819f37708`

The original dataset recorded `seed: null`. This repeat uses the last AutoSim
commit before the original generation began; its CNS preset matches the
original resolved config byte-for-byte. A later AutoSim version changed split
seed spacing, so using current `main` would not be the same data-generating
procedure. The separate stats commit matches the procedure and timestamp that
created the published `stats.yml`.

No files are changed in `/home/u6eo/ltcx7228.u6eo/autosim`. Stage 0 archives
the two commits above into temporary directories and applies the new seed,
split sizes, output directory, and visualization settings as CLI overrides in
`00_generate_data_interactive.sh`. Dataset checks live in
`validate_dataset.py`; shared paths and commit identities live in `pipeline.sh`.

## Outputs and dependency gates

```text
0 data ──┬──> 1 CRPS
         └──> 2 AE ──> 3 cache ──> 4 FM
```

- data: external dataset target above
- CRPS: `outputs/2026-08-01/main_comparison_cns_seed43/crps_vit_azula_large`
- AE: `outputs/2026-08-01/main_comparison_cns_seed43/ae_dc_large`
- cache: `outputs/2026-08-01/main_comparison_cns_seed43/ae_dc_large/cached_latents`
- FM: `outputs/2026-08-01/main_comparison_cns_seed43/fm_vit_large`

Each script defaults to preview mode. The exact commands that arm a stage are:

```bash
RUN=true ./slurm_scripts/comparison/main_comparison_cns_seed43/00_generate_data_interactive.sh
SUBMIT=true ./slurm_scripts/comparison/main_comparison_cns_seed43/01_submit_crps.sh
SUBMIT=true ./slurm_scripts/comparison/main_comparison_cns_seed43/02_submit_autoencoder.sh
RUN=true ./slurm_scripts/comparison/main_comparison_cns_seed43/03_cache_latents_interactive.sh
SUBMIT=true ./slurm_scripts/comparison/main_comparison_cns_seed43/04_submit_flow_matching.sh
```

Steps 0 and 3 use the short interactive reservation. Steps 1, 2, and 4 are
four-GPU batch jobs submitted through the repository's normal Slurm launcher.
Run them one at a time, reviewing completion before arming the next stage; the
scripts do not queue the full pipeline automatically.

No downstream stage can start until its required files exist and the dataset
validator has written its success marker. Cache generation requires the stable
final `autoencoder.ckpt`; it never falls back to a temporary checkpoint. FM
validates the cached data and AE configs before submission.
