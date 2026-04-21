# Comparison study: CRPS/FM × ambient/latent × 4 datasets

Head-to-head comparison of ensemble-CRPS and flow-matching (FM) in both ambient
and latent space, across 4 datasets.

**Datasets:** `advection_diffusion`, `conditioned_navier_stokes`,
`gpe_laser_wake_only`, `gray_scott`

## Variant layout

| variant | subcommand | Hydra config | scripts |
|---|---|---|---|
| AE (shared) | `autocast ae` | `ae/<dataset>/dc_large.yaml` | `ae/` |
| CRPS ambient (baseline concat) | `autocast epd` | `epd/<dataset>/crps_vit_azula_large.yaml` | `epd/submit_crps_*.sh` |
| CRPS via AE latent core (primary) | `autocast epd` | `epd/<dataset>/crps_vit_azula_large_ae_ambient.yaml` | `epd/submit_crps_ae_ambient_*.sh` |
| FM ambient | `autocast epd` | `epd/<dataset>/fm_vit_large.yaml` | `epd/submit_fm_ambient_*.sh` |
| CRPS latent (cached, ablation) | `autocast processor` | `processor/<dataset>/crps_vit_azula_large.yaml` | `cached_latents/submit_crps_latent_*.sh` |
| FM latent (cached, primary FM latent) | `autocast processor` | `processor/<dataset>/fm_vit_large.yaml` | `cached_latents/submit_fm_*.sh` |

Hydra configs live at `local_hydra/local_experiment/{ae,cache_latents,epd,processor}/<dataset>/`.
Scripts in `cached_latents/` first cache latents via `submit_cache_latents.sh`,
then run the latent-space processor variants.
For cache generation, `local_hydra/local_experiment/cache_latents/...` remains
the source of truth; scripts fail fast if its `datamodule.use_normalization`
does not match `<ae_run_dir>/resolved_autoencoder_config.yaml`.
All latent submit scripts now fail fast if
`<ae_run_dir>/cached_latents/autoencoder_config.yaml` does not match
`<ae_run_dir>/resolved_autoencoder_config.yaml` for critical datamodule fields
(`data_path`, `n_steps_input`, `n_steps_output`, `stride`, `use_normalization`,
`normalization_path`).

## Submission order

1. `ae/submit_ae_timing.sh` — per-epoch timing for AE
2. `ae/submit_ae_large.sh` — full AE training (provides `<ae_run_dir>`)
3. `cached_latents/submit_cache_latents.sh` — cache latents from trained AE
4. Primary runs: `epd/submit_crps_ae_ambient_*.sh`,
   `epd/submit_fm_ambient_*.sh`, and `cached_latents/submit_fm_*.sh`.
5. `cached_latents/submit_crps_latent_*.sh` is kept as an ablation.

## Model-size matrix (~80M params, DiT-aligned)

All 4 processor variants target ~80M trainable parameters (AE params excluded)
with DiT-canonical proportions: depth=12, heads=8, head_dim≈64. Width is the
only dial that differs between CRPS and FM (different backbone APIs).

| variant | backbone | hidden | depth | heads | patch | tokens | ~params |
|---|---|---|---|---|---|---|---|
| CRPS ambient | `vit_azula_large` | 568 | 12 | 8 | 4 | 16×16=256 | 80.75M |
| FM ambient | `vit` (flow-matching) | 704 | 12 | 8 | 4 | 16×16=256 | 80.04M |
| CRPS latent | `vit_azula_large` | 568 | 12 | 8 | 1 | 8×8=64 | 80.72M |
| FM latent | `vit` (flow-matching) | 704 | 12 | 8 | 1 | 8×8=64 | 79.91M |

The CNS-only ~160M follow-up sweep that scales these ambient baselines lives in
`slurm_scripts/ablations/model_size/README.md`.

## Effective-batch parity (FM vs CRPS)

`ProcessorModelEnsemble` / `EncoderProcessorDecoderEnsemble` repeat the batch
by `n_members` internally before the forward pass, so the CRPS *effective* batch
is `bs × n_members`. FM/diffusion must be matched on this effective count:

```
bs_fm = bs_crps × n_members   →   256 = 32 × 8
```

All configs use `bs_crps=32`, `n_members=8`, `bs_fm=256` **per GPU** (4 GPUs →
global batches of 128 and 1024 respectively). If an FM run OOMs, step down
(256 → 128 → 64 per GPU) and record the reason in the script.

## Critical: latent patch_size

`vit_azula_large` defaults to `patch_size=4`. On an 8×8 latent this produces
only 2×2=4 tokens — nearly no spatial structure. The
`processor/<dataset>/crps_vit_azula_large.yaml` configs explicitly override
`patch_size: 1` (64 tokens). The FM `vit` backbone already defaults to
`patch_size=1`, so no override is needed for FM-latent.

## Conditioning mechanism

The four variants pass conditioning (e.g. `constant_scalars`) through two
different paths, and this is intentional — not an oversight:

| variant | encoder | global_cond on backbone | conditioning path |
|---|---|---|---|
| CRPS ambient | `permute_concat` (`with_constants: true`) | **off** (processor `include_global_cond: false`) | spatial channels |
| FM ambient | `identity` | **on** (backbone default) | AdaLN modulation |
| CRPS latent | — (cached latents) | **on** (explicit override) | AdaLN modulation |
| FM latent | — (cached latents) | **on** (backbone default + auto-detect) | AdaLN modulation |

Setup auto-fills `global_cond_channels` from `encoder.encode_cond(batch)` /
`batch.global_cond`. For CRPS ambient, `AzulaViTProcessor.include_global_cond`
is hard-coded false in `vit_azula_large.yaml`, so the auto-detected value is
ignored — conditioning flows only through `permute_concat`'s spatial
concatenation.

Planned ablation (not in this submission round): CRPS ambient with
`identity` encoder + `include_global_cond: true` to match FM ambient's
conditioning path exactly and isolate the encoder effect.

## Cosine schedule (per-dataset, 24h budget)

Each `(variant, dataset)` pair gets its own `COSINE_EPOCHS` so every run
fills its 24h wall-clock budget — no model is held back by another
dataset's slower epoch times. This lets us claim each (method, dataset)
result was given its best shot within the fixed 24h budget, which is the
load-bearing constraint for the CRPS vs FM comparison. Values are
extracted per-dataset from `*_timing.sh` via
`uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24`
and live in `COSINE_EPOCHS_BY_DATASET` at the top of each `*_large.sh`.

Ambient (timing 2026-04-18; CRPS from `timing_efficient_crps/`, FM from `timing/`):

| variant | gray_scott | gpe_laser_only_wake | cond_navier_stokes | advection_diffusion |
|---|---|---|---|---|
| CRPS ambient (permute_concat) | **399** (212.0 s/ep) | 477 (177.2) | 473 (178.8) | 478 (177.0) |
| CRPS-via-AE ambient (EPD)     | **49**  (1724.6 s/ep) | 85  (991.0) | 85  (985.0) | 58  (1436.9) |
| FM ambient                    | **2631** (32.2 s/ep)  | 3171 (26.7) | 2982 (28.4) | 3264 (25.9) |

Latent (timing 2026-04-18, FM only — full 4-dataset CRPS-latent timing pending):

| variant | gray_scott | gpe_laser_only_wake | cond_navier_stokes | advection_diffusion |
|---|---|---|---|---|
| FM latent (cached) | **2830** (29.9 s/ep) | 3411 (24.8) | 3223 (26.3) | 3314 (25.6) |
| CRPS latent (cached) | 1080 (placeholder) | 1080 | 1080 | 1080 |

CNS-only ablations (timing 2026-04-18):

| variant | conditioned_navier_stokes |
|---|---|
| CRPS ambient (identity + global_cond AdaLN) | 469 (180.3 s/ep) |
| CRPS latent (cached, ablation)              | 345 (245.0 s/ep) |

All `s/ep` values are the mean across the n=5 epoch durations recorded by
`TrainingTimerCallback` in `timing.ckpt` (saved after `on_train_end`, so
it captures the final epoch — unlike `last.ckpt`, which is saved during
`on_train_epoch_end` and only holds the first 4 durations).

Each script saves quarter-schedule checkpoints (every `cosine_epochs / 4`)
plus `last.ckpt` at train-end (guaranteed final state). Quarter boundaries
differ per dataset, which is fine — within-dataset curves are what we
compare, and absolute losses are only compared within a dataset.
