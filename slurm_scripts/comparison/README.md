# Comparison study: CRPS/FM × ambient/latent × 4 datasets

Head-to-head comparison of ensemble-CRPS and flow-matching (FM) in both ambient
and latent space, across 4 datasets.

**Datasets:** `advection_diffusion`, `conditioned_navier_stokes`,
`gpe_laser_wake_only`, `gray_scott`

## Variant layout

| variant | subcommand | Hydra config | scripts |
|---|---|---|---|
| AE (shared) | `autocast ae` | `ae/<dataset>/dc_large.yaml` | `ae/` |
| CRPS ambient | `autocast epd` | `epd/<dataset>/crps_vit_azula_large.yaml` | `epd/submit_crps_*.sh` |
| FM ambient | `autocast epd` | `epd/<dataset>/fm_vit_large.yaml` | `epd/submit_fm_ambient_*.sh` |
| CRPS latent | `autocast processor` | `processor/<dataset>/crps_vit_azula_large.yaml` | `cached_latents/submit_crps_latent_*.sh` |
| FM latent | `autocast processor` | `processor/<dataset>/fm_vit_large.yaml` | `cached_latents/submit_fm_*.sh` |

Hydra configs live at `local_hydra/local_experiment/{ae,cache_latents,epd,processor}/<dataset>/`.
Scripts in `cached_latents/` first cache latents via `submit_cache_latents.sh`,
then run the latent-space processor variants.

## Submission order

1. `ae/submit_ae_timing.sh` — per-epoch timing for AE
2. `ae/submit_ae_large.sh` — full AE training (provides `<ae_run_dir>`)
3. `cached_latents/submit_cache_latents.sh` — cache latents from trained AE
4. All remaining scripts in `epd/` and `cached_latents/` are independent and
   can be submitted in parallel once their dependencies above are complete.

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
