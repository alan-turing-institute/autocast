# Rayleigh-Benard comparison submitters

This folder contains the maintained Rayleigh-Benard submitters for the
post-merge `283/` Rayleigh-Benard stack. Some configs referenced here live on
parallel `283/` branches today, so these entrypoints are intended for the final
state after the RB ambient, latent FM, masked FM, diffusion, and latent CRPS
branches have all landed on `main`.

## Config inventory

| Group | Run | Config |
| --- | --- | --- |
| 24h comparison | CRPS ambient, LOLA pixel ViT | `the_well/rayleigh_benard/crps_vit_azula_lola_pixel_ambient` |
| 24h comparison | CRPS ambient | `the_well/rayleigh_benard/crps_vit_azula_large_ambient` |
| 24h comparison | CRPS latent | `the_well/rayleigh_benard/crps_vit_azula_large_latent` |
| 24h comparison | FM latent | `the_well/rayleigh_benard/fm_vit_large` |
| LOLA 4096-step comparison | FM latent | `the_well/rayleigh_benard/fm_vit_large` |
| LOLA 4096-step comparison | FM latent, masked window | `the_well/rayleigh_benard/fm_vit_large_masked_window` |
| LOLA 4096-step comparison | Diffusion latent | `the_well/rayleigh_benard/diffusion_vit_large_lola` |

Additional ambient FM presets are available under
`local_hydra/local_experiment/the_well/rayleigh_benard/` for future ablations,
but are not part of these maintained submitters because they are expected to be
more computationally expensive than the selected 24h comparison matrix.

## Entry points

Use `DRY_RUN_ONLY=true` to only emit SLURM dry runs. By default each submitter
runs one dry-run submission and then the real submission, matching the existing
comparison scripts.

```bash
# Timing probes or final 24h comparison runs.
bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_24h.sh timing all
bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_24h.sh final all

# LOLA-equivalent 4096 epoch x 64 train-batch runs.
bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_lola4096.sh all

# Diffusion eval modes: Euler, AB order 3, and DDPM.
bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_diffusion_eval.sh all
```

Each entrypoint also accepts individual run keys, for example
`submit_24h.sh final crps_latent fm_latent` or
`submit_diffusion_eval.sh ab_o3`.

For diffusion eval, `EVAL_MAX_ROLLOUT_STEPS` controls the requested rollout
windows. The script derives `DATAMODULE_MAX_ROLLOUT_STEPS` from that value and
`EVAL_ROLLOUT_START` so the rollout dataloader exposes enough truth frames for
the requested horizon.
