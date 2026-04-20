# Ablations

Sensitivity sweeps, comparisons, and ablations that sit on top of the main
4-dataset comparison in `slurm_scripts/comparison/`. "Ablation" is used
loosely here for all three — true ablations (EMA on/off), comparisons
(FM vs diffusion, ViT vs U-Net), and sweeps (ensemble size, noise
channels) — to match how ML papers usually label this section.

Everything is **CNS-only for now**. A second dataset will be added once
the main 4-dataset comparison results are in. Each script uses a
`declare -A DATASETS` mapping so adding a second dataset is a one-line
change per ablation.

## Status table

| ablation | type | datasets | runs | status |
|---|---|---|---|---|
| ensemble_size (m=16, fixed bs=32) | sweep | CNS | 1 | ready |
| ensemble_size (m=16, fixed global eff. bs=1024) | sweep | CNS | 1 | ready |
| noise_channels | sweep | CNS | 1+ | stub |
| crps_variants (AlphaFair / Fair / CRPS) | comparison | CNS | 3 | stub |
| fm_vs_diffusion | comparison | CNS | 1 | stub |
| arch_unet_fno_vit | comparison | CNS | 2 | stub |
| model_size | sweep | CNS | 1 | stub |
| cached_latent_crps | comparison | CNS | 1 (done, 2026-04-19) | stub |
| cond_global_vs_permute | comparison | CNS | 1 (done for CRPS-ViT, 2026-04-18) | stub |
| eval_only/ode_steps | eval-only | FM runs | 0 | stub |
| eval_only/ema | eval-only | EMA ckpts | 0 | stub |

"Done" entries refer to runs already produced by
`slurm_scripts/comparison/` that double as the CNS data point for this
ablation — no new training required, but they should be eval'd through
the same pipeline.

## Design notes

- **Flexible by construction.** Each ablation is a self-contained
  subdirectory. Changing the knob values, swapping to a different
  baseline, or dropping an ablation is a localized edit. Adding a second
  dataset is a one-line change in each submit script's `DATASETS` map.
- **Baselines stay in `local_hydra/local_experiment/{epd,processor}/`.**
  Ablation configs extend those via Hydra `defaults`. When the sweep is
  a one-liner (e.g. ensemble size → `model.n_members` +
  `datamodule.batch_size`), the submit script uses CLI overrides and no
  new config file is created. When the ablation materially changes the
  architecture (model size, arch comparison), each variant gets its own
  yaml under `local_hydra/local_experiment/ablations/<name>/<dataset>/`.
- **Timing first, then 24h schedule.** Same two-step pattern as
  `slurm_scripts/comparison/`: each ablation has a `*_timing.sh` (5-epoch
  run → `timing.ckpt`) and a `*_large.sh` (24h run with cosine epochs
  computed from timing).

## Submission workflow

1. `submit_*_timing.sh` — 5-epoch timing runs, producing `timing.ckpt`.
2. Extract per-combo `cosine_epochs` via
   `uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24`
   and paste into `submit_*_large.sh` (matches `comparison/` flow).
3. `submit_*_large.sh` — 24h production runs, dry-run first.
4. `slurm_scripts/comparison/eval/` — eval with matching `eval.batch_size`.
