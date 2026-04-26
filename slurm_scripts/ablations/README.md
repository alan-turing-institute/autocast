# Ablations

Sensitivity sweeps, comparisons, and ablations that sit on top of the main
4-dataset comparison in `slurm_scripts/comparison/`. "Ablation" is used
loosely here for all three — true ablations (EMA on/off), comparisons
(FM vs diffusion, ViT vs U-Net), and sweeps (ensemble size, noise
channels) — to match how ML papers usually label this section.

Most ablations are still **CNS-only for now**. The current exception is
`ensemble_size` under the `eff_bs1024` regime, which now extends to the
other three main comparison datasets (`gray_scott`,
`gpe_laser_only_wake`, `advection_diffusion`) in addition to CNS. Each
script keeps dataset coverage local so widening an ablation remains a
small edit.

## Status table

| ablation | type | datasets | runs | status |
|---|---|---|---|---|
| ensemble_size (m=16, fixed bs=32) | sweep | CNS | 1 | ready |
| ensemble_size (m=16, fixed global eff. bs=1024) | sweep | GS / GPE / CNS / AD | 4 | timing ready |
| planned_cns batch | mixed | CNS | 8 | timing scripted |
| noise_channels | sweep | CNS | 1 | config + planned |
| crps_variants (AlphaFair / Fair / CRPS) | comparison | CNS | 2 new (+baseline) | config + planned |
| fm_vs_diffusion | comparison | CNS | 1 | config + planned |
| arch_unet_fno_vit | comparison | CNS | 1 U-Net (+ViT baseline) | config + planned |
| model_size | sweep | CNS | 2 active (+2 staged) | in progress |
| vit_mae_pretrain | pretrain | CNS | 1 | staged |
| cached_latent_crps | comparison | CNS | 1 (basis: 2026-04-20) | eval ready |
| cond_global_vs_permute | comparison | CNS | 1 planned rerun (+old 2026-04-18 point) | config ready |
| eval_only/ode_steps | eval-only | FM runs | 0 | stub |
| eval_only/ema | eval-only | EMA ckpts | 0 | stub |

"Done" entries refer to runs already produced by
`slurm_scripts/comparison/` that double as the CNS data point for this
ablation — no new training required, but they should be eval'd through
the same pipeline.

## Planned CNS batch

The current planned CNS batch is centralized in
`submit_planned_cns_timing.sh` and `submit_planned_cns_large.sh` so the
cross-ablation run list can be submitted consistently after timing. It covers:

| planned run | study folder | implementation |
|---|---|---|
| U-Net m=8 CRPS CNS | `arch_unet_fno_vit` | `crps_unet_azula_80m`, ~80.9M params |
| Diffusion CNS | `fm_vs_diffusion` | diffusion processor with the FM 704/12/8 ViT backbone |
| CNS m=8 fair CRPS | `crps_variants` | FairCRPS loss on the 80M CRPS ViT |
| CNS m=8 CRPS | `crps_variants` | plain CRPS loss on the 80M CRPS ViT |
| CNS ViT noise channels=256 | `noise_channels` | CRPS ViT with `n_noise_channels=256`, `hidden_dim=704` (~79.9M params) |
| CNS m=4 ViT | `ensemble_size` | canonical CRPS ViT plus `n_members=4`, `batch_size=64` |
| CNS m=8 latent CRPS | `cached_latent_crps` | 2026-04-20 cached-latent CRPS basis |
| CNS m=8 CRPS ViT global cond | `cond_global_vs_permute` | identity encoder + `include_global_cond=true` |

Use the 2026-04-24 CRPS ambient runs for current CRPS comparison numbers and
the 2026-04-20 `diff_*` cached-latent runs as the FM/diff basis. The comparison
eval scripts have those dates wired in.

## Design notes

- **Flexible by construction.** Each ablation is a self-contained
  subdirectory. Changing the knob values, swapping to a different
  baseline, or dropping an ablation is a localized edit. Dataset
  coverage lives inside each ablation's submit scripts, so extending one
  ablation does not spill into the others.
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
4. Eval from the script local to the study:
   `slurm_scripts/comparison/eval/` for the canonical comparison suite, and
   `slurm_scripts/ablations/<name>/eval/` for ablation-only run sets that have
   not been promoted into the main comparison yet.
