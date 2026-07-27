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
| planned_01 batch | mixed | CNS | 8 | timing scripted |
| planned_02 batch | mixed | GS / GPE / AD | 6 | timing + production scripted |
| planned_updates_01 batch | MC dropout | GS / GPE / CNS / AD | 4 | timing + production scripted |
| planned_updates_02 batch | MC dropout MSE + L2 | CNS | 1 | config + timing + production + eval scripted |
| planned_updates_03 batch | FNO architecture | GS / GPE / CNS / AD | 4 | timing + production + eval scripted |
| noise_channels | sweep | CNS | 1 | config + planned |
| mc_dropout (FFN, p=0.1) | comparison | GS / GPE / CNS / AD | 4 | timing + production scripted |
| mc_dropout MSE + L2 (FFN, p=0.1) | comparison | CNS | 1 | ready |
| crps_variants (AlphaFair / Fair / CRPS) | comparison | CNS | 2 new (+baseline) | config + planned |
| fm_vs_diffusion | comparison | CNS | 1 | config + planned |
| arch_unet_fno_vit | comparison | all 4 FNO; CNS U-Net | 4 FNO + 1 U-Net (+ViT baselines) | FNO timing ready |
| model_size | sweep | CNS | 2 active (+2 staged) | in progress |
| vit_mae_pretrain | pretrain | CNS | 1 | staged |
| cached_latent_crps | comparison | CNS | 1 (basis: 2026-04-20) | eval ready |
| cond_global_vs_permute | comparison | CNS | 1 planned rerun (+old 2026-04-18 point) | config ready |
| eval_only/ode_steps | eval-only | GS / GPE / CNS / AD | 4 | ready |
| eval_only/ema | eval-only | GS / GPE / CNS / AD | 4 | ready |

"Done" entries refer to runs already produced by
`slurm_scripts/comparison/` that double as the CNS data point for this
ablation — no new training required, but they should be eval'd through
the same pipeline.

## Planned Batch 01

The current planned CNS batch is centralized in
`submit_planned_01_timing.sh` and `submit_planned_01_large.sh` so the
cross-ablation run list can be submitted consistently after timing. It covers:

| planned run | study folder | implementation |
|---|---|---|
| U-Net m=8 CRPS CNS | `arch_unet_fno_vit` | `crps_unet_azula_80m`, ~81.3M params |
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

## Planned Batch 02

The follow-up batch lives in `submit_planned_02_timing.sh`,
`submit_planned_02_large.sh`,
`submit_planned_02_m4_followup_timing.sh`, and
`submit_planned_02_m4_followup_large.sh` so planned batches can keep
extending without repurposing earlier scripts. It covers:

| planned run | study folder | implementation |
|---|---|---|
| GS m=8 latent CRPS | `cached_latent_crps` | `processor/gray_scott/crps_vit_azula_large` with cached GS latents |
| GPE m=8 latent CRPS | `cached_latent_crps` | `processor/gpe_laser_wake_only/crps_vit_azula_large` with cached GPE latents |
| AD m=8 latent CRPS | `cached_latent_crps` | `processor/advection_diffusion/crps_vit_azula_large` with cached AD latents |
| GS m=4 ViT | `ensemble_size` | canonical GS CRPS ViT plus `n_members=4`, `batch_size=64` |
| GPE m=4 ViT | `ensemble_size` | canonical GPE CRPS ViT plus `n_members=4`, `batch_size=64` |
| AD m=4 ViT | `ensemble_size` | canonical AD CRPS ViT plus `n_members=4`, `batch_size=64` |

The m=4 GPE/AD follow-up follows the same timing-then-production pattern:
`submit_planned_02_m4_followup_timing.sh` first, then
`submit_planned_02_m4_followup_large.sh` after retrieving timing outputs.

## Planned Updates Batch 01

The first post-comparison update batch contains the parameter-matched
four-dataset MC-dropout CRPS ablation. Its orchestration lives in
`submit_planned_updates_01_timing.sh` and
`submit_planned_updates_01_large.sh`; the reusable experiment configs and
design notes remain under `ablations/mc_dropout/`.

## Planned Updates Batch 02

The second post-comparison update batch is a CNS MC-dropout MSE baseline. It
uses the same parameter-matched architecture and `p=0.1` sampler as the CRPS
ablation, plus an explicit processor-local L2 penalty with coefficient `1e-5`.
Evaluation uses 50 stochastic forward passes. GS, GPE and AD configs remain
available but are commented out in the submission scripts. Its orchestration
lives
in `submit_planned_updates_02_timing.sh`,
`submit_planned_updates_02_large.sh`, and
`submit_eval_planned_updates_02.sh`.

## Planned Updates Batch 03

The third post-comparison update batch adds parameter-matched CRPS FNO
architecture runs for all four comparison datasets. Its orchestration lives in
`submit_planned_updates_03_timing.sh` and
`submit_planned_updates_03_large.sh`, with evaluation in
`submit_eval_planned_updates_03.sh`. The experiment config and design notes
remain under `ablations/arch_unet_fno_vit/`.

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
   and paste into `submit_*_large.sh`, or use a large script that derives
   them from matching timing checkpoints.
3. `submit_*_large.sh` — 24h production runs, dry-run first.
4. Eval from the corresponding central `submit_eval_planned*.sh` script or a
   study-local `eval/` submitter when the run set has not been centralized.
