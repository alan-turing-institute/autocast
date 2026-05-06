# Ensemble size ablation

First-pass defaults focus on `n_members=16` under two batch-size
regimes. The planned CNS batch also includes a compute-matched
`n_members=4` ViT point via the root-level planned submitter:
`model.n_members=4` and `datamodule.batch_size=64`, preserving the baseline
effective per-GPU batch of 256.

For the current production submission pass,
`submit_ensemble_large.sh` is pared down to just three `eff_bs1024` runs
on `gray_scott`, `gpe_laser_only_wake`, and `advection_diffusion`; the
CNS production entries and `fixed_bs32` combo are left commented for
later reuse. Timing-derived schedules are still cached for both CNS
regimes. All runs inherit from the matching per-dataset
`local_hydra/local_experiment/epd/<dataset>/crps_vit_azula_large.yaml`;
the ablation is a pure CLI override on `model.n_members` +
`datamodule.batch_size`, so no new experiment configs are needed.

## Knob map

Main baseline is `bs_crps=32 × n_members=8 × 4 GPUs = 1024 global
effective` (i.e. `256 effective per-GPU`).

### Fixed batch size = 32/GPU (same as baseline)

Keep `datamodule.batch_size=32` and set `n_members=16`.
This doubles effective batch vs baseline.

| n_members | bs_per_gpu | effective per-GPU | effective global |
|---:|---:|---:|---:|
| 16 | 32 | 512 | 2048 |

### Fixed global effective batch = 1024 (matches baseline compute budget)

Keep `bs_crps × n_members × 4 GPUs = 1024`. With `n_members=16`,
`bs_per_gpu=16`.

| n_members | bs_per_gpu | effective per-GPU | effective global |
|---:|---:|---:|---:|
| 4 | 64 | 256 | 1024 |
| 16 | 16 | 256 | 1024 |

## Dataset coverage

| dataset | `fixed_bs32` | `eff_bs1024` |
|---|---:|---:|
| `conditioned_navier_stokes` | yes | yes |
| `gray_scott` | no | yes |
| `gpe_laser_only_wake` | no | yes |
| `advection_diffusion` | no | yes |

This keeps the original CNS pilot in reserve while the active submit
scripts target only the three compute-matched (`1024` effective global
batch) CRPS ablations on the other comparison datasets.

## Files

| file | purpose |
|---|---|
| `submit_ensemble_timing.sh` | 5-epoch timing for the current `eff_bs1024` timing set (`conditioned_navier_stokes`, `gray_scott`, `gpe_laser_only_wake`, `advection_diffusion`) -> `timing.ckpt` per run |
| `submit_ensemble_large.sh`  | 24h production runs for the same three active runs, using cached or timing-derived cosine schedules |
| `eval/submit_eval_crps_ambient.sh` | ambient eval for the current `m=16` CRPS run set (CNS `fixed_bs32` pilot plus all available `eff_bs1024` runs), with conservative `eval.batch_size=4` and explicit `eval.n_members=10` to match the comparison-study eval regime |
| `eval_0p50/submit_eval_crps_ambient.sh` | ambient eval for each run's `snapshot-0p50-*.ckpt` progress checkpoint, falling back to legacy quarter checkpoints |
| `eval_0p75/submit_eval_crps_ambient.sh` | ambient eval for each run's `snapshot-0p75-*.ckpt` progress checkpoint, falling back to legacy quarter checkpoints |

## Extending the sweep

Add more lines to `COMBOS` in both submit scripts. Invariants are checked
per regime so bad tuples fail fast before any submission:

- `fixed_bs32`: require `bs_per_gpu=32`; vary `n_members`.
- `eff_bs1024`: require `bs_per_gpu × n_members × 4 GPUs = 1024`.

Dataset coverage is controlled separately via `REGIMES_BY_DATASET` in
each submit script, so extending `eff_bs1024` without broadening
`fixed_bs32` is a one-line change per dataset.

## Eval placement

Ensemble-size eval now lives under `slurm_scripts/ablations/ensemble_size/`
rather than `slurm_scripts/comparison/eval/`. The reason is organizational:
the run set is still partly ablation-only (`fixed_bs32`) even though the
`eff_bs1024` subset may later graduate into the main comparison baseline.

We keep two sibling eval directories here:

- `eval/` for the standard final-checkpoint evals.
- `eval_0p50/` and `eval_0p75/` for 50% and 75% progress-checkpoint
  evals, so those
  partial-schedule outputs do not mix with the canonical final-checkpoint
  metrics and videos.

If the `eff_bs1024` subset is later promoted, move the promoted run dirs into a
comparison-level eval script and leave only the genuinely ablation-only runs
here.

## Scheduling

`submit_ensemble_large.sh` first checks `COSINE_EPOCHS_BY_COMBO`. If a
key is missing, it looks for the matching timing run
`outputs/*/crps_<dataset>_<regime>_m<n_members>/timing.ckpt` and derives
`trainer.max_epochs` on the fly with:

`uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02`

The current CRPS `m=16` timing-derived schedules are cached in
`submit_ensemble_large.sh`:

| dataset | regime | seconds/epoch | `trainer.max_epochs` |
|---|---|---:|---:|
| `conditioned_navier_stokes` | `fixed_bs32` | 334.6 | 253 |
| `conditioned_navier_stokes` | `eff_bs1024` | 340.0 | 249 |
| `advection_diffusion` | `eff_bs1024` | 346.6 | 244 |
| `gpe_laser_only_wake` | `eff_bs1024` | 345.8 | 244 |
| `gray_scott` | `eff_bs1024` | 414.1 | 204 |

Those values come from the 2026-04-20/21 timing runs with a 24h budget and 2%
margin. The separate `timing_model_size` CNS results belong to the model-size
ablation and do not change this ensemble-size CRPS schedule.
