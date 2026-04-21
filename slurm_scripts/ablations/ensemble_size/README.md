# Ensemble size ablation

First-pass defaults focus on `n_members=16` under two batch-size
regimes. For the current submission pass, the active scripts are pared
down to just three `eff_bs1024` runs on `gray_scott`,
`gpe_laser_only_wake`, and `advection_diffusion`; the CNS entries and
`fixed_bs32` combo are left commented for later reuse. All runs inherit
from the matching per-dataset
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
| `submit_ensemble_timing.sh` | 5-epoch timing for the three active `eff_bs1024` runs (`gray_scott`, `gpe_laser_only_wake`, `advection_diffusion`) → `timing.ckpt` per run |
| `submit_ensemble_large.sh`  | 24h production runs for the same three active runs, using cached or timing-derived cosine schedules |

## Extending the sweep

Add more lines to `COMBOS` in both submit scripts. Invariants are checked
per regime so bad tuples fail fast before any submission:

- `fixed_bs32`: require `bs_per_gpu=32`; vary `n_members`.
- `eff_bs1024`: require `bs_per_gpu × n_members × 4 GPUs = 1024`.

Dataset coverage is controlled separately via `REGIMES_BY_DATASET` in
each submit script, so extending `eff_bs1024` without broadening
`fixed_bs32` is a one-line change per dataset.

## Scheduling

`submit_ensemble_large.sh` first checks `COSINE_EPOCHS_BY_COMBO`. If a
key is missing, it looks for the matching timing run
`outputs/*/crps_<dataset>_<regime>_m<n_members>/timing.ckpt` and derives
`trainer.max_epochs` on the fly with:

`uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02`

That means the added `gray_scott`, `gpe_laser_only_wake`, and
`advection_diffusion` `eff_bs1024` runs become submit-ready as soon as
their timing jobs finish, without another script edit.
