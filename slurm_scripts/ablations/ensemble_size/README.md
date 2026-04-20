# Ensemble size ablation

First-pass CNS defaults focus on `n_members=16` under two batch-size
regimes, but the submit scripts are combo-driven and meant to be
extended. All runs inherit from
`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`;
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

## Datasets

CNS only for the first pass. `DATASETS` in each submit script has the
CNS entry; adding a second dataset means uncommenting the relevant line.

## Files

| file | purpose |
|---|---|
| `submit_ensemble_timing.sh` | 5-epoch timing for the 2 `m=16` combos → `timing.ckpt` per run |
| `submit_ensemble_large.sh`  | 24h production runs for the same 2 combos with per-combo cosine schedule |

## Extending the sweep

Add more lines to `COMBOS` in both submit scripts. Invariants are checked
per regime so bad tuples fail fast before any submission:

- `fixed_bs32`: require `bs_per_gpu=32`; vary `n_members`.
- `eff_bs1024`: require `bs_per_gpu × n_members × 4 GPUs = 1024`.

When adding a combo, also add its key to `COSINE_EPOCHS_BY_COMBO` in
`submit_ensemble_large.sh` after timing results are available.

## Scheduling

Per-combo `cosine_epochs` is populated from `timing.ckpt` via
`uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24`.
Placeholders live in `COSINE_EPOCHS_BY_COMBO` at the top of
`submit_ensemble_large.sh` — replace before submitting the 24h runs.
