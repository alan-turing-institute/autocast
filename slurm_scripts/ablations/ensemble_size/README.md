# Ensemble size ablation

Sweep `n_members` for the CRPS ambient baseline under two batch-size
regimes. All runs inherit from
`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`;
the ablation is a pure CLI override on `model.n_members` +
`datamodule.batch_size`, so no new experiment configs are needed.

## Knob map

Main baseline is `bs_crps=32 × n_members=8 × 4 GPUs = 1024 global
effective` (i.e. `256 effective per-GPU`).

### Fixed global effective batch = 1024 (matches main compute budget)

Keep `bs_crps × n_members × 4 GPUs = 1024`. Sweeps n_members while
holding per-step total compute constant.

| n_members | bs_per_gpu | effective per-GPU | effective global |
|---:|---:|---:|---:|
| 4  | 64 | 256 | 1024 |
| 16 | 16 | 256 | 1024 |
| 32 | 8  | 256 | 1024 |

(n_members=8, bs=32 is already the main run — not rerun here.)

### Fixed per-GPU effective batch = 128 (smaller, faster budget)

Keep `bs_crps × n_members = 128 per GPU` (effective after member
expansion). Sweeps n_members while holding per-GPU throughput constant.

| n_members | bs_per_gpu | effective per-GPU | effective global |
|---:|---:|---:|---:|
| 4  | 32 | 128 | 512 |
| 16 | 8  | 128 | 512 |

## Datasets

CNS only for the first pass. `DATASETS` in each submit script has the
CNS entry; adding a second dataset means uncommenting the relevant line.

## Files

| file | purpose |
|---|---|
| `submit_ensemble_timing.sh` | 5-epoch timing across all 5 combos → `timing.ckpt` per run |
| `submit_ensemble_large.sh`  | 24h production runs with per-combo cosine schedule |

## Scheduling

Per-combo `cosine_epochs` is populated from `timing.ckpt` via
`uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24`.
Placeholders live in `COSINE_EPOCHS_BY_COMBO` at the top of
`submit_ensemble_large.sh` — replace before submitting the 24h runs.
