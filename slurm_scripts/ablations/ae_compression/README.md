# AE compression ablation (CNS-only)

CNS autoencoder trained at 8×8×8 latent (24× compression) for comparison
with the main-study AE at 16×16×8 (4×). See
`slurm_scripts/comparison/README.md` for the main-study design.

## Configuration

| | main study | this ablation |
|---|---|---|
| latent | 16×16×8 | 8×8×8 |
| compression | 4× | 24× |
| encoder `hid_channels` | `[128,256,512]` | `[64,128,256,512]` |
| encoder `hid_blocks` | `[3,3,3]` | `[3,3,3,3]` |
| latent channels | 8 | 8 |
| trainable params | ~49M (24.5M enc + 24.5M dec) | ~50M (+2.1%) |
| schedule | `COSINE_EPOCHS=512` | `COSINE_EPOCHS=512` |

The DC encoder downsamples between adjacent levels, so N levels gives N-1
downsamples (3 → 64 → 16; 4 → 64 → 8).

## Files

| file | purpose |
|---|---|
| `local_hydra/local_experiment/ablations/ae_compression/conditioned_navier_stokes/ae_dc_large_f8.yaml` | overrides on top of the baseline AE config |
| `submit_ae_compression_large.sh` | 24h production run, `COSINE_EPOCHS=512` |
| `submit_ae_compression_timing.sh` | 5-epoch timing run, fallback if 512 wall-clocks out |

## Workflow

1. `bash submit_ae_compression_large.sh`
2. If the run hits the 24h wall: `bash submit_ae_compression_timing.sh`,
   then extract a fitted value via
   `uv run autocast time-epochs --from-checkpoint <timing_path>/timing.ckpt -b 24`
   and re-run with that `COSINE_EPOCHS`.
