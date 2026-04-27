# AE compression ablation (CNS-only)

CNS autoencoder trained at 8×8×8 latent (24× compression) for comparison
with the main-study AE at 16×16×8 (4×). See
`slurm_scripts/comparison/README.md` for the main-study design.

## Configuration

The base ablation matches the main-study AE param budget; the 100M and
150M variants form a capacity sweep at fixed 24× compression to test
whether the f8 en/decoder is under-provisioned at the harder bottleneck.

| | main study | f8 base | f8 100M | f8 150M |
|---|---|---|---|---|
| latent | 16×16×8 | 8×8×8 | 8×8×8 | 8×8×8 |
| compression | 4× | 24× | 24× | 24× |
| encoder `hid_channels` | `[128,256,512]` | `[64,128,256,512]` | `[90,180,360,720]` | `[112,224,448,896]` |
| encoder `hid_blocks` | `[3,3,3]` | `[3,3,3,3]` | `[3,3,3,3]` | `[3,3,3,3]` |
| latent channels | 8 | 8 | 8 | 8 |
| trainable params | ~49M | ~50M | ~99M (1.98×) | ~153M (3.06×) |
| schedule | `COSINE_EPOCHS=512` | `COSINE_EPOCHS=512` | `COSINE_EPOCHS=256` | `COSINE_EPOCHS=256` |

Widths follow a strict 1:2:4:8 ratio across all variants; the 100M and
150M widths are sized to land at their target param counts. Both 100M and
150M use the same `COSINE_EPOCHS=256` for an apples-to-apples comparison
on a shared schedule. FLOPs/epoch scale ~linearly with params at fixed
latent grid, so the 100M run (~14h linear) sits comfortably within the
24h wall, while the 150M run (~21h linear) is expected to wall before
the schedule fully completes — checkpoint callbacks preserve the best
state.

The DC encoder downsamples between adjacent levels, so N levels gives N-1
downsamples (3 → 64 → 16; 4 → 64 → 8).

## Files

| file | purpose |
|---|---|
| `local_hydra/local_experiment/ablations/ae_compression/conditioned_navier_stokes/ae_dc_large_f8.yaml` | base ablation (50M, 4-level f8) |
| `local_hydra/local_experiment/ablations/ae_compression/conditioned_navier_stokes/ae_dc_large_f8_100m.yaml` | widths `[90,180,360,720]`, ~99M |
| `local_hydra/local_experiment/ablations/ae_compression/conditioned_navier_stokes/ae_dc_large_f8_150m.yaml` | widths `[112,224,448,896]`, ~153M |
| `submit_ae_compression_large.sh` | base 50M run, `COSINE_EPOCHS=512` |
| `submit_ae_compression_100m.sh` | 100M run, `COSINE_EPOCHS=256` |
| `submit_ae_compression_150m.sh` | 150M run, `COSINE_EPOCHS=256` |
| `submit_ae_compression_timing.sh` | 5-epoch timing run, fallback if a run wall-clocks out |

## Workflow

1. `bash submit_ae_compression_large.sh` (base 50M)
2. `bash submit_ae_compression_100m.sh` and `bash submit_ae_compression_150m.sh` (capacity sweep)
3. If a run hits the 24h wall: `bash submit_ae_compression_timing.sh`,
   then extract a fitted value via
   `uv run autocast time-epochs --from-checkpoint <timing_path>/timing.ckpt -b 24`
   and re-run with that `COSINE_EPOCHS`.
