# Architecture comparison: U-Net, FNO, ViT

Compare U-Net and FNO backbones against the ViT (Azula) baseline on the
CRPS ambient path.

**Status:** U-Net run complete; FNO config and timing/production/eval scripts
ready.

## Baseline

`local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml`
(ViT-Azula, ~80.8M real scalar parameters).

## Knob

Swap `model.processor` backbone while trying to match parameter count
(~80M), global spatial resolution, and the 24h training budget.

| variant | spatial setting | depth | width | ~real scalar params |
|---|---|---:|---:|---:|
| ViT baseline | patch 4 -> 16x16 tokens | 12 | 568 | 80.8M |
| U-Net | four 2x scales | 3 blocks/scale | `[62,124,248,496]` | 81.3M |
| FNO | 16 modes/axis | 4 | 264 | 81.1M |

The planned U-Net run uses
`local_hydra/local_experiment/ablations/arch_unet_fno_vit/conditioned_navier_stokes/crps_unet_azula_80m.yaml`.
It matches the ambient baseline's encoder/decoder/loss and uses an Azula U-Net
channel ladder `[62, 124, 248, 496]`, measured at ~81.3M processor params for
CNS ambient shapes.

The FNO run uses
`local_hydra/local_experiment/ablations/arch_unet_fno_vit/conditioned_navier_stokes/crps_fno_80m.yaml`.
On the 64x64 ambient grid, the ViT's patch size of 4 produces a 16x16 token
lattice. Retaining 16 Fourier modes per axis gives the FNO a comparable global
spatial bandwidth, while the FNO pointwise path still operates on the full
64x64 grid.

FNO spectral weights are complex. The parameter match counts each complex
coefficient as two real scalar degrees of freedom. The configured FNO has
about 41.0M PyTorch tensor elements but 81.1M real scalar parameters. Matching
the raw `numel()` count instead would give the FNO roughly twice the real
storage and degrees of freedom of the ViT.

## Noise and shared training setup

The ViT and U-Net use a 1024-dimensional global noise vector through
conditional normalization. The FNO has no conditional-normalization path in
the current wrapper, so it uses `ConcatenatedNoiseInjector` with one spatial
white-noise channel. Ensemble expansion happens before injection, giving each
of the eight members an independent noise field.

All other practical settings follow the CNS CRPS ViT baseline:

- `AlphaFairCRPSLoss` and the same train/validation metrics;
- `n_members=8`, batch size 32/GPU, and four-GPU DDP;
- normalized ambient training with `permute_concat` and `channels_last`;
- AdamW, learning rate `2e-4`, no warmup, and the same checkpoint callbacks;
- a timing-derived cosine schedule filling the same 24h wall-clock budget.

## Datasets

CNS only for now.

## Run sequence

1. Submit `../submit_planned_updates_03_timing.sh`.
2. Retrieve
   `outputs/<date>/timing_planned_updates_03/fno_m8_crps_cns/`.
3. Run `../submit_planned_updates_03_large.sh`. It derives the 24h epoch count
   from the latest timing checkpoint; `COSINE_EPOCHS=<n>` can override it
   explicitly.
4. After production, run `../submit_eval_planned_updates_03.sh`. It finds the
   latest matching run automatically, or accepts `FNO_RUN_DIR=<path>`.

Start evaluation at batch size 4/GPU because FNO keeps full-resolution feature
maps. Override with `EVAL_BATCH_SIZE` after confirming memory headroom.

The first timing job is also the memory check. If batch size 32/GPU does not
fit, use batch size 16 with two gradient-accumulation steps so the optimizer
still sees the baseline's effective batch.
