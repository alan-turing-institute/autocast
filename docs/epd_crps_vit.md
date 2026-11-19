# Vision transformer in the EPD CRPS comparison

The [Slurm launcher](../slurm_scripts/comparison/epd/submit_crps_large.sh) runs
`autocast epd --mode slurm` and selects a Hydra `local_experiment` for each
dataset. All four experiments train an encoder–processor–decoder ensemble with
eight members. The processor is `AzulaViTProcessor`, which wraps a spatial ViT.

## Configuration

The [EPD root config](../src/autocast/configs/encoder_processor_decoder.yaml)
adds `local_hydra` to Hydra's search path. Each of the following experiments
selects `vit_azula_large` as `model.processor` and overrides the same four
processor settings.

- [Gray–Scott](../local_hydra/local_experiment/epd/gray_scott/crps_vit_azula_large.yaml)
- [GPE laser wake](../local_hydra/local_experiment/epd/gpe_laser_wake_only/crps_vit_azula_large.yaml) (the datamodule key is `gpe_laser_only_wake`)
- [Conditioned Navier–Stokes](../local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml)
- [Advection–diffusion](../local_hydra/local_experiment/epd/advection_diffusion/crps_vit_azula_large.yaml)

| Effective processor settings | Where they come from |
| --- | --- |
| `hidden_dim: 568`, `n_layers: 12`, `num_heads: 8`, `n_noise_channels: 1024` | Dataset experiment overrides |
| `patch_size: 4` (4 × 4 spatial patches), `temporal_method: none`, `noise_mode: global`, `include_global_cond: false` | [Shared processor preset](../src/autocast/configs/processor/vit_azula_large.yaml) |
| Input and output channel counts and spatial resolution | Inferred from the data and encoder/decoder by `setup_epd_model` in [setup.py](../src/autocast/scripts/setup.py) |

## Implementation and data flow

1. `PermuteConcat` folds the input time dimension into channels and appends
   constant fields and scalars when present. This supplies a channel-first
   spatial field to the processor. [Encoder implementation](../src/autocast/encoders/permute_concat.py)
2. `AzulaViTProcessor.map()` draws a 1,024-dimensional Gaussian vector per
   example and ensemble member. Each vector is shared across that member's
   spatial tokens and reused across transformer blocks for adaptive layer
   normalization (AdaLN). `forward()` adds a singleton time dimension and
   passes the noise as precomputed modulation. [Processor implementation](../src/autocast/processors/azula_vit.py)
3. `SpatialViTBackbone` constructs `azula.nn.vit.ViT`. Attention operates over
   spatial patch tokens with query/key normalization and relative positional
   bias enabled. The shared `TemporalBackboneBase` handles tensor layout and
   modulation. The patch projections and transformer blocks live in the
   installed Azula package, pinned to 0.7.0 in the lockfile.
   [ViT wrapper](../src/autocast/nn/vit.py) ·
   [Backbone base](../src/autocast/nn/base.py) · [Dependency lockfile](../uv.lock)
4. `EncoderProcessorDecoderEnsemble` repeats each input eight times and uses
   the same processor weights for all members. `ChannelsLast` restores the
   output time and channel axes. With `train_in_latent_space: false`,
   `AlphaFairCRPSLoss` scores the decoded ensemble in field space.
   [Ensemble model](../src/autocast/models/encoder_processor_decoder_ensemble.py) ·
   [Decoder](../src/autocast/decoders/channels_last.py) ·
   [Loss implementation](../src/autocast/losses/ensemble.py)

The processor regression tests cover the default global noise path, spatial
modulation variants, and backpropagation through ensemble CRPS.
[Azula ViT noise tests](../tests/processors/test_azula_vit_spatial_noise.py)

## Mixed precision

Training and evaluation each default to FP32. Set the precision independently
for each run. Both accept `bf16-mixed` and `16-mixed` through Lightning.

- Train with `TRAINING_PRECISION=bf16-mixed bash slurm_scripts/comparison/epd/submit_crps_large.sh`
  or pass `trainer.precision=bf16-mixed` directly to `autocast epd`.
- Evaluate with `uv run autocast eval --workdir PATH_TO_RUN eval.precision=bf16-mixed`.
