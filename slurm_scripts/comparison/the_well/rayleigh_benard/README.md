# Rayleigh-Benard comparison submitters

This folder contains the maintained Rayleigh-Benard submitters for the
post-merge `283/` Rayleigh-Benard stack. Some configs referenced here live on
parallel `283/` branches today, so these entrypoints are intended for the final
state after the RB ambient, latent FM, masked FM, diffusion, and latent CRPS
branches have all landed on `main`.

## Config inventory

| Group | Run | Config |
| --- | --- | --- |
| 24h comparison | CRPS ambient, LOLA pixel ViT | `the_well/rayleigh_benard/crps_vit_azula_lola_pixel_ambient` |
| 24h comparison | CRPS ambient | `the_well/rayleigh_benard/crps_vit_azula_large_ambient` |
| 24h comparison | CRPS latent | `the_well/rayleigh_benard/crps_vit_azula_large_latent` |
| 24h comparison | FM latent | `the_well/rayleigh_benard/fm_vit_large` |
| LOLA 4096-step comparison | FM latent | `the_well/rayleigh_benard/fm_vit_large` |
| LOLA 4096-step comparison | FM latent, masked window | `the_well/rayleigh_benard/fm_vit_large_masked_window` |
| LOLA 4096-step comparison | Diffusion latent | `the_well/rayleigh_benard/diffusion_vit_large_lola` |

Additional ambient FM presets are available under
`local_hydra/local_experiment/the_well/rayleigh_benard/` for future ablations,
but are not part of these maintained submitters because they are expected to be
more computationally expensive than the selected 24h comparison matrix.

## Entry points

Use `DRY_RUN_ONLY=true` to only emit SLURM dry runs. By default each submitter
runs one dry-run submission and then the real submission, matching the existing
comparison scripts.

```bash
# Timing probes or final 24h comparison runs.
bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_24h.sh timing all
bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_24h.sh final all

# LOLA-equivalent 4096 epoch x 64 train-batch runs.
bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_lola4096.sh all

# Diffusion eval modes: Euler, AB order 3, and DDPM.
bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_diffusion_eval.sh all
```

Each entrypoint also accepts individual run keys, for example
`submit_24h.sh final crps_latent fm_latent` or
`submit_diffusion_eval.sh ab_o3`.

For diffusion eval, `EVAL_MAX_ROLLOUT_STEPS` controls the requested rollout
windows. The script derives `DATAMODULE_MAX_ROLLOUT_STEPS` from that value and
`EVAL_ROLLOUT_START` so the rollout dataloader exposes enough truth frames for
the requested horizon.

## Spatiotemporal RB extension (FM and CRPS)

These additive presets use the shared `spatiotemporal_vit` backbone; the older
time-in-channels configs above are unchanged. They require fresh training.
The [library guide](../../../../docs/spatiotemporal_vit.md) covers the reusable
backbone, processor presets and configurable noise/masking interfaces.
The settings and reference comparisons below belong to this RB experiment.

| Setting | LOLA reference | New FM | New CRPS |
| --- | --- | --- | --- |
| Transformer | Width 1024, 16 blocks, 8 heads, modulation 256 | Same core | Same core |
| Transformer trunk parameters | 222,689,344 at this C64 input shape, excluding conditioning embeddings | Same | Same; noise projection counted separately |
| Latent patches / tokens | `[1,1,1]`; 320 at 5×16×4 | Same | Same |
| Epochs × optimizer updates | 4096 × 64 = 262,144 | Same | Same |
| Global distinct windows / update | 256 | 256 (64/GPU) | 32 (8/GPU) |
| Global member-window predictions / update | 256 | 256 | 256 (8 members) |
| AdamW / cosine | LR 1e-4, betas (0.9,0.99), zero decay/warmup, clip 1 | Same | Same |
| Training context | Random prefix of five-frame window | Same pinned code policy | Same pinned code policy |
| Objective | Denoising with LOLA weighting/time sampling | Existing velocity FM | Existing alpha-fair CRPS (alpha 0.95) |
| Validation | 16 batches, every epoch | 16 batches, every 8 epochs | 16 batches, every 8 epochs |
| Hardware | 8 H100s (reported latent training) | 4 GH200s | 4 GH200s |

The architecture and update schedule match the
[pinned LOLA implementation](https://github.com/francois-rozet/lola/tree/21a4354b327e6e5ee06da5075ba3bd1dd88c61f1).
The library guide describes the numerical architecture checks. These
experiments extend our benchmark rather than reproduce the complete LOLA
method: they retain our fixed forecasting evaluation,
method-specific time/noise conditioning, and member-inclusive batch convention.
Consequently FM sees eight times as many distinct training windows as CRPS;
matching updates and predictions does **not** match real-example exposure.
Validation also sees 4096 real windows for FM versus 512 for CRPS per check.
We reuse the existing frozen LOLA `dcae_f32c64_large` cache and its preprocessing,
not a newly trained autoencoder. Neither cached-latent loader adds online
spatial augmentation; LOLA's latent training branch also reads cached windows.
Our finite shuffled training loader and sequential validation loader differ
from LOLA's infinite shuffled loaders. For comparison with our report, the
evaluation launcher must explicitly select `eval.rollout_start=16` and the
same rollout horizon and member count. These training presets do not set up
the complete rollout evaluation; the generic start defaults to zero. Start16
is a deliberate difference from LOLA's frame-zero protocol.

Both methods draw one prefix mask per real example using the pinned code's
Poisson rate 1, prefix probability 1 and minimum context 1. The modulo-wrapped
context length is 1-4 of the five frames. This follows the executable code;
the paper instead describes Poisson rate 2 with truncation. Training and
validation loss average over unobserved elements within each example, then
over examples, so longer forecasts do not receive extra weight. This loss
reduction is an explicit comparison choice, distinct from LOLA's full-window
denoising objective. Forecast metrics and rollouts always observe the fixed
first frame and predict four frames.

The CRPS preset trains from scratch with eight members and our global-noise
adapter; it does not reproduce the separate noise heads, bidirectional masks or
deterministic-checkpoint fine-tuning in the
[retrofitting study](https://arxiv.org/abs/2603.01949v2).
Its processor owns the ensemble loss (`model.loss_func: null`): masks and
clean context are shared across members, with independent global noise per
member. The unreduced scoring rule and mask sampler are configurable
components; the backbone contains neither policy nor objective logic.

Matching the original LOLA transformer does not fix the stochastic CRPS
adapter: the original neural solver is deterministic. Our adapter mixes a
1024-dimensional noise vector (linear projection to 256 features) with the
physical embedding before each block's existing AdaLN head. This retains
our benchmark's global-noise/shared-head strategy, with a new 1024-to-256
projection; the previous RB CRPS model used 1024 modulation features
directly. We keep that projection as an explicit choice for this experiment.
The retrofitting code instead maps 32-dimensional noise through an MLP and
LayerNorm, then adds a separate noise AdaLN head in every block (about 18M
extra parameters).
Those are different stochastic parameterizations of the same attention trunk.

| CRPS setting | This extension | Retrofitting reference |
| --- | --- | --- |
| Initialization | Fresh training | Pretrained deterministic LOLA checkpoint |
| Latent cache channels | 64 | 16 |
| Score | Alpha-fair CRPS, alpha 0.95 | Fair CRPS |
| Mask direction | Prefix only | Prefix/suffix, prefix probability 0.66 in code |
| Training members / real windows per update | 8 / 32 | 4 / 32 for RB |
| Optimization | One LR 1e-4, no warmup; 4096×64 updates | Backbone/noise LRs 1e-4/3e-4, 10-epoch warmup; 100×512 updates after pretraining |

The CRPS masking/normalization mechanics agree with the pinned retrofitting
code when evaluated with fair CRPS and the same masks; score and prediction
gradients were checked directly. Initialization and noise modulation are
material differences, so this is a from-scratch objective comparison using
a LOLA-compatible backbone, not a reproduction of published retrofitting
results. The reference reduces real batch size by the member count too;
our member-inclusive accounting follows that convention, but its nominal
prediction budget and total exposure differ.
Sources: [pinned CRPS config](https://github.com/cddcam/lola_crps/blob/7643376c2949717ee5c2c840584689f529ba77a5/experiments/configs/train_crps.yaml),
[noise wrapper](https://github.com/cddcam/lola_crps/blob/7643376c2949717ee5c2c840584689f529ba77a5/lola/surrogate.py),
[noise heads](https://github.com/cddcam/lola_crps/blob/7643376c2949717ee5c2c840584689f529ba77a5/lola/nn/vit.py),
[paper training table](https://arxiv.org/html/2603.01949v2).

```bash
# Dry run, then actual timing only (four GPUs per job, 20-minute cap each).
DRY_RUN_ONLY=true bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_spacetime_timing.sh
bash slurm_scripts/comparison/the_well/rayleigh_benard/submit_spacetime_timing.sh
```

`fm_spacetime_latent` and `crps_spacetime_latent` are the experiment names under
`local_experiment=the_well/rayleigh_benard/`. Timing run and output identifiers
retain their recorded spelling; the presets now use the canonical
`spatiotemporal_vit` library names. Remove the temporary `spacetime_vit`
import/config aliases after collecting the timing results and migrating any
retained configs/checkpoints. The timing script now runs 16
64-update epochs including two validation passes and an epoch-boundary checkpoint,
without shortening the 4096-epoch cosine horizon. It uses unique timestamped
output groups; `OUTPUT_BASE` can point to an existing HPC output root. No full
training jobs are submitted by this script. Use its measured epoch times to
estimate the full schedule; previous 64-token timings are not applicable.
These presets compute gradient diagnostics and synchronize training-loss logs
every 64 optimizer updates, matching `trainer.log_every_n_steps`; the logged
loss remains the global mean for that update. Other configs retain their
existing every-step defaults, and validation logging is unchanged.
The full presets checkpoint at completed epochs every 128 epochs, retain best
validation/calibration checkpoints, and stop within the 24-hour Slurm limit;
finishing a longer 4096-epoch schedule requires a subsequent resume.
Resume a concrete completed-epoch `snapshot-*.ckpt` with full state, keeping
`trainer.max_epochs=4096` and `optimizer.cosine_epochs=4096` as total horizons.
Set `reset_resume_time_budget=true` to give the new allocation a fresh 23h45
stop budget; this does not reset the epoch-based LR schedule. Do not use
weights-only loading or assume the final wall-clock-stop checkpoint completed
an epoch. The CPU resume regression checks the first resumed LR, optimizer,
scheduler, EMA and diagnostic cadence against uninterrupted training.
The new output group and W&B names include `masked`, keeping these probes
separate from the fixed-context measurements below. Use `TIMING_EPOCHS=8`
for one validation cycle if necessary. The latest CRPS measurement is below;
FM still needs retiming with the logging changes. Full training is a separate step.

Measured on Isambard-AI, 2026-10-15 (code `796d6673`, fixed-context training,
before enabling random context in these presets):

| Timing run | Parameters | Seconds / 64-update epoch (training only) | Mean including validation/checkpoints | Projected 4096 epochs |
| --- | ---: | ---: | ---: | ---: |
| [FM, job 7430708](https://wandb.ai/turing-core/autocast/runs/7ecp1a0v) | 222.89M | 19.4 | 21.6 | 24.6 h |
| [CRPS, job 7430710](https://wandb.ai/turing-core/autocast/runs/ididdaws) | 223.02M | 19.2 | 21.3 | 24.2 h |

Both completed 512 updates, validation and checkpoint writes, using 8.3
node-minutes together. The projection extrapolates the measured epoch cycles,
not the one-off Slurm startup and W&B shutdown. Eight epochs give a rough
estimate: initial warmup, best-checkpoint frequency and the timing-only
epoch-eight snapshot affect overhead. These measurements initially suggested
allowing roughly 27 hours per method, including a resume across the allocation
limit. The full preset's 23h45 training limit leaves room for checkpointing
and shutdown. These timings use our validation cadence above, not LOLA's
every-epoch validation; previous packed-time runtimes remain separate.
These fixed-context measurements precede the masked configs above and should
not be used as their current runtime projection.

HPC source: `~/autocast-rb-spacetime` on branch `spacetime-vit-backbone`, using
`~/autocast-03/.venv` with `PYTHONPATH` pointing at the new checkout's `src`.
Outputs and `timing_summary.json` are under
`~/autocast-03/outputs/timing/rb_spacetime_20261015T084800Z/`.

For multi-node Isambard runs, add `hydra/launcher=slurm_isambard` alongside
`distributed=ddp_4gpu_2node_slurm`. This opt-in launcher loads the site's
`brics/nccl/2.26.6-1` module before `srun`, enabling the OFI plugin and Slingshot
settings without changing the generic launcher or model configuration.
The existing PyTorch environment retains its bundled NCCL library. When checking
transport, set `NCCL_DEBUG=INFO` after loading the module and confirm the logs
report `AWS Libfabric` with the `cxi` provider; an `hsn` socket interface alone
does not confirm RDMA. See the [Isambard NCCL guide](https://docs.isambard.ac.uk/user-documentation/guides/nccl/).

Transport check `7455810` (2026-10-15, two nodes/eight GPUs, 54 seconds) confirmed
Socket in the old environment and `AWS Libfabric`/`cxi`/`GDRDMA` on all ranks with
this launcher setup. Both passed collective correctness checks. Mean all-reduce
times changed from 6.70 to 0.45 ms at 25 MiB, but from 60.97 to 94.22 ms at
256 MiB; the larger-message slowdown remains unexplained. This confirms the
transport fix, not a full-training speedup. Evidence is under
`~/autocast-03/outputs/network/rb_nccl_transport_20261015T205720Z/`.

Masked CRPS logging check (2026-10-16, [job 7462449](https://wandb.ai/turing-core/autocast/runs/mmi7dgd3)):
one node/four GPUs completed 16 epochs and 1,024 updates in 6m26s. Mean epoch
time, including validation/checkpoints, fell from 21.30 to 20.68 seconds;
the 4,096-epoch projection fell from 24.23 to 23.53 hours. This leaves only
13 minutes below the 23h45 training guard, so retain a checkpoint-resume
contingency. FM has not been retimed with these logging changes. Source
snapshots, validation results and timings are saved under
`~/autocast-03/outputs/timing/rb_crps_spatiotemporal_4gpu_logging_20261016T045534Z/`.

The masked FM probe before the logging fix ([job 7447965](https://wandb.ai/turing-core/autocast/runs/ujoxz7xv))
averaged 21.14 seconds per epoch, projecting 24.05 hours on four GH200s.
Thus FM/CRPS projections are about 96/94 GPU-hours, versus the
[LOLA paper's](https://arxiv.org/html/2507.02608v4#A2) reported 48 hours on eight
H100s (384 GPU-hours) per latent RB emulator. Both use cached latents; the
paper's emulator timing excludes autoencoder training. This gap is unresolved,
not an established hardware speedup. Our validation is eight times less
frequent, and the [reference training loop](https://github.com/francois-rozet/lola/blob/21a4354b327e6e5ee06da5075ba3bd1dd88c61f1/experiments/train_diffusion.py)
writes model weights and synchronizes every epoch. Objectives, real-example
exposure for CRPS, and runtime environments also differ. Matching the trunk
and update schedule does not make these end-to-end timing comparisons exact;
our short probes are projections, not completed 4,096-epoch runs.
