# MC dropout

Replace the CRPS baseline's sampled conditional-normalization modulation with
Monte Carlo dropout in the Azula ViT feed-forward blocks. Dropout uses
`p=0.1`, remains active during evaluation, and resamples independently on
every forward call and autoregressive rollout step.

This first run deliberately excludes attention-projection dropout, stochastic
depth, and rollout-locked masks. It is the simplest Azula-native MC-dropout
comparison.

## CNS MSE + L2 baseline

The `planned_updates_02` scripts schedule one
`conditioned_navier_stokes` run. Reusable GS, GPE and AD configs remain in the
repository, and their script entries are commented out rather than deleted.

The run trains one MC-dropout sample per case with mean-reduced MSE and evaluates
50 stochastic samples. It holds the processor, `p=0.1`, conditioning and
sampling mechanism fixed against the CRPS MC-dropout run, making the scoring
objective the principal intended axis. The two training estimators still use
different numbers of unique cases and masks per update, as detailed under
comparison controls.

The loss is:

```text
MSE + 1e-5 * sum(W^2)
```

Here `W` comprises all trainable processor parameters with at least two
dimensions: input/output projections, attention and FFN matrices, and any other
multidimensional processor parameters. It is not restricted to the layers
followed by dropout. Biases, normalization vectors and the frozen
encoder/decoder are excluded. AdamW weight decay is zero so this explicit L2
term is the only weight regularizer. L2 is active only in training; validation
and test loss are plain MSE so checkpoint selection follows forecast fit rather
than weight norm. This deliberately differs from Keras's compiled validation
loss in the WeatherBench implementation.

For a seed-42 instantiation of the parameter-matched 704-wide, 12-block
processor, the initial eligible weight norm is approximately
`sum(W^2)=2.66e4`, so the L2 term initially contributes about `0.266`. During
the timing run, `train_loss` includes L2 while `train_mse` is the data term.
Their scale and the continued decrease of `train_mse` must be checked before
the production submission.

## Literature basis

| design choice | evidence | consequence here |
|---|---|---|
| MSE, `p=0.1`, 50 inference members | [WeatherBench Probability](https://arxiv.org/abs/2205.00865) tested `p` in `{0, 0.1, 0.2, 0.5}`, selected `0.1` by RMSE/CRPS, and used 50 samples to match IFS. | Direct empirical precedent; 50 is an evaluation protocol, not a theoretical requirement. |
| Explicit L2 coefficient `1e-5` | An [official `p=0.1` repository config](https://github.com/sagar-garg/WeatherBench/blob/f41f497ac45377d363dc30bfa77daf50d7b28afd/nn_configs/B/81-resnet_d3_dr_0.1.yml) uses latitude-weighted MSE and `l2: 1e-5`; the [network code](https://github.com/sagar-garg/WeatherBench/blob/f41f497ac45377d363dc30bfa77daf50d7b28afd/src/networks.py) applies Keras L2 to convolution kernels. | We copy this code-level coefficient and sum-of-squares convention, but deliberately limit it to the trainable processor. The paper itself does not state `1e-5`. |
| Approximate-Bayesian interpretation | Under the mean-over-data objective in [Gal and Ghahramani](https://proceedings.mlr.press/v48/gal16.html), the matrix-weight coefficient is `p_keep l^2 / (2 tau N)`; bias scaling differs. Their predictive covariance also adds `tau^-1 I`, and their construction places dropout before every weight layer. | Here `p_drop=0.1` means `p_keep=0.9`, but `N` is ambiguous for overlapping trajectory windows and no prior length scale or observation precision is specified. FFN-only dropout plus L2 on all processor matrices is therefore not that exact VI construction. |
| Point loss versus CRPS | [U-Cast](https://arxiv.org/html/2604.09041) uses `p=0.1`, first trains with weighted MAE, and then fine-tunes with fair CRPS using two members per update. It attributes point-loss MC-dropout underdispersion to the lack of a reward for ensemble spread and reports 50-member evaluation. | It motivates the scoring-rule axis and `p=0.1`, but not this exact MSE objective, eight-member AlphaFair CRPS estimator, or `1e-5` coefficient. |
| PDE/fluid relevance | [LE-PDE-UQ](https://arxiv.org/abs/2402.08383) benchmarks MSE plus `p=0.5` dropout on turbulent 2-D Navier–Stokes. It finds learned UQ stronger and argues that fixed-rate dropout is often better viewed as a shared-parameter ensemble than a data-dependent posterior. Its Adam weight-decay sweep favours zero over `1e-5`. | CNS is an appropriate illustrative domain, but its dropout rate and regularization mechanism are not precedents for this exact configuration. |
| Earlier spatiotemporal MC dropout | The [2021 study](https://arxiv.org/abs/2105.11982) evaluates air quality, traffic and COVID mortality with `p=0.05` and 50 passes for model uncertainty. Its air-quality ConvLSTM uses a weighted MAE over weather covariates and PM2.5, and it reports no MC-dropout L2 term. | It supports the broad baseline, not this loss or coefficient, and is not a weather-forecasting study. |

The WeatherBench repository is pinned above at commit
`f41f497ac45377d363dc30bfa77daf50d7b28afd`. It does not contain the expected
paper-specific `P/002` config. The cited `B/81` file records a matching
repository recipe (`p=0.1`, latitude-weighted MSE and `l2=1e-5`); the L2 value
is code-level evidence rather than an explicit paper setting.

WeatherBench Probability found its `p=0.1` MC-dropout ensemble strongly
underdispersive (spread-skill ratios `0.40`, `0.34`, `0.39` and `0.13` for
Z500, T850, T2M and precipitation). We therefore describe this run as an
**MC-dropout shared-parameter ensemble and epistemic proxy**, not a calibrated
Bayesian posterior. Unlike WeatherBench's convolutional ResNet, this processor
places dropout only in the Azula ViT FFNs; the run is an Azula-native baseline,
not an architectural reproduction.

The final comparison has three distinct roles:

1. MSE + MC dropout + L2: single-model shared-parameter/epistemic proxy.
2. CRPS + the same MC dropout: the same sampler trained for predictive spread.
3. Flow matching: an expressive conditional generative distribution.

Evaluation should report ensemble-mean RMSE, CRPS, spread-skill ratio and
coverage or rank histograms. Otherwise the expected MSE-dropout underdispersion
will be hidden. A single CNS run demonstrates behavior in this fluid benchmark;
it cannot establish cross-domain generality or seed stability.

## Comparison controls

| control | CRPS MC dropout | CNS MSE + L2 |
|---|---:|---:|
| processor | 704-wide, 12 blocks, 8 heads | same |
| dropout | FFN, `p=0.1` | same |
| stochastic modulation noise | off | off |
| members per update | 8 | 1 |
| cases per GPU | 32 | 256 |
| processor evaluations per GPU update | 256 | 256 |
| objective | AlphaFair CRPS | mean MSE + explicit L2 |
| learning rate | `2e-4` | `2e-4` |

The different batch sizes match processor work per update, not gradient
semantics: CRPS sees eight masks for each of 32 cases, while MSE sees one mask
for each of 256 cases. CNS simulation constants remain present through the
`PermuteConcat` encoder (`with_constants=true`); the separate adaptive global
conditioning path is off.

Removing the 1024-channel modulation projection reduces the original
`hidden_dim=568` model from approximately 80.8M to 53.3M parameters. These
configs use `hidden_dim=704`, producing approximately 79.8M parameters while
preserving depth and head count. This follows the parameter-matching approach
used by the noise-channel ablation.

## Files

| file | purpose |
|---|---|
| `../submit_planned_updates_01_timing.sh` | Five-epoch timing jobs for all four datasets |
| `../submit_planned_updates_01_large.sh` | Timing-derived 24h production jobs |
| `../submit_planned_updates_02_timing.sh` | Four-dataset MSE + L2 five-epoch timing jobs |
| `../submit_planned_updates_02_large.sh` | Four-dataset MSE + L2 24h production jobs |
| `../submit_eval_planned_updates_02.sh` | Four deferred final-checkpoint evaluations with 50 MC samples |
| `local_hydra/local_experiment/ablations/mc_dropout/<dataset>/crps_vit_azula_mc_dropout_large.yaml` | Dataset-specific parameter-matched experiment |
| `local_hydra/local_experiment/ablations/mc_dropout/<dataset>/mse_vit_azula_mc_dropout_large.yaml` | Reusable dataset-specific MSE + L2 experiment |

## Submission workflow

1. Run `slurm_scripts/ablations/submit_planned_updates_01_timing.sh`.
2. Retrieve the timing outputs using the command printed by that script.
3. Run `slurm_scripts/ablations/submit_planned_updates_01_large.sh`. It
   locates each timing checkpoint and derives the 24h cosine schedule with a
   2% margin.

The production script performs a dry-run submission before each real
submission, matching the main CRPS comparison workflow. Set `TRAINING_SEED`
to create an independent repeat; it defaults to 42.

For the MSE + L2 baselines, run `submit_planned_updates_02_timing.sh`, inspect
the MSE/L2 scale, then run `submit_planned_updates_02_large.sh`. Preview and
submit the 50-sample evaluations with:

```bash
RUN_ROOT=outputs/YYYY-MM-DD/planned_updates_02 \
  ./slurm_scripts/ablations/submit_eval_planned_updates_02.sh
RUN_ROOT=outputs/YYYY-MM-DD/planned_updates_02 SUBMIT=true \
  ./slurm_scripts/ablations/submit_eval_planned_updates_02.sh
```

The evaluation jobs use `afterany` dependencies and resolve each run's final
`encoder_processor_decoder.ckpt` only after its training job leaves the queue.
