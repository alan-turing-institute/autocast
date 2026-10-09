# Evaluating a trained model

With a trained model checkpoint in hand, you can run the `eval` command to compute metrics and visualisations on the test set.

## Basic usage

The simplest invocation points `--workdir` at a training run's output folder.
AutoCast will pick up the saved configuration and checkpoint automatically.
For example, to evaluate the EPD model we trained [previously](./epd.md):

```
uv run autocast eval \
    --workdir ../full_epd_output
```

Evaluation has two main 'modes'.

The first is to perform single-step predictions, i.e., given a test set input at time `t`, the model predicts the output at time `{t+1, ..., t+N}` (where `N` is the number of output time steps; this is the `n_steps_output` parameter in the datamodule, which is 4 for many datasets in AutoCast).
Then, the predictions are compared to the ground truth and a suite of metrics is computed.
This is averaged over all time points and all trajectories in the test set.

The second is to perform autoregressive rollouts, where the model is run from the `t = 0` initial starting point, and its own predictions are fed back as input for the next time step.
That is, the model predicts `{t+1, ..., t+N}` from the initial input, then is given its own predicted `t+N` data point as input to predict `{t+N+1, ..., t+2N}`, and so on.
Again metrics are computed by comparing the model's predictions to the ground truth and averaged over all trajectories.

The [default suite of metrics](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/eval/default.yaml) includes both deterministic ones (MSE, RMSE, etc.), as well as probabilistic ones if the model is an ensemble (CRPS, energy score, spread–skill ratio, etc.).

## Output structure

After evaluation completes, you'll find a new `eval` subfolder inside the working directory:

```
full_epd_output
├── ...
└── eval
    ├── benchmark_metrics.csv
    ├── encoder_processor_decoder.log
    ├── evaluation_metrics.csv
    ├── resolved_eval_config.yaml
    ├── rollout_coverage_window_....{csv,png}
    ├── rollout_metrics.csv
    ├── rollout_metrics_per_timestep_channel_....csv
    ├── test_coverage_window_all.{csv,png}
    └── videos
        ├── ...
        └── snapshots
            └── ...
```

The CSV files `evaluation_metrics.csv` and `rollout_metrics.csv` contain the single-step and rollout metrics.

Furthermore, AutoCast will time model inference and report this in `benchmark_metrics.csv`.

## Coverage

Many of the other outputs here relate to *coverage* metrics for ensemble model predictions, i.e., how often the ground truth falls within the model's predicted uncertainty intervals.
At a given value of $\alpha$ (the *nominal coverage*), the *empirical coverage* is the fraction of test samples for which the ground truth falls within the model's predicted $\alpha$-level uncertainty interval.

`test_coverage_window_all.csv` contains the empirical coverage for each nominal coverage level for the single-step predictions, while `rollout_coverage_window_....csv` contains the same for the autoregressive rollouts.

Note that coverage can only be computed for models that produce uncertainty estimates.
For this to happen, the `eval.n_members` parameter must be greater than 1
By default this is set to 10, but you can override it to a different value if you want to use more or fewer ensemble members at evaluation time.
Increasing `n_members` at eval time gives a smoother estimate of the model's predictive distribution, at the cost of more computation.

## Ambient-space evaluation of latent-space models

If your model was trained in latent space (for example, the flow matching processor we trained), then simply running `uv run autocast eval` on that directory will calculate metrics in latent space.
These may not be comparable with metrics in the original ambient space.

To evaluate the model in ambient space, you need to pass the `autoencoder_checkpoint` configuration option to the `eval` command:

```
uv run autocast eval \
    --workdir ../proc_output \
    ++autoencoder_checkpoint=/path/to/parent_folder/ae_output/autoencoder.ckpt
```

To explicitly control whether the evaluation is done in latent or ambient space, you can also set the `eval.mode` parameter to `latent`, `ambient`, or `encode_once`.
`latent` and `ambient` mean that all predictions and metrics are computed in the respective space; in contrast, `encode_once` means that the test set is first encoded into latent space, then the model is run in latent space, and finally the predictions are decoded back into ambient space for metric computation.
In general this is automatically inferred from the model being evaluated so you do not need to set it manually:

- If it's a full encoder-processor-decoder stack, then `eval.mode` is set to `ambient`.
- For a processor-only model, `eval.mode` is set to `latent` if `autoencoder_checkpoint` is not provided, and `encode_once` if it is provided.
