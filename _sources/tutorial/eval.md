# Evaluating models

To run a series of preset evaluation tests on a saved model checkpoint, including single-step predictions and autoregressive rollout, you can use the `eval` subcommand and set `--workdir` to the run folder containing the configuration and model checkpoint to evaluate.

```bash
uv run autocast eval \
    --workdir /path/to/outputs
```

Some useful Hydra options for further controlling the evaluation are:

- `autoencoder_checkpoint`: the path to the autoencoder checkpoint to use for evaluation (if applicable).
  This is used if you trained a standalone processor (i.e., `uv run autocast processor`) in latent space.
  If you trained a full encoder-processor-decoder stack (i.e., `uv run autocast epd`), the autoencoder is already part of the model checkpoint, so does not need to be supplied separately.
- `eval.metrics`: a list of metrics to compute during evaluation.
- `eval.n_members`: the number of members to use for ensemble evaluation. Increasing this allows you to get a smoother estimate of the model's uncertainty.
