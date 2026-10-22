# Output paths

In the [autoencoder walkthrough](../walkthrough/autoencoder.md) we used the `--workdir` option to specify where the output of the training should be saved.
There are, however, other ways to specify the output path which may be more convenient in some situations.

If `--workdir` is not specified, the output will be saved to `<OUTPUT_BASE>/<RUN_GROUP>/<RUN_ID>`.
Each of these three path components, `<OUTPUT_BASE>`, `<RUN_GROUP>`, and `<RUN_ID>`, can be specified via separate command-line flags `--output-base OUTPUT_BASE`, `--run-group GROUP`, and `--run-id ID`.

The defaults for each of them are the string `outputs`, today's date, and an automatically generated identifier.

However, note that `--run-id ID` is also used for [Weights and Biases logging](../howto/wandb.md), so if you also want to log to W&B you may find it easier to set `--run-group` and `--run-id` so that both the output directory and the W&B run have the same name.
