# Available commands

Before running anything, let's first look at the 'base' configurations for datasets and model architectures are stored in `src/autocast/configs`.
If you want more lower-level information about how configurations work, please see [the Hydra documentation](https://hydra.cc/).

`autocast` provides some subcommands for training standard model stacks:

| Command                         | Description                                | Default config                                                                                               |
| ---------------------------     | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `uv run autocast ae`            | Train an autoencoder                       | [`src/autocast/configs/autoencoder.yaml`][1]
| `uv run autocast cache-latents` | Cache latents from an encoder              | [`src/autocast/configs/cache_latents.yaml`][2]                                                               |
| `uv run autocast processor`     | Train a processor (frozen encoder/decoder) | [`src/autocast/configs/processor.yaml`][3]                                                                   |
| `uv run autocast epd`           | Train an encoder-processor-decoder         | [`src/autocast/configs/encoder_processor_decoder.yaml`][4]                                                   |
| `uv run autocast eval`          | Evaluate a trained model                   | [`src/autocast/configs/eval/encoder_processor_decoder.yaml`][5]                                              |

[1]: https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/autoencoder.yaml
[2]: https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/cache_latents.yaml
[3]: https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/processor.yaml
[4]: https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/encoder_processor_decoder.yaml
[5]: https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/eval/encoder_processor_decoder.yaml

Notice that each of these YAML files in turn refer to a number of _other_ YAML files.
For example, `src/autocast/configs/autoencoder.yaml` specifies (amongst other things)

```yaml
defaults:
  - model: autoencoder
  - logging: wandb
```

which in turn point to [`src/autocast/configs/model/autoencoder.yaml`][6] and [`src/autocast/configs/logging/wandb.yaml`][7] respectively.
In this way, configurations can be built up from smaller pieces in a modular way.
We'll talk more about configurations on the next page

[6]: https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/model/autoencoder.yaml
[7]: https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/logging/wandb.yaml

## Two useful command-line flags

### Running on SLURM

`autocast` supports running experiments on SLURM clusters by adding the `--mode slurm` flag.
This automatically generates a submission Bash script and submits it to the cluster, so you don't have to worry about writing your own submission scripts.

### Performing a dry run

Add `--dry-run` to print the commands that will be executed without actually running them.
