# Training an autoencoder

We'll begin by training an autoencoder on the AD dataset we just simulated.

`autocast` provides a high-level tool for training autoencoders: `uv run autocast ae <options>`.
The default options are stored in YAML configuration files, which can be found in `src/autocast/configs` directory.
For example, the default configuration for autoencoders is [here](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/autoencoder.yaml).

:::{note}
Both `autocast` and `autosim` use Hydra, a library which allows you to compose YAML configurations to form full specifications for experiments.
You can read more about Hydra in [its docs](https://hydra.cc/).
:::

If you look at the default configuration linked above, it points to a `datamodule` called `autosim`, which is itself a configuration file [here](https://github.com/alan-turing-institute/autocast/blob/main/src/autocast/configs/datamodule/autosim.yaml).

The datamodule is a specification of where the dataset is, how to load it, and other details such as normalisation.
In this case, the `autosim` datamodule is a blank slate which lets us pass the path to any `autosim`-generated dataset.
It has a `data_path` field which we can set to the path of the dataset we generated in the previous section.

We'll specify this by providing the `datamodule.data_path` as a command-line argument:

```bash
uv run autocast ae \
    datamodule.data_path=../advection_diffusion_toy_data \
    trainer.max_epochs=10
```
