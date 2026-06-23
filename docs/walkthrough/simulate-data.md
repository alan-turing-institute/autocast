# Simulating data

To begin with, we're going to need some spatiotemporal data to work with.

You can get large-scale datasets from sources such as [The Well](https://polymathic-ai.org/the_well/), but these are too large to work with in this tutorial!
Instead of that, we'll generate some synthetic data by forward simulation of a partial differential equation (PDE).
Our sister tool [`autosim`](https://alan-turing-institute.github.io/autosim/) provides this functionality, with several built-in PDEs.

For this example we'll choose the advection–diffusion (AD) equation, which models the transport of a substance due to both diffusion (movement from high to low concentration) and advection (movement due to flow of the medium that the substance is in).
Details of the actual equation can be found on, e.g., [Wikipedia](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation).

First, install `autosim`.

```bash
git clone https://github.com/alan-turing-institute/autosim.git
cd autosim
uv sync
```

For the purposes of this tutorial, we'll assume that you've already [installed `autocast`](../installation.md), and that `autosim` and `autocast` have been cloned to the same parent folder.
In other words, your directory structure looks like the following.
This isn't mandatory: you can have a different directory structure, but you will need to adjust the paths in the later commands accordingly.

```
parent_folder
├── autocast
│   ├── pyproject.toml
│   ├── src
│   └── ...
└── autosim
    ├── pyproject.toml
    ├── src
    └── ...
```

:::{note}
This is also a good time to install `ffmpeg` if you haven't already: it will let you visualise the generated data as a video.
:::

Then, from the `autosim` directory, run:

```bash
uv run autosim \
    simulator=spatiotemporal/advection_diffusion \
    simulator.n=16 \
    seed=42 \
    simulator.T=1.0 simulator.dt=0.1 \
    dataset.n_train=10 dataset.n_valid=3 dataset.n_test=3 \
    dataset.output_dir=../advection_diffusion_toy_data
```

This will generate a dataset with a spatial grid of size 16×16, with 10 trajectories (i.e. different sets of forward simulations) in the test set, 3 in the validation set, and 3 in the test set.

`T` and `dt` respectively refer to the total time of the simulation and the time step size.
In this case, we should expect each trajectory to contain 11 frames (including the starting point at time 0).

We won't go into more detail about `autosim` here, but you can find more information in the [documentation](https://alan-turing-institute.github.io/autosim/).

Your directory structure should now look similar to this:

```
parent_folder
├── advection_diffusion_toy_data
│   ├── cli.log
│   ├── examples
│   │   └── train
│   │       ├── batch_0.mp4
│   │       ├── batch_1.mp4
│   │       ├── batch_2.mp4
│   │       └── batch_3.mp4
│   ├── resolved_config.yaml
│   ├── stats.yml
│   ├── test
│   │   └── data.pt
│   ├── train
│   │   └── data.pt
│   └── valid
│       └── data.pt
├── autocast
└── autosim
```

There are a few files of interest here.
You won't need to handle any of them manually, but it's good to know what they are:

- `stats.yml` contains normalisation statistics for the dataset, calculated on the training set.
- `batch_*.mp4` are example videos of the generated trajectories.
  If you open these videos you will likely see not much going on: this is just because of the very short simulation we did!
  In a more realistic scenario the videos would be more interesting.
- `data.pt` files contain the actual data, stored as PyTorch tensors.
- `resolved_config.yaml` contains the configuration used to generate the dataset.
  This includes both the custom settings we used in the command-line invocation above, as well as other defaults that we didn't override.
  This is useful if you want to later see what settings were used to generate the dataset, or if you want to reproduce the dataset exactly.

Now that we have some data in hand, let's move on to training a model on it!
