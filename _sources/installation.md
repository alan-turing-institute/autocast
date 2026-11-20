# Installation

Autocast is currently not on PyPI, so to install the library you will need to clone it from GitHub.
Installation is most easily done using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# Clone the repo
git clone https://github.com/alan-turing-institute/autocast.git
cd autocast

# Install dependencies
uv sync
```

[`ffmpeg`](https://ffmpeg.org) is an optional binary dependency that is used to generate videos during model evaluation.
Running `uv sync` will not install `ffmpeg` for you.
If you want to use this feature, you will need to install `ffmpeg` via your system package manager (for example, `brew install ffmpeg` on macOS).

Once you have the dependencies set up, you should be able to run the `autocast` command:

```bash
uv run autocast --help
```

If you are also interested in contributing to the codebase, there are optional development dependencies that can be installed.
Please see the [Contributing](contributing.md) guide for more information.
