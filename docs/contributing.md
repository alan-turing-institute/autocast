# Contributing

Autocast welcomes contributions from users in the form of either issues or PRs!

## Setting up a development environment

```bash
# Clone the repository
git clone https://github.com/alan-turing-institute/autocast
cd autocast
uv sync
```

This should get you an environment with all the dependencies installed; you can then work on changes locally.

## Building the docs

The project documentation is built with [Jupyter Book](https://jupyterbook.org/).
The documentation can be built locally with the following steps:

```bash
# Install the necessary dependencies
uv sync --extra docs
# Build the docs
uv run jupyter-book build docs
# Serve locally
uv run python -m http.server -d docs/_build/html
```

Then open `http://localhost:8000` in your browser to view the docs.

## Tests and linting

To run tests, use:

```bash
# Install the necessary dependencies
uv sync --extra dev
# Run tests
uv run pytest
```

Pull requests on GitHub additionally include checks for linting.
The easiest way to run all these checks locally is to use [`pre-commit`](https://pre-commit.com/).
First install `pre-commit` following the instructions on their website, then run:

```bash
pre-commit install
pre-commit run --all-files
```
