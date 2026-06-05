# Contributing to AutoCast

We welcome all contributions to AutoCast! This guide will help you get started.

## Getting started

1. Fork the repository on GitHub
2. Clone your fork:

   ```bash
   git clone https://github.com/YOUR-USERNAME/autocast.git
   cd autocast
   ```

3. Install development dependencies:

   ```bash
   uv sync --extra dev
   ```

4. Set up pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Development workflow

1. Create a new branch for your feature or fix:

   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes and ensure tests pass:

   ```bash
   uv run pytest
   ```

3. Run the linter and type checker:

   ```bash
   uv run ruff check .
   uv run pyright
   ```

4. Commit your changes and push to your fork:

   ```bash
   git add .
   git commit -m "Add my feature"
   git push origin feature/my-feature
   ```

5. Open a pull request on GitHub.

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [Pyright](https://github.com/microsoft/pyright) for type checking. Pre-commit hooks will run these automatically on each commit.

- Line length: 88 characters
- Docstring convention: NumPy style
- Type hints are encouraged

## Testing

Tests are located in the `tests/` directory. Run them with:

```bash
uv run pytest
```

For test coverage:

```bash
uv run pytest --cov=autocast
```

## Reporting issues

- Use the [GitHub issue tracker](https://github.com/alan-turing-institute/autocast/issues) to report bugs
- Include a minimal reproducible example when possible
- Describe the expected vs actual behaviour
