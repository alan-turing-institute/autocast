"""Optimizer configuration utilities."""

from typing import Any


def get_optimizer_config(
    learning_rate: float = 1e-4,
    optimizer: str = "adam",
    scheduler: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create an optimizer configuration dict.

    Convenience function for quickly creating optimizer configs in notebooks
    and scripts without needing full Hydra configuration.

    Args:
        learning_rate: Learning rate for the optimizer. Default 1e-4.
        optimizer: Optimizer name ('adam', 'adamw', 'sgd'). Default 'adam'.
        scheduler: Optional scheduler name ('cosine', 'cosine_with_restarts',
            'step', 'plateau').
            Default None (no scheduler).
        **kwargs: Additional optimizer parameters (e.g., betas, weight_decay,
            step_size, gamma).

    Returns
    -------
        Dict containing optimizer configuration compatible with OptimizerMixin.
    """
    config = {
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "scheduler": scheduler,
    }
    config.update(kwargs)
    return config
