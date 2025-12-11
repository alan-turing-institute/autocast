from lightning.pytorch.loggers import WandbLogger
from torch import nn

from autocast.logging import (
    create_notebook_logger,
    create_wandb_logger,
    maybe_watch_model,
)


class DummyModel(nn.Module):
    """Minimal model for testing watch functionality."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)


def test_create_wandb_logger_disabled():
    """Test that disabled config returns None."""
    cfg = {"wandb": {"enabled": False}}
    logger, watch = create_wandb_logger(cfg, experiment_name="test")
    assert logger is None
    assert watch is None


def test_create_wandb_logger_enabled():
    """Test that enabled config creates logger in offline mode."""
    cfg = {
        "wandb": {
            "enabled": True,
            "project": "test-project",
            "name": "test-run",
            "mode": "disabled",  # Prevent network calls
        }
    }
    logger, watch = create_wandb_logger(cfg, experiment_name="test")
    assert isinstance(logger, WandbLogger)
    assert watch is not None


def test_create_wandb_logger_with_tags():
    """Test logger creation with tags."""
    cfg = {
        "wandb": {
            "enabled": True,
            "project": "test-project",
            "tags": ["test", "ci"],
            "mode": "disabled",
        }
    }
    logger, _ = create_wandb_logger(cfg, experiment_name="test")
    assert isinstance(logger, WandbLogger)


def test_create_notebook_logger_disabled():
    """Test notebook logger with enabled=False."""
    logger, watch = create_notebook_logger(enabled=False)
    assert logger is None
    assert watch is None


def test_create_notebook_logger_enabled():
    """Test notebook logger creation with default settings."""
    logger, watch = create_notebook_logger(
        project="test-project",
        name="test-notebook",
        tags=["notebook"],
        enabled=True,
    )
    assert isinstance(logger, WandbLogger)
    assert watch is not None


def test_create_notebook_logger_minimal():
    """Test notebook logger with minimal arguments."""
    logger, _ = create_notebook_logger()
    assert isinstance(logger, WandbLogger)


def test_maybe_watch_model_with_none_logger():
    """Test watch does nothing when logger is None."""
    model = DummyModel()
    maybe_watch_model(None, model, None)  # Should not raise


def test_maybe_watch_model_with_none_model():
    """Test watch does nothing when model is None."""
    cfg = {"wandb": {"enabled": True, "mode": "disabled"}}
    logger, watch = create_wandb_logger(cfg, experiment_name="test")
    maybe_watch_model(logger, None, watch)  # Should not raise


def test_maybe_watch_model_with_none_watch_cfg():
    """Test watch does nothing when watch config is None."""
    model = DummyModel()
    cfg = {"wandb": {"enabled": True, "mode": "disabled"}}
    logger, _ = create_wandb_logger(cfg, experiment_name="test")
    maybe_watch_model(logger, model, None)  # Should not raise


def test_create_wandb_logger_with_config_payload():
    """Test logger creation with config payload."""
    cfg = {
        "wandb": {
            "enabled": True,
            "project": "test-project",
            "mode": "disabled",
            "config": {"learning_rate": 0.001, "batch_size": 32},
        }
    }
    logger, _ = create_wandb_logger(
        cfg,
        experiment_name="test",
        config={"model": "test-model"},
    )
    assert isinstance(logger, WandbLogger)


def test_create_notebook_logger_with_watch():
    """Test notebook logger with watch mode specified."""
    logger, watch = create_notebook_logger(
        project="test-project",
        name="test-run",
        watch="gradients",
    )
    assert isinstance(logger, WandbLogger)
    assert watch is not None
    assert watch.log == "gradients"
