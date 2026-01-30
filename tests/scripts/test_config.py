"""Unit tests for autocast.scripts.config module."""

import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from autocast.scripts.config import save_resolved_config


def test_save_resolved_config_saves_yaml_file():
    cfg = OmegaConf.create({"key": "value", "nested": {"a": 1}})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_resolved_config(cfg, Path(tmpdir), "test.yaml")
        assert path.exists()
        assert path.name == "test.yaml"


def test_save_resolved_config_resolves_interpolations():
    cfg = OmegaConf.create({"base": 10, "derived": "${base}"})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_resolved_config(cfg, Path(tmpdir))
        content = path.read_text()
        assert "derived: 10" in content
