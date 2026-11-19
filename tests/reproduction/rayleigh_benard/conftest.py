"""Use the campaign's selected library checkout for its configuration fixtures."""

from pathlib import Path

import pytest

import autocast


@pytest.fixture
def REPO_ROOT():
    """Resolve configs from the same checkout as the imported scientific code."""
    return Path(autocast.__file__).resolve().parents[2]
