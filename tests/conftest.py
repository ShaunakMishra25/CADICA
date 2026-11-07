"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the src-layout package importable without an editable install.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def manifest() -> pd.DataFrame:
    """Load the frame sampling manifest, skipping when it is absent."""
    from cadica.config import SAMPLED_FRAMES_PATH

    if not SAMPLED_FRAMES_PATH.exists():
        pytest.skip(f"Missing manifest: {SAMPLED_FRAMES_PATH}")
    return pd.read_csv(SAMPLED_FRAMES_PATH)


@pytest.fixture(scope="session")
def valid_manifest_indices(manifest: pd.DataFrame) -> list[int]:
    """Return manifest indices whose image files exist on this machine."""
    from cadica.data import resolve_image_path

    indices = [
        index
        for index, path in enumerate(manifest["image_path"])
        if resolve_image_path(path).exists()
    ]
    if not indices:
        pytest.skip("No manifest image paths exist on this machine")
    return indices
