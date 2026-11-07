"""Training CLI import and sampling tests."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from cadica.cli import train  # noqa: E402


def test_train_cli_is_importable() -> None:
    """Guard against the old `from dataset import ...` style imports."""
    assert callable(train.main)
    assert callable(train.build_sampler)
    assert callable(train.evaluate_patient_level)


def test_build_sampler_weights_every_frame() -> None:
    train_df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3", "p3", "p3"],
            "severity": ["<20%", "<20%", "20-50%", ">70%", ">70%", ">70%"],
        }
    )

    sampler = train.build_sampler(train_df)
    weights = list(sampler.weights)

    assert len(weights) == len(train_df)
    assert all(weight > 0 for weight in weights)

    # Each patient should contribute the same total weight.
    assert weights[0] + weights[1] == pytest.approx(weights[2])
    assert weights[0] + weights[1] == pytest.approx(sum(weights[3:]))
