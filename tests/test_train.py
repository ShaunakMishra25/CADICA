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


def test_should_stop_respects_min_epochs_and_patience() -> None:
    # min_epochs protects the LR schedule from stopping too early.
    assert not train.should_stop(
        epoch=19, epochs_without_improvement=100, patience=5, min_epochs=20
    )
    assert not train.should_stop(
        epoch=20, epochs_without_improvement=4, patience=5, min_epochs=20
    )
    assert train.should_stop(
        epoch=20, epochs_without_improvement=5, patience=5, min_epochs=20
    )


def test_compute_class_weights_use_patients_not_frames() -> None:
    train_df = pd.DataFrame(
        {
            "patient_id": [
                "p1", "p1", "p1", "p2", "p3", "p3",
                "p4", "p5", "p5", "p5", "p6",
            ],
            "severity": [
                "<20%", "<20%", "<20%", "<20%", "20-50%", "20-50%",
                ">70%", ">70%", ">70%", ">70%", ">70%",
            ],
        }
    )

    weights = train.compute_class_weights(train_df)

    # Patient counts are 2, 1, 3 -> middle class is the rarest by patient.
    assert float(weights.mean()) == pytest.approx(1.0)
    assert float(weights[1]) > float(weights[0]) > float(weights[2])


def test_write_predictions_roundtrip(tmp_path) -> None:
    results = {
        "p1": {"label": 0, "prediction": 1, "probabilities": [0.2, 0.7, 0.1]},
        "p2": {"label": 2, "prediction": 2, "probabilities": [0.05, 0.05, 0.9]},
    }

    path = tmp_path / "predictions.csv"
    train.write_predictions(path, results)

    table = pd.read_csv(path)

    assert list(table["patient_id"]) == ["p1", "p2"]
    assert list(table["actual"]) == [0, 2]
    assert list(table["predicted"]) == [1, 2]
    assert table["p_>70%"].tolist() == pytest.approx([0.1, 0.9])
