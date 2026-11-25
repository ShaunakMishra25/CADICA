"""Tests for fold aggregation of patient-level predictions."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from cadica.cli import aggregate  # noqa: E402
from cadica.config import CLASS_NAMES, N_SPLITS  # noqa: E402


def write_predictions(path, actual, predicted) -> None:
    pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(len(actual))],
            "actual": actual,
            "predicted": predicted,
        }
    ).to_csv(path, index=False)


def test_predictions_path_variants(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(aggregate, "RESULTS_DIR", tmp_path)

    last = aggregate.predictions_path("e2", 3, "last")
    best = aggregate.predictions_path("e2", 3, "best")

    assert last.name == "e2_patient_predictions_fold_3.csv"
    assert best.name == "e2_best_patient_predictions_fold_3.csv"

    with pytest.raises(ValueError):
        aggregate.predictions_path("e2", 3, "other")


def test_fold_metrics_reports_per_class_columns() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [0, 0, 1, 2],
            "predicted": [0, 1, 1, 2],
        }
    )

    metrics = aggregate.fold_metrics(predictions)

    assert metrics["patients"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(
        (2 / 3 + 2 / 3 + 1.0) / 3, rel=1e-6
    )
    for class_name in CLASS_NAMES:
        assert f"recall_{class_name}" in metrics
        assert f"support_{class_name}" in metrics

    # Ordinal metrics: one adjacent error out of four patients.
    assert metrics["ordinal_mae"] == pytest.approx(0.25)
    assert metrics["qwk"] < 1.0


def test_aggregate_folds_mean_and_std(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(aggregate, "RESULTS_DIR", tmp_path)

    perfect = ([0, 1, 2], [0, 1, 2])
    mistakes = ([0, 0, 1, 2], [0, 1, 1, 2])

    for fold in range(1, N_SPLITS + 1):
        actual, predicted = mistakes if fold == 1 else perfect
        write_predictions(
            aggregate.predictions_path("run", fold, "last"), actual, predicted
        )

    per_fold, summary = aggregate.aggregate_folds("run", "last")

    assert len(per_fold) == N_SPLITS
    assert per_fold.loc[1, "accuracy"] == pytest.approx(0.75)
    assert per_fold.loc[1, "support_<20%"] == pytest.approx(2)
    assert per_fold.loc[1, "ordinal_mae"] == pytest.approx(0.25)

    accuracy_row = summary[summary["metric"] == "accuracy"].iloc[0]
    assert accuracy_row["mean"] == pytest.approx((0.75 + 4) / 5)
    assert accuracy_row["std"] == pytest.approx(0.1118, abs=1e-3)


def test_aggregate_folds_fails_clearly_when_a_fold_is_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(aggregate, "RESULTS_DIR", tmp_path)

    with pytest.raises(SystemExit, match="Missing predictions for fold 1"):
        aggregate.aggregate_folds("absent", "last")
