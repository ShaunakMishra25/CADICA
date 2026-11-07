"""Patient-level fold integrity tests."""

from __future__ import annotations

import pandas as pd
import pytest

from cadica import config


@pytest.fixture(scope="module")
def patient_folds() -> pd.DataFrame:
    if not config.PATIENT_FOLDS_PATH.exists():
        pytest.skip(f"Missing fold table: {config.PATIENT_FOLDS_PATH}")
    return pd.read_csv(config.PATIENT_FOLDS_PATH)


def test_fold_table_schema(patient_folds: pd.DataFrame) -> None:
    assert {"patient_id", "severity", "fold"} <= set(patient_folds.columns)
    assert not patient_folds.empty


def test_each_patient_belongs_to_one_fold(patient_folds: pd.DataFrame) -> None:
    counts = patient_folds.groupby("patient_id")["fold"].nunique()
    assert counts.max() == 1


def test_folds_are_complete(patient_folds: pd.DataFrame) -> None:
    expected = set(range(1, config.N_SPLITS + 1))
    assert set(patient_folds["fold"]) == expected


def test_severity_labels_are_supported(patient_folds: pd.DataFrame) -> None:
    assert set(patient_folds["severity"]) <= set(config.LABEL_MAP)


def test_split_files_have_no_patient_leakage() -> None:
    split_paths = [
        path
        for fold in range(1, config.N_SPLITS + 1)
        for path in (config.train_fold_path(fold), config.val_fold_path(fold))
    ]
    if not all(path.exists() for path in split_paths):
        pytest.skip("Fold split manifests have not been generated yet")

    all_patients: set[str] = set()
    for fold in range(1, config.N_SPLITS + 1):
        train = pd.read_csv(config.train_fold_path(fold))
        val = pd.read_csv(config.val_fold_path(fold))

        train_patients = set(train["patient_id"])
        val_patients = set(val["patient_id"])

        assert train_patients.isdisjoint(val_patients)
        assert set(val["fold"]) == {fold}
        assert fold not in set(train["fold"])

        all_patients |= train_patients | val_patients

    assert all_patients
