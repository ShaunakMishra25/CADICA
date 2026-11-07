"""Verify patient-level fold integrity and absence of leakage."""

from __future__ import annotations

import pandas as pd

from cadica.config import (
    N_SPLITS,
    SAMPLED_FRAMES_PATH,
    train_fold_path,
    val_fold_path,
)


def main() -> None:
    """Run integrity checks over the sampling manifest and fold splits."""
    sampled = pd.read_csv(SAMPLED_FRAMES_PATH)

    patient_fold_counts = sampled.groupby("patient_id")["fold"].nunique()
    repeated = patient_fold_counts[patient_fold_counts > 1]
    if not repeated.empty:
        raise AssertionError(
            f"Patients assigned to multiple folds:\n{repeated.to_string()}"
        )

    expected_folds = set(range(1, N_SPLITS + 1))
    observed_folds = set(sampled["fold"].unique())
    if observed_folds != expected_folds:
        raise AssertionError(
            f"Expected folds {sorted(expected_folds)}, found {sorted(observed_folds)}"
        )

    print("Patients per fold:")
    print(sampled.groupby("fold")["patient_id"].nunique().to_string())

    print("\nVideos per fold:")
    print(sampled.groupby("fold")["video_id"].nunique().to_string())

    print("\nFrames per patient:")
    print(sampled.groupby("patient_id").size().describe().to_string())

    all_patients = set(sampled["patient_id"])

    for fold in range(1, N_SPLITS + 1):
        train_df = pd.read_csv(train_fold_path(fold))
        val_df = pd.read_csv(val_fold_path(fold))

        train_patients = set(train_df["patient_id"])
        val_patients = set(val_df["patient_id"])

        if not train_patients.isdisjoint(val_patients):
            raise AssertionError(
                f"Patient leakage in fold {fold}: "
                f"{sorted(train_patients & val_patients)}"
            )
        if set(train_df["fold"]) & {fold}:
            raise AssertionError(f"Fold {fold} appears in its own training split")
        if set(val_df["fold"]) != {fold}:
            raise AssertionError(
                f"Validation split {fold} contains other folds: "
                f"{sorted(set(val_df['fold']))}"
            )
        if train_patients | val_patients != all_patients:
            raise AssertionError(
                f"Fold {fold} splits do not cover every patient"
            )

    print(f"\nAll patient-level fold checks passed for {len(all_patients)} patients.")


if __name__ == "__main__":
    main()
