"""Build patient-level stratified cross-validation folds."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from cadica.config import N_SPLITS, PATIENT_FOLDS_PATH, SEED
from cadica.data.loader import build_patient_index


def main() -> None:
    """Assign every patient to exactly one stratified fold and save the table."""
    patients = build_patient_index()

    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )

    patients["fold"] = -1
    for fold, (_, val_indices) in enumerate(
        splitter.split(patients["patient_id"], patients["severity"]),
        start=1,
    ):
        patients.loc[val_indices, "fold"] = fold

    PATIENT_FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    patients.to_csv(PATIENT_FOLDS_PATH, index=False)

    print("PATIENTS PER FOLD")
    print(patients["fold"].value_counts().sort_index().to_string())

    print("\nCLASS DISTRIBUTION PER FOLD")
    print(pd.crosstab(patients["fold"], patients["severity"]).to_string())

    print(f"\nTOTAL PATIENTS: {patients['patient_id'].nunique()}")
    print(f"Fold assignments saved to {PATIENT_FOLDS_PATH}")


if __name__ == "__main__":
    main()
