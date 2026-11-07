"""Create leave-one-fold-out training and validation manifests."""

from __future__ import annotations

import pandas as pd

from cadica.config import (
    FOLDS_DIR,
    N_SPLITS,
    SAMPLED_FRAMES_PATH,
    train_fold_path,
    val_fold_path,
)


def main() -> None:
    """Split the sampling manifest into one train/validation pair per fold."""
    sampled = pd.read_csv(SAMPLED_FRAMES_PATH)
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)

    for fold in range(1, N_SPLITS + 1):
        val_df = sampled[sampled["fold"] == fold].copy()
        train_df = sampled[sampled["fold"] != fold].copy()

        if val_df.empty:
            raise ValueError(f"Fold {fold} has no validation frames")

        train_patients = set(train_df["patient_id"])
        val_patients = set(val_df["patient_id"])
        if not train_patients.isdisjoint(val_patients):
            raise AssertionError(
                f"Patient leakage detected in fold {fold}: "
                f"{sorted(train_patients & val_patients)}"
            )

        train_df.to_csv(train_fold_path(fold), index=False)
        val_df.to_csv(val_fold_path(fold), index=False)

        print(f"\nFold {fold}")
        print(f"  Training frames:   {len(train_df)}")
        print(f"  Validation frames: {len(val_df)}")
        print(f"  Training patients:   {train_df['patient_id'].nunique()}")
        print(f"  Validation patients: {val_df['patient_id'].nunique()}")

    print("\nAll folds prepared successfully.")


if __name__ == "__main__":
    main()
