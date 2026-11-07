"""Print patient-level and per-fold class distributions."""

from __future__ import annotations

import pandas as pd

from cadica.config import PATIENT_FOLDS_PATH


def main() -> None:
    """Display the class balance of the patient-level fold table."""
    patients = pd.read_csv(PATIENT_FOLDS_PATH)

    print("PATIENT-LEVEL CLASS DISTRIBUTION")
    print(patients["severity"].value_counts().to_string())

    print("\nCLASS DISTRIBUTION PER FOLD")
    print(pd.crosstab(patients["fold"], patients["severity"]).to_string())


if __name__ == "__main__":
    main()
