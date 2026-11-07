"""Sanity-check the raw CADICA dataset layout and metadata coverage."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from cadica.config import DATA_ROOT, METADATA_PATH, SELECTED_ROOT, TARGET_COL
from cadica.data.loader import collect_image_paths


def main() -> None:
    print(f"Dataset root: {DATA_ROOT}")
    print("Selected videos folder exists:", SELECTED_ROOT.exists())

    if not SELECTED_ROOT.exists():
        raise SystemExit(f"Missing dataset folder: {SELECTED_ROOT}")

    patient_dirs = [
        path
        for path in SELECTED_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("p")
    ]
    print("Patient folders:", len(patient_dirs))

    images = collect_image_paths()
    print("PNG frames found:", len(images))

    counts = Counter(images["patient_id"])
    print("Unique patients represented:", len(counts))
    print("Frames for first 5 patients:")
    for patient_id, count in sorted(counts.items())[:5]:
        print(" ", patient_id, count)

    metadata = pd.read_excel(METADATA_PATH)
    metadata["patient_id"] = metadata["Patient ID"].apply(lambda x: f"p{int(x)}")

    print("\nMetadata shape:", metadata.shape)
    print("Metadata columns:", metadata.columns.tolist())
    print(metadata.head().to_string())

    print("\nUnique metadata patients:", metadata["patient_id"].nunique())
    print("Missing patient IDs:", metadata["patient_id"].isna().sum())
    print("\nSeverity counts:")
    print(metadata[TARGET_COL].value_counts(dropna=False).to_string())

    image_patients = set(counts)
    metadata_patients = set(metadata["patient_id"])

    print(
        "\nPatients in images but not metadata:",
        sorted(image_patients - metadata_patients),
    )
    print(
        "Patients in metadata but not images:",
        sorted(metadata_patients - image_patients),
    )


if __name__ == "__main__":
    main()
