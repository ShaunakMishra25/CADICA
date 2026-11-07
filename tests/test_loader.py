"""Raw dataset indexing tests."""

from __future__ import annotations

import pytest

from cadica import config
from cadica.data.loader import build_patient_index, normalize_severity


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<20%", "<20%"),
        (" 20-50% ", "20-50%"),
        ("> 70 %", ">70%"),
    ],
)
def test_normalize_severity(raw: str, expected: str) -> None:
    assert normalize_severity(raw) == expected


def test_normalize_severity_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        normalize_severity("50-70%")


def test_patient_index_schema() -> None:
    if not config.METADATA_PATH.exists():
        pytest.skip(f"Missing metadata workbook: {config.METADATA_PATH}")

    patients = build_patient_index()

    assert list(patients.columns) == ["patient_id", "image_count", "severity"]
    assert not patients.empty
    assert set(patients["severity"]) <= set(config.LABEL_MAP)
    assert patients["patient_id"].is_unique
