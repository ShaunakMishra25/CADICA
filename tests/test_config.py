"""Configuration and label-definition tests."""

from __future__ import annotations

from cadica import config


def test_label_map_matches_class_names() -> None:
    assert list(config.LABEL_MAP) == config.CLASS_NAMES
    assert list(config.LABEL_MAP.values()) == [0, 1, 2]
    assert config.NUM_CLASSES == 3


def test_derived_paths_live_under_data_dir() -> None:
    assert config.PATIENT_FOLDS_PATH.parent == config.FOLDS_DIR
    assert config.SAMPLED_FRAMES_PATH.parent == config.MANIFESTS_DIR
    assert config.FOLDS_DIR.is_relative_to(config.DATA_DIR)
    assert config.MANIFESTS_DIR.is_relative_to(config.DATA_DIR)


def test_fold_path_helpers() -> None:
    assert config.train_fold_path(1).name == "train_fold_1.csv"
    assert config.val_fold_path(5).name == "val_fold_5.csv"
    assert config.train_fold_path(1).parent == config.FOLDS_DIR
