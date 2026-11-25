"""Aggregate patient-level fold predictions into mean and standard deviation metrics.

Reads the per-fold prediction CSVs written by ``cadica-train`` and produces a
per-fold table plus a summary table. Use a unique ``--run-tag`` per condition and
seed so runs never overwrite each other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from cadica.config import CLASS_NAMES, N_SPLITS, RESULTS_DIR


def predictions_path(run_tag: str, fold: int, variant: str = "last") -> Path:
    """Return the prediction CSV path for a run tag, fold and reporting variant."""
    if variant == "best":
        stem = f"{run_tag}_best_patient_predictions_fold_{fold}"
    elif variant == "last":
        stem = f"{run_tag}_patient_predictions_fold_{fold}"
    else:
        raise ValueError(f"Unknown variant {variant!r}; expected 'last' or 'best'")
    return RESULTS_DIR / f"{stem}.csv"


def fold_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute patient-level metrics for one fold's prediction table."""
    y_true = predictions["actual"].astype(int)
    y_pred = predictions["predicted"].astype(int)
    labels = list(range(len(CLASS_NAMES)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    qwk = float(
        cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    )
    if pd.isna(qwk):
        qwk = 0.0

    metrics: dict[str, float] = {
        "patients": float(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "qwk": qwk,
        "ordinal_mae": float(mean_absolute_error(y_true, y_pred)),
    }

    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f"precision_{class_name}"] = float(precision[index])
        metrics[f"recall_{class_name}"] = float(recall[index])
        metrics[f"f1_{class_name}"] = float(f1[index])
        metrics[f"support_{class_name}"] = float(support[index])

    return metrics


def aggregate_folds(
    run_tag: str,
    variant: str = "last",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(per_fold, summary)`` tables for a run tag.

    ``per_fold`` has one row per fold. ``summary`` has one row per metric with
    ``mean`` and ``std`` columns across folds.
    """
    rows: dict[int, dict[str, float]] = {}

    for fold in range(1, N_SPLITS + 1):
        path = predictions_path(run_tag, fold, variant)

        if not path.exists():
            raise SystemExit(
                f"Missing predictions for fold {fold}: {path}\n"
                f"Run `cadica-train --fold {fold} --run-tag {run_tag}` first."
            )

        rows[fold] = fold_metrics(pd.read_csv(path))

    per_fold = pd.DataFrame.from_dict(rows, orient="index")
    per_fold.index.name = "fold"

    summary = per_fold.agg(["mean", "std"]).T.reset_index()
    summary = summary.rename(columns={"index": "metric"})

    return per_fold, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--run-tag",
        default="baseline",
        help="Run tag used when the folds were trained.",
    )

    parser.add_argument(
        "--variant",
        choices=("last", "best"),
        default="last",
        help=(
            "Which reporting variant to aggregate. 'last' is the fixed-budget "
            "headline; 'best' is the early-stopping/checkpoint result."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    per_fold, summary = aggregate_folds(args.run_tag, args.variant)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_fold_path = RESULTS_DIR / f"{args.run_tag}_{args.variant}_per_fold.csv"
    summary_path = RESULTS_DIR / f"{args.run_tag}_{args.variant}_summary.csv"

    per_fold.to_csv(per_fold_path, index=True)
    summary.to_csv(summary_path, index=False)

    print("=" * 55)
    print("CADICA FOLD AGGREGATION")
    print("=" * 55)
    print(f"Run tag: {args.run_tag}")
    print(f"Variant: {args.variant}")
    print(f"Folds: {len(per_fold)}")

    print("\nPer-fold patient-level metrics:")
    print(
        per_fold[["patients", "accuracy", "balanced_accuracy", "macro_f1"]]
        .round(4)
        .to_string()
    )

    print("\nMean +/- std across folds:")

    headline = summary[
        summary["metric"].isin(
            ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "ordinal_mae"]
        )
    ]

    for row in headline.itertuples(index=False):
        print(f"  {row.metric:<18} {row.mean:.4f} +/- {row.std:.4f}")

    print("\nPer-class recall:")
    recall_rows = summary[summary["metric"].str.startswith("recall_")]

    for row in recall_rows.itertuples(index=False):
        print(f"  {row.metric:<18} {row.mean:.4f} +/- {row.std:.4f}")

    print("\nSaved outputs:")
    print("Per-fold:", per_fold_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
