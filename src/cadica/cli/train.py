"""Train and evaluate the baseline CADICA severity classifier on one patient-level fold."""

from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import DataLoader, WeightedRandomSampler

from cadica.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    N_SPLITS,
    RESULTS_DIR,
    SEED,
    train_fold_path,
    val_fold_path,
)

from cadica.data import CADICADataset
from cadica.models import CADICACNN


# --------------------------------------------------
# 1. ARGUMENTS
# --------------------------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        choices=range(1, N_SPLITS + 1),
        help="Validation fold to hold out",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="cuda or cpu. Automatically detected if omitted.",
    )

    return parser.parse_args()


# --------------------------------------------------
# 2. REPRODUCIBILITY
# --------------------------------------------------


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# 3. PATIENT- AND CLASS-BALANCED SAMPLER
# --------------------------------------------------


def build_sampler(
    train_df: pd.DataFrame,
) -> WeightedRandomSampler:
    """
    Balance sampling across classes and patients.

    Each patient receives equal total weight within its class.
    Each class receives equal total sampling weight.
    """

    patient_counts = train_df["patient_id"].value_counts()

    patient_labels = train_df[["patient_id", "severity"]].drop_duplicates()

    class_counts = patient_labels["severity"].value_counts()

    patient_weights = {}

    for patient_id, severity in patient_labels.itertuples(index=False):
        patient_weights[patient_id] = (1.0 / class_counts[severity]) / patient_counts[
            patient_id
        ]

    sample_weights = [
        patient_weights[patient_id] for patient_id in train_df["patient_id"]
    ]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# --------------------------------------------------
# 4. PATIENT-LEVEL EVALUATION
# --------------------------------------------------


def evaluate_patient_level(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    """
    Predict each frame independently.

    Average frame probabilities belonging to the same patient.
    Return one prediction per patient.
    """

    model.eval()

    patient_predictions = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            ).cpu()

            for index, patient_id in enumerate(batch["patient_id"]):
                if patient_id not in patient_predictions:
                    patient_predictions[patient_id] = {
                        "probabilities": [],
                        "label": int(labels[index].item()),
                    }

                patient_predictions[patient_id]["probabilities"].append(
                    probabilities[index]
                )

    results = {}

    for patient_id, result in patient_predictions.items():
        average_probability = torch.stack(result["probabilities"]).mean(dim=0)

        results[patient_id] = {
            "label": result["label"],
            "prediction": int(average_probability.argmax().item()),
            "probabilities": average_probability.tolist(),
        }

    return results


# --------------------------------------------------
# 5. CALCULATE PATIENT-LEVEL METRICS
# --------------------------------------------------


def calculate_metrics(
    results: dict[str, dict[str, object]],
) -> dict[str, object]:

    y_true = [int(result["label"]) for result in results.values()]

    y_pred = [int(result["prediction"]) for result in results.values()]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
    }

    return metrics


# --------------------------------------------------
# 6. MAIN TRAINING FUNCTION
# --------------------------------------------------


def main() -> None:

    args = parse_args()

    set_seed(SEED)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 55)
    print("CADICA CNN TRAINING")
    print("=" * 55)

    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # --------------------------------------------------
    # 7. LOAD DATA
    # --------------------------------------------------

    train_csv = train_fold_path(args.fold)
    val_csv = val_fold_path(args.fold)

    if not train_csv.exists() or not val_csv.exists():
        raise SystemExit(
            "Missing fold manifests. Run `make pipeline` first.\n"
            f"Expected: {train_csv} and {val_csv}"
        )

    train_df = pd.read_csv(train_csv)

    train_dataset = CADICADataset(train_csv)
    val_dataset = CADICADataset(val_csv)

    print("\nDATASET INFORMATION")

    print("Training frames:", len(train_dataset))
    print("Validation frames:", len(val_dataset))

    print(
        "Training patients:",
        train_df["patient_id"].nunique(),
    )

    print(
        "Validation patients:",
        pd.read_csv(val_csv)["patient_id"].nunique(),
    )

    # --------------------------------------------------
    # 8. DATA LOADERS
    # --------------------------------------------------

    sampler = build_sampler(train_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --------------------------------------------------
    # 9. MODEL, LOSS AND OPTIMIZER
    # --------------------------------------------------

    model = CADICACNN(num_classes=len(CLASS_NAMES)).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    history = []

    # --------------------------------------------------
    # 10. TRAINING AND EPOCH-WISE MONITORING
    # --------------------------------------------------

    print("\nTRAINING STARTED")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_train_loss = total_loss / len(train_loader)

        # ----------------------------------------------
        # VALIDATION
        # ----------------------------------------------

        val_results = evaluate_patient_level(
            model,
            val_loader,
            device,
        )

        val_metrics = calculate_metrics(val_results)

        val_accuracy = val_metrics["accuracy"]

        val_balanced_accuracy = val_metrics["balanced_accuracy"]

        val_macro_f1 = val_metrics["macro_f1"]

        # ----------------------------------------------
        # SAVE EPOCH METRICS
        # ----------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy": (val_balanced_accuracy),
                "val_macro_f1": val_macro_f1,
            }
        )

        # ----------------------------------------------
        # PRINT EPOCH RESULTS
        # ----------------------------------------------

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Val Balanced Acc: {val_balanced_accuracy:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

    # --------------------------------------------------
    # 11. FINAL PATIENT-LEVEL EVALUATION
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("FINAL PATIENT-LEVEL VALIDATION")
    print("=" * 55)

    final_results = evaluate_patient_level(
        model,
        val_loader,
        device,
    )

    final_metrics = calculate_metrics(final_results)

    for patient_id, result in final_results.items():
        print(
            f"{patient_id}: Actual={result['label']}, Predicted={result['prediction']}"
        )

    print(
        "\nValidation patients:",
        len(final_results),
    )

    print(f"Patient-level accuracy: {final_metrics['accuracy']:.4f}")

    print(f"Patient-level balanced acc.: {final_metrics['balanced_accuracy']:.4f}")

    print(f"Patient-level macro F1: {final_metrics['macro_f1']:.4f}")

    print("\nConfusion matrix (actual rows, predicted columns):")

    print(final_metrics["confusion_matrix"])

    print("\nClassification report:")

    print(final_metrics["classification_report"])

    # --------------------------------------------------
    # 12. SAVE RESULTS
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save training history

    training_log_path = RESULTS_DIR / f"training_log_fold_{args.fold}.csv"

    pd.DataFrame(history).to_csv(
        training_log_path,
        index=False,
    )

    # Save patient predictions

    prediction_rows = []

    for patient_id, result in final_results.items():
        row = {
            "patient_id": patient_id,
            "actual": result["label"],
            "predicted": result["prediction"],
        }

        for class_name, probability in zip(
            CLASS_NAMES,
            result["probabilities"],
            strict=True,
        ):
            row[f"p_{class_name}"] = probability

        prediction_rows.append(row)

    prediction_path = RESULTS_DIR / f"patient_predictions_fold_{args.fold}.csv"

    pd.DataFrame(prediction_rows).to_csv(
        prediction_path,
        index=False,
    )

    # Save final-epoch checkpoint

    checkpoint_path = CHECKPOINTS_DIR / f"baseline_fold_{args.fold}.pt"

    torch.save(
        {
            "fold": args.fold,
            "epochs": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "final_metrics": {
                "accuracy": final_metrics["accuracy"],
                "balanced_accuracy": (final_metrics["balanced_accuracy"]),
                "macro_f1": final_metrics["macro_f1"],
            },
            "config": {
                "seed": SEED,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "num_classes": len(CLASS_NAMES),
                "class_names": CLASS_NAMES,
            },
        },
        checkpoint_path,
    )

    # --------------------------------------------------
    # 13. OUTPUT LOCATIONS
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("SAVED OUTPUTS")
    print("=" * 55)

    print("Training log:", training_log_path)

    print("Patient predictions:", prediction_path)

    print("Checkpoint:", checkpoint_path)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
