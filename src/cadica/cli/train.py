"""Train the baseline CADICA severity classifier on one patient-level fold."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        choices=range(1, N_SPLITS + 1),
        help="Validation fold to hold out (default: 1)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda or cpu (default: auto-detect)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """Balance classes and patients by weighting each frame inversely.

    Every patient contributes the same total weight, and every class
    contributes the same total weight, regardless of how many frames or
    patients it contains.
    """
    patient_counts = train_df["patient_id"].value_counts()
    class_counts = train_df.drop_duplicates("patient_id")["severity"].value_counts()

    patient_weights = {}
    for patient_id, severity in (
        train_df[["patient_id", "severity"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        patient_weights[patient_id] = (
            1.0 / class_counts[severity]
        ) / patient_counts[patient_id]

    sample_weights = [
        patient_weights[patient_id] for patient_id in train_df["patient_id"]
    ]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def evaluate_patient_level(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    """Average frame probabilities per patient and return patient predictions."""
    model.eval()
    patient_predictions: dict[str, dict[str, object]] = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            probabilities = torch.softmax(model(images), dim=1).cpu()

            for index, patient_id in enumerate(batch["patient_id"]):
                if patient_id not in patient_predictions:
                    patient_predictions[patient_id] = {
                        "probabilities": [],
                        "label": int(labels[index].item()),
                    }
                patient_predictions[patient_id]["probabilities"].append(
                    probabilities[index]
                )

    results: dict[str, dict[str, object]] = {}
    for patient_id, result in patient_predictions.items():
        average_probability = torch.stack(result["probabilities"]).mean(dim=0)
        results[patient_id] = {
            "label": result["label"],
            "prediction": int(average_probability.argmax().item()),
            "probabilities": average_probability.tolist(),
        }
    return results


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=build_sampler(train_df),
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CADICACNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        history.append({"epoch": epoch, "train_loss": average_loss})
        print(f"Epoch [{epoch}/{args.epochs}] Training Loss: {average_loss:.4f}")

    results = evaluate_patient_level(model, val_loader, device)

    y_true = [int(result["label"]) for result in results.values()]
    y_pred = [int(result["prediction"]) for result in results.values()]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    print("\nPATIENT-LEVEL VALIDATION")
    for patient_id, result in results.items():
        print(
            f"{patient_id}: Actual={result['label']}, "
            f"Predicted={result['prediction']}"
        )

    print(f"\nValidation patients: {len(results)}")
    print(f"Patient-level accuracy:        {accuracy:.4f}")
    print(f"Patient-level balanced acc.:   {balanced_accuracy:.4f}")
    print(f"Patient-level macro F1:        {macro_f1:.4f}")
    print("\nConfusion matrix (actual rows, predicted columns):")
    print(matrix)
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / f"training_log_fold_{args.fold}.csv", index=False
    )

    prediction_rows = []
    for patient_id, result in results.items():
        row = {
            "patient_id": patient_id,
            "actual": result["label"],
            "predicted": result["prediction"],
        }
        for class_name, probability in zip(
            CLASS_NAMES, result["probabilities"], strict=True
        ):
            row[f"p_{class_name}"] = probability
        prediction_rows.append(row)

    pd.DataFrame(prediction_rows).to_csv(
        RESULTS_DIR / f"patient_predictions_fold_{args.fold}.csv", index=False
    )

    checkpoint_path = CHECKPOINTS_DIR / f"baseline_fold_{args.fold}.pt"
    torch.save(
        {
            "fold": args.fold,
            "epochs": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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

    print(f"\nTraining log:  {RESULTS_DIR / f'training_log_fold_{args.fold}.csv'}")
    print(
        f"Predictions:   {RESULTS_DIR / f'patient_predictions_fold_{args.fold}.csv'}"
    )
    print(f"Checkpoint:    {checkpoint_path}")


if __name__ == "__main__":
    main()
