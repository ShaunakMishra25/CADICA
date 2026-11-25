"""Train and evaluate the baseline CADICA severity classifier on one patient-level fold."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from cadica.data import CADICADataset, build_eval_transform, build_train_transform
from cadica.models import CADICACNN, FocalLoss

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
        "--run-tag",
        default="baseline",
        help=(
            "Prefix for output files. Use a unique tag per condition and seed, "
            "for example e2_aug_s42."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Maximum epochs. This is the fixed budget when early stopping is off.",
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

    # --------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------

    parser.add_argument(
        "--scheduler",
        choices=("plateau", "none"),
        default="plateau",
        help="Learning-rate scheduler. Plateau monitors patient-level macro F1.",
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.3,
        help="Multiplicative factor applied when the plateau scheduler fires.",
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="Epochs without improvement before the LR is reduced.",
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
    )

    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------

    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable early stopping on patient-level macro F1. Disable for the "
            "fixed-budget comparison protocol."
        ),
    )

    parser.add_argument(
        "--es-patience",
        type=int,
        default=10,
        help="Epochs without improvement before training stops.",
    )

    parser.add_argument(
        "--es-min-delta",
        type=float,
        default=0.02,
        help="Minimum macro-F1 improvement that counts as progress.",
    )

    parser.add_argument(
        "--es-min-epochs",
        type=int,
        default=20,
        help="Do not stop before this epoch, so the LR schedule has a chance.",
    )

    # --------------------------------------------------
    # Imbalance handling and preprocessing
    # --------------------------------------------------

    parser.add_argument(
        "--intensity-norm",
        choices=("none", "standardize"),
        default="none",
        help=(
            "Per-image intensity normalization applied to both splits. "
            "'standardize' removes per-frame exposure differences."
        ),
    )

    parser.add_argument(
        "--sampler",
        choices=("balanced", "none"),
        default="balanced",
        help="'balanced' uses the patient- and class-balanced WeightedRandomSampler.",
    )

    parser.add_argument(
        "--loss",
        choices=("ce", "weighted_ce", "focal"),
        default="ce",
        help="Cross-entropy, class-weighted cross-entropy, or focal loss.",
    )

    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter for focal loss.",
    )

    # --------------------------------------------------
    # Augmentation (training split only)
    # --------------------------------------------------

    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply on-the-fly augmentation to training frames.",
    )

    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=7.0,
        help="Maximum absolute rotation in degrees. Keep within 5-10.",
    )

    parser.add_argument(
        "--translate",
        type=float,
        default=0.05,
        help="Maximum translation as a fraction of image size.",
    )

    parser.add_argument(
        "--brightness",
        type=float,
        default=0.15,
        help="Brightness jitter magnitude in [0, 1].",
    )

    parser.add_argument(
        "--contrast",
        type=float,
        default=0.15,
        help="Contrast jitter magnitude in [0, 1].",
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Gaussian noise standard deviation in [0, 1] intensity units.",
    )

    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of horizontal flip. Defaults to 0 because flips change "
            "coronary laterality; enable only after an ablation."
        ),
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
# 6. TRAINING CONTROL
# --------------------------------------------------


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU copy of a model's state, detached from the live parameters."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def should_stop(
    epoch: int,
    epochs_without_improvement: int,
    patience: int,
    min_epochs: int,
) -> bool:
    """Return True when early stopping should end training after ``epoch``."""
    return epoch >= min_epochs and epochs_without_improvement >= patience


def write_predictions(path: Path, results: dict[str, dict[str, object]]) -> None:
    """Write patient-level predictions and class probabilities to CSV."""
    rows = []

    for patient_id, result in results.items():
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

        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def scalar_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """Return only the numeric metrics that belong in a checkpoint or log."""
    return {
        key: float(metrics[key])
        for key in ("accuracy", "balanced_accuracy", "macro_f1")
    }


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Inverse patient-frequency class weights in ``CLASS_NAMES`` order.

    The unit of analysis is the patient, so counts are taken over unique
    patients, not frames. Weights are normalised to mean 1.
    """
    patient_labels = train_df[["patient_id", "severity"]].drop_duplicates()
    counts = patient_labels["severity"].value_counts()

    inverse = {
        name: 1.0 / max(int(counts.get(name, 0)), 1) for name in CLASS_NAMES
    }
    mean_inverse = sum(inverse.values()) / len(inverse)

    return torch.tensor(
        [inverse[name] / mean_inverse for name in CLASS_NAMES],
        dtype=torch.float32,
    )


# --------------------------------------------------
# 7. MAIN TRAINING FUNCTION
# --------------------------------------------------


def main() -> None:

    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 55)
    print("CADICA CNN TRAINING")
    print("=" * 55)

    print(f"Device: {device}")
    print(f"Run tag: {args.run_tag}")
    print(f"Seed: {args.seed}")
    print(f"Fold: {args.fold}")
    print(f"Max epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    print("\nOPTIMIZATION CONTROL")

    print(f"Scheduler: {args.scheduler}")

    print(f"Sampler: {args.sampler}")
    print(f"Loss: {args.loss}")
    if args.loss == "focal":
        print(f"  focal gamma={args.focal_gamma}")
    print(f"Intensity normalization: {args.intensity_norm}")

    if args.scheduler == "plateau":
        print(f"  factor={args.lr_factor} patience={args.lr_patience} min_lr={args.min_lr}")

    print(f"Early stopping: {args.early_stopping}")

    if args.early_stopping:
        print(
            f"  patience={args.es_patience} min_delta={args.es_min_delta} "
            f"min_epochs={args.es_min_epochs}"
        )
        print(
            "  Note: early-stopped epochs are selected on the validation fold. "
            "Use --no-early-stopping for fixed-budget comparisons."
        )

    # --------------------------------------------------
    # 8. LOAD DATA
    # --------------------------------------------------

    train_csv = train_fold_path(args.fold)
    val_csv = val_fold_path(args.fold)

    if not train_csv.exists() or not val_csv.exists():
        raise SystemExit(
            "Missing fold manifests. Run `make pipeline` first.\n"
            f"Expected: {train_csv} and {val_csv}"
        )

    train_df = pd.read_csv(train_csv)

    val_transform = build_eval_transform(intensity_norm=args.intensity_norm)

    if args.augment:
        train_transform = build_train_transform(
            rotation_degrees=args.rotation_degrees,
            translate=args.translate,
            brightness=args.brightness,
            contrast=args.contrast,
            gaussian_std=args.noise_std,
            horizontal_flip_p=args.hflip_prob,
            intensity_norm=args.intensity_norm,
        )
    else:
        train_transform = val_transform

    train_dataset = CADICADataset(train_csv, transform=train_transform)
    val_dataset = CADICADataset(val_csv, transform=val_transform)

    print("\nDATASET INFORMATION")

    print("Training frames:", len(train_dataset))
    print("Validation frames:", len(val_dataset))

    print("\nAUGMENTATION")

    if args.augment:
        print(f"Rotation: +/-{args.rotation_degrees} degrees")
        print(f"Translation: +/-{args.translate:.2%} of image size")
        print(f"Brightness/contrast jitter: {args.brightness}/{args.contrast}")
        print(f"Gaussian noise std: {args.noise_std}")
        print(f"Horizontal flip probability: {args.hflip_prob}")
    else:
        print("Disabled (evaluation transform applied to both splits)")

    print(
        "Training patients:",
        train_df["patient_id"].nunique(),
    )

    print(
        "Validation patients:",
        pd.read_csv(val_csv)["patient_id"].nunique(),
    )

    # --------------------------------------------------
    # 9. DATA LOADERS
    # --------------------------------------------------

    sampler = build_sampler(train_df) if args.sampler == "balanced" else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # --------------------------------------------------
    # 10. MODEL, LOSS, OPTIMIZER AND SCHEDULER
    # --------------------------------------------------

    model = CADICACNN(num_classes=len(CLASS_NAMES)).to(device)

    class_weights = compute_class_weights(train_df).to(device)

    if args.loss == "ce":
        criterion: nn.Module = nn.CrossEntropyLoss()
    elif args.loss == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)

    weight_text = ", ".join(
        f"{name}={weight:.2f}"
        for name, weight in zip(CLASS_NAMES, class_weights.tolist(), strict=True)
    )
    print(f"\nCLASS WEIGHTS (patient-level): {weight_text}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = None

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    history = []

    best_metric = float("-inf")
    best_epoch = 0
    best_state = clone_state_dict(model)
    epochs_without_improvement = 0
    stopped_early = False

    # --------------------------------------------------
    # 11. TRAINING AND EPOCH-WISE MONITORING
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

        current_lr = optimizer.param_groups[0]["lr"]

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
        # BEST-CHECKPOINT TRACKING
        # ----------------------------------------------

        is_best = val_macro_f1 > best_metric + args.es_min_delta

        if is_best:
            best_metric = val_macro_f1
            best_epoch = epoch
            best_state = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(val_macro_f1)

        # ----------------------------------------------
        # SAVE EPOCH METRICS
        # ----------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": average_train_loss,
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy": (val_balanced_accuracy),
                "val_macro_f1": val_macro_f1,
                "is_best": is_best,
            }
        )

        # ----------------------------------------------
        # PRINT EPOCH RESULTS
        # ----------------------------------------------

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Val Balanced Acc: {val_balanced_accuracy:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
            f"{' | best' if is_best else ''}"
        )

        # ----------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------

        if args.early_stopping and should_stop(
            epoch,
            epochs_without_improvement,
            args.es_patience,
            args.es_min_epochs,
        ):
            stopped_early = True
            print(
                f"\nEarly stopping at epoch {epoch}: no improvement for "
                f"{epochs_without_improvement} epochs "
                f"(patience {args.es_patience})."
            )
            break

    # --------------------------------------------------
    # 12. FINAL PATIENT-LEVEL EVALUATION
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("FINAL PATIENT-LEVEL VALIDATION")
    print("=" * 55)

    # Fixed-budget / last-epoch result: the unbiased headline comparison.
    last_results = evaluate_patient_level(
        model,
        val_loader,
        device,
    )

    last_metrics = calculate_metrics(last_results)

    # Preserve the final weights before overwriting them with the best state.
    last_state = clone_state_dict(model)

    # Best-epoch result: restore the best weights before scoring.
    model.load_state_dict(best_state)

    best_results = evaluate_patient_level(
        model,
        val_loader,
        device,
    )

    best_metrics = calculate_metrics(best_results)

    print(
        f"\nLast epoch: {len(history)}"
        f"{' (early stopped)' if stopped_early else ' (fixed budget)'}"
    )

    print(f"  accuracy         : {last_metrics['accuracy']:.4f}")
    print(f"  balanced accuracy: {last_metrics['balanced_accuracy']:.4f}")
    print(f"  macro F1         : {last_metrics['macro_f1']:.4f}")

    print(f"\nBest epoch: {best_epoch} (macro F1 {best_metric:.4f})")

    print(f"  accuracy         : {best_metrics['accuracy']:.4f}")
    print(f"  balanced accuracy: {best_metrics['balanced_accuracy']:.4f}")
    print(f"  macro F1         : {best_metrics['macro_f1']:.4f}")

    print("\nBest-epoch confusion matrix (actual rows, predicted columns):")

    print(best_metrics["confusion_matrix"])

    print("\nBest-epoch classification report:")

    print(best_metrics["classification_report"])

    # --------------------------------------------------
    # 13. SAVE RESULTS
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

    training_log_path = RESULTS_DIR / f"{args.run_tag}_training_log_fold_{args.fold}.csv"

    pd.DataFrame(history).to_csv(
        training_log_path,
        index=False,
    )

    # Save patient predictions for both reporting variants

    last_prediction_path = (
        RESULTS_DIR / f"{args.run_tag}_patient_predictions_fold_{args.fold}.csv"
    )

    best_prediction_path = (
        RESULTS_DIR / f"{args.run_tag}_best_patient_predictions_fold_{args.fold}.csv"
    )

    write_predictions(last_prediction_path, last_results)

    write_predictions(best_prediction_path, best_results)

    # Shared provenance for both checkpoints

    config = {
        "run_tag": args.run_tag,
        "seed": args.seed,
        "fold": args.fold,
        "max_epochs": args.epochs,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "min_lr": args.min_lr,
        "early_stopping": args.early_stopping,
        "es_patience": args.es_patience,
        "es_min_delta": args.es_min_delta,
        "es_min_epochs": args.es_min_epochs,
        "intensity_norm": args.intensity_norm,
        "sampler": args.sampler,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "class_weights": class_weights.tolist(),
        "augment": args.augment,
        "rotation_degrees": args.rotation_degrees,
        "translate": args.translate,
        "brightness": args.brightness,
        "contrast": args.contrast,
        "noise_std": args.noise_std,
        "hflip_prob": args.hflip_prob,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
    }

    # Last-epoch checkpoint: full training state.

    last_checkpoint_path = (
        CHECKPOINTS_DIR / f"{args.run_tag}_fold_{args.fold}_last.pt"
    )

    torch.save(
        {
            "fold": args.fold,
            "epoch": len(history),
            "model_state_dict": last_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "final_metrics": scalar_metrics(last_metrics),
            "config": config,
        },
        last_checkpoint_path,
    )

    # Best-epoch checkpoint: weights selected by patient-level macro F1.

    best_checkpoint_path = (
        CHECKPOINTS_DIR / f"{args.run_tag}_fold_{args.fold}_best.pt"
    )

    torch.save(
        {
            "fold": args.fold,
            "epoch": best_epoch,
            "model_state_dict": best_state,
            "final_metrics": scalar_metrics(best_metrics),
            "config": config,
        },
        best_checkpoint_path,
    )

    # --------------------------------------------------
    # 14. OUTPUT LOCATIONS
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("SAVED OUTPUTS")
    print("=" * 55)

    print("Training log:", training_log_path)
    print("Last predictions:", last_prediction_path)
    print("Best predictions:", best_prediction_path)
    print("Last checkpoint:", last_checkpoint_path)
    print("Best checkpoint:", best_checkpoint_path)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
