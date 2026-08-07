"""Evaluate the saved SteelCNN checkpoint on the held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from steel_defect.dataset import SteelDataset, build_file_list, create_splits
from steel_defect.model import SteelCNN
from steel_defect.preprocessing import build_val_transforms
from steel_defect.utils import CHECKPOINT_PATH, CLASS_NAMES, DEVICE, NUM_CLASSES


def evaluate(checkpoint_path: Path = CHECKPOINT_PATH, batch_size: int = 32) -> dict:
    """Return accuracy, per-class metrics, and confusion matrix for the test set."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train first.")

    _, _, test_list = create_splits(build_file_list())
    test_loader = DataLoader(
        SteelDataset(test_list, transform=build_val_transforms()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = SteelCNN(num_classes=checkpoint.get("num_classes", NUM_CLASSES)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    expected: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())
            expected.extend(labels.tolist())

    labels = list(range(NUM_CLASSES))
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "validation_accuracy_at_checkpoint": float(checkpoint.get("best_val_acc", 0.0)),
        "test_samples": len(expected),
        "test_accuracy": float(accuracy_score(expected, predicted)),
        "class_names": CLASS_NAMES,
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=labels,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
