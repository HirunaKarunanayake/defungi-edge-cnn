"""
evaluate.py
Owner: Member 3 (Evaluation & Metrics)

Loads a saved checkpoint and evaluates it on the test set.
Reports:
  - Overall accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrix (saved as PNG)

Usage:
  python src/evaluate.py --checkpoint checkpoints/model_a_adam_best.pth
"""

import os
import sys
import argparse
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data_loader import get_dataloaders
from model_a import ModelA
from model_b import ModelB


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint. Returns (model, class_names, meta)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "a")
    class_names = ckpt.get("class_names", ["H1", "H2", "H3", "H5", "H6"])

    if model_name == "a":
        model = ModelA(num_classes=len(class_names))
    elif model_name == "b":
        model = ModelB(num_classes=len(class_names))
    else:
        # SOTA model - load via torchvision
        import torchvision.models as tvm
        if "mobilenet" in model_name:
            model = tvm.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
        elif "squeezenet" in model_name:
            model = tvm.squeezenet1_1(weights=None)
            model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=1)
            model.num_classes = len(class_names)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, ckpt


def evaluate_model(checkpoint_path: str, plot_dir: str = "./plots",
                   batch_size: int = 32, num_workers: int = 0):
    """Evaluate a saved model checkpoint on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names, ckpt = load_model(checkpoint_path, device)
    model_name = ckpt.get("model_name", "?")
    optimizer_name = ckpt.get("optimizer_name", "?")
    tag = f"model_{model_name}_{optimizer_name}"

    _, _, test_loader, _ = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"  Evaluation: {tag}")
    print(f"{'='*60}")
    print(f"  Test Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=4))

    # Confusion matrix plot
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix – {tag}\n(Test Acc: {acc*100:.2f}%)")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(plot_dir, f"{tag}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved: {cm_path}")

    # Save metrics as JSON for report generation
    metrics = {
        "tag": tag,
        "test_accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    json_path = os.path.join(plot_dir, f"{tag}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a DeFungi model checkpoint.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint file.")
    parser.add_argument("--plot-dir", default="./plots",
                        help="Where to save confusion matrix plots.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers",    type=int, default=0)
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        plot_dir=args.plot_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
