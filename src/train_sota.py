"""
train_sota.py
Owner: Member 4 (SOTA Fine-Tuning & Evaluation)

Fine-tunes MobileNetV2 and SqueezeNet1_1 (pre-trained on ImageNet) on the
DeFungi dataset (5-class microscopic fungal infection images).

Key differences from train.py (custom CNNs):
  - Uses a lower LR (1e-4) since backbone weights already carry useful features.
  - Uses Adam optimizer exclusively (standard choice for fine-tuning).
  - Trains for fewer epochs (10 on CPU) since convergence is faster.
  - Uses class-weighted loss to handle DeFungi's class imbalance.

Usage:
  python src/train_sota.py --model mobilenet --epochs 10
  python src/train_sota.py --model squeezenet --epochs 10
"""

import os
import sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data_loader import get_dataloaders, get_class_weights
from sota_models import get_mobilenet, get_squeezenet, count_parameters


def get_sota_model(name: str, num_classes: int = 5, pretrained: bool = True):
    name = name.lower()
    if name == "mobilenet":
        return get_mobilenet(num_classes=num_classes, pretrained=pretrained)
    elif name == "squeezenet":
        return get_squeezenet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown SOTA model '{name}'. Choose mobilenet | squeezenet.")


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    """Run one epoch. Returns (avg_loss, accuracy)."""
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def save_plots(history: dict, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(tag, fontsize=13)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#4C72B0")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#DD8452")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#4C72B0")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="#DD8452")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{tag}_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def train_sota(
    model_name: str = "mobilenet",
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 32,
    checkpoint_dir: str = "./checkpoints",
    plot_dir: str = "./plots",
    num_workers: int = 0,
    pretrained: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SOTA Model: {model_name}   Epochs: {epochs}   "
          f"LR: {lr}   Device: {device}")
    print(f"{'='*60}")

    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )

    model = get_sota_model(model_name, num_classes=len(class_names),
                           pretrained=pretrained).to(device)
    class_weights = get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    print(f"  Parameters: {count_parameters(model):,}")

    tag = f"sota_{model_name}"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, f"{tag}_best.pth")
    patience_counter = 0
    early_stop_patience = 8

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      device, training=False)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        scheduler.step(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": vl_acc,
                "model_name": model_name,
                "optimizer_name": "adam",
                "class_names": class_names,
            }, best_path)
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>2}/{epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
              f"[{elapsed:.1f}s]{marker}")

        if patience_counter >= early_stop_patience:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    plot_path = save_plots(history, plot_dir, tag)
    print(f"\n  Best val_acc: {best_val_acc:.4f}")
    print(f"  Checkpoint saved: {best_path}")
    print(f"  Plot saved:       {plot_path}")

    json_path = os.path.join(plot_dir, f"{tag}_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    return best_val_acc, best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a SOTA model on DeFungi.")
    parser.add_argument("--model",      default="mobilenet",
                        choices=["mobilenet", "squeezenet"])
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--workers",    type=int,   default=0)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train SOTA models from scratch (not recommended).")
    args = parser.parse_args()

    train_sota(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pretrained=not args.no_pretrained,
    )
