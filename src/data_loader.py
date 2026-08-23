"""
data_loader.py
Owner: Member 1 (Data Preparation)

Downloads the EuroSAT dataset (64x64 RGB land-use/land-cover scenes, 10 classes),
splits it 70/15/15 into train/val/test, and exposes ready-to-use DataLoaders.

Everyone else on the team should import get_dataloaders() rather than writing
their own data code, so all models train/validate/test on identical splits.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_ROOT = "./data"          # where EuroSAT will be downloaded/extracted
IMAGE_SIZE = 64                # EuroSAT is natively 64x64, kept here for clarity/safety
BATCH_SIZE = 32
SEED = 42                      # fixed seed so every team member gets the SAME split
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Standard ImageNet-style normalization works fine here and also matches what
# the pretrained SOTA models (Member 4's step) expect, so we reuse the same
# transform everywhere for consistency.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_transforms():
    """Basic transform: resize (safety no-op at 64x64) + tensor + normalize."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_datasets():
    """Downloads EuroSAT (if not already present) and returns train/val/test Subsets."""
    os.makedirs(DATA_ROOT, exist_ok=True)

    full_dataset = datasets.EuroSAT(
        root=DATA_ROOT,
        download=True,
        transform=get_transforms(),
    )

    n_total = len(full_dataset)
    n_train = int(TRAIN_FRAC * n_total)
    n_val = int(VAL_FRAC * n_total)
    n_test = n_total - n_train - n_val  # remainder avoids rounding gaps

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    return train_set, val_set, test_set, full_dataset.classes


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=2):
    """Returns (train_loader, val_loader, test_loader, class_names)."""
    train_set, val_set, test_set, class_names = get_datasets()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick sanity check you can run standalone: python data_loader.py
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} "
          f"| Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}  (expect [B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}])")
    print(f"Label sample: {labels[:8].tolist()}")
