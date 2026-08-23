"""
data_loader.py
Owner: Member 1 (Data Preparation)

Downloads the DeFungi dataset (UCI ML Repository, dataset #773) - microscopic
fungal infection images across 5 classes - resizes to 64x64, splits it
70/15/15 into train/val/test, and exposes ready-to-use DataLoaders.

Source: https://archive.ics.uci.edu/dataset/773/defungi
Reference: Sopo, Hajati & Gheisari (2021), "DeFungi: Direct mycological
examination of microscopic fungi images", Medical Mycology.

Classes (folder names -> organism):
  H1 -> Candida albicans        (4,404 images)
  H2 -> Aspergillus niger       (2,334 images)
  H3 -> Trichophyton rubrum     (  819 images)
  H5 -> Trichophyton mentagrophytes (818 images)
  H6 -> Epidermophyton floccosum (  739 images)

NOTE: classes are imbalanced (H1 is ~6x larger than H5/H6). Plain accuracy
alone is misleading here - use the class weights below in your loss function,
and always report precision/recall/confusion matrix, not just accuracy.

Everyone else on the team should import get_dataloaders() rather than writing
their own data code, so all models train/validate/test on identical splits.
"""

import os
import io
import zipfile
import urllib.request

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_URL = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/773/defungi.zip"
DATA_ROOT = "./data"                       # where DeFungi will be downloaded/extracted
RAW_DIR = os.path.join(DATA_ROOT, "defungi_raw")
IMAGE_SIZE = 64                             # per assignment: downscale to max 64x64
BATCH_SIZE = 32
SEED = 42                                   # fixed seed so every team member gets the SAME split
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Standard ImageNet-style normalization also matches what the pretrained SOTA
# models (Member 4's step) expect, so we reuse the same transform everywhere.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["H1", "H2", "H3", "H5", "H6"]
CLASS_COUNTS = {"H1": 4404, "H2": 2334, "H3": 819, "H5": 818, "H6": 739}


def get_transforms():
    """Resize raw (larger, variable-size) images down to 64x64 + tensor + normalize."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def download_and_extract():
    """Downloads the DeFungi zip from UCI and extracts it, if not already present."""
    os.makedirs(DATA_ROOT, exist_ok=True)

    # Already extracted? skip.
    if os.path.isdir(RAW_DIR) and any(
        os.path.isdir(os.path.join(RAW_DIR, c)) for c in CLASS_NAMES
    ):
        return

    print(f"Downloading DeFungi (~149 MB) from {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL) as response:
        zip_bytes = response.read()

    print("Extracting ...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(RAW_DIR)

    # The zip may extract into a nested folder (e.g. defungi/H1/...); if so,
    # flatten it so RAW_DIR directly contains H1/, H2/, etc.
    if not any(os.path.isdir(os.path.join(RAW_DIR, c)) for c in CLASS_NAMES):
        subdirs = [d for d in os.listdir(RAW_DIR)
                   if os.path.isdir(os.path.join(RAW_DIR, d))]
        for sub in subdirs:
            nested = os.path.join(RAW_DIR, sub)
            if any(os.path.isdir(os.path.join(nested, c)) for c in CLASS_NAMES):
                for c in os.listdir(nested):
                    os.rename(os.path.join(nested, c), os.path.join(RAW_DIR, c))
                break


def get_class_weights():
    """Inverse-frequency class weights, useful for a weighted loss (imbalanced classes)."""
    total = sum(CLASS_COUNTS.values())
    weights = [total / (len(CLASS_NAMES) * CLASS_COUNTS[c]) for c in CLASS_NAMES]
    return torch.tensor(weights, dtype=torch.float32)


def get_datasets():
    """Downloads DeFungi (if not already present) and returns train/val/test Subsets."""
    download_and_extract()

    full_dataset = datasets.ImageFolder(root=RAW_DIR, transform=get_transforms())
    # Sanity check: ImageFolder sorts classes alphabetically -> should match CLASS_NAMES
    assert full_dataset.classes == CLASS_NAMES, (
        f"Unexpected class folders: {full_dataset.classes}, expected {CLASS_NAMES}"
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
    print(f"Class weights (for imbalanced loss, if needed): {get_class_weights().tolist()}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}  (expect [B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}])")
    print(f"Label sample: {labels[:8].tolist()}")
