"""
model_a.py
Owner: Member 1 (Custom Architecture Design - Model A)

Model A (Standard CNN): interleaves standard 2D convolutions with max-pooling,
per the assignment spec. Designed for 64x64x3 input, 5 output classes
(DeFungi: H1, H2, H3, H5, H6 - microscopic fungal infection images, UCI #773).

Architecture summary (documented here so it's easy to paste into the report):

  Input:            3 x 64 x 64
  Conv1 (3->16, 3x3, pad=1) + ReLU
  MaxPool (2x2)               -> 16 x 32 x 32
  Conv2 (16->32, 3x3, pad=1) + ReLU
  MaxPool (2x2)               -> 32 x 16 x 16
  Conv3 (32->64, 3x3, pad=1) + ReLU
  MaxPool (2x2)               -> 64 x 8 x 8
  Flatten                     -> 4096
  FC1 (4096 -> 128) + ReLU
  FC2 (128 -> num_classes)    (softmax applied implicitly via CrossEntropyLoss)

Activation function justification (hardware-aware):
  ReLU is used throughout instead of sigmoid/tanh. ReLU is max(0, x) - a single
  comparison, with no exponential term. Sigmoid/tanh require computing e^x, which
  is expensive on microcontrollers/edge devices lacking a hardware FPU or fast
  transciable exponential unit. ReLU also avoids vanishing-gradient issues that
  slow convergence, which matters when you have a limited training/compute budget.

Trainable parameter count (verified by count_parameters() below; independent
of num_classes=10 vs 5 except in the final FC2 layer):
  Conv1: (3*3*3 + 1) * 16   = 448
  Conv2: (3*3*16 + 1) * 32  = 4,640
  Conv3: (3*3*32 + 1) * 64  = 18,496
  FC1:   (4096 + 1) * 128   = 524,416
  FC2:   (128 + 1) * 5      = 645       (5 classes for DeFungi)
  ---------------------------------
  Total                     = 548,645 params
"""

import torch
import torch.nn as nn


class ModelA(nn.Module):
    def __init__(self, num_classes=5, in_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 64x64 -> 32x32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 16x16 -> 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logits; use nn.CrossEntropyLoss (applies softmax internally)


def count_parameters(model):
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check you can run standalone: python model_a.py
    model = ModelA(num_classes=5)
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)

    print(model)
    print(f"\nOutput shape: {output.shape}  (expect [1, 5])")
    print(f"Total trainable parameters: {count_parameters(model):,}")
