"""
model_b.py
Owner: Member 2 (Custom Architecture Design - Model B)

Model B (Lightweight DSConv CNN): replaces every standard Conv2d with a
Depthwise-Separable Convolution (depthwise + pointwise), per the assignment
spec. Designed for 64x64x3 input, 5 output classes
(DeFungi: H1, H2, H3, H5, H6 - microscopic fungal infection images, UCI #773).

Constraint: total trainable parameters MUST be < 100,000.

Architecture summary:

  Input:                    3 x 64 x 64
  DSConv1 (3->16)  + BN + ReLU + MaxPool2d   -> 16 x 32 x 32
  DSConv2 (16->32) + BN + ReLU + MaxPool2d   -> 32 x 16 x 16
  DSConv3 (32->64) + BN + ReLU + MaxPool2d   -> 64 x  8 x  8
  DSConv4 (64->96) + BN + ReLU + MaxPool2d   -> 96 x  4 x  4
  GlobalAveragePooling2d                      -> 96
  Dropout(0.3)
  FC (96 -> num_classes)                      -> 5 (logits)

A Depthwise-Separable Convolution (DSConv) factors a standard k×k convolution
(C_in -> C_out) into:
  1. Depthwise:  C_in filters of shape k×k×1  (one filter per input channel)
  2. Pointwise:  C_out filters of shape 1×1×C_in  (linear recombination)

This reduces FLOPs and parameters by ~1/C_out + 1/k^2 vs standard Conv2d.

Why BatchNorm?
  BN normalises activations per-batch. On edge/embedded deployments this is
  folded into the preceding conv weights at export time (zero runtime cost),
  while making training significantly more stable and faster to converge.

Why GlobalAveragePooling instead of Flatten->FC?
  After 4 MaxPool layers the feature map is only 4x4. Flattening to 96x4x4=1536
  and then adding a hidden FC would cost ~196k params alone - breaking the <100k
  budget. GAP collapses each channel to its spatial mean (no learnable params),
  giving a 96-d descriptor that a single FC head can classify.

Trainable parameter count (verified by count_parameters() below):
  DSConv1 depthwise:  (3*3*3)   =   27
  DSConv1 pointwise:  (16*3)    =   48 ; BN: 32   -> 107
  DSConv2 depthwise:  (3*3*16)  =  144
  DSConv2 pointwise:  (32*16)   =  512 ; BN: 64   -> 720
  DSConv3 depthwise:  (3*3*32)  =  288
  DSConv3 pointwise:  (64*32)   = 2048 ; BN: 128  -> 2,464
  DSConv4 depthwise:  (3*3*64)  =  576
  DSConv4 pointwise:  (96*64)   = 6144 ; BN: 192  -> 6,912
  FC (96+1)*5                   =  485
  -----------------------------------
  Total                         ~ 10,688 params   OK well under 100,000
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block:
      depthwise Conv2d  (groups=in_channels, kernel=3x3, pad=1)
      pointwise Conv2d  (1x1, in_channels -> out_channels)
      BatchNorm2d
      ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise: one filter per channel
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, groups=in_channels, bias=False),
            # Pointwise: mix channels
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ModelB(nn.Module):
    """
    Lightweight edge CNN using Depthwise-Separable Convolutions.
    Total params ~ 10,688 (well under the 100,000 cap).
    """
    def __init__(self, num_classes: int = 5, in_channels: int = 3,
                 dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 16),
            nn.MaxPool2d(2),   # 64->32

            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(2),   # 32->16

            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),   # 16->8

            DepthwiseSeparableConv(64, 96),
            nn.MaxPool2d(2),   # 8->4
        )

        # Global Average Pooling collapses H×W to 1×1 per channel - no params.
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x  # raw logits; use nn.CrossEntropyLoss (applies softmax internally)


def count_parameters(model: nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check: python src/model_b.py
    model = ModelB(num_classes=5)
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)

    n_params = count_parameters(model)
    print(model)
    print(f"\nOutput shape: {output.shape}  (expect [1, 5])")
    print(f"Total trainable parameters: {n_params:,}")
    assert n_params < 100_000, f"OVER BUDGET: {n_params:,} >= 100,000!"
    print("Parameter count is within the 100,000 budget.")
