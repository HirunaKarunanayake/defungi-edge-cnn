"""
sota_models.py
Owner: Member 4 (SOTA Fine-Tuning)

Provides two pre-trained lightweight SOTA models adapted for DeFungi (5 classes):
  - MobileNetV2  (from torchvision.models)
  - SqueezeNet1_1 (from torchvision.models)

Strategy:
  1. Load ImageNet pre-trained weights (backbone already knows edges, textures).
  2. Replace only the final classification head with a 5-class output.
  3. Fine-tune the ENTIRE network (not just the head) with a low learning rate,
     so backbone features adapt to microscopic fungal texture rather than natural
     images. Full fine-tuning consistently outperforms head-only tuning when the
     source (ImageNet) and target (microscopy) domains are moderately different.

Why MobileNetV2?
  Designed for mobile/embedded inference. Uses Inverted Residuals + Linear
  Bottlenecks and Depthwise-Separable Convolutions throughout. Total ImageNet
  params: ~3.4 M. Far smaller than ResNet-50 (~25 M) while still competitive.

Why SqueezeNet1_1?
  ~1.2 M params; achieves AlexNet-level accuracy with 50× fewer parameters.
  Uses Fire modules (squeeze + expand) for extreme compression. Ideal reference
  for edge deployment comparison.

Usage:
  from sota_models import get_mobilenet, get_squeezenet, count_parameters
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def get_mobilenet(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    MobileNetV2 pre-trained on ImageNet with final classifier replaced for
    `num_classes` output classes.
    """
    weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.mobilenet_v2(weights=weights)

    # model.classifier is Sequential([Dropout, Linear(1280, 1000)])
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def get_squeezenet(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    SqueezeNet 1.1 pre-trained on ImageNet with final conv replaced for
    `num_classes` output classes.
    """
    weights = tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.squeezenet1_1(weights=weights)

    # SqueezeNet's head: model.classifier = Sequential([Dropout, Conv2d(512,1000,1), ReLU, AdaptiveAvgPool])
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes

    return model


def count_parameters(model: nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy = torch.randn(1, 3, 64, 64)

    mob = get_mobilenet(num_classes=5)
    out = mob(dummy)
    print(f"MobileNetV2   output: {out.shape}  params: {count_parameters(mob):,}")

    sq = get_squeezenet(num_classes=5)
    out = sq(dummy)
    print(f"SqueezeNet1_1 output: {out.shape}  params: {count_parameters(sq):,}")
