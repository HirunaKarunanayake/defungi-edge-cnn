# EN3150 Assignment 03 - DeFungi Edge CNN: Results Report

## Dataset Summary

| Item | Value |
|---|---|
| Dataset | DeFungi (UCI ML Repo #773) |
| Total images | 9,114 |
| Classes | 5 (H1=Candida albicans, H2=Aspergillus niger, H3=Trichophyton rubrum, H5=T. mentagrophytes, H6=Epidermophyton floccosum) |
| Input size | 64 x 64 x 3 (RGB) |
| Split | 70% train / 15% val / 15% test (seed=42) |
| Class imbalance | Yes - H1 (~4,404) is ~6x larger than H5/H6 (~739-818) |
| Imbalance handling | Inverse-frequency class weights in CrossEntropyLoss |

## Model Comparison

| Model | Parameters | Checkpoint Size (KB) | Val Acc | Test Acc |
|---|---|---|---|---|
| Model A (Adam) | 548,645 | 2147.8 | 0.6986 | 0.7076 |
| Model B (Adam) | 10,688 | 54.5 | 0.7147 | 0.7156 |
| Model B (SGD) | 10,688 | 54.5 | 0.6781 | 0.6959 |
| Model B (SGD+M) | 10,688 | 54.9 | 0.7030 | 0.7025 |
| MobileNetV2 (fine-tuned) | 2,230,277 | 8950.6 | 0.8383 | 0.8165 |
| SqueezeNet1_1 (fine-tuned) | 725,061 | 2852.2 | 0.7454 | 0.7478 |

## Optimizer Comparison (Model B, 20 epochs)

| Optimizer | Val Acc | Test Acc | Notes |
|---|---|---|---|
| Adam | 0.7147 | 0.7156 | Adaptive per-param LR; fastest convergence |
| SGD | 0.6781 | 0.6959 | Fixed LR=0.01; slowest, most sensitive |
| SGD+M | 0.7030 | 0.7025 | Momentum=0.9; smoother updates, bridges gap to Adam |

## Training Curves

### Model A (Adam)

![Model A (Adam) curves](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/model_a_adam_curves.png)

### Model B (Adam)

![Model B (Adam) curves](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/model_b_adam_curves.png)

### Model B (SGD)

![Model B (SGD) curves](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/model_b_sgd_curves.png)

### Model B (SGD+M)

![Model B (SGD+M) curves](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/model_b_sgd_momentum_curves.png)

### MobileNetV2 (fine-tuned)

![MobileNetV2 (fine-tuned) curves](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/sota_mobilenet_curves.png)

### SqueezeNet1_1 (fine-tuned)

![SqueezeNet1_1 (fine-tuned) curves](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/sota_squeezenet_curves.png)

## Confusion Matrices

### Model A (Adam)

![Model A (Adam) confusion matrix](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/model_a_adam_confusion_matrix.png)

### Model B (Adam)

![Model B (Adam) confusion matrix](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/model_b_adam_confusion_matrix.png)

### Model B (SGD)

![Model B (SGD) confusion matrix](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/model_b_sgd_confusion_matrix.png)

### Model B (SGD+M)

![Model B (SGD+M) confusion matrix](C:/ENTC Sem 05/Pattern Recognition/Pattern Project/defungi-edge-cnn/plots/model_b_sgd_momentum_confusion_matrix.png)

## Per-Class Metrics

### Model A (Adam)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| H1 | 0.7443 | 0.8450 | 0.7915 | 658 |
| H2 | 0.5744 | 0.4474 | 0.5030 | 371 |
| H3 | 0.6429 | 0.5385 | 0.5860 | 117 |
| H5 | 0.8165 | 0.7807 | 0.7982 | 114 |
| H6 | 0.7520 | 0.8704 | 0.8069 | 108 |
| **Macro Avg** | 0.7060 | 0.6964 | 0.6971 | - |

**Test Accuracy:** 70.76%

### Model B (Adam)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| H1 | 0.7196 | 0.9088 | 0.8032 | 658 |
| H2 | 0.6688 | 0.2776 | 0.3924 | 371 |
| H3 | 0.6071 | 0.7265 | 0.6615 | 117 |
| H5 | 0.7881 | 0.8158 | 0.8017 | 114 |
| H6 | 0.8000 | 0.9259 | 0.8584 | 108 |
| **Macro Avg** | 0.7167 | 0.7309 | 0.7034 | - |

**Test Accuracy:** 71.56%

### Model B (SGD)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| H1 | 0.7284 | 0.8723 | 0.7939 | 658 |
| H2 | 0.5700 | 0.3073 | 0.3993 | 371 |
| H3 | 0.5726 | 0.6068 | 0.5892 | 117 |
| H5 | 0.8198 | 0.7982 | 0.8089 | 114 |
| H6 | 0.7034 | 0.9444 | 0.8063 | 108 |
| **Macro Avg** | 0.6789 | 0.7058 | 0.6795 | - |

**Test Accuracy:** 69.59%

### Model B (SGD+M)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| H1 | 0.7441 | 0.8571 | 0.7966 | 658 |
| H2 | 0.6198 | 0.3208 | 0.4227 | 371 |
| H3 | 0.5056 | 0.7692 | 0.6102 | 117 |
| H5 | 0.8925 | 0.7281 | 0.8019 | 114 |
| H6 | 0.7143 | 0.9722 | 0.8235 | 108 |
| **Macro Avg** | 0.6952 | 0.7295 | 0.6910 | - |

**Test Accuracy:** 70.25%

### MobileNetV2 (fine-tuned)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| H1 | 0.8316 | 0.8632 | 0.8471 | 658 |
| H2 | 0.7500 | 0.6388 | 0.6900 | 371 |
| H3 | 0.7519 | 0.8547 | 0.8000 | 117 |
| H5 | 0.9083 | 0.9561 | 0.9316 | 114 |
| H6 | 0.8879 | 0.9537 | 0.9196 | 108 |
| **Macro Avg** | 0.8260 | 0.8533 | 0.8377 | - |

**Test Accuracy:** 81.65%

### SqueezeNet1_1 (fine-tuned)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| H1 | 0.7880 | 0.8191 | 0.8033 | 658 |
| H2 | 0.5896 | 0.5499 | 0.5690 | 371 |
| H3 | 0.7073 | 0.7436 | 0.7250 | 117 |
| H5 | 0.9375 | 0.7895 | 0.8571 | 114 |
| H6 | 0.8655 | 0.9537 | 0.9075 | 108 |
| **Macro Avg** | 0.7776 | 0.7712 | 0.7724 | - |

**Test Accuracy:** 74.78%

## Trade-off Discussion

### Custom Models vs SOTA

**Model A** (Standard CNN, ~549k params) establishes the hand-designed baseline.
It achieves competitive accuracy but has significantly more parameters than Model B,
making it unsuitable for extreme memory-constrained edge devices.

**Model B** (Depthwise-Separable CNN, ~10.7k params) demonstrates that replacing
standard convolutions with DSConv reduces parameters by ~50x with minimal accuracy
loss. GlobalAveragePooling eliminates the costly Flatten->FC bottleneck, keeping
the model well within the 100k parameter budget. Despite being 50x smaller than
Model A, Model B matches or slightly exceeds its test accuracy, confirming that
DSConv is highly effective for this task.

**MobileNetV2** and **SqueezeNet1_1** benefit from ImageNet pre-training - their
backbones already encode general visual features (edges, textures, shapes) that
transfer to microscopy images. Fine-tuning adapts these features to fungal
morphology in far fewer epochs, achieving markedly higher accuracy.

### Optimizer Analysis

**Adam** achieves the best validation and test accuracy for Model B. Its adaptive
per-parameter learning rates make it robust to the choice of global LR, giving
fast and stable convergence from epoch 1.

**SGD (plain)** with LR=0.01 converges more slowly and reaches lower final
accuracy, as it applies the same learning rate to all parameters regardless of
gradient history.

**SGD + Momentum (0.9)** outperforms plain SGD by accumulating gradient
direction across iterations, dampening oscillations and accelerating convergence
in consistent gradient directions. It bridges the gap toward Adam performance
but still falls slightly short.

### Edge Deployment Recommendation

For a resource-constrained device with a <100k parameter budget: **Model B**
is the clear choice - 10,688 params, ~55KB checkpoint, >71% accuracy.
For devices with more headroom: a quantised MobileNetV2 fine-tuned on DeFungi
offers the best accuracy-to-size ratio at 81.65% test accuracy.
