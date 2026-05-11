# Efficient Dataset Generation for Canine White Blood Cell Detection
### Using a Custom CNN and Semi-Automated Annotation in Blood Smear Imaging

**Course:** 01204466 Deep Learning — Section 450, Faculty of Engineering, Kasetsart University

**Authors:**
- 6510552663 Napat Praphakamol — Custom CNN Design, Hyperparameter Tuning, Balanced Augmentation, Evaluation
- 6510552787 Varintorn Marktoum — YOLOv8 Training, Evaluation, Comparison Analysis, GitHub Repository

---
### Full Report
[Download Final Report](final_report.pdf)

## 1. Project Topic

White Blood Cell (WBC) Classification and Detection using Deep Learning — comparing a **Custom CNN** (Classification) against **YOLOv8** (Object Detection) on microscope images of canine white blood cells.

**5 Classes:** Basophils · Eosinophils · Lymphocytes · Monocytes · Neutrophils

---

## 2. Why This Topic?

WBC differential count is critical for diagnosing infectious diseases, leukemia, and immune deficiency disorders. Current lab-based analysis is time-consuming, error-prone, and unavailable in remote areas. Deep Learning can automate this process — improving speed, reducing human error, and supporting medical professionals.

---

## 3. Why Deep Learning?

| Method | Strengths | Weaknesses |
|---|---|---|
| Rule-based / Thresholding | Simple, interpretable | Fails with lighting/staining variation |
| Traditional ML (SVM, Random Forest) | Requires less data | Needs manual feature engineering, lower accuracy |
| **Deep Learning (CNN) ✓** | Automatically learns features, high accuracy | Requires large datasets and compute |

---

## 4. Repository Structure

```
DLWBC/
├── Custom-CNN/
│   ├── train_custom_cnn.py       # Main CNN training script
│   ├── augment_balance.py        # Balanced augmentation script
│   ├── hyperparam_compare.py     # Hyperparameter comparison
│   └── compare_models.py         # CNN vs YOLO comparison charts
├── semi-automated-labeling/
│   ├── dino.py                   # GroundingDINO pseudo-label generation
│   ├── server.py                 # FastAPI annotation backend
│   ├── autolabel.py              # YOLO-based auto-labeling
│   └── split_manifest.json       # Dataset split configuration
└── WBC_dataset_yolo/             # Dataset (YOLO format)
```

---

## 5. Model Architecture — CustomWBC_CNN

```
Input (128×128×3)
    ↓
Conv Block 1: Conv2d(3→32) + BatchNorm + ReLU + MaxPool  →  64×64×32
    ↓
Conv Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool →  32×32×64
    ↓
Conv Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool → 16×16×128
    ↓
Flatten  →  32,768
    ↓
FC1: Linear(32768→512) + ReLU
    ↓
Dropout(p=0.5)
    ↓
FC2: Linear(512→5)
    ↓
Output (5 class scores)
```

**Total Trainable Parameters:** 16,873,989

| Component | Role |
|---|---|
| Conv2d | Extracts feature maps with learnable 3×3 filters |
| BatchNorm | Stabilizes training, handles staining variation |
| ReLU | Non-linear activation f(x) = max(0, x) |
| MaxPool2d | Halves spatial size, reduces computation |
| Dropout(0.5) | Prevents overfitting on small dataset |
| CrossEntropyLoss | Multi-class loss (includes Softmax internally) |

---

## 6. Code Overview

### `train_custom_cnn.py`
[Source](https://github.com/varilltosh/DLWBC/blob/main/Custom-CNN/train_custom_cnn.py)

**Part 1 — Data Pipeline**
Uses `torchvision.datasets.ImageFolder` with the following augmentations:
- `RandomHorizontalFlip()` — horizontal flip
- `RandomVerticalFlip()` — vertical flip
- `RandomRotation(15)` — random rotation ±15°
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` — random color adjustment
- `Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])` — normalize to [-1, 1]

**Part 2 — Model Definition**
`CustomWBC_CNN` inherits from `nn.Module`. Layers defined in `__init__()`, forward pass in `forward()`.

**Part 3 — Training Loop**
100 epochs, 2 phases per epoch:
- **Train:** CrossEntropyLoss (with class weights) + Adam optimizer + backpropagation
- **Validation:** `torch.no_grad()` evaluation, save best model by val accuracy
- **LR Scheduler:** StepLR — reduces LR × 0.5 every 10 epochs

**Part 4 — Evaluation**
Loads `best_custom_wbc_cnn.pth`, evaluates on test set using `sklearn.metrics` (Accuracy, Precision, Recall, F1, Confusion Matrix).

### `augment_balance.py`
[Source](https://github.com/varilltosh/DLWBC/blob/main/Custom-CNN/augment_balance.py)

Counts images per class, augments minority classes to match the largest class using: flip H/V, rotation 90°/180°/270°, brightness/contrast ±15%.

---

## 7. Dataset

**Source:** Canine white blood cell images collected from the Faculty of Veterinary Medicine, Kasetsart University.
**Dataset:** https://github.com/varilltosh/DLWBC/tree/main/WBC_dataset_yolo

### Split Distribution (60% / 10% / 30%)

| Class | Train | Test | Validate | Total |
|---|---|---|---|---|
| Basophils | 31 | 5 | 10 | 46 |
| Eosinophils | 173 | 29 | 58 | 260 |
| Lymphocytes | 313 | 52 | 104 | 469 |
| Monocytes | 70 | 12 | 23 | 105 |
| Neutrophils | 316 | 53 | 105 | 474 |
| **Total** | **903** | **151** | **300** | **1,354** |

### Semi-Automated Labeling Pipeline

1. **`dino.py`** — Uses GroundingDINO with text prompt `"purple round shape cell with nucleus"` to generate initial pseudo-labels (bounding boxes) for all images automatically
2. **`server.py`** — FastAPI backend: accepts image uploads, runs YOLO inference, returns predictions as JSON, and saves verified annotations in YOLO format
3. **`autolabel.py`** — Applies trained `best1.pt` to new images for auto-annotation; feeds corrections back to improve the dataset iteratively
4. **`split_manifest.json`** — Records dataset split (seed=42), ensuring reproducibility across experiments

---

## 8. Training Configuration

### Hyperparameter Tuning

Compared LR × Batch Size at epochs 50, 100, 200:

| Config | LR | Batch | Val Acc @50 | Val Acc @100 | Val Acc @200 | F1 @100 |
|---|---|---|---|---|---|---|
| **Config 1 ✓** | **0.001** | **8** | **94.35%** | **95.09%** | **95.09%** | **89.54%** |
| Config 2 | 0.0001 | 8 | 95.33% | 95.09% | 94.84% | 87.94% |
| Config 3 | 0.001 | 16 | 93.12% | 94.10% | 94.10% | 86.81% |
| Config 4 | 0.0001 | 16 | 94.59% | 94.35% | 94.35% | 85.93% |

**Selected: Config 1** — highest macro F1, converged by epoch 100 (no gain from 100→200).

### Final Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss (with class weights) |
| LR Scheduler | StepLR — ×0.5 every 10 epochs |
| Epochs | 100 |
| Batch Size | 8 |
| Image Size | 128 × 128 |
| Device | NVIDIA GeForce GTX 1050 Ti (CUDA) |
| Model Selection | Best by Validation Accuracy |

---

## 9. Results

### Class Imbalance Handling Comparison

| Method | Accuracy | Precision | Recall | F1 macro | Basophils Recall |
|---|---|---|---|---|---|
| ① Baseline | 95.62% | 94.37% | 85.00% | 87.06% | 40% |
| ② + Class Weights | 91.24% | 90.91% | 86.41% | 87.50% | 60% |
| **③ + Balanced Augmentation** | **95.62%** | **92.38%** | **94.68%** | **93.31%** | **100%** |

Balanced Augmentation achieved the best result: F1 improved from 87.06% → 93.31% and Basophils Recall from 40% → 100%, without reducing overall accuracy.

### Final Test Results — Custom CNN (Balanced Augmentation)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Basophils | 83.33% | 100.00% | 90.91% | 5 |
| Eosinophils | 92.86% | 100.00% | 96.30% | 26 |
| Lymphocytes | 97.83% | 95.74% | 96.77% | 47 |
| Monocytes | 90.00% | 81.82% | 85.71% | 11 |
| Neutrophils | 97.87% | 95.83% | 96.84% | 48 |
| **Macro Avg** | **92.38%** | **94.68%** | **93.31%** | **137** |

### Model Comparison: Custom CNN vs YOLOv8s

| Metric | Custom CNN | YOLOv8s |
|---|---|---|
| Task | Classification | Object Detection |
| Architecture | 3-Block Custom CNN | YOLOv8 (pretrained) |
| Epochs | 100 | 100 |
| Accuracy | **95.62%** | 85.08% |
| Precision (macro) | **92.38%** | 86.47% |
| Recall (macro) | **94.68%** | 83.69% |
| F1-Score (macro) | **93.31%** | 85.06% |

The Custom CNN outperformed YOLOv8s on all classification metrics. Note that the tasks differ — CNN performs classification only, while YOLOv8 simultaneously localizes and classifies cells, making direct comparison valid only in terms of classification accuracy.

---

## 10. References

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
2. Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. *arXiv:1804.02767*.
3. Jocher, G. et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
4. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*, 25.
5. Shahin, A. I. et al. (2019). White blood cells identification system based on convolutional deep neural learning networks. *Computer Methods and Programs in Biomedicine*, 168, 69–80.
6. He, K. et al. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
7. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training. *ICML*.
8. Srivastava, N. et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929–1958.
