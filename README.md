# Efficient Dataset Generation for Canine White Blood Cell Detection

## Overview

This project focuses on canine white blood cell (WBC) classification and detection using Deep Learning techniques. The system compares a custom Convolutional Neural Network (CNN) with YOLOv8 and introduces a semi-automated annotation pipeline to reduce manual labeling effort.

Classes:
- Basophils
- Eosinophils
- Lymphocytes
- Monocytes
- Neutrophils

---

# Project Objectives

- Develop a custom CNN for WBC classification
- Compare Custom CNN with YOLOv8
- Reduce annotation effort using weak supervision
- Build a semi-automatic annotation system
- Improve minority-class performance using augmentation

---

# Dataset

The dataset consists of canine blood smear microscope images collected from the Faculty of Veterinary Medicine, Kasetsart University.

## Dataset Distribution

| Class | Train | Test | Validation | Total |
|---|---|---|---|---|
| Basophils | 31 | 5 | 10 | 46 |
| Eosinophils | 173 | 29 | 58 | 260 |
| Lymphocytes | 313 | 52 | 104 | 469 |
| Monocytes | 70 | 12 | 23 | 105 |
| Neutrophils | 316 | 53 | 105 | 474 |

Total images: 1354

---

# Custom CNN Architecture

Input size: `128x128x3`

Architecture:
1. Conv Block 1
2. Conv Block 2
3. Conv Block 3
4. Fully Connected Layer
5. Output Layer

Features:
- Batch Normalization
- ReLU activation
- MaxPooling
- Dropout (0.5)

Total parameters: `16,873,989`

---

# Data Augmentation

Training augmentation:
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation(15)
- ColorJitter
- Normalization

Balanced augmentation:
- Rotation (90°, 180°, 270°)
- Brightness adjustment
- Contrast adjustment

---

# Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 8 |
| Epochs | 100 |
| Loss Function | CrossEntropyLoss |
| LR Scheduler | StepLR |
| Device | NVIDIA GTX 1050 Ti |

---

# Hyperparameter Tuning

Configurations tested:
- Learning rates: `0.001`, `0.0001`
- Batch sizes: `8`, `16`
- Epochs: `50`, `100`, `200`

Best configuration:
- Learning Rate = `0.001`
- Batch Size = `8`
- Epochs = `100`

---

# Handling Class Imbalance

Three methods were compared:

| Method | Accuracy | F1 Macro |
|---|---|---|
| Baseline | 95.62% | 87.06% |
| Class Weights | 91.24% | 87.50% |
| Balanced Augmentation | 95.62% | 93.31% |

Balanced augmentation achieved the best results.

---

# Final Results

## Custom CNN (Balanced Augmentation)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Basophils | 83.33% | 100.00% | 90.91% |
| Eosinophils | 92.86% | 100.00% | 96.30% |
| Lymphocytes | 97.83% | 95.74% | 96.77% |
| Monocytes | 90.00% | 81.82% | 85.71% |
| Neutrophils | 97.87% | 95.83% | 96.84% |

Macro averages:
- Precision: `92.38%`
- Recall: `94.68%`
- F1-score: `93.31%`

---

# Custom CNN vs YOLOv8

| Metric | Custom CNN | YOLOv8 |
|---|---|---|
| Accuracy | 95.62% | 85.08% |
| Precision | 92.38% | 86.47% |
| Recall | 94.68% | 83.69% |
| F1-score | 93.31% | 85.06% |

---

# Semi-Automated Annotation Pipeline

Pipeline:
1. Raw images
2. GroundingDINO pseudo-label generation
3. YOLO format conversion
4. YOLOv8 training
5. Human verification
6. Dataset refinement

Technologies:
- GroundingDINO
- Ultralytics SAM
- YOLOv8
- FastAPI

---

# Project Structure

```bash
Custom-CNN/
├── train_custom_cnn.py
├── augment_balance.py

semi-automated-labeling/
├── server.py
├── dino.py
├── autolabel.py
├── split_manifest.json
```

---

# Key Features

- Custom CNN for WBC classification
- Semi-automatic annotation system
- Weakly supervised dataset generation
- Balanced augmentation pipeline
- Web-based annotation interface
- YOLO-compatible dataset export

---

# References

1. LeCun et al. — Deep Learning
2. YOLOv3
3. Ultralytics YOLOv8
4. AlexNet
5. WBC Identification using CNN
6. ResNet
7. Batch Normalization
8. Dropout

---

# Contributors

| Name | Contribution |
|---|---|
| Napat Praphakamol | Custom CNN, augmentation, evaluation |
| Varintorn Marktoum | YOLOv8, evaluation, GitHub repository |
