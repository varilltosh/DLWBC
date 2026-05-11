"""
Custom CNN for White Blood Cell (WBC) Classification
204466 Deep Learning - Final Project

Architecture: CustomWBC_CNN (3 Conv Blocks + Fully Connected)
Classes: Basophils, Eosinophils, Lymphocytes, Monocytes, Neutrophils
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import seaborn as sns

# ==============================================================
# CONFIG — แก้ path ตรงนี้ให้ตรงกับเครื่องของคุณ
# ==============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # โฟลเดอร์เดียวกับไฟล์นี้

TRAIN_DIR = os.path.join(BASE_DIR, "preview", "train")
VAL_DIR   = os.path.join(BASE_DIR, "preview", "val")
TEST_DIR  = os.path.join(BASE_DIR, "preview", "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # ไฟล์ผลลัพธ์จะบันทึกที่นี่
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE    = 8
LEARNING_RATE = 0.001
EPOCHS        = 100
IMAGE_SIZE    = 128
NUM_WORKERS   = 0

# ==============================================================
# DEVICE
# ==============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==============================================================
# DATA TRANSFORMS & LOADERS
# ==============================================================
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_data   = datasets.ImageFolder(VAL_DIR,   transform=val_test_transforms)
test_data  = datasets.ImageFolder(TEST_DIR,  transform=val_test_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

CLASS_NAMES  = train_data.classes
NUM_CLASSES  = len(CLASS_NAMES)
print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# ==============================================================
# MODEL — Custom WBC CNN (Own Design)
# ==============================================================
class CustomWBC_CNN(nn.Module):
    """
    Custom CNN for White Blood Cell classification.

    Architecture:
        3x [Conv2d → BatchNorm → ReLU → MaxPool]
        Flatten → FC(512) → Dropout(0.5) → FC(num_classes)

    Input : (B, 3, 128, 128)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes: int):
        super(CustomWBC_CNN, self).__init__()

        # Block 1: 3 → 32 channels | 128x128 → 64x64
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 32 → 64 channels | 64x64 → 32x32
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 64 → 128 channels | 32x32 → 16x16
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier: 128*16*16 = 32768
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


model = CustomWBC_CNN(NUM_CLASSES).to(device)

# Class weights — แก้ปัญหา imbalanced data
# class ที่มีข้อมูลน้อย (เช่น Basophils) จะได้ weight สูงขึ้นอัตโนมัติ
class_counts  = torch.tensor(
    [len(list((Path(TRAIN_DIR) / cls).glob("*.jpg"))) for cls in CLASS_NAMES],
    dtype=torch.float
)
class_weights = (1.0 / class_counts)
class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(device)
print(f"\nClass Weights (imbalance correction):")
for cls, w, c in zip(CLASS_NAMES, class_weights.cpu(), class_counts):
    print(f"  {cls:<15}: weight={w:.4f}  (train samples={int(c)})")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("\nModel Architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}\n")

# ==============================================================
# TRAINING LOOP
# ==============================================================
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

best_val_acc  = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_custom_wbc_cnn.pth")

print("=" * 60)
print("Starting Training...")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total         += labels.size(0)
        correct        += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100.0 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ── Validate ───────────────────────────────────────────────
    model.eval()
    v_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            v_loss   += loss.item()
            _, predicted = torch.max(outputs, 1)
            total    += labels.size(0)
            correct  += (predicted == labels).sum().item()

    val_loss = v_loss / len(val_loader)
    val_acc  = 100.0 * correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step()

    # บันทึก best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        saved_marker = " ← best saved"
    else:
        saved_marker = ""

    print(f"Epoch [{epoch:>3}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
          f"{saved_marker}")

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

# ==============================================================
# PLOT — Loss & Accuracy Curves
# ==============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label="Train Loss",      color="steelblue")
axes[0].plot(val_losses,   label="Validation Loss", color="tomato")
axes[0].set_title("Training & Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label="Train Accuracy",      color="steelblue")
axes[1].plot(val_accs,   label="Validation Accuracy", color="tomato")
axes[1].set_title("Training & Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(curve_path, dpi=150)
plt.show()
print(f"Saved: {curve_path}")

# ==============================================================
# TEST EVALUATION — โหลด Best Model มาทดสอบ
# ==============================================================
print("\n" + "=" * 60)
print("Evaluating on Test Set (Best Model)...")
print("=" * 60)

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc  = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
recall    = recall_score(all_labels, all_preds,    average="macro", zero_division=0) * 100
f1        = f1_score(all_labels, all_preds,        average="macro", zero_division=0) * 100

print(f"\n  Test Accuracy  : {test_acc:.2f}%")
print(f"  Precision (macro): {precision:.2f}%")
print(f"  Recall    (macro): {recall:.2f}%")
print(f"  F1-Score  (macro): {f1:.2f}%")

print("\nPer-Class Report:")
print(classification_report(all_labels, all_preds,
                             target_names=CLASS_NAMES, digits=4))

# ==============================================================
# CONFUSION MATRIX
# ==============================================================
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Confusion Matrix — Custom WBC CNN (Test Set)")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"Saved: {cm_path}")

print("\nDone! All outputs saved to:", OUTPUT_DIR)
