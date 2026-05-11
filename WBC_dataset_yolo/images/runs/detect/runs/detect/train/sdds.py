from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# PATHS
# =====================================================
run_dir = "/home/skuba/skuba_ws/src/try_vision_project/wbc_project/website/WBC_dataset_yolo/images/runs/detect/runs/detect/train"

csv_path = os.path.join(run_dir, "results.csv")
model_path = os.path.join(run_dir, "weights/best.pt")

# =====================================================
# LOAD TRAINING CSV
# =====================================================
df = pd.read_csv(csv_path)

# Remove spaces in column names
df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns)

# =====================================================
# CREATE TRAIN + VALIDATION LOSS GRAPH
# =====================================================
plt.figure(figsize=(14, 5))

# ---------------- LEFT: LOSS ----------------
plt.subplot(1, 2, 1)

# Total train loss
train_loss = (
    df['train/box_loss'] +
    df['train/cls_loss'] +
    df['train/dfl_loss']
)

# Total validation loss
val_loss = (
    df['val/box_loss'] +
    df['val/cls_loss'] +
    df['val/dfl_loss']
)

plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')

plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# ---------------- RIGHT: METRICS ----------------
plt.subplot(1, 2, 2)

# Use mAP50 as "accuracy"
train_acc = df['metrics/precision(B)'] * 100
val_acc = df['metrics/mAP50(B)'] * 100

plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')

plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()

# =====================================================
# SAVE GRAPH
# =====================================================
graph_save_path = os.path.join(run_dir, "training_validation_graph.png")

plt.savefig(graph_save_path, dpi=300)
plt.close()

print(f"\nGraph saved to:\n{graph_save_path}")

# =====================================================
# LOAD MODEL
# =====================================================
model = YOLO(model_path)

# =====================================================
# VALIDATE MODEL
# =====================================================
metrics = model.val()

# =====================================================
# EXTRACT PER-CLASS METRICS
# =====================================================
class_names = model.names

precision = metrics.box.p
recall = metrics.box.r
f1 = metrics.box.f1
ap50 = metrics.box.ap50

# =====================================================
# CREATE REPORT TABLE
# =====================================================
report_data = []

for i, class_name in class_names.items():
    report_data.append({
        "Class": class_name,
        "Precision": round(float(precision[i]), 4),
        "Recall": round(float(recall[i]), 4),
        "F1-Score": round(float(f1[i]), 4),
        "mAP50": round(float(ap50[i]), 4)
    })

df_report = pd.DataFrame(report_data)

# =====================================================
# SAVE REPORT
# =====================================================
csv_report_path = os.path.join(run_dir, "per_class_report.csv")
txt_report_path = os.path.join(run_dir, "per_class_report.txt")

df_report.to_csv(csv_report_path, index=False)

with open(txt_report_path, "w") as f:
    f.write("Per-Class YOLOv8 Report\n")
    f.write("========================\n\n")
    f.write(df_report.to_string(index=False))

# =====================================================
# SAVE FINAL METRICS
# =====================================================
precision_mean = df['metrics/precision(B)'].iloc[-1]
recall_mean = df['metrics/recall(B)'].iloc[-1]
map50 = df['metrics/mAP50(B)'].iloc[-1]
map5095 = df['metrics/mAP50-95(B)'].iloc[-1]

f1_score = 2 * (precision_mean * recall_mean) / (
    precision_mean + recall_mean + 1e-8
)

final_metrics_path = os.path.join(run_dir, "final_metrics.txt")

with open(final_metrics_path, "w") as f:
    f.write("Final YOLOv8 Metrics\n")
    f.write("====================\n")
    f.write(f"Precision      : {precision_mean:.4f}\n")
    f.write(f"Recall         : {recall_mean:.4f}\n")
    f.write(f"F1-Score       : {f1_score:.4f}\n")
    f.write(f"mAP@50         : {map50:.4f}\n")
    f.write(f"mAP@50-95      : {map5095:.4f}\n")

# =====================================================
# PRINT RESULTS
# =====================================================
print("\nPer-Class Report:")
print(df_report)

print(f"\nSaved files:")
print(f"- Graph: {graph_save_path}")
print(f"- CSV Report: {csv_report_path}")
print(f"- TXT Report: {txt_report_path}")
print(f"- Final Metrics: {final_metrics_path}")