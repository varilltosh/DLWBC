from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# PATHS
# =====================================================

run_dir = "/home/skuba/skuba_ws/src/try_vision_project/wbc_project/website/WBC_dataset_yolo/runs/detect/final_exp_batch16_0.0001"

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

train_loss = (
    df['train/box_loss'] +
    df['train/cls_loss'] +
    df['train/dfl_loss']
)

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

graph_save_path = os.path.join(
    run_dir,
    "training_validation_graph.png"
)

plt.savefig(graph_save_path, dpi=300)
plt.close()

print(f"\nGraph saved to:\n{graph_save_path}")

# =====================================================
# EVALUATE MULTIPLE EPOCH CHECKPOINTS
# =====================================================

epochs_to_check = [50, 100, 200]

all_reports_path = os.path.join(
    run_dir,
    "all_epoch_reports.txt"
)

with open(all_reports_path, "w") as output_file:

    for ep in epochs_to_check:

        # -------------------------------------------------
        # CHECKPOINT PATH
        # -------------------------------------------------

        checkpoint_path = os.path.join(
            run_dir,
            f"weights/epoch{ep}.pt"
        )

        # If checkpoint does not exist -> use best.pt
        if not os.path.exists(checkpoint_path):

            checkpoint_path = os.path.join(
                run_dir,
                "weights/best.pt"
            )

            actual_epoch = "latest/best"

        else:

            actual_epoch = ep

        print(f"\nEvaluating Epoch {actual_epoch}")

        # -------------------------------------------------
        # LOAD MODEL
        # -------------------------------------------------

        model = YOLO(checkpoint_path)

        # -------------------------------------------------
        # VALIDATE
        # -------------------------------------------------

        metrics = model.val()

        # -------------------------------------------------
        # EXTRACT METRICS
        # -------------------------------------------------

        class_names = model.names

        precision = metrics.box.p
        recall = metrics.box.r
        f1 = metrics.box.f1
        ap50 = metrics.box.ap50

        # -------------------------------------------------
        # CREATE REPORT
        # -------------------------------------------------

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

        # -------------------------------------------------
        # OVERALL METRICS
        # -------------------------------------------------

        precision_mean = metrics.results_dict['metrics/precision(B)']
        recall_mean = metrics.results_dict['metrics/recall(B)']
        map50 = metrics.results_dict['metrics/mAP50(B)']
        map5095 = metrics.results_dict['metrics/mAP50-95(B)']

        f1_score = 2 * (
            precision_mean * recall_mean
        ) / (
            precision_mean + recall_mean + 1e-8
        )

        # -------------------------------------------------
        # WRITE TO FILE
        # -------------------------------------------------

        output_file.write(f"\nEpoch {actual_epoch}\n")
        output_file.write("Final YOLOv8 Metrics\n")
        output_file.write("====================\n")

        output_file.write(
            f"Precision      : {precision_mean:.4f}\n"
        )

        output_file.write(
            f"Recall         : {recall_mean:.4f}\n"
        )

        output_file.write(
            f"F1-Score       : {f1_score:.4f}\n"
        )

        output_file.write(
            f"mAP@50         : {map50:.4f}\n"
        )

        output_file.write(
            f"mAP@50-95      : {map5095:.4f}\n\n"
        )

        output_file.write(
            "Per-Class YOLOv8 Report\n"
        )

        output_file.write(
            "========================\n\n"
        )

        output_file.write(
            df_report.to_string(index=False)
        )

        output_file.write("\n\n")

print(f"\nSaved all reports to:\n{all_reports_path}")