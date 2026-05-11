from groundingdino.util.inference import load_model, load_image, predict
from ultralytics.models.sam import SAM3SemanticPredictor
import os
import torch
import random
from tqdm import tqdm
import cv2
# -------------------------
# PATH CONFIG
# -------------------------

CONFIG_PATH = "/home/skuba/skuba_ws/src/try_vision_project/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHT_PATH = "/home/skuba/skuba_ws/src/try_vision_project/GroundingDINO/weights/groundingdino_swint_ogc.pth"

DATASET_PATH = "/home/skuba/skuba_ws/src/try_vision_project/WBC_dataset"

OUTPUT_ROOT = "/home/skuba/skuba_ws/src/try_vision_project/yolo_dataset"

IMG_TRAIN = os.path.join(OUTPUT_ROOT,"images/train")
IMG_VAL   = os.path.join(OUTPUT_ROOT,"images/val")
IMG_TEST  = os.path.join(OUTPUT_ROOT,"images/test")

LAB_TRAIN = os.path.join(OUTPUT_ROOT,"labels/train")
LAB_VAL   = os.path.join(OUTPUT_ROOT,"labels/val")
LAB_TEST  = os.path.join(OUTPUT_ROOT,"labels/test")

for p in [IMG_TRAIN,IMG_VAL,IMG_TEST,LAB_TRAIN,LAB_VAL,LAB_TEST]:
    os.makedirs(p,exist_ok=True)

# -------------------------
# PROMPT CONFIG
# -------------------------

TEXT_PROMPT = "purple round shape cell with nucleus"
BOX_THRESHOLD = 0.45
TEXT_THRESHOLD = 0.8

# -------------------------
# CLASS MAP
# -------------------------

class_map = {
    "Basophils":0,
    "Eosinophils":1,
    "Lymphocytes":2,
    "Monocytes":3,
    "Neutrophils":4
}

# -------------------------
# LOAD MODELS
# -------------------------
total_images = 0

for class_name in os.listdir(DATASET_PATH):
    class_folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_folder):
        continue

    total_images += len([
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".jpg",".jpeg",".png"))
    ])
pbar = tqdm(total=total_images, desc="Processing images", unit="img")
print("Loading GroundingDINO...")
model = load_model(CONFIG_PATH, WEIGHT_PATH)

print("Loading SAM3...")
predictor = SAM3SemanticPredictor(
    overrides=dict(
        conf=0.85,
        task="segment",
        mode="predict",
        model="/home/skuba/skuba_ws/src/try_vision_project/sam3/sam3.pt",
        half=True,
        save=False,
    ),
    bpe_path="/home/skuba/skuba_ws/src/try_vision_project/sam3/bpe_simple_vocab_16e6.txt.gz"
)

print("Models loaded.\n")
dataset_stats = {}

# -------------------------
# PROCESS DATASET
# -------------------------

for class_name in os.listdir(DATASET_PATH):

    class_folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_folder):
        continue

    if class_name not in class_map:
        print("Skipping unknown class:", class_name)
        continue

    CLASS_ID = class_map[class_name]
    dataset_stats[class_name] = {
    "train": 0,
    "val": 0,
    "test": 0
    }
    print("\nProcessing class:", class_name)

    image_list = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".jpg",".jpeg",".png"))
    ]

    random.shuffle(image_list)

    total = len(image_list)

    train_split = int(total * 0.7)
    val_split   = int(total * 0.85)

    train_imgs = image_list[:train_split]
    val_imgs   = image_list[train_split:val_split]
    test_imgs  = image_list[val_split:]

    split_map = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split, files in split_map.items():

        print(f"{split}: {len(files)} images")
        pbar.set_description(f"{class_name}-{split}")
        for filename in files:

            image_path = os.path.join(class_folder, filename)

            image_source, image = load_image(image_path)

            H, W, _ = image_source.shape

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )

            if len(boxes) == 0:
                pbar.update(1)
                continue

            boxes = boxes * torch.tensor([W, H, W, H])

            box_prompts = []

            for box in boxes:

                x_center, y_center, bw, bh = box.tolist()

                x1 = max(0, int(x_center - bw/2))
                y1 = max(0, int(y_center - bh/2))
                x2 = min(W, int(x_center + bw/2))
                y2 = min(H, int(y_center + bh/2))

                box_prompts.append([x1, y1, x2, y2])

            new_name = f"{class_name}_{filename}"

            if split == "train":
                img_out = IMG_TRAIN
                lab_out = LAB_TRAIN
            elif split == "val":
                img_out = IMG_VAL
                lab_out = LAB_VAL
            else:
                img_out = IMG_TEST
                lab_out = LAB_TEST

            save_img_path = os.path.join(img_out,new_name)
            img = cv2.imread(image_path)
            cv2.imwrite(save_img_path, img)

            label_path = os.path.join(
                lab_out,
                new_name.rsplit(".",1)[0] + ".txt"
            )

            with open(label_path,"w") as f:

                for box in box_prompts:

                    x1,y1,x2,y2 = box

                    cx = ((x1+x2)/2)/W
                    cy = ((y1+y2)/2)/H
                    bw = (x2-x1)/W
                    bh = (y2-y1)/H

                    f.write(f"{CLASS_ID} {cx} {cy} {bw} {bh}\n")
            dataset_stats[class_name][split] += 1
            pbar.update(1)

pbar.close()
print("Dataset generation finished.")
print("\nDataset Split Summary")
print("-"*40)

for class_name, splits in dataset_stats.items():

    train_count = splits["train"]
    val_count   = splits["val"]
    test_count  = splits["test"]

    print(f"{class_name:<12} train: {train_count:<4} val: {val_count:<4} test: {test_count:<4}")
