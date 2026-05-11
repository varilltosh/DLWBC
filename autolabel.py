from ultralytics import YOLO
import cv2

MODEL_PATH = "/home/skuba/skuba_ws/src/try_vision_project/wbc_project/wbc_training/yolo11_wbc10/weights/best1.pt"

model = YOLO(MODEL_PATH)


def autolabel_image(image_path):

    img = cv2.imread(image_path)

    results = model(img)[0]

    boxes = []

    for b in results.boxes:

        x1,y1,x2,y2 = b.xyxy[0].tolist()
        cls = int(b.cls[0])

        boxes.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "class": cls
        })

    return boxes