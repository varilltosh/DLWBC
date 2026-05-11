from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import shutil
import os
import cv2
import json

#from wbc_project.website.autolabel import autolabel_image
from autolabel import autolabel_image

app = FastAPI()

UPLOAD_DIR = "uploads"
LABEL_DIR = "labels"
PREVIEW_DIR = "preview"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)


templates = Jinja2Templates(directory="templates")

app.mount("/preview", StaticFiles(directory="preview"), name="preview")


CLASS_NAMES = {
    0: "Basophil",
    1: "Eosinophil",
    2: "Lymphocyte",
    3: "Monocyte",
    4: "Neutrophils"
}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/autolabel")
async def autolabel(files: List[UploadFile] = File(...)):

    async def process():

        results = []
        count = 0

        for file in files:

            if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(UPLOAD_DIR, file.filename)

            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            boxes = autolabel_image(image_path)

            img = cv2.imread(image_path)

            if img is None:
                continue

            H, W = img.shape[:2]

            preview_path = os.path.join(PREVIEW_DIR, file.filename)
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)

            cv2.imwrite(preview_path, img)

            results.append({
                "image": f"/preview/{file.filename}",
                "name": file.filename,
                "boxes": boxes,
                "width": W,
                "height": H
            })

            count += 1

            # send progress update
            yield json.dumps({"progress": count}) + "\n"

        # send final dataset
        yield json.dumps({
            "done": True,
            "data": results
        }) + "\n"

    return StreamingResponse(process(), media_type="text/plain")


@app.post("/save")
async def save_labels(payload: dict):

    image_name = payload["image"]
    boxes = payload["boxes"]
    W = payload["width"]
    H = payload["height"]

    label_path = os.path.join(
        LABEL_DIR,
        image_name.rsplit(".", 1)[0] + ".txt"
    )

    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as f:

        for b in boxes:

            x1 = b["x1"]
            y1 = b["y1"]
            x2 = b["x2"]
            y2 = b["y2"]
            cls = b["class"]

            bw = x2 - x1
            bh = y2 - y1

            cx = x1 + bw / 2
            cy = y1 + bh / 2

            # convert to YOLO format
            cx /= W
            cy /= H
            bw /= W
            bh /= H

            f.write(f"{cls} {cx} {cy} {bw} {bh}\n")

    return {"status": "saved"}