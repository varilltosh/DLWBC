import cv2
import os

base = "images"

for root, _, files in os.walk(base):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg")):
            path = os.path.join(root, f)

            img = cv2.imread(path)

            if img is None:
                continue

            # overwrite clean JPEG
            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])