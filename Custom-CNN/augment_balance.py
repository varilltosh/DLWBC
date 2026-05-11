"""
Balanced Augmentation — ทำให้ทุก class มีจำนวนรูปเท่ากัน
ใช้ augmentation เบาๆ ไม่เปลี่ยนลักษณะเซลล์มากเกินไป
"""

from pathlib import Path
from PIL import Image, ImageEnhance
import random
import itertools

# ==============================================================
# CONFIG
# ==============================================================
TRAIN_DIR = Path(__file__).parent / "preview" / "train"
CLASSES   = ["Basophils", "Eosinophils", "Lymphocytes", "Monocytes", "Neutrophils"]
random.seed(42)

# ==============================================================
# AUGMENTATION เบาๆ (ไม่เปลี่ยนลักษณะเซลล์มากเกินไป)
# ==============================================================
MILD_RECIPES = [
    ("fh",  lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
    ("fv",  lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
    ("r90", lambda img: img.rotate(90,  expand=False)),
    ("r180",lambda img: img.rotate(180, expand=False)),
    ("r270",lambda img: img.rotate(270, expand=False)),
    ("fh_r90",  lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90,  expand=False)),
    ("fv_r90",  lambda img: img.transpose(Image.FLIP_TOP_BOTTOM).rotate(90,  expand=False)),
    ("fh_r270", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=False)),
    ("br+",  lambda img: ImageEnhance.Brightness(img).enhance(1.15)),
    ("br-",  lambda img: ImageEnhance.Brightness(img).enhance(0.85)),
    ("ct+",  lambda img: ImageEnhance.Contrast(img).enhance(1.15)),
    ("ct-",  lambda img: ImageEnhance.Contrast(img).enhance(0.90)),
    ("br+fh",lambda img: ImageEnhance.Brightness(img.transpose(Image.FLIP_LEFT_RIGHT)).enhance(1.10)),
    ("br-fv",lambda img: ImageEnhance.Brightness(img.transpose(Image.FLIP_TOP_BOTTOM)).enhance(0.90)),
    ("ct+r90",lambda img: ImageEnhance.Contrast(img.rotate(90, expand=False)).enhance(1.10)),
]

def get_orig_files(cls_dir: Path):
    """เอาเฉพาะรูปต้นฉบับ (ไม่รวมที่ aug ไว้แล้ว)"""
    return [f for f in sorted(cls_dir.glob("*.jpg")) if "_aug_" not in f.name]

def get_all_files(cls_dir: Path):
    return list(cls_dir.glob("*.jpg"))

# ==============================================================
# นับจำนวนรูปแต่ละ class
# ==============================================================
print("=" * 50)
print("Current image counts:")
print("=" * 50)
counts = {}
for cls in CLASSES:
    cls_dir = TRAIN_DIR / cls
    orig  = len(get_orig_files(cls_dir))
    total = len(get_all_files(cls_dir))
    counts[cls] = total
    aug_count = total - orig
    print(f"  {cls:<15}: {total:>4} total  ({orig} original + {aug_count} augmented)")

TARGET = max(counts.values())
print(f"\nTarget (max class): {TARGET} images per class\n")

# ==============================================================
# AUGMENT แต่ละ class ให้ถึง TARGET
# ==============================================================
print("=" * 50)
print("Augmenting...")
print("=" * 50)

for cls in CLASSES:
    cls_dir    = TRAIN_DIR / cls
    orig_files = get_orig_files(cls_dir)
    current    = len(get_all_files(cls_dir))
    needed     = TARGET - current

    if needed <= 0:
        print(f"  {cls:<15}: ครบแล้ว ({current} รูป) ข้าม")
        continue

    print(f"  {cls:<15}: ต้องเพิ่ม {needed} รูป ({current} → {TARGET})")

    # วนซ้ำ recipe + รูป จนครบ
    recipe_cycle = itertools.cycle(MILD_RECIPES)
    img_cycle    = itertools.cycle(orig_files)
    created      = 0

    while created < needed:
        img_path            = next(img_cycle)
        recipe_name, recipe_fn = next(recipe_cycle)
        save_name           = f"{img_path.stem}_aug_{recipe_name}_{created:04d}.jpg"
        save_path           = cls_dir / save_name

        if not save_path.exists():
            img = Image.open(img_path).convert("RGB")
            aug = recipe_fn(img)
            aug.save(save_path, quality=95)
            created += 1

    print(f"  {cls:<15}: สร้างเพิ่ม {created} รูป → รวม {TARGET} รูป ✓")

# ==============================================================
# สรุป
# ==============================================================
print("\n" + "=" * 50)
print("Final counts:")
print("=" * 50)
for cls in CLASSES:
    total = len(get_all_files(TRAIN_DIR / cls))
    print(f"  {cls:<15}: {total} images")

print("\nDone! Now run: python train_custom_cnn.py")
