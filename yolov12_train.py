# =========================
# 1. KURULUM
# =========================
!pip install -q ultralytics roboflow pyyaml matplotlib

# =========================
# 2. GPU KONTROL
# =========================
import torch
print("GPU:", torch.cuda.is_available())

# =========================
# 3. DATASETLER
# =========================
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")

ds1 = rf.workspace("defect-detection-edbnh").project("concrete-5november-final-2").version(1).download("yolov8")
ds2 = rf.workspace("defect-detection-edbnh").project("concrete-14nov-2").version(1).download("yolov8")
ds3 = rf.workspace("team5").project("wall-crack-yccht").version(1).download("yolov8")

# =========================
# 4. HEDEF CLASSLAR
# =========================
target_classes = [
    "Crack",
    "Efflorescence",
    "Exposed Rebar",
    "Spallin"
]

# =========================
# 5. MERGE + FILTER
# =========================
import os, shutil, yaml

base = "/kaggle/working/merged"

for p in ["images/train","images/val","labels/train","labels/val"]:
    os.makedirs(f"{base}/{p}", exist_ok=True)

def process_dataset(ds):

    with open(ds.location + "/data.yaml") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    # mapping
    mapping = {}
    new_id = 0

    for i, n in enumerate(names):
        if n in target_classes:
            mapping[i] = new_id
            new_id += 1

    def process_split(split):

        # bazı datasetlerde "valid" yerine "val" olabilir
        split_name = split

        if not os.path.exists(f"{ds.location}/{split_name}"):
            if split == "valid" and os.path.exists(f"{ds.location}/val"):
                split_name = "val"

        img_dir = f"{ds.location}/{split_name}/images"
        lbl_dir = f"{ds.location}/{split_name}/labels"

        if not os.path.exists(img_dir):
            return

        for file in os.listdir(img_dir):

            if not (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")):
                continue

            img_path = os.path.join(img_dir, file)

            txt_name = (
                file.replace(".jpg", ".txt")
                    .replace(".png", ".txt")
                    .replace(".jpeg", ".txt")
            )

            lbl_path = os.path.join(lbl_dir, txt_name)

            new_lines = []

            if os.path.exists(lbl_path):

                with open(lbl_path) as f:
                    lines = f.readlines()

                for l in lines:

                    parts = l.strip().split()

                    if len(parts) < 5:
                        continue

                    cls = int(parts[0])

                    if cls in mapping:
                        parts[0] = str(mapping[cls])
                        new_lines.append(" ".join(parts) + "\n")

            # sadece hedef class varsa kaydet
            if len(new_lines) > 0:

                dst_img = f"{base}/images/train/{file}"
                dst_lbl = f"{base}/labels/train/{txt_name}"

                shutil.copy(img_path, dst_img)

                with open(dst_lbl, "w") as f:
                    f.writelines(new_lines)

    process_split("train")
    process_split("valid")

# 3 dataseti işle
process_dataset(ds1)
process_dataset(ds2)
process_dataset(ds3)

print("Dataset hazır ✔")

# =========================
# 6. TRAIN / VAL SPLIT
# =========================
import random

all_images = os.listdir(f"{base}/images/train")

random.shuffle(all_images)

split = int(len(all_images) * 0.8)

train_files = all_images[:split]
val_files = all_images[split:]

for f in val_files:

    src_img = f"{base}/images/train/{f}"
    dst_img = f"{base}/images/val/{f}"

    src_lbl = (
        f"{base}/labels/train/"
        f"{f.replace('.jpg','.txt').replace('.png','.txt').replace('.jpeg','.txt')}"
    )

    dst_lbl = (
        f"{base}/labels/val/"
        f"{f.replace('.jpg','.txt').replace('.png','.txt').replace('.jpeg','.txt')}"
    )

    if os.path.exists(src_img):
        shutil.move(src_img, dst_img)

    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

# =========================
# 7. DATA.YAML
# =========================
data_yaml = {
    "train": f"{base}/images/train",
    "val": f"{base}/images/val",
    "nc": 4,
    "names": target_classes
}

with open(f"{base}/data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("data.yaml oluşturuldu ✔")

# =========================
# 8. YOLOv12 MODEL
# =========================
from ultralytics import YOLO

# YOLOv12 medium modeli
model = YOLO("yolo12m.pt")

# =========================
# 9. EĞİTİM
# =========================
results = model.train(
    data=f"{base}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    optimizer="AdamW",
    lr0=0.001,
    augment=True,
    patience=20,
    workers=2,
    cache=True,
    name="kaggle_concrete_yolov12"
)

# =========================
# 10. METRİKLER
# =========================
metrics = model.val()

print("\n===== GENEL METRİKLER =====")

print(f"Precision      : {metrics.box.mp:.4f}")
print(f"Recall         : {metrics.box.mr:.4f}")
print(f"mAP50          : {metrics.box.map50:.4f}")
print(f"mAP50-95       : {metrics.box.map:.4f}")

print("\n===== SINIF BAZLI mAP =====")

for i, n in metrics.names.items():
    print(f"{n}: {metrics.box.maps[i]:.4f}")

# =========================
# 11. METRİK TABLOSU
# =========================
import pandas as pd

metric_table = pd.DataFrame({
    "Metric": ["Precision", "Recall", "mAP50", "mAP50-95"],
    "Value": [
        metrics.box.mp,
        metrics.box.mr,
        metrics.box.map50,
        metrics.box.map
    ]
})

print("\n===== METRİK TABLOSU =====")
print(metric_table)

# =========================
# 12. GRAFİKLER
# =========================
import matplotlib.pyplot as plt

results_path = "runs/detect/kaggle_concrete_yolov12/results.csv"

df = pd.read_csv(results_path)

# -------------------------
# LOSS GRAFİĞİ
# -------------------------
plt.figure(figsize=(8,5))

plt.plot(df["train/box_loss"], label="Train Box Loss")
plt.plot(df["val/box_loss"], label="Val Box Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()

plt.show()

# -------------------------
# mAP GRAFİĞİ
# -------------------------
plt.figure(figsize=(8,5))

plt.plot(df["metrics/mAP50(B)"], label="mAP50")
plt.plot(df["metrics/mAP50-95(B)"], label="mAP50-95")

plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("mAP Performance")
plt.legend()

plt.show()

# -------------------------
# PRECISION / RECALL
# -------------------------
plt.figure(figsize=(8,5))

plt.plot(df["metrics/precision(B)"], label="Precision")
plt.plot(df["metrics/recall(B)"], label="Recall")

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision vs Recall")
plt.legend()

plt.show()

# =========================
# 13. BEST MODEL YOLU
# =========================
print("\n===== MODEL KAYDI =====")
print("Best model:")
print("runs/detect/kaggle_concrete_yolov12/weights/best.pt")
