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
        img_dir = f"{ds.location}/{split}/images"
        lbl_dir = f"{ds.location}/{split}/labels"

        if not os.path.exists(img_dir):
            return

        for file in os.listdir(img_dir):

            img_path = os.path.join(img_dir, file)
            lbl_path = os.path.join(lbl_dir, file.replace(".jpg",".txt").replace(".png",".txt"))

            new_lines = []

            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    lines = f.readlines()

                for l in lines:
                    cls = int(l.split()[0])
                    if cls in mapping:
                        parts = l.split()
                        parts[0] = str(mapping[cls])
                        new_lines.append(" ".join(parts))

            # sadece hedef class varsa kaydet
            if len(new_lines) > 0:
                dst_img = f"{base}/images/train/{file}"
                dst_lbl = f"{base}/labels/train/{file.replace('.jpg','.txt').replace('.png','.txt')}"

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

split = int(len(all_images)*0.8)

train_files = all_images[:split]
val_files = all_images[split:]

for f in val_files:
    shutil.move(f"{base}/images/train/{f}", f"{base}/images/val/{f}")
    shutil.move(f"{base}/labels/train/{f.replace('.jpg','.txt').replace('.png','.txt')}",
                f"{base}/labels/val/{f.replace('.jpg','.txt').replace('.png','.txt')}")

# =========================
# 7. DATA.YAML
# =========================
data_yaml = {
    "train": f"{base}/images/train",
    "val": f"{base}/images/val",
    "nc": 4,
    "names": target_classes
}

with open(f"{base}/data.yaml","w") as f:
    yaml.dump(data_yaml,f)

# =========================
# 8. MODEL
# =========================
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

# =========================
# 9. EĞİTİM (~7-8 SAAT)
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
    name="kaggle_concrete_3ds"
)
