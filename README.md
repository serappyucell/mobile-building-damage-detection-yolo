# AI-Based Mobile Building Damage Detection System (YOLOv8 / YOLOv11 / YOLOv12)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v11%20%7C%20v12-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-Computer%20Vision-ee4c2c.svg)](https://pytorch.org/)
[![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-6706CE.svg)](https://roboflow.com/)

## 📌 Project Overview & Problem Statement
Following major earthquakes, rapid and reliable structural damage assessment of reinforced concrete buildings is critical. Traditional field inspections are time-consuming, and expert engineering teams cannot reach every affected structure simultaneously.

This graduation project presents an **end-to-end, AI-powered mobile pre-assessment system** capable of detecting structural surface damages from building photographs, calculating risk scores, and generating instant inspection reports. The system compares three generations of real-time object detection architectures (**YOLOv8**, **YOLOv11**, and **YOLOv12**) across four critical structural damage classes.

---

## 👩‍💻 Team & Role Distribution
Developed as a Graduation Project at **Kahramanmaraş Sütçü İmam University** under the supervision of **Assoc. Prof. Dr. Yavuz Canbay**.

* **Serap YÜCEL** *(My Contributions — AI & Computer Vision Pipeline)*:
  * **Multi-Dataset Curation & Preprocessing:** Merged three distinct concrete defect datasets from **Roboflow**, built custom class-filtering and label-remapping scripts in Python, and structured the dataset into an 80% training / 20% validation split.
  * **Deep Learning Model Training & Benchmarking:** Trained, tuned, and evaluated **YOLOv8m**, **YOLOv11**, and **YOLOv12m** architectures over 50 epochs (`AdamW` optimizer, `imgsz=640`, data augmentation) and conducted per-class mAP, Precision, Recall, and loss curve analyses.
* **Ayşe ELGÖRMÜŞ** *(Team Member — Backend API, Networking & Mobile Application)*:
  * Developed the **Android mobile client** (Kotlin, Retrofit), **Google Firebase** integration (Authentication, Cloud Firestore for analysis history and GPS coordinates), **Python REST API** inference server, and **Tailscale VPN** secure client-server communication.

---

## 🏗️ End-to-End System Architecture
1. **Mobile Client (Android / Kotlin):** User captures or uploads a structural image, selects the YOLO model (`YOLOv8`, `YOLOv11`, or `YOLOv12`), and attaches building metadata (UAVT code, address, GPS location).
2. **Secure Communication:** The image is transmitted securely to the inference server over a **Tailscale VPN** tunnel via **REST API** (`Retrofit`).
3. **AI Inference Engine (Python + YOLO):** The server runs the trained YOLO weights (`best.pt`) to detect damage bounding boxes, confidence scores, and overall structural risk levels (*Low / Medium / High*).
4. **Cloud Storage & Reporting:** Results are returned to the mobile app in JSON format, stored in **Firebase Firestore**, and exported as a shareable **PDF Structural Inspection Report**.

---

## 🗂️ Dataset & Target Damage Classes
Three public concrete surface defect datasets from **Roboflow** (`concrete-5november-final-2`, `concrete-14nov-2`, and `wall-crack-yccht`) were programmatically merged and filtered to standardize labels across **4 structural damage classes**:

1. **Crack (*Çatlak*):** Structural fractures and separations on concrete surfaces.
2. **Efflorescence (*Tuzlanma*):** Moisture-induced chemical surface deposits and whitening.
3. **Exposed Rebar (*Demir Görünmesi*):** Spalling of concrete cover exposing underlying steel reinforcement bars.
4. **Spalling (*Beton Dökülmesi*):** Surface concrete delamination and hazardous material loss.

---

## 📊 Experimental Results (50 Epochs)
All models were trained under identical conditions (`50 epochs`, `640x640` resolution, `batch=16`, `AdamW`, `lr0=0.001`). **YOLOv8** achieved the highest performance across every evaluation metric:

| Model Architecture | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50-95 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8** *(Selected Best Model)* | **93.15%** | **88.63%** | **93.97%** | **80.96%** |
| **YOLOv11** | 81.21% | 57.18% | 58.58% | 41.27% |
| **YOLOv12** | 83.92% | 59.65% | 64.39% | 45.56% |

**Key Findings:**
* **YOLOv8** outperformed newer architectures on this multi-source structural dataset, reaching **93.97% mAP@50** and **80.96% mAP@50-95**, making it the primary production model for reliable field detection.
* **YOLOv12** demonstrated improved feature extraction over **YOLOv11** (+5.81% mAP@50), though both required larger sample volumes to match YOLOv8's convergence stability within 50 epochs.

---

## 📂 Repository Structure
```text
├── yolov8_train.py     # Multi-dataset Roboflow merging, label filtering & YOLOv8m training script
└── yolov12_train.py    # Dataset preprocessing, YOLOv12m training, per-class mAP & metric plotting
