# Ada Mata - Machine Learning Engineer Take Home Test
### Bottle Cap Detection System (YOLOv8 + ONNX Optimization)

![CI/CD Status](https://github.com/wikan1602/ada-mata-mle-test/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-available-blue)

## 📋 Project Overview
Repository ini berisi solusi lengkap untuk tantangan **Machine Learning Engineer** dari Ada Mata. Sistem ini dirancang untuk mendeteksi tutup botol dan mengklasifikasikannya menjadi 3 kategori: **Light Blue**, **Dark Blue**, dan **Others**.

Solusi mencakup:
1.  **Model Development:** Auto-labeling, Training YOLOv8n, dan Optimasi ke ONNX.
2.  **ML Pipeline:** CLI Tool (`bsort`), Docker Containerization, dan CI/CD Pipeline.

---

## 🚀 Task 1: Model Development & Experimentation

### 1. Data Preprocessing (Auto-Labeling)
Dataset awal hanya memiliki label kelas `0`. Saya melakukan analisis warna menggunakan ruang warna **HSV** untuk memisahkan kelas secara otomatis:
- **Light Blue (0):** `Hue > 90` AND `Value > 88` (Bright)
- **Dark Blue (1):** `Hue > 90` AND `Value <= 88` (Dark)
- **Others (2):** `Hue < 90` (Orange/Green/Yellow)

### 2. Model Selection & Optimization Strategy
Tantangan utama adalah mencapai inferensi **5-10ms** pada Edge Device (Raspberry Pi 5).
- **Base Model:** `yolov8n.pt` (Nano) dipilih karena arsitekturnya paling ringan.
- **Optimization:** Mengkonversi model ke format **ONNX** (Open Neural Network Exchange).
- **Resolution Trade-off:**
    - Pada input `640x640`, kecepatan inferensi ~43ms (CPU).
    - Saya menurunkan input ke **`320x320`** untuk meningkatkan kecepatan secara drastis hingga 4x lipat.

### 3. Evaluation Results
Pengujian dilakukan pada CPU Laptop (Intel i5-1235U). Pada Raspberry Pi 5 (ARM Cortex-A76), performa diharapkan setara atau lebih baik dengan optimasi ONNX Runtime.

| Metric | Configuration (640px) | **Configuration (320px) [CHOSEN]** |
| :--- | :---: | :---: |
| **Inference Speed (CPU)** | ~43.34 ms | **~11.55 ms (🚀 ~86 FPS)** |
| **mAP@50 (Accuracy)** | 98.6% | **93.28%** |

> **Kesimpulan:** Penurunan resolusi input menyebabkan penurunan akurasi minor (~5%), namun memberikan peningkatan kecepatan 400%, yang krusial untuk memenuhi batasan *latency* pada edge device.

**Experiment Tracking:**
Hasil training lengkap dapat dilihat di Weights & Biases:
[🔗 Link ke Project WandB Kamu](https://wandb.ai/USERNAME_KAMU/ada-mata-bottle-cap) *(Ganti dengan link asli kamu)*

---

## 🛠 Task 2: ML Pipeline & Installation

Project ini menggunakan **Poetry** untuk manajemen dependensi dan **Typer** untuk CLI.

### Prerequisites
- Python 3.11
- Docker Desktop (Optional)

### 1. Local Installation (Using Poetry)
```bash
# Clone repository
git clone [https://github.com/USERNAME_KAMU/NAMA_REPO.git](https://github.com/USERNAME_KAMU/NAMA_REPO.git)
cd NAMA_REPO

# Install dependencies
pip install poetry
poetry install
```
### 2. Using the CLI (`bsort`)
Program CLI `bsort` memiliki dua fitur utama:

**A. Training Model**
```bash
poetry run bsort train --config settings.yaml
```
**B. Inference (Deteksi Gambar)**
```bash
poetry run bsort infer --config settings.yaml --image sample.jpg
```
## 🐳 Docker Usage
Aplikasi ini sudah dibungkus dalam Docker Container (Debian Slim) dan dioptimalkan ukurannya (menggunakan PyTorch CPU version).

### Build Image
```bash
docker build -t ada-mata-bsort .
```
## Run Container
```bash
# Menampilkan menu bantuan
docker run --rm ada-mata-bsort --help

# Menjalankan Inferensi (Mount folder lokal ke dalam docker)
# Catatan: $(pwd) digunakan di Linux/Mac/PowerShell. Untuk CMD Windows ganti dengan %cd%
docker run --rm -v $(pwd):/app ada-mata-bsort infer --config settings.yaml --image sample.jpg
```
## CI/CD Pipeline
CI/CD Pipeline
Repository ini terintegrasi dengan GitHub Actions yang secara otomatis menjalankan:
### 1. Code Quality Check: Black (Formatting), Isort (Imports), Pylint (Linting).
### 2. Unit Testing: Pytest untuk memvalidasi fungsi config dan CLI.
### 3. Docker Build: Memastikan Dockerfile valid dan image berhasil dibangun tanpa error.

## Project Structure
## 📂 Project Structure
```text
ada_mata_mle/
├── .github/workflows/  # CI/CD Configurations
├── datasets/           # Dataset (Train/Val)
├── notebooks/          # Jupyter Notebook (Eksperimen Task 1)
├── src/
│   └── ada_mata_mle/   # Source Code (CLI Tool)
├── tests/              # Unit Tests
├── Dockerfile          # Docker Configuration
├── pyproject.toml      # Dependencies & Metadata
├── settings.yaml       # Model Configuration
└── README.md           # Documentation
```

## Author : Wikan Priambudi
