import sys
from pathlib import Path

import typer
import yaml
from ultralytics import YOLO

# Inisialisasi Aplikasi CLI
app = typer.Typer(help="Ada Mata MLE Take Home Test - Bottle Cap Detector")


def load_config(config_path: str):
    """Membaca file konfigurasi YAML dengan error handling."""
    path = Path(config_path)
    if not path.exists():
        print(f"❌ Error: Config file '{config_path}' not found.")
        sys.exit(1)

    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_best_model(base_dir: Path, experiment_name: str):
    """
    Mencari model terbaik secara otomatis.
    Prioritas:
    1. best.onnx (Dioptimalkan)
    2. best.pt (Standar)
    """
    # Cek folder training dari CLI run
    train_dir = base_dir / experiment_name / "weights"

    # Cek juga folder training dari Notebook (fallback)
    notebook_dirs = [
        Path("runs/train/final_v8_optimized/weights"),  # Hasil optimasi terakhir kita
        Path("runs/train/exp_yolo11_aug/weights"),
        Path("runs/train/exp_final/weights"),
    ]

    search_paths = [train_dir] + notebook_dirs

    for folder in search_paths:
        # Prioritas 1: ONNX
        onnx_path = folder / "best.onnx"
        if onnx_path.exists():
            return onnx_path

        # Prioritas 2: PT
        pt_path = folder / "best.pt"
        if pt_path.exists():
            return pt_path

    return None


@app.command()
def train(config: str = typer.Option("settings.yaml", help="Path to config file")):
    """
    Melatih model YOLO berdasarkan konfigurasi di settings.yaml.
    Usage: bsort train --config settings.yaml
    """
    cfg = load_config(config)
    print(f"🚀 Memulai Training Project: {cfg.get('project_name', 'Unknown')}")
    print(
        f"   Model: {cfg['train']['model_type']} | Epochs: {cfg['train']['epochs']} | ImgSize: {cfg['dataset']['img_size']}"
    )

    # Load Model
    model = YOLO(cfg["train"]["model_type"])

    # Start Training
    model.train(
        data=cfg["dataset"]["path"],
        epochs=cfg["train"]["epochs"],
        imgsz=cfg["dataset"]["img_size"],
        batch=cfg["train"]["batch_size"],
        device=cfg["train"]["device"],
        project="runs/train",
        name=cfg["experiment_name"],
        lr0=cfg["train"]["learning_rate"],
        patience=10,  # Early stopping standar
        plots=True,
        exist_ok=True,
    )

    print(
        f"✅ Training Selesai! Hasil tersimpan di runs/train/{cfg['experiment_name']}"
    )

    # Auto-export ke ONNX setelah training selesai (biar siap inference cepat)
    print("📦 Meng-export model terbaik ke ONNX...")
    model.export(
        format="onnx", imgsz=cfg["dataset"]["img_size"], dynamic=False, opset=12
    )


@app.command()
def infer(
    config: str = typer.Option("settings.yaml", help="Path to config file"),
    image: str = typer.Option(..., help="Path to input image for inference"),
):
    """
    Melakukan deteksi pada satu gambar menggunakan model terbaik yang ditemukan.
    Usage: bsort infer --config settings.yaml --image sample.jpg
    """
    cfg = load_config(config)
    img_path = Path(image)

    if not img_path.exists():
        print(f"❌ Error: Image '{image}' not found.")
        sys.exit(1)

    # Cari model terbaik
    weights_path = find_best_model(Path("runs/train"), cfg["experiment_name"])

    if not weights_path:
        print("❌ Error: Model tidak ditemukan!")
        print(
            "   Pastikan kamu sudah menjalankan training (bsort train) atau notebook eksperimen."
        )
        sys.exit(1)

    print(f"🔍 Menggunakan Model: {weights_path}")
    print(f"🖼️  Memproses Gambar: {image} (Resize: {cfg['dataset']['img_size']}px)")

    # Load & Predict
    model = YOLO(str(weights_path))

    results = model.predict(
        source=image,
        imgsz=cfg["dataset"]["img_size"],
        conf=cfg["inference"]["confidence_threshold"],
        save=True,
        project=cfg["inference"]["output_dir"],
        name="predict",
        exist_ok=True,
    )

    print(f"✅ Deteksi Selesai!")
    print(f"📂 Hasil tersimpan di: {cfg['inference']['output_dir']}/predict")


# Entry point
if __name__ == "__main__":
    app()
