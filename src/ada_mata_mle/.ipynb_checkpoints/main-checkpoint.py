import sys
from pathlib import Path

import cv2
import typer
import yaml
from ultralytics import YOLO

# Inisialisasi Aplikasi CLI
app = typer.Typer(help="Ada Mata MLE Take Home Test - Bottle Cap Detector")

def load_config(config_path: str):
    """Membaca file konfigurasi YAML."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)

@app.command()
def train(config: str = typer.Option("settings.yaml", help="Path to config file")):
    """
    Melatih model YOLO berdasarkan konfigurasi.
    Usage: bsort train --config settings.yaml
    """
    cfg = load_config(config)
    print(f"Memulai Training untuk project: {cfg['project_name']}")
    
    # Load Model
    model = YOLO(cfg['train']['model_type'])
    
    # Start Training
    model.train(
        data=cfg['dataset']['path'],
        epochs=cfg['train']['epochs'],
        imgsz=cfg['dataset']['img_size'],
        batch=cfg['train']['batch_size'],
        device=cfg['train']['device'],
        project="runs/train",
        name=cfg['experiment_name'],
        lr0=cfg['train']['learning_rate']
    )
    
    print("Training Selesai!")

@app.command()
def infer(
    config: str = typer.Option("settings.yaml", help="Path to config file"),
    image: str = typer.Option(..., help="Path to input image for inference")
):
    """
    Melakukan deteksi pada satu gambar.
    Usage: bsort infer --config settings.yaml --image sample.jpg
    """
    cfg = load_config(config)
    img_path = Path(image)
    
    if not img_path.exists():
        print(f"Error: Image '{image}' not found.")
        sys.exit(1)
        
    print(f"Menjalankan inferensi pada: {image} dengan ukuran {cfg['dataset']['img_size']}px")
    
    # Load model terbaik (asumsi training sudah pernah jalan)
    # Nanti kita buat lebih dinamis, tapi untuk sekarang hardcode path hasil training notebook/cli
    # Pastikan path ini sesuai dengan tempat modelmu disimpan
    weights_path = Path("runs/train") / cfg['experiment_name'] / "weights/best.pt"
    
    # Fallback ke model notebook jika hasil CLI belum ada
    if not weights_path.exists():
        print(f"Model CLI belum ditemukan di {weights_path}")
        print("Mencoba mencari model hasil Notebook (Task 1)...")
        weights_path = Path("notebooks/runs/train/exp_final/weights/best.pt") # Path notebook tadi
        
    if not weights_path.exists():
        print("Model tidak ditemukan! Jalankan training dulu.")
        sys.exit(1)

    model = YOLO(str(weights_path))
    
    # Inference
    results = model.predict(
        source=image,
        imgsz=cfg['dataset']['img_size'],
        conf=cfg['inference']['confidence_threshold'],
        save=True,
        project=cfg['inference']['output_dir']
    )
    
    print(f"Hasil tersimpan di folder: {cfg['inference']['output_dir']}")

# Agar script bisa dijalankan langsung
if __name__ == "__main__":
    app()