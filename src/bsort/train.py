"""
Training module for the Bottle Cap Detector.
Handles YOLO training and OpenVINO export pipelines.
"""

import shutil
from pathlib import Path
from typing import Any, Dict

import wandb
from ultralytics import YOLO, settings

# Ensure WandB integration is on
settings.update({"wandb": True})


def train_model(config: Dict[str, Any]) -> str:
    """
    Executes the training pipeline: Train -> Validate -> Export (FP16).

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        str: Path to the exported OpenVINO model directory.
    """
    # Extract config
    data_path = config.get("dataset_path", "datasets/bottle_caps/data.yaml")
    model_name = config.get("model_name", "yolov8n.pt")
    epochs = config.get("epochs", 50)
    img_size = config.get("img_size", 320)
    batch_size = config.get("batch_size", 16)
    project_name = config.get("project_name", "Bottle-Cap-Detection")
    exp_name = config.get("exp_name", "production_build")

    print(f"🚀 Starting Training Pipeline: {exp_name} @ {img_size}px")

    # 1. Initialize WandB
    if wandb.run is not None:
        wandb.finish()

    wandb.init(
        project=project_name,
        name=exp_name,
        job_type="production_train",
        config=config,
    )

    # 2. Setup Model & Augmentation
    model = YOLO(model_name)

    train_args = {
        "data": data_path,
        "epochs": epochs,
        "imgsz": img_size,
        "batch": batch_size,
        "patience": 15,
        "project": project_name,
        "name": exp_name,
        "device": "cpu",
        "plots": True,
        "exist_ok": True,
        "mosaic": 0.0,
        "scale": 0.1,
        "degrees": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }

    # 3. Train
    print("🔄 Training started...")
    model.train(**train_args)

    # 4. Export to OpenVINO FP16
    print("📦 Exporting to OpenVINO FP16...")

    save_dir = Path(f"{project_name}/{exp_name}/weights")
    for direct in save_dir.glob("*openvino*"):
        shutil.rmtree(direct, ignore_errors=True)

    exported_path = model.export(
        format="openvino", imgsz=img_size, half=True, dynamic=False
    )

    print(f"✅ Export Success: {exported_path}")
    wandb.finish()

    return str(exported_path)
