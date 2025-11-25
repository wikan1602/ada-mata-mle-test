"""
Inference module using OpenVINO Runtime (via Ultralytics wrapper).
"""

import time
from pathlib import Path

from ultralytics import YOLO


def run_inference(config: dict, image_path: str) -> None:
    """
    Runs inference on a single image using the exported OpenVINO model.

    Args:
        config (dict): Configuration dictionary containing model paths.
        image_path (str): Path to the input image file.
    """
    project_name = config.get("project_name", "Bottle-Cap-Detection")
    exp_name = config.get("exp_name", "production_build")

    # Auto-detect OpenVINO folder inside weights
    weights_dir = Path(f"{project_name}/{exp_name}/weights")
    ov_dirs = list(weights_dir.glob("*openvino*"))

    if not ov_dirs:
        raise FileNotFoundError(
            f"❌ No OpenVINO model found in {weights_dir}. Please train first."
        )

    model_path = str(ov_dirs[0])
    img_size = config.get("img_size", 320)

    print(f"⚡ Loading Model: {model_path}")
    model = YOLO(model_path, task="detect")

    print(f"🖼️ Processing Image: {image_path}")

    # Warmup
    for _ in range(3):
        model.predict(source=image_path, imgsz=img_size, verbose=False)

    # Inference
    start_time = time.time()
    results = model.predict(
        source=image_path,
        imgsz=img_size,
        conf=0.25,
        save=True,  # Saves to runs/detect/predict
        verbose=True,
    )
    end_time = time.time()

    latency = (end_time - start_time) * 1000
    print("✅ Inference Complete.")
    print(f"⏱️ Latency: {latency:.2f} ms")
    print(f"📂 Result saved to: {results[0].save_dir}")
