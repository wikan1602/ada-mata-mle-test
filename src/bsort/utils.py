"""
Utility functions for Bottle Cap Sorting.
Includes color classification logic and configuration loading.
"""

import cv2
import numpy as np
import yaml  # Pastikan pyyaml terinstall


def get_color_label(image_crop: np.ndarray) -> int:
    """
    Determines the color class of a bottle cap using HSV thresholding.

    Args:
        image_crop (np.ndarray): Cropped image of the object (BGR format).

    Returns:
        int: Class ID (0: Light Blue, 1: Dark Blue, 2: Others).
    """
    # pylint: disable=no-member
    # Convert to HSV color space
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)

    # Calculate dominant Hue (average color) in the center area
    height, width, _ = hsv.shape
    # Focus on center 50%
    center_hsv = hsv[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4, :]
    mean_hue = np.mean(center_hsv[:, :, 0])

    # Note: Saturation (mean_sat) is calculated but logic below relies on Hue currently.
    # mean_sat = np.mean(center_hsv[:, :, 1])

    # --- LOGIKA THRESHOLD ---
    # Light Blue
    if 90 < mean_hue < 110:
        return 0

    # Dark Blue
    if 110 <= mean_hue < 130:
        return 1

    # Others
    return 2


def load_config(config_path: str) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the .yaml config file.

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    # Fix: encoding='utf-8' added
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
