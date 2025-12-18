import cv2
import os
from pathlib import Path
import json

# Folder paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load ROI config
CONFIG_PATH = Path("config/roi_config.json")
if not CONFIG_PATH.exists():
    print("[ERROR] roi_config.json not found. Run show_roi.py first.")
    exit()

with open(CONFIG_PATH, "r") as f:
    ROI_COORDS = json.load(f)

def crop_rois(file_path):
    """
    Crops ribbon, round, and timer regions from a raw screenshot.
    """
    img = cv2.imread(file_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {file_path}")
        return

    base_name = Path(file_path).stem
    cropped_paths = {}

    for key, coords in ROI_COORDS.items():
        x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
        roi = img[y1:y2, x1:x2]
        save_path = PROCESSED_DIR / f"{base_name}_{key}.png"
        cv2.imwrite(save_path, roi)
        cropped_paths[key] = save_path
        print(f"[INFO] Saved {key} ROI: {save_path}")

    return cropped_paths

# Example usage
if __name__ == "__main__":
    raw_files = sorted(RAW_DIR.glob("*.png"))
    if raw_files:
        crop_rois(str(raw_files[-1]))
    else:
        print("[INFO] No screenshots found in raw folder.")
