import cv2
from pathlib import Path
from icon_detector import detect_icon

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SLOTS_DIR = PROJECT_ROOT / "data/processed/slots"

slot_files = sorted(SLOTS_DIR.glob("*.png"))

for slot_path in slot_files:
    img = cv2.imread(str(slot_path))
    label = detect_icon(img)
    print(f"{slot_path.name} → {label}")
