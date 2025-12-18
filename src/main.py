import time
from datetime import datetime
from pathlib import Path
import cv2

# ===== IMPORT COMPONENTS =====
from src.capture.screen_capture import capture_screen_once
from src.capture.roi import crop_rois
from src.vision.ribbon_detector import split_ribbon
from src.vision.icon_detector import detect_icon


# ===== CONFIG =====
INTERVAL = 40  # seconds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "predictions.csv"

# File to store last round number
ROUND_FILE = LOG_DIR / "last_round.txt"

# Initialize CSV log file
if not log_file.exists():
    with open(log_file, "w") as f:
        f.write("timestamp,round,slot0,slot1,slot2,slot3,slot4\n")

# Initialize starting round
# Initialize starting round
if ROUND_FILE.exists():
    with open(ROUND_FILE, "r") as f:
        content = f.read().strip()
        if content.isdigit():
            current_round = int(content)
        else:
            current_round = int(input("Enter starting round number: "))
else:
    current_round = int(input("Enter starting round number: "))


# ===== HELPER FUNCTION =====
def detect_icons_in_slots(slot_paths):
    """
    Detect icons for all slot images.
    Returns list of detected labels.
    """
    results = []
    for slot_path in slot_paths:
        img = cv2.imread(str(slot_path))
        if img is None:
            print(f"[WARN] Could not read {slot_path}")
            results.append("unknown")
            continue
        label = detect_icon(img)
        results.append(label)
    return results

# ===== MAIN LOOP =====
def run_pipeline():
    global current_round

    while True:
        # 1️⃣ Capture screenshot
        screenshot_path = capture_screen_once()

        # 2️⃣ Crop ribbon ROI
        cropped_paths = crop_rois(screenshot_path)
        ribbon_path = cropped_paths["ribbon"]

        # 3️⃣ Split ribbon into slots
        slot_paths = split_ribbon(ribbon_path)  # returns list of slot paths

        # 4️⃣ Detect icons in slots
        predictions = detect_icons_in_slots(slot_paths)

        # 5️⃣ Log results with timestamp and round number
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp},{current_round},{','.join(predictions)}\n")

        print(f"[INFO] Round {current_round} → {predictions}")

        # 6️⃣ Increment round and save for persistence
        current_round += 1
        with open(ROUND_FILE, "w") as f:
            f.write(str(current_round))

        # 7️⃣ Wait until next capture
        print(f"[INFO] Waiting {INTERVAL} seconds...")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    run_pipeline()
