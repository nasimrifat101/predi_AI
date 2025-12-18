# src/capture/screen_capture.py
import mss
import cv2
import numpy as np
import os
from datetime import datetime

# Directory to save screenshots
SAVE_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
os.makedirs(SAVE_DIR, exist_ok=True)

# Define the region of your mirrored phone window
PHONE_WINDOW = {
    "top": 50,
    "left": 10,
    "width": 400,
    "height": 1000
}

def capture_screen_once():
    """Capture one screenshot and return the saved file path."""
    with mss.mss() as sct:
        screenshot = np.array(sct.grab(PHONE_WINDOW))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(SAVE_DIR, f'screen_{timestamp}.png')
        cv2.imwrite(file_path, screenshot)
        print(f"[INFO] Screenshot saved: {file_path}")
        return file_path
