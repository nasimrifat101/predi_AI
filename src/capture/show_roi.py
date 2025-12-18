import cv2
from pathlib import Path
import json
import os

# Load the latest screenshot
RAW_DIR = Path("data/raw")
raw_files = sorted(RAW_DIR.glob("*.png"))

if not raw_files:
    print("[ERROR] No screenshots found in raw folder.")
    exit()

img_path = str(raw_files[-1])
img = cv2.imread(img_path)
img_copy = img.copy()

# Dictionary to store ROIs
roi_coords = {}
roi_names = ["ribbon"]
current_roi = 0
drawing = False
x_start, y_start = -1, -1

def select_roi(event, x, y, flags, param):
    global x_start, y_start, drawing, img_copy, current_roi

    if current_roi >= len(roi_names):
        return  # all ROIs selected

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start = x, y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img_copy.copy()
            cv2.rectangle(temp_img, (x_start, y_start), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROIs", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        name = roi_names[current_roi]
        roi_coords[name] = {"x1": x_start, "y1": y_start, "x2": x, "y2": y}
        print(f"[INFO] ROI '{name}' selected: {roi_coords[name]}")
        current_roi += 1
        # Draw permanent rectangle
        cv2.rectangle(img_copy, (x_start, y_start), (x, y), (0, 0, 255), 2)
        cv2.imshow("Select ROIs", img_copy)

cv2.imshow("Select ROIs", img_copy)
cv2.setMouseCallback("Select ROIs", select_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ensure config folder exists
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../../config')
os.makedirs(CONFIG_DIR, exist_ok=True)

# Save JSON file
roi_file_path = os.path.join(CONFIG_DIR, "roi_config.json")
with open(roi_file_path, "w") as f:
    json.dump(roi_coords, f, indent=4)
    print(f"[INFO] ROI coordinates saved to {roi_file_path}")
