import cv2
from pathlib import Path

# ===== PATHS =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LABELED_DIR = PROJECT_ROOT / "data/labels"

# ===== CONFIG =====
MATCH_THRESHOLD = 0.75

# ===== LOAD TEMPLATES =====
TEMPLATES = {}  # {label: [img1, img2, ...]}

for label_dir in LABELED_DIR.iterdir():
    if not label_dir.is_dir():
        continue

    TEMPLATES[label_dir.name] = []
    for img_path in label_dir.glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            TEMPLATES[label_dir.name].append(img)

print(f"[INFO] Loaded templates: { {k: len(v) for k,v in TEMPLATES.items()} }")

# ===== DETECTOR =====
def detect_icon(slot_img):
    gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)

    best_label = "unknown"
    best_score = 0

    for label, templates in TEMPLATES.items():
        for template in templates:
            if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                continue

            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            score = res.max()

            if score > best_score:
                best_score = score
                best_label = label

    if best_score < MATCH_THRESHOLD:
        return "unknown"

    return best_label
