import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ===== IMPORT COMPONENTS =====
from src.capture.screen_capture import capture_screen_once
from src.capture.roi import crop_rois
from src.vision.ribbon_detector import split_ribbon
from src.vision.icon_detector import detect_icon
from src.model.prepare_dataset import prepare_dataset, OUTPUT_PATH

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ===== CONFIG =====
INTERVAL = 40  # seconds between captures
SEQ_LEN = 20
NUM_SLOTS = 5
NUM_CLASSES = 8  # total distinct slot classes (excluding unknown)
IDX_TO_SLOT = ["leg", "hotdog", "carrot", "tomato", "ballon", "horse", "cycle", "car"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "predictions.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "slot_predictor.h5"

ROUND_FILE = LOG_DIR / "last_round.txt"

# Initialize CSV log file
if not log_file.exists():
    with open(log_file, "w") as f:
        f.write("timestamp,round,slot1,slot2,slot3,slot4,slot5\n")

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

# ===== SLOT MAP =====
SLOT_MAP = {
    "leg": 0,
    "hotdog": 1,
    "carrot": 2,
    "tomato": 3,
    "ballon": 4,
    "horse": 5,
    "cycle": 6,
    "car": 7,
    "unknown": -1
}

# ===== HELPER FUNCTIONS =====
def detect_icons_in_slots(slot_paths):
    """Detect icons in each slot image."""
    results = []
    for slot_path in slot_paths:
        img = cv2.imread(str(slot_path))
        if img is None:
            results.append("unknown")
            continue
        label = detect_icon(img)
        results.append(label)
    return results

def build_model():
    """Create and compile LSTM model."""
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQ_LEN, NUM_SLOTS)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# ===== TRAIN / PREDICT FUNCTION =====
def train_or_load_model(round_number):
    """Train every 50 rounds or load existing model."""
    retrain = False
    if not MODEL_PATH.exists():
        retrain = True
        print("[INFO] No model found. Training new model...")
    elif round_number % 50 == 0:
        retrain = True
        print(f"[INFO] Round {round_number} → Retraining model...")

    if retrain:
        prepare_dataset()
        data = np.load(OUTPUT_PATH)
        X, y = data["X"], data["y"]
        y_next = y[:, 0]
        y_onehot = to_categorical(y_next, num_classes=NUM_CLASSES)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.2, shuffle=True, random_state=42
        )
        model = build_model()
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, verbose=0)
        model.save(MODEL_PATH)
        print(f"[OK] Model trained & saved → {MODEL_PATH}")
    else:
        print("[INFO] Using pretrained model...")
        model = load_model(MODEL_PATH)
    return model


def preload_rolling_window():
    if not log_file.exists():
        print("[WARN] No log file found")
        return []

    df = pd.read_csv(log_file)
    df.columns = df.columns.str.strip()

    required = ["slot1", "slot2", "slot3", "slot4", "slot5"]
    for c in required:
        if c not in df.columns:
            print(f"[ERROR] Missing column {c}")
            return []

    if len(df) < SEQ_LEN:
        print("[INFO] Collecting initial sequence...")
        return []

    window = []
    for _, row in df.tail(SEQ_LEN).iterrows():
        window.append([
            SLOT_MAP.get(row["slot1"], -1),
            SLOT_MAP.get(row["slot2"], -1),
            SLOT_MAP.get(row["slot3"], -1),
            SLOT_MAP.get(row["slot4"], -1),
            SLOT_MAP.get(row["slot5"], -1),
        ])

    print(f"[OK] Rolling window loaded ({len(window)} rounds)")
    return window




# ===== MAIN LOOP =====
def run_pipeline():
    global current_round

    rolling_window = preload_rolling_window()  # store last SEQ_LEN rounds as indices

    # Load model at start
    model = train_or_load_model(current_round)

    while True:
        # 1️⃣ Capture screenshot
        screenshot_path = capture_screen_once()

        # 2️⃣ Crop ribbon ROI
        cropped_paths = crop_rois(screenshot_path)
        ribbon_path = cropped_paths["ribbon"]

        # 3️⃣ Split ribbon into slots
        slot_paths = split_ribbon(ribbon_path)

        # 4️⃣ Detect icons in slots
        predictions = detect_icons_in_slots(slot_paths)

        # 5️⃣ Log results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp},{current_round},{','.join(predictions)}\n")

        print(f"[INFO] Round {current_round} → {predictions}")

        # 6️⃣ Update rolling window
        current_indices = [SLOT_MAP.get(slot, -1) for slot in predictions]
        rolling_window.append(current_indices)
        if len(rolling_window) > SEQ_LEN:
            rolling_window.pop(0)

        # 7️⃣ Retrain every 50 rounds
        if current_round % 50 == 0:
            model = train_or_load_model(current_round)

        # 8️⃣ Predict next round if enough data
        if len(rolling_window) == SEQ_LEN:
            input_seq = np.array([rolling_window], dtype=np.int8)
            probs = model.predict(input_seq, verbose=0)
            top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
            top3 = [(IDX_TO_SLOT[idx], float(probs[0, idx])) for idx in top3_idx[0]]
            print("[INFO] Predicted next round top-3 with probabilities:")
            for name, prob in top3:
                print(f"  {name}: {prob:.2f}")
        else:
            print("[INFO] Collecting initial sequence...")

        # 9️⃣ Increment round
        current_round += 1
        with open(ROUND_FILE, "w") as f:
            f.write(str(current_round))

        # 🔟 Wait for next capture
        print(f"[INFO] Waiting {INTERVAL} seconds...\n")
        time.sleep(INTERVAL)
        
        


if __name__ == "__main__":
    run_pipeline()
