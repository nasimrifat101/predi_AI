import pandas as pd
import numpy as np
from pathlib import Path

SEQ_LEN = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/logs"
CSV_PATH = DATA_DIR / "predictions.csv"
OUTPUT_PATH = DATA_DIR / "dataset.npz"


SLOT_MAP = {
    # food
    "leg": 0,
    "hotdog": 1,
    "carrot": 2,
    "tomato": 3,

    # toys
    "ballon": 4,
    "horse": 5,
    "cycle": 6,
    "car": 7,

    "unknown": -1
}


def encode_row(row):
    return [
        SLOT_MAP.get(row["slot1"], -1),
        SLOT_MAP.get(row["slot2"], -1),
        SLOT_MAP.get(row["slot3"], -1),
        SLOT_MAP.get(row["slot4"], -1),
        SLOT_MAP.get(row["slot5"], -1),
    ]

    
def prepare_dataset():
    print("[INFO] Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # 🔥 FIX: remove leading/trailing spaces from headers
    df.columns = df.columns.str.strip()

    print("[DEBUG] Columns:", df.columns.tolist())

    print("[INFO] Encoding slots...")
    encoded_slots = df.apply(encode_row, axis=1, result_type="expand")
    encoded_slots.columns = ["s0", "s1", "s2", "s3", "s4"]

    data = encoded_slots.values

    X, y = [], []

    print("[INFO] Building sequences...")
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i : i + SEQ_LEN])
        y.append(data[i + SEQ_LEN])

    X = np.array(X, dtype=np.int8)
    y = np.array(y, dtype=np.int8)

    print(f"[OK] X shape: {X.shape}")
    print(f"[OK] y shape: {y.shape}")

    np.savez(OUTPUT_PATH, X=X, y=y)
    print(f"[OK] Dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_dataset()


