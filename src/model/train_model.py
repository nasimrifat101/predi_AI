import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# ===== CONFIG =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/logs"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
DATASET_PATH = DATA_DIR / "dataset.npz"
MODEL_PATH = MODEL_DIR / "slot_predictor.h5"

SEQ_LEN = 50       # last 50 rounds used as input
NUM_SLOTS = 5      # slots per round (ignored for output)
NUM_CLASSES = 8    # number of possible items

# ===== SLOT MAP =====
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

# Reverse map for predictions
IDX_TO_SLOT = {v: k for k, v in SLOT_MAP.items() if v >= 0}

# ===== LOAD DATA =====
print("[INFO] Loading dataset...")
data = np.load(DATASET_PATH)
X, y = data["X"], data["y"]
print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

# ===== PREPARE TARGET =====
y_next = y[:, 0]  # shape (num_samples,)
y_onehot = to_categorical(y_next, num_classes=NUM_CLASSES)
print(f"[INFO] y_onehot shape: {y_onehot.shape}")

# ===== TRAIN-TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, shuffle=True, random_state=42
)
print(f"[INFO] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ===== BUILD MODEL =====
model = Sequential()
model.add(LSTM(64, input_shape=(SEQ_LEN, NUM_SLOTS)))
model.add(Dense(32, activation="relu"))
model.add(Dense(NUM_CLASSES, activation="softmax"))  # predict probability for each item

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# ===== TRAIN MODEL =====
print("[INFO] Training model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# ===== SAVE MODEL =====
model.save(MODEL_PATH)
print(f"[OK] Model saved to {MODEL_PATH}")

# ===== PREDICT TOP-3 WINNERS WITH NAMES =====
def predict_top3_with_probs(x_input):
    """
    Predict top-3 most probable winners for the given input sequences.
    Returns list of tuples: (name, probability)
    """
    probs = model.predict(x_input)  # shape: (num_samples, NUM_CLASSES)
    top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # top 3 indices per sample

    top3_results = []
    for sample_idx, sample in enumerate(top3_idx):
        sample_result = [(IDX_TO_SLOT[idx], float(probs[sample_idx, idx])) for idx in sample]
        top3_results.append(sample_result)
    return top3_results

# Example usage
top3_winners_with_probs = predict_top3_with_probs(X_test[:1])
print("Top 3 predicted winners with probabilities for first test sample:")
for name, prob in top3_winners_with_probs[0]:
    print(f"{name}: {prob:.2f}")