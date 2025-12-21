import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random

# ===== Terminal Colors =====
from colorama import init, Fore, Style
init(autoreset=True)

# ===== IMPORT COMPONENTS =====
from src.capture.screen_capture import capture_screen_once
from src.capture.roi import crop_rois
from src.vision.ribbon_detector import split_ribbon
from src.vision.icon_detector import detect_icon

# ===== CONFIG =====
INTERVAL = 40  # seconds between captures
LOOKBACK = 30
COOLDOWN = 3
NUM_SLOTS = 5

IDX_TO_SLOT = ["leg", "hotdog", "carrot", "tomato", "ballon", "horse", "cycle", "car"]
SLOT_MAP = {name: i for i, name in enumerate(IDX_TO_SLOT)}
SLOT_MAP["unknown"] = -1

FOOD = {"leg", "hotdog", "carrot", "tomato"}
TOYS = {"ballon", "horse", "cycle", "car"}
MULTIPLIER = {"leg":5,"hotdog":5,"carrot":5,"tomato":5,"ballon":10,"horse":15,"cycle":25,"car":45}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "predictions.csv"
ROUND_FILE = LOG_DIR / "last_round.txt"

# Initialize CSV log file
if not log_file.exists():
    with open(log_file, "w") as f:
        f.write("timestamp,round,slot1,slot2,slot3,slot4,slot5\n")

# Initialize starting round
if ROUND_FILE.exists():
    with open(ROUND_FILE, "r") as f:
        content = f.read().strip()
        current_round = int(content) if content.isdigit() else int(input("Enter starting round number: "))
else:
    current_round = int(input("Enter starting round number: "))

# ===== HELPER FUNCTIONS =====
def detect_icons_in_slots(slot_paths):
    results = []
    for slot_path in slot_paths:
        img = cv2.imread(str(slot_path))
        if img is None:
            results.append("unknown")
            continue
        label = detect_icon(img)
        results.append(label)
    return results

def compute_streaks(df):
    streaks = defaultdict(int)
    if df.empty:
        return streaks
    last_values = df["slot1"].tolist()
    last_value = None
    count = 0
    for val in reversed(last_values):
        if val == last_value:
            count += 1
        else:
            count = 1
            last_value = val
        streaks[val] = max(streaks[val], count)
    return streaks

def detect_short_term_repeats(df, window=5):
    """Detect recent repeats in slot1 and boost prediction"""
    if len(df) < window:
        return {}
    recent = df['slot1'].tail(window).tolist()
    repeats = Counter(recent)
    repeat_scores = {}
    for item, count in repeats.items():
        if count > 1:
            repeat_scores[item] = count * 0.5
    return repeat_scores

def predict_top4_slot1(df):
    if df.empty:
        return random.sample(FOOD, 2) + random.sample(TOYS, 2)

    history = df.tail(LOOKBACK)
    freq = Counter(history["slot1"])
    overdue_scores = defaultdict(float)
    streaks = compute_streaks(df)
    repeats = detect_short_term_repeats(df, window=5)

    for item in freq:
        last_seen = df[df["slot1"] == item].index.max()
        gap = len(df) - last_seen
        overdue_scores[item] = np.log1p(gap) + freq[item]*0.3 + streaks.get(item,0)*0.5
        overdue_scores[item] += repeats.get(item, 0)  # repeat boost
        overdue_scores[item] += random.uniform(0,0.1)

    # Penalize very recent items slightly
    recent = list(df.tail(COOLDOWN)["slot1"])
    for item in recent:
        overdue_scores[item] -= 0.1

    # Split food & toy
    food_scores = {k:v for k,v in overdue_scores.items() if k in FOOD}
    toy_scores  = {k:v for k,v in overdue_scores.items() if k in TOYS}

    top_food = sorted(food_scores, key=food_scores.get, reverse=True)[:2]
    top_toy  = sorted(toy_scores, key=toy_scores.get, reverse=True)[:2]

    # Ensure always 2 food & 2 toys
    while len(top_food)<2:
        top_food.append(random.choice(list(FOOD)))
    while len(top_toy)<2:
        top_toy.append(random.choice(list(TOYS)))

    return top_food + top_toy

def allocate_bets(top4):
    bets = {}
    for i, slot in enumerate(top4):
        if slot in FOOD:
            bets[slot] = 100 if i==0 else 30
        else:
            bets[slot] = 20 if i>=2 else 10
    return bets

def calculate_profit(detected, bets):
    """Profit based on previous round bets"""
    if not bets:
        return 0
    total_invested = sum(bets.values())
    slot1_item = detected[0]
    earned = 0
    for item, bet_amount in bets.items():
        if item == slot1_item:
            earned += bet_amount * MULTIPLIER.get(item, 1)
    profit = earned - total_invested
    return profit

# ===== MAIN LOOP =====
def run_pipeline():
    global current_round
    cumulative_profit = 0
    prev_bets = None
    next_round_prediction = None

    while True:
        screenshot_path = capture_screen_once()
        cropped_paths = crop_rois(screenshot_path)
        ribbon_path = cropped_paths["ribbon"]
        slot_paths = split_ribbon(ribbon_path)
        detected = detect_icons_in_slots(slot_paths)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp},{current_round},{','.join(detected)}\n")

        print(Fore.CYAN + f"[INFO] Round {current_round} → detected: {detected}")

        df = pd.read_csv(log_file)
        df.columns = df.columns.str.strip()

        # Calculate profit from previous round
        profit = calculate_profit(detected, prev_bets)
        cumulative_profit += profit
        if prev_bets:
            print(Fore.GREEN + f"[PROFIT] Previous round profit: {profit} points")
            print(Fore.MAGENTA + f"[CUMULATIVE PROFIT]: {cumulative_profit} points")

        # Predict top4 for next round
        top4 = predict_top4_slot1(df)
        next_round_prediction = allocate_bets(top4)
        print(Fore.YELLOW + f"[NEXT ROUND] Top-4 Slot1 prediction: {top4}")
        print(Fore.YELLOW + f"[NEXT ROUND] Suggested bets: {next_round_prediction}")

        # Prepare for next round
        prev_bets = next_round_prediction
        current_round += 1
        with open(ROUND_FILE, "w") as f:
            f.write(str(current_round))

        print(Fore.BLUE + f"[INFO] Waiting {INTERVAL} seconds...\n")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_pipeline()
