import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
from typing import List, Dict, Tuple
import os

# ===== Terminal Colors =====
from colorama import init, Fore, Style
init(autoreset=True)

# ===== IMPORT COMPONENTS =====
from src.capture.screen_capture import capture_screen_once
from src.capture.roi import crop_rois
from src.vision.ribbon_detector import split_ribbon
from src.vision.icon_detector import detect_icon

# ===== CONFIG =====
INTERVAL = 40
LOOKBACK = 30
COOLDOWN = 3

FOOD = {"leg", "hotdog", "carrot", "tomato"}
TOYS = {"ballon", "horse", "cycle", "car"}

MULTIPLIER = {
    "leg": 5, "hotdog": 5, "carrot": 5, "tomato": 5,
    "ballon": 10, "horse": 15, "cycle": 25, "car": 45
}

# ===== FIND YOUR EXISTING DATA FOLDER =====
# Direct path to your data folder
existing_data_folder = Path(__file__).resolve().parent.parent / "data" / "logs"
existing_data_folder.mkdir(parents=True, exist_ok=True)

log_file = existing_data_folder / "predictions.csv"
ROUND_FILE = existing_data_folder / "last_round.txt"

# ===== CSV INIT =====
if not log_file.exists():
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("timestamp,round,slot1,slot2,slot3,slot4,slot5\n")

# ===== ROUND NUMBER =====
if ROUND_FILE.exists():
    with open(ROUND_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.isdigit():
            current_round = int(content)
        else:
            current_round = int(input("Enter starting round number: "))
else:
    current_round = int(input("Enter starting round number: "))

# ===== HELPERS =====
def detect_icons_in_slots(slot_paths):
    """Detect icons in slot images"""
    results = []
    for slot_path in slot_paths:
        img = cv2.imread(str(slot_path))
        if img is None:
            results.append("unknown")
            continue
        label = detect_icon(img)
        results.append(label)
    return results

def calculate_profit(detected, bets):
    """Calculate profit from bets"""
    if not bets:
        return 0, {}
    
    total_invested = sum(bets.values())
    slot1_item = detected[0]
    earned = 0
    
    for item, bet in bets.items():
        if item == slot1_item:
            earned += bet * MULTIPLIER.get(item, 1)
    
    profit = earned - total_invested
    details = {
        "slot1": slot1_item,
        "invested": total_invested,
        "earned": earned,
        "profit": profit,
        "won": earned > 0,
        "winning_bet": bets.get(slot1_item, 0)
    }
    
    return profit, details

# ===== PREDICTION LOGIC =====
def predict_top4_slot1(df):
    """Predict top 4 items for next slot1 - Your winning formula"""
    if df.empty or len(df) < 5:
        return ["leg", "hotdog", "ballon", "cycle"]
    
    df.columns = df.columns.str.strip()
    scores = defaultdict(float)
    recent = df.tail(LOOKBACK)
    
    # 1. Frequency analysis
    freq = Counter(recent["slot1"])
    total = sum(freq.values())
    if total > 0:
        for item, count in freq.items():
            if item != "unknown":
                scores[item] += (count / total) * 3.0
    
    # 2. Slot2 → Slot1 pattern
    if len(df) >= 2:
        last_slot2 = df.iloc[-1]["slot2"]
        if last_slot2 != "unknown":
            scores[last_slot2] += 2.5
    
    # 3. Streak detection
    last_items = df["slot1"].tail(5).tolist()
    if len(last_items) >= 2:
        current_item = last_items[-1]
        streak_len = 1
        for item in reversed(last_items[:-1]):
            if item == current_item:
                streak_len += 1
            else:
                break
        
        if streak_len >= 2:
            scores[current_item] += streak_len * 2.0
        
        if streak_len >= 4:
            for item in FOOD | TOYS:
                if item != current_item:
                    scores[item] += 1.5
    
    # 4. Position analysis
    for _, row in recent.iterrows():
        scores[row["slot2"]] += 1.5
        scores[row["slot3"]] += 0.8
    
    # 5. Cooldown penalty
    cooldown_items = df.tail(COOLDOWN)["slot1"].tolist()
    for item in cooldown_items:
        if item in scores:
            scores[item] -= 1.0
    
    # 6. Food-toy correlation
    if len(df) > 0:
        last_slot1 = df.iloc[-1]["slot1"]
        correlations = {
            "leg": "ballon",
            "hotdog": "horse",
            "carrot": "cycle",
            "tomato": "car",
        }
        if last_slot1 in correlations:
            correlated = correlations[last_slot1]
            scores[correlated] += 1.8
    
    # Remove unknowns
    if "unknown" in scores:
        del scores["unknown"]
    
    # Ensure all items have minimal score
    for item in FOOD | TOYS:
        if item not in scores:
            scores[item] = 0.1
    
    # Get ranked items
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Show top predictions with confidence percentages
    print(Fore.CYAN + "[PREDICTION CONFIDENCE]:")
    for i, (item, score) in enumerate(ranked[:4]):
        # Convert score to percentage (normalize to 0-100%)
        max_score = ranked[0][1] if ranked else 1.0
        confidence = min(99, int((score / max_score) * 100)) if max_score > 0 else 0
        print(Fore.CYAN + f"  {i+1}. {item}: {confidence}%")
    
    # Separate food and toys
    food_items = [item for item, _ in ranked if item in FOOD][:2]
    toy_items = [item for item, _ in ranked if item in TOYS][:2]
    
    # Fill missing
    while len(food_items) < 2:
        food_items.append(random.choice(list(FOOD)))
    while len(toy_items) < 2:
        toy_items.append(random.choice(list(TOYS)))
    
    return food_items + toy_items

def allocate_bets(top4):
    """Allocate bets based on your successful pattern"""
    bets = {}
    for i, item in enumerate(top4):
        if item in FOOD:
            bets[item] = 100 if i == 0 else 30
        else:
            bets[item] = 20
    return bets

# ===== STATISTICS TRACKER =====
class StatisticsTracker:
    def __init__(self):
        self.total_rounds = 0
        self.rounds_won = 0
        self.total_profit = 0
        self.prediction_accuracy = []  # Track if predictions were correct
        
    def update(self, won: bool, profit: int, predicted_top4: List[str], actual_slot1: str):
        """Update statistics"""
        self.total_rounds += 1
        if won:
            self.rounds_won += 1
        self.total_profit += profit
        
        # Check prediction accuracy (if actual slot1 was in top4 predictions)
        prediction_correct = actual_slot1 in predicted_top4
        self.prediction_accuracy.append(prediction_correct)
        
    def get_stats(self) -> Dict:
        """Get current statistics"""
        if self.total_rounds == 0:
            return {
                "win_rate": 0,
                "avg_profit": 0,
                "prediction_accuracy": 0,
                "total_profit": 0
            }
        
        win_rate = (self.rounds_won / self.total_rounds) * 100
        avg_profit = self.total_profit / self.total_rounds
        
        # Calculate prediction accuracy (last 20 rounds)
        recent_accuracy = self.prediction_accuracy[-20:] if len(self.prediction_accuracy) > 20 else self.prediction_accuracy
        pred_accuracy = (sum(recent_accuracy) / len(recent_accuracy) * 100) if recent_accuracy else 0
        
        return {
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "prediction_accuracy": pred_accuracy,
            "total_profit": self.total_profit,
            "total_rounds": self.total_rounds
        }

# ===== MAIN LOOP =====
def run_pipeline():
    global current_round
    
    cumulative_profit = 0
    prev_bets = None
    tracker = StatisticsTracker()
    
    print(Fore.GREEN + "="*60)
    print(Fore.GREEN + "🎯 PREDICTION SYSTEM 🎯")
    print(Fore.GREEN + f"Starting round: {current_round}")
    print(Fore.GREEN + "="*60 + "\n")
    
    try:
        while True:
            try:
                # 1. Capture
                print(Fore.CYAN + f"\n[ROUND {current_round}] Capturing...")
                screenshot_path = capture_screen_once()
                cropped_paths = crop_rois(screenshot_path)
                ribbon_path = cropped_paths["ribbon"]
                slot_paths = split_ribbon(ribbon_path)
                
                # 2. Detect
                detected = detect_icons_in_slots(slot_paths)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(Fore.CYAN + f"[RESULT] Slot1: {detected[0]} | Full: {detected}")
                
                # 3. Write to CSV
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp},{current_round},{','.join(detected)}\n")
                
                # 4. Load data for prediction
                try:
                    df = pd.read_csv(log_file)
                    df.columns = df.columns.str.strip()
                except:
                    df = pd.DataFrame()
                
                # 5. Calculate profit from previous round
                if prev_bets:
                    profit, details = calculate_profit(detected, prev_bets)
                    cumulative_profit += profit
                    
                    # Update tracker with previous prediction (top4_prev was in prev iteration)
                    tracker.update(details["won"], profit, list(prev_bets.keys()), detected[0])
                    
                    stats = tracker.get_stats()
                    
                    if details["won"]:
                        roi = (details["profit"] / details["invested"] * 100) if details["invested"] > 0 else 0
                        print(Fore.GREEN + "━"*40)
                        print(Fore.GREEN + f"✅ WIN! +{profit} points (ROI: {roi:.0f}%)")
                        print(Fore.GREEN + f"   Bet {details['winning_bet']} on {details['slot1']}")
                        print(Fore.GREEN + "━"*40)
                    else:
                        print(Fore.RED + "━"*40)
                        print(Fore.RED + f"❌ LOSS: {profit} points")
                        print(Fore.RED + "━"*40)
                    
                    # Display statistics
                    print(Fore.MAGENTA + f"[STATS] Win rate: {stats['win_rate']:.1f}%")
                    print(Fore.MAGENTA + f"[STATS] Prediction accuracy: {stats['prediction_accuracy']:.1f}%")
                    print(Fore.MAGENTA + f"[STATS] Avg profit/round: {stats['avg_profit']:+.1f}")
                    print(Fore.MAGENTA + f"[TOTAL PROFIT] {cumulative_profit:+}")
                
                # 6. Predict next round
                if not df.empty and len(df) >= 5:
                    top4 = predict_top4_slot1(df)
                else:
                    top4 = ["leg", "hotdog", "ballon", "cycle"]
                    print(Fore.YELLOW + "[INFO] Using default prediction")
                
                # 7. Allocate bets
                bets = allocate_bets(top4)
                
                print(Fore.YELLOW + f"[NEXT ROUND] Prediction: {top4}")
                print(Fore.YELLOW + f"[BETS] {bets} (Total: {sum(bets.values())} points)")
                
                # 8. Calculate potential ROI for each bet
                total_bet = sum(bets.values())
                if total_bet > 0:
                    print(Fore.CYAN + "[POTENTIAL ROI]:")
                    for item, bet in bets.items():
                        potential_profit = (bet * MULTIPLIER[item]) - total_bet
                        roi = (potential_profit / bet * 100) if bet > 0 else 0
                        if potential_profit > 0:
                            print(Fore.GREEN + f"  {item}: {bet} pts → {roi:+.0f}% ROI")
                        else:
                            print(Fore.YELLOW + f"  {item}: {bet} pts → {roi:+.0f}% ROI")
                
                # 9. Prepare for next round
                prev_bets = bets
                current_round += 1
                
                # Save round number
                with open(ROUND_FILE, "w", encoding="utf-8") as f:
                    f.write(str(current_round))
                
                # 10. Wait
                print(Fore.BLUE + f"\n⏰ Next round in {INTERVAL} seconds...")
                print(Fore.BLUE + "-"*60)
                
                # Simple countdown
                for i in range(INTERVAL, 0, -10):
                    if i <= 30:
                        print(Fore.BLUE + f"   {i} seconds...")
                    time.sleep(min(10, i))
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(Fore.RED + f"[ERROR] {e}")
                time.sleep(5)
    
    except KeyboardInterrupt:
        # Final statistics
        stats = tracker.get_stats()
        print(Fore.RED + "\n🛑 Stopping prediction system...")
        print(Fore.MAGENTA + "="*60)
        print(Fore.MAGENTA + "FINAL STATISTICS:")
        print(Fore.MAGENTA + f"  Total rounds: {stats['total_rounds']}")
        print(Fore.MAGENTA + f"  Win rate: {stats['win_rate']:.1f}%")
        print(Fore.MAGENTA + f"  Prediction accuracy: {stats['prediction_accuracy']:.1f}%")
        print(Fore.MAGENTA + f"  Avg profit/round: {stats['avg_profit']:+.1f}")
        print(Fore.MAGENTA + f"  Final profit: {cumulative_profit:+}")
        
        if cumulative_profit > 0:
            print(Fore.GREEN + "💰 PROFITABLE SESSION! 💰")
        else:
            print(Fore.RED + "📉 UNPROFITABLE SESSION")
        
        print(Fore.MAGENTA + "="*60)

if __name__ == "__main__":
    run_pipeline()