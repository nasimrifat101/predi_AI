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
import pickle

# ===== Terminal Colors =====
from colorama import init, Fore, Style
init(autoreset=True)

# ===== IMPORT COMPONENTS =====
from src.capture.screen_capture import capture_screen_once
from src.capture.roi import crop_rois
from src.vision.ribbon_detector import split_ribbon
from src.vision.icon_detector import detect_icon

# ===== QUICK-LEARN PREDICTOR =====
class QuickLearnPredictor:
    def __init__(self):
        self.symbols = ['leg', 'hotdog', 'carrot', 'tomato', 'ballon', 'horse', 'cycle', 'car']
        self.food_symbols = ['leg', 'hotdog', 'carrot', 'tomato']
        self.toy_symbols = ['ballon', 'horse', 'cycle', 'car']
        
        # Core game mechanics
        self.history = []  # slot1 history
        self.slot2_history = []  # slot2 history
        self.actual_results = []  # What actually happened
        
        # Quick learning
        self.prediction_log = []  # Track predictions
        self.learning_phase = True
        self.learned_patterns = defaultdict(lambda: defaultdict(int))
        
        # High-accuracy betting
        self.balance = 10000
        self.base_unit = 40
        self.total_budget = 220
        
        # Start betting after just 3 rounds of observation
        self.min_observation = 3
        
        print(Fore.GREEN + "[QUICK-LEARN] Ready to learn and bet fast!")
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load any existing data to accelerate learning"""
        log_file = Path(__file__).resolve().parent.parent / "data" / "logs" / "predictions.csv"
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                if 'slot1' in df.columns and 'slot2' in df.columns:
                    # Load last 20 rounds
                    for _, row in df.tail(20).iterrows():
                        slot1 = row['slot1']
                        slot2 = row['slot2']
                        
                        if slot1 in self.symbols:
                            self.history.append(slot1)
                            self.actual_results.append(slot1)
                        
                        if slot2 in self.symbols:
                            self.slot2_history.append(slot2)
                    
                    # Learn from existing data
                    for i in range(len(self.history) - 1):
                        if i < len(self.slot2_history):
                            predicted = self.slot2_history[i]
                            actual = self.history[i + 1]
                            self.learned_patterns[predicted][actual] += 1
                    
                    print(Fore.GREEN + f"[QUICK-LEARN] Loaded {len(self.history)} existing rounds")
                    
                    # If we have enough data, skip learning phase
                    if len(self.history) >= self.min_observation:
                        self.learning_phase = False
                        print(Fore.GREEN + "[QUICK-LEARN] Ready to bet!")
                        
            except Exception as e:
                print(Fore.YELLOW + f"[QUICK-LEARN] Error loading data: {e}")
    
    def predict_with_confidence(self, current_state):
        """Make prediction with calculated confidence"""
        predictions = []
        confidences = []
        
        # Rule 1: Slot2 is primary predictor (60% confidence if we have data)
        if len(current_state) >= 2:
            slot2 = current_state[1]
            if slot2 in self.symbols:
                predictions.append(slot2)
                
                # Calculate confidence based on learned patterns
                if slot2 in self.learned_patterns:
                    total = sum(self.learned_patterns[slot2].values())
                    # Base confidence for slot2 prediction
                    base_confidence = 0.6
                    
                    # Boost if we've seen this pattern before
                    pattern_boost = min(0.3, total / 10)  # Up to 30% boost
                    confidence = base_confidence + pattern_boost
                else:
                    confidence = 0.6  # Base confidence for slot2
                
                confidences.append(min(0.9, confidence))
        
        # Rule 2: Add complementary predictions
        if predictions:  # If we have a slot2 prediction
            main_pred = predictions[0]
            
            # Add backup predictions based on type
            if main_pred in self.food_symbols:
                # Main is food, add toy backups
                toy_options = [t for t in self.toy_symbols if t != main_pred]
                for toy in toy_options[:2]:  # Add up to 2 toys
                    predictions.append(toy)
                    confidences.append(0.3)  # Lower confidence for backups
                
                # Add one more food backup
                other_foods = [f for f in self.food_symbols if f != main_pred]
                if other_foods:
                    predictions.append(random.choice(other_foods))
                    confidences.append(0.4)
            
            else:  # Main is toy
                # Add food backups
                food_options = [f for f in self.food_symbols]
                for food in food_options[:2]:  # Add up to 2 foods
                    predictions.append(food)
                    confidences.append(0.4)  # Higher confidence for food
                
                # Add one more toy backup
                other_toys = [t for t in self.toy_symbols if t != main_pred]
                if other_toys:
                    predictions.append(random.choice(other_toys))
                    confidences.append(0.3)
        
        # Ensure we have at least 4 predictions
        while len(predictions) < 4:
            if len(predictions) < 2:
                # Need more food
                available_food = [f for f in self.food_symbols if f not in predictions]
                if available_food:
                    predictions.append(random.choice(available_food))
                    confidences.append(0.3)
                else:
                    predictions.append('leg')
                    confidences.append(0.3)
            else:
                # Need more toys
                available_toys = [t for t in self.toy_symbols if t not in predictions]
                if available_toys:
                    predictions.append(random.choice(available_toys))
                    confidences.append(0.25)
                else:
                    predictions.append('ballon')
                    confidences.append(0.25)
        
        # Calculate overall confidence
        overall_confidence = max(confidences) if confidences else 0.5
        
        return predictions[:5], confidences[:5], overall_confidence
    
    def calculate_smart_bets(self, predictions, confidences, overall_confidence):
        """Calculate smart bets based on your strategy"""
        if self.learning_phase and len(self.history) < self.min_observation:
            return {}, "learning phase"
        
        # Only bet if confidence is reasonable
        if overall_confidence < 0.4:
            return {}, f"low confidence ({overall_confidence:.0%})"
        
        bets = {}
        
        # Your betting strategy: main food high, others balanced
        bet_allocations = {
            0: 120,  # Main prediction (highest confidence)
            1: 60,   # Secondary
            2: 40,   # Backup
            3: 30,   # First toy/backup
            4: 20    # Last backup
        }
        
        total_bet = 0
        for i, (symbol, confidence) in enumerate(zip(predictions, confidences)):
            if i >= len(bet_allocations):
                break
            
            base_bet = bet_allocations[i]
            
            # Adjust based on confidence
            confidence_multiplier = 0.7 + (confidence * 0.6)  # 0.7-1.3x
            
            # Adjust based on symbol type
            if symbol in self.food_symbols:
                type_multiplier = 1.0
            else:
                type_multiplier = 0.8
            
            # Adjust based on overall confidence
            overall_multiplier = 0.8 + (overall_confidence * 0.4)  # 0.8-1.2x
            
            bet = int(base_bet * confidence_multiplier * type_multiplier * overall_multiplier)
            bet = max(self.base_unit, bet)  # Ensure minimum bet
            
            bets[symbol] = bet
            total_bet += bet
        
        # Ensure total is reasonable
        if total_bet > self.total_budget * 1.5:
            scale_factor = (self.total_budget * 1.5) / total_bet
            for symbol in list(bets.keys()):
                bets[symbol] = int(bets[symbol] * scale_factor)
                if bets[symbol] < self.base_unit:
                    del bets[symbol]
        
        return bets, f"confident ({overall_confidence:.0%})"
    
    def update_learning(self, actual_symbol, current_state, bets_made):
        """Update learning from results"""
        # Add to history
        if actual_symbol in self.symbols:
            self.history.append(actual_symbol)
            self.actual_results.append(actual_symbol)
            
            # Track slot2 from previous state (for learning)
            if len(self.history) >= 2 and len(current_state) >= 2:
                prev_slot2 = current_state[1]
                if prev_slot2 in self.symbols:
                    self.slot2_history.append(prev_slot2)
                    
                    # Learn pattern: slot2 -> next slot1
                    self.learned_patterns[prev_slot2][actual_symbol] += 1
            
            # Track prediction accuracy if we made bets
            if bets_made:
                was_correct = actual_symbol in bets_made
                self.prediction_log.append({
                    'round': len(self.history),
                    'predicted': list(bets_made.keys()),
                    'actual': actual_symbol,
                    'correct': was_correct,
                    'bets': bets_made
                })
            
            # Check if we can exit learning phase
            if self.learning_phase and len(self.history) >= self.min_observation:
                self.learning_phase = False
                print(Fore.GREEN + "[QUICK-LEARN] Learning phase complete! Ready to bet.")
    
    def get_accuracy(self):
        """Get current accuracy"""
        if not self.prediction_log:
            return 0.0
        
        correct = sum(1 for log in self.prediction_log if log['correct'])
        return correct / len(self.prediction_log)

# ===== CONFIG =====
INTERVAL = 40
MULTIPLIER = {
    "leg": 5, "hotdog": 5, "carrot": 5, "tomato": 5,
    "ballon": 10, "horse": 15, "cycle": 25, "car": 45
}

# ===== FOLDER SETUP =====
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

# ===== MAIN LOOP =====
def run_pipeline():
    global current_round
    
    cumulative_profit = 0
    prev_bets = None
    predictor = QuickLearnPredictor()
    
    print(Fore.GREEN + "="*70)
    print(Fore.GREEN + "⚡ QUICK-LEARN PREDICTOR ⚡")
    print(Fore.GREEN + f"Starting round: {current_round}")
    print(Fore.GREEN + f"Balance: {predictor.balance}")
    print(Fore.GREEN + f"Learning: {predictor.min_observation} rounds then betting")
    print(Fore.GREEN + "="*70 + "\n")
    
    try:
        while True:
            try:
                print(Fore.CYAN + f"\n[ROUND {current_round}]")
                print(Fore.CYAN + "─" * 40)
                
                # 1. Capture
                print(Fore.CYAN + "[1/5] Capturing...")
                screenshot_path = capture_screen_once()
                cropped_paths = crop_rois(screenshot_path)
                ribbon_path = cropped_paths["ribbon"]
                slot_paths = split_ribbon(ribbon_path)
                
                # 2. Detect
                print(Fore.CYAN + "[2/5] Detecting...")
                detected = detect_icons_in_slots(slot_paths)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(Fore.CYAN + f"[RESULT] Slot1: {detected[0]} | Slots: {detected}")
                
                # 3. Log results
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp},{current_round},{','.join(detected)}\n")
                
                # 4. Calculate previous round profit
                if prev_bets:
                    profit, details = calculate_profit(detected, prev_bets)
                    cumulative_profit += profit
                    predictor.balance += profit
                    
                    # Update predictor learning
                    predictor.update_learning(detected[0], detected, prev_bets)
                    
                    if details["won"]:
                        roi = (details["profit"] / details["invested"] * 100) if details["invested"] > 0 else 0
                        print(Fore.GREEN + "━" * 40)
                        print(Fore.GREEN + f"✅ QUICK WIN! +{profit} (ROI: {roi:.0f}%)")
                        print(Fore.GREEN + f"   Bet {details['winning_bet']} on {details['slot1']}")
                        print(Fore.GREEN + "━" * 40)
                    else:
                        print(Fore.RED + "━" * 40)
                        print(Fore.RED + f"❌ LOSS: {profit}")
                        print(Fore.RED + "━" * 40)
                    
                    accuracy = predictor.get_accuracy()
                    print(Fore.MAGENTA + f"[LEARNING] Accuracy: {accuracy:.1%}")
                    print(Fore.MAGENTA + f"[BALANCE] {predictor.balance}")
                    print(Fore.MAGENTA + f"[TOTAL] Profit: {cumulative_profit:+}")
                else:
                    # Still update learning
                    predictor.update_learning(detected[0], detected, {})
                
                # 5. Make prediction for next round
                print(Fore.CYAN + "[3/5] Quick analysis...")
                predictions, confidences, overall_confidence = predictor.predict_with_confidence(detected)
                
                print(Fore.YELLOW + f"[PREDICTIONS] {predictions}")
                print(Fore.YELLOW + f"[CONFIDENCES] {[f'{c:.0%}' for c in confidences]}")
                print(Fore.YELLOW + f"[OVERALL] {overall_confidence:.0%} confidence")
                
                # Show key insight
                if len(detected) >= 2:
                    print(Fore.CYAN + f"[KEY] Current Slot2 ({detected[1]}) → Next Slot1")
                
                # 6. Calculate bets
                print(Fore.CYAN + "[4/5] Smart betting...")
                bets, bet_reason = predictor.calculate_smart_bets(predictions, confidences, overall_confidence)
                
                if bets:
                    total_bet = sum(bets.values())
                    
                    # Show bet distribution
                    food_bets = {k:v for k,v in bets.items() if k in predictor.food_symbols}
                    toy_bets = {k:v for k,v in bets.items() if k in predictor.toy_symbols}
                    
                    print(Fore.GREEN + f"  Total bet: {total_bet} ({bet_reason})")
                    
                    if food_bets:
                        print(Fore.CYAN + "  🍎 FOOD:")
                        for symbol, bet in sorted(food_bets.items(), key=lambda x: x[1], reverse=True):
                            print(Fore.GREEN + f"    {symbol}: {bet}")
                    
                    if toy_bets:
                        print(Fore.CYAN + "  🎪 TOYS:")
                        for symbol, bet in sorted(toy_bets.items(), key=lambda x: x[1], reverse=True):
                            print(Fore.YELLOW + f"    {symbol}: {bet}")
                    
                    # Show potential
                    print(Fore.CYAN + "[POTENTIAL]:")
                    main_pred = predictions[0]
                    for item, bet in bets.items():
                        potential = (bet * MULTIPLIER[item]) - total_bet
                        roi = (potential / total_bet * 100) if total_bet > 0 else 0
                        
                        if item == main_pred:
                            marker = "🎯"
                            color = Fore.GREEN
                        elif potential > 0:
                            marker = "💰"
                            color = Fore.YELLOW
                        else:
                            marker = "🛡️"
                            color = Fore.CYAN
                        
                        print(color + f"    {marker} {item}: {bet} → ROI: {roi:+.0f}%")
                else:
                    print(Fore.YELLOW + f"  No bets ({bet_reason})")
                    bets = {}
                
                # 7. Prepare for next round
                prev_bets = bets
                current_round += 1
                
                # Save round number
                with open(ROUND_FILE, "w", encoding="utf-8") as f:
                    f.write(str(current_round))
                
                # 8. Wait for next round
                print(Fore.BLUE + f"\n[5/5] Next quick analysis in {INTERVAL}s...")
                print(Fore.BLUE + "─" * 40)
                
                # Fast countdown
                for i in range(INTERVAL, 0, -5):
                    if i <= 15 or i % 10 == 0:
                        print(Fore.BLUE + f"   {i}s...")
                    time.sleep(min(5, i))
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(Fore.RED + f"[ERROR] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    
    except KeyboardInterrupt:
        # Final results
        accuracy = predictor.get_accuracy()
        
        print(Fore.RED + "\n🛑 STOPPED")
        print(Fore.MAGENTA + "="*70)
        print(Fore.MAGENTA + "QUICK-LEARN RESULTS:")
        print(Fore.MAGENTA + f"  Rounds observed: {len(predictor.history)}")
        print(Fore.MAGENTA + f"  Bets made: {len(predictor.prediction_log)}")
        print(Fore.MAGENTA + f"  Accuracy: {accuracy:.1%}")
        print(Fore.MAGENTA + f"  Final balance: {predictor.balance}")
        print(Fore.MAGENTA + f"  Total profit: {cumulative_profit:+}")
        
        # Show learned patterns
        if predictor.learned_patterns:
            print(Fore.MAGENTA + "\nLEARNED PATTERNS (Slot2 → Next Slot1):")
            for slot2, outcomes in list(predictor.learned_patterns.items())[:5]:
                total = sum(outcomes.values())
                if total > 0:
                    most_common = max(outcomes.items(), key=lambda x: x[1])
                    reliability = most_common[1] / total
                    print(Fore.CYAN + f"  {slot2} → {most_common[0]} ({reliability:.0%} reliable)")
        
        print(Fore.MAGENTA + "="*70)

if __name__ == "__main__":
    run_pipeline()