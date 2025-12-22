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

# ===== BALANCED PREDICTOR =====
class BalancedPredictor:
    def __init__(self):
        self.symbols = ['leg', 'hotdog', 'carrot', 'tomato', 'ballon', 'horse', 'cycle', 'car']
        self.food_symbols = ['leg', 'hotdog', 'carrot', 'tomato']
        self.toy_symbols = ['ballon', 'horse', 'cycle', 'car']
        
        # Core insight: slot2 is next round's slot1
        self.slot2_predictions = []
        self.history = []
        
        # Pattern tracking
        self.transition_patterns = defaultdict(lambda: defaultdict(int))
        self.symbol_frequencies = defaultdict(int)
        self.streak_patterns = []
        
        # Performance
        self.accuracy_history = []
        self.win_history = []
        
        # Balanced betting
        self.balance = 10000
        self.base_unit = 40  # Base betting unit
        self.total_budget = 220  # Total per round (like you mentioned)
        
        # Learning requirements
        self.min_data_for_betting = 20  # Need 20 rounds of data
        
        print(Fore.GREEN + "[BALANCED] Predictor initialized")
        self.load_history()
    
    def load_history(self):
        """Load historical data"""
        log_file = Path(__file__).resolve().parent.parent / "data" / "logs" / "predictions.csv"
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                if 'slot1' in df.columns:
                    self.history = df['slot1'].tail(100).tolist()
                    
                    # Calculate frequencies
                    for symbol in self.history:
                        if symbol in self.symbols:
                            self.symbol_frequencies[symbol] += 1
                    
                    print(Fore.GREEN + f"[BALANCED] Loaded {len(self.history)} rounds")
            except:
                pass
    
    def analyze_patterns(self, current_state):
        """Analyze all patterns for balanced prediction"""
        patterns = defaultdict(float)
        
        # 1. Slot2 is strongest predictor (40% weight)
        if len(current_state) >= 2:
            slot2 = current_state[1]
            if slot2 in self.symbols:
                patterns[slot2] += 0.40
                # Also consider transitions from this symbol
                if slot2 in self.transition_patterns:
                    transitions = self.transition_patterns[slot2]
                    total = sum(transitions.values())
                    if total > 0:
                        for next_sym, count in transitions.items():
                            patterns[next_sym] += (count / total) * 0.10
        
        # 2. Food/Toy balance pattern (25% weight)
        food_count = sum(1 for s in current_state if s in self.food_symbols)
        
        if food_count >= 4:  # Many food -> toy likely
            for toy in self.toy_symbols:
                patterns[toy] += 0.25 / len(self.toy_symbols)
        elif food_count <= 1:  # Many toys -> food likely
            for food in self.food_symbols:
                patterns[food] += 0.25 / len(self.food_symbols)
        else:  # Balanced -> slightly favor food
            for food in self.food_symbols:
                patterns[food] += 0.15 / len(self.food_symbols)
            for toy in self.toy_symbols:
                patterns[toy] += 0.10 / len(self.toy_symbols)
        
        # 3. Recent frequency (20% weight)
        recent = self.history[-10:] if len(self.history) >= 10 else self.history
        freq = Counter(recent)
        total_freq = sum(freq.values())
        if total_freq > 0:
            for symbol, count in freq.items():
                if symbol in self.symbols:
                    patterns[symbol] += (count / total_freq) * 0.20
        
        # 4. Streak patterns (15% weight)
        if len(self.history) >= 3:
            last_symbol = self.history[-1]
            streak = 1
            for i in range(2, min(4, len(self.history)) + 1):
                if self.history[-i] == last_symbol:
                    streak += 1
                else:
                    break
            
            if streak >= 3:
                # After 3+ streak, different symbol likely
                for symbol in self.symbols:
                    if symbol != last_symbol:
                        patterns[symbol] += 0.15 / (len(self.symbols) - 1)
            elif streak == 2:
                # Moderate chance to continue
                patterns[last_symbol] += 0.10
        
        return patterns
    
    def predict_balanced(self, current_state):
        """Get balanced predictions with confidence"""
        patterns = self.analyze_patterns(current_state)
        
        if not patterns:
            # Default balanced prediction
            return {
                'predictions': ['carrot', 'tomato', 'leg', 'ballon', 'horse'],
                'confidences': [0.3, 0.2, 0.15, 0.2, 0.15],
                'total_confidence': 0.3
            }
        
        # Sort by probability
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 5 predictions (2-3 food, 2-3 toys)
        top_predictions = []
        top_confidences = []
        
        food_selected = 0
        toy_selected = 0
        
        for symbol, confidence in sorted_patterns:
            if symbol in self.food_symbols and food_selected < 3:
                top_predictions.append(symbol)
                top_confidences.append(confidence)
                food_selected += 1
            elif symbol in self.toy_symbols and toy_selected < 3:
                top_predictions.append(symbol)
                top_confidences.append(confidence)
                toy_selected += 1
            
            if len(top_predictions) >= 5:
                break
        
        # Ensure we have at least 2 of each
        while food_selected < 2:
            remaining_food = [f for f in self.food_symbols if f not in top_predictions]
            if remaining_food:
                symbol = random.choice(remaining_food)
                top_predictions.append(symbol)
                top_confidences.append(0.05)
                food_selected += 1
        
        while toy_selected < 2:
            remaining_toys = [t for t in self.toy_symbols if t not in top_predictions]
            if remaining_toys:
                symbol = random.choice(remaining_toys)
                top_predictions.append(symbol)
                top_confidences.append(0.05)
                toy_selected += 1
        
        # Calculate total confidence (gap between top and average)
        if len(top_confidences) >= 2:
            top_confidence = top_confidences[0]
            avg_others = sum(top_confidences[1:]) / len(top_confidences[1:])
            total_confidence = min(0.9, max(0.1, top_confidence - avg_others))
        else:
            total_confidence = 0.3
        
        return {
            'predictions': top_predictions,
            'confidences': top_confidences,
            'total_confidence': total_confidence
        }
    
    def calculate_balanced_bets(self, prediction_data):
        """Calculate balanced bets like you described"""
        predictions = prediction_data['predictions']
        confidences = prediction_data['confidences']
        total_confidence = prediction_data['total_confidence']
        
        # Only bet if we have enough data and reasonable confidence
        if len(self.history) < self.min_data_for_betting or total_confidence < 0.3:
            return {}
        
        bets = {}
        total_bet = 0
        
        # Sort by confidence
        sorted_items = sorted(zip(predictions, confidences), key=lambda x: x[1], reverse=True)
        
        # Betting strategy:
        # 1st (highest confidence): 120 if food, 100 if toy
        # 2nd: 60
        # 3rd: 40  
        # 4th: 30
        # 5th: 20
        # Total: ~270 (adjusts based on confidence)
        
        bet_weights = [120, 60, 40, 30, 20]
        
        for i, (symbol, confidence) in enumerate(sorted_items[:5]):
            if i >= len(bet_weights):
                break
            
            # Adjust bet based on confidence and position
            base_bet = bet_weights[i]
            
            # First position gets confidence boost
            if i == 0:
                confidence_multiplier = min(1.5, 0.8 + total_confidence)
            else:
                confidence_multiplier = 0.7 + (confidence * 0.5)
            
            # Adjust for symbol type
            if symbol in self.food_symbols:
                type_multiplier = 1.0
            else:
                type_multiplier = 0.8  # Toys slightly less
            
            # Adjust based on recent accuracy
            accuracy_multiplier = 1.0
            if self.accuracy_history:
                recent_accuracy = sum(self.accuracy_history[-5:]) / min(5, len(self.accuracy_history))
                accuracy_multiplier = 0.7 + (recent_accuracy * 0.6)
            
            bet = int(base_bet * confidence_multiplier * type_multiplier * accuracy_multiplier)
            
            # Ensure minimum bet
            bet = max(self.base_unit, bet)
            
            # Don't exceed total budget
            if total_bet + bet > self.total_budget * 1.5:  # Allow 50% over budget for high confidence
                bet = max(self.base_unit, self.total_budget * 1.5 - total_bet)
            
            if bet > 0:
                bets[symbol] = bet
                total_bet += bet
        
        # Normalize to total budget
        if total_bet > 0:
            scale_factor = min(1.0, self.total_budget / total_bet)
            if scale_factor < 0.9:  # If over budget by more than 10%, scale down
                for symbol in list(bets.keys()):
                    bets[symbol] = int(bets[symbol] * scale_factor)
                    if bets[symbol] < self.base_unit:
                        del bets[symbol]
        
        return bets
    
    def update_learning(self, actual_symbol, bets_made):
        """Update learning with new results"""
        if actual_symbol in self.symbols:
            # Update history
            self.history.append(actual_symbol)
            if len(self.history) > 200:
                self.history.pop(0)
            
            # Update frequencies
            self.symbol_frequencies[actual_symbol] += 1
            
            # Update transitions
            if len(self.history) >= 2:
                prev = self.history[-2]
                self.transition_patterns[prev][actual_symbol] += 1
            
            # Update accuracy
            if bets_made:
                was_correct = actual_symbol in bets_made
                self.accuracy_history.append(was_correct)
                if len(self.accuracy_history) > 50:
                    self.accuracy_history.pop(0)
                
                # Update win/loss
                self.win_history.append(was_correct)
                if len(self.win_history) > 100:
                    self.win_history.pop(0)
    
    def get_recent_accuracy(self):
        """Get recent accuracy"""
        if not self.accuracy_history:
            return 0.0
        recent = self.accuracy_history[-10:] if len(self.accuracy_history) >= 10 else self.accuracy_history
        return sum(recent) / len(recent)
    
    def get_win_rate(self):
        """Get overall win rate"""
        if not self.win_history:
            return 0.0
        return sum(self.win_history) / len(self.win_history)

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
    predictor = BalancedPredictor()
    
    print(Fore.GREEN + "="*70)
    print(Fore.GREEN + "🎯 BALANCED BETTING STRATEGY 🎯")
    print(Fore.GREEN + f"Starting round: {current_round}")
    print(Fore.GREEN + f"Balance: {predictor.balance}")
    print(Fore.GREEN + f"Budget per round: ~{predictor.total_budget}")
    print(Fore.GREEN + "="*70 + "\n")
    
    # Strategy explanation
    print(Fore.CYAN + "📊 BETTING STRATEGY:")
    print(Fore.CYAN + "  • Main food: 100-120 points")
    print(Fore.CYAN + "  • Secondary food: 50-60 points")
    print(Fore.CYAN + "  • Backup food: 30-40 points")
    print(Fore.CYAN + "  • Primary toy: 40-50 points")
    print(Fore.CYAN + "  • Backup toy: 20-30 points")
    print(Fore.CYAN + "  • Total: ~220-270 points")
    print(Fore.CYAN + "  • Adjusts based on confidence and accuracy\n")
    
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
                    predictor.update_learning(detected[0], list(prev_bets.keys()))
                    
                    if details["won"]:
                        roi = (details["profit"] / details["invested"] * 100) if details["invested"] > 0 else 0
                        print(Fore.GREEN + "━" * 40)
                        print(Fore.GREEN + f"✅ BALANCED WIN! +{profit} (ROI: {roi:.0f}%)")
                        print(Fore.GREEN + f"   Bet {details['winning_bet']} on {details['slot1']}")
                        print(Fore.GREEN + "━" * 40)
                    else:
                        print(Fore.RED + "━" * 40)
                        print(Fore.RED + f"❌ LOSS: {profit}")
                        print(Fore.RED + "━" * 40)
                    
                    accuracy = predictor.get_recent_accuracy()
                    win_rate = predictor.get_win_rate()
                    print(Fore.MAGENTA + f"[LEARNING] Recent accuracy: {accuracy:.1%}")
                    print(Fore.MAGENTA + f"[LEARNING] Overall win rate: {win_rate:.1%}")
                    print(Fore.MAGENTA + f"[BALANCE] {predictor.balance}")
                    print(Fore.MAGENTA + f"[TOTAL] Profit: {cumulative_profit:+}")
                else:
                    # Still update learning
                    predictor.update_learning(detected[0], [])
                
                # 5. Make balanced prediction
                print(Fore.CYAN + "[3/5] Analyzing for balanced bets...")
                prediction_data = predictor.predict_balanced(detected)
                
                print(Fore.YELLOW + f"[PREDICTIONS] {prediction_data['predictions']}")
                print(Fore.YELLOW + f"[CONFIDENCE] {prediction_data['total_confidence']:.0%}")
                print(Fore.YELLOW + f"[RECENT ACCURACY] {predictor.get_recent_accuracy():.1%}")
                
                # Show slot2 insight
                if len(detected) >= 2:
                    print(Fore.CYAN + f"[INSIGHT] Slot2 ({detected[1]}) → Next Slot1")
                
                # 6. Calculate balanced bets
                print(Fore.CYAN + "[4/5] Calculating balanced bets...")
                bets = predictor.calculate_balanced_bets(prediction_data)
                
                if bets:
                    total_bet = sum(bets.values())
                    
                    # Categorize bets
                    food_bets = {k:v for k,v in bets.items() if k in predictor.food_symbols}
                    toy_bets = {k:v for k,v in bets.items() if k in predictor.toy_symbols}
                    
                    print(Fore.GREEN + f"  Total bet: {total_bet}")
                    
                    if food_bets:
                        print(Fore.CYAN + "  🍎 FOOD BETS:")
                        for symbol, bet in sorted(food_bets.items(), key=lambda x: x[1], reverse=True):
                            print(Fore.GREEN + f"    {symbol}: {bet}")
                    
                    if toy_bets:
                        print(Fore.CYAN + "  🎪 TOY BETS:")
                        for symbol, bet in sorted(toy_bets.items(), key=lambda x: x[1], reverse=True):
                            print(Fore.YELLOW + f"    {symbol}: {bet}")
                    
                    # Calculate potential returns
                    print(Fore.CYAN + "[POTENTIAL RETURNS]:")
                    for item, bet in bets.items():
                        potential_profit = (bet * MULTIPLIER[item]) - total_bet
                        roi = (potential_profit / total_bet * 100) if total_bet > 0 else 0
                        
                        if potential_profit > 0:
                            color = Fore.GREEN
                        elif potential_profit > -bet:
                            color = Fore.YELLOW
                        else:
                            color = Fore.RED
                        
                        print(color + f"    {item}: {bet} → ROI: {roi:+.0f}%")
                else:
                    reason = "learning phase" if len(predictor.history) < predictor.min_data_for_betting else "low confidence"
                    print(Fore.YELLOW + f"  No bets ({reason})")
                
                # 7. Prepare for next round
                prev_bets = bets
                current_round += 1
                
                # Save round number
                with open(ROUND_FILE, "w", encoding="utf-8") as f:
                    f.write(str(current_round))
                
                # 8. Wait for next round
                print(Fore.BLUE + f"\n[5/5] Next balanced analysis in {INTERVAL}s...")
                print(Fore.BLUE + "─" * 40)
                
                # Wait with progress
                wait_time = INTERVAL
                while wait_time > 0:
                    if wait_time <= 10 or wait_time % 10 == 0:
                        print(Fore.BLUE + f"   {wait_time}s...")
                    time.sleep(min(5, wait_time))
                    wait_time -= 5
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(Fore.RED + f"[ERROR] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    
    except KeyboardInterrupt:
        # Final statistics
        accuracy = predictor.get_recent_accuracy()
        win_rate = predictor.get_win_rate()
        
        print(Fore.RED + "\n🛑 STOPPED")
        print(Fore.MAGENTA + "="*70)
        print(Fore.MAGENTA + "BALANCED STRATEGY RESULTS:")
        print(Fore.MAGENTA + f"  Total rounds: {len(predictor.history)}")
        print(Fore.MAGENTA + f"  Recent accuracy: {accuracy:.1%}")
        print(Fore.MAGENTA + f"  Overall win rate: {win_rate:.1%}")
        print(Fore.MAGENTA + f"  Final balance: {predictor.balance}")
        print(Fore.MAGENTA + f"  Total profit: {cumulative_profit:+}")
        
        # Strategy evaluation
        if cumulative_profit > predictor.total_budget * 2:  # More than 2 rounds of budget
            print(Fore.GREEN + "\n🎯 EXCELLENT BALANCED STRATEGY! 🎯")
        elif cumulative_profit > 0:
            print(Fore.GREEN + "\n💰 PROFITABLE BALANCED STRATEGY 💰")
        elif cumulative_profit > -predictor.total_budget * 3:  # Less than 3 rounds loss
            print(Fore.YELLOW + "\n📊 REASONABLE LOSS (Good risk management)")
        else:
            print(Fore.YELLOW + "\n📉 LOSS (But balanced bets minimized damage)")
        
        # Show betting distribution
        print(Fore.MAGENTA + "\nFINAL BETTING INSIGHTS:")
        if predictor.symbol_frequencies:
            print(Fore.CYAN + "  Most frequent symbols:")
            for symbol, count in sorted(predictor.symbol_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]:
                frequency = count / len(predictor.history) if predictor.history else 0
                print(Fore.CYAN + f"    {symbol}: {frequency:.1%}")
        
        print(Fore.MAGENTA + "="*70)

if __name__ == "__main__":
    run_pipeline()