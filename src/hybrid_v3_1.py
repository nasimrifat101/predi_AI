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
import math

# ===== Terminal Colors =====
from colorama import init, Fore, Style
init(autoreset=True)

# ===== IMPORT COMPONENTS =====
from src.capture.screen_capture import capture_screen_once
from src.capture.roi import crop_rois
from src.vision.ribbon_detector import split_ribbon
from src.vision.icon_detector import detect_icon

# ===== SAFE OPTIMIZED PREDICTOR =====
class OptimizedPredictor:
    def __init__(self):
        self.symbols = ['leg', 'hotdog', 'carrot', 'tomato', 'ballon', 'horse', 'cycle', 'car']
        self.food_symbols = ['leg', 'hotdog', 'carrot', 'tomato']
        self.toy_symbols = ['ballon', 'horse', 'cycle', 'car']
        
        # Payout multipliers
        self.MULTIPLIER = {
            "leg": 5, "hotdog": 5, "carrot": 5, "tomato": 5,
            "ballon": 10, "horse": 15, "cycle": 25, "car": 45
        }
        
        # Core data structures - CORRECTED
        self.history = []  # All slot1 history
        self.transition_counts = defaultdict(lambda: defaultdict(int))  # (prev1, prev2) -> next
        self.symbol_counts = Counter()
        
        # Betting with SAFETY FIRST
        self.balance = 10000
        self.base_unit = 10  # Much smaller
        self.min_observation = 20  # Need more data
        
        # Performance tracking
        self.prediction_log = []
        
        # Load existing data
        self.load_existing_data()
        
        print(Fore.GREEN + "[SAFE-PREDICTOR] Ready with conservative betting!")
    
    def load_existing_data(self):
        """Load and analyze historical data SAFELY"""
        log_file = Path(__file__).resolve().parent.parent / "data" / "logs" / "predictions.csv"
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                if 'slot1' in df.columns:
                    for _, row in df.iterrows():
                        slot1 = str(row['slot1']).strip()
                        if slot1 in self.symbols:
                            self.history.append(slot1)
                            self.symbol_counts[slot1] += 1
                    
                    # Learn transitions SAFELY
                    for i in range(len(self.history) - 2):
                        state = (self.history[i], self.history[i+1])
                        next_symbol = self.history[i+2]
                        self.transition_counts[state][next_symbol] += 1
                    
                    print(Fore.GREEN + f"[DATA] Loaded {len(self.history)} rounds safely")
                    
                    # Show actual distribution
                    print(Fore.CYAN + "[ACTUAL DISTRIBUTION]:")
                    total = sum(self.symbol_counts.values())
                    for symbol, count in self.symbol_counts.most_common():
                        pct = (count / total * 100)
                        print(Fore.CYAN + f"  {symbol}: {count} times ({pct:.1f}%)")
                        
            except Exception as e:
                print(Fore.RED + f"[ERROR] Could not load data: {e}")
    
    def calculate_realistic_probabilities(self, last_two_symbols):
        """Calculate REALISTIC probabilities with smoothing"""
        # BASE PROBABILITIES from actual data
        total_observed = sum(self.symbol_counts.values())
        base_probs = {}
        
        if total_observed > 0:
            for symbol in self.symbols:
                base_probs[symbol] = self.symbol_counts[symbol] / total_observed
        else:
            # Equal probability if no data
            for symbol in self.symbols:
                base_probs[symbol] = 1 / len(self.symbols)
        
        # If we have transition data, blend with base probabilities
        if (last_two_symbols and last_two_symbols in self.transition_counts and 
            len(self.history) >= 10):  # Need enough data
            
            transition_data = self.transition_counts[last_two_symbols]
            total_transitions = sum(transition_data.values())
            
            # Only trust transitions if we have enough samples
            if total_transitions >= 3:
                for symbol in self.symbols:
                    transition_prob = transition_data.get(symbol, 0) / total_transitions
                    # Blend: 70% transition, 30% base (smoothing)
                    base_probs[symbol] = 0.7 * transition_prob + 0.3 * base_probs[symbol]
        
        # Ensure no probability is too extreme
        min_prob = 0.02  # 2% minimum
        max_prob = 0.40  # 40% maximum (no symbol should have >40% probability)
        
        for symbol in self.symbols:
            base_probs[symbol] = max(min_prob, min(max_prob, base_probs[symbol]))
        
        # Normalize
        total = sum(base_probs.values())
        return {k: v/total for k, v in base_probs.items()}
    
    def calculate_expected_value(self, probabilities):
        """Calculate expected value for each symbol"""
        ev_results = {}
        
        for symbol, prob in probabilities.items():
            payout = self.MULTIPLIER.get(symbol, 1)
            ev = (prob * payout) - 1  # Expected value formula
            ev_results[symbol] = {
                'probability': prob,
                'payout': payout,
                'ev': ev,
                'kelly_fraction': (prob * payout - 1) / (payout - 1) if payout > 1 else 0
            }
        
        return ev_results
    
    def predict_safely(self, current_round_symbols):
        """Make SAFE predictions with realistic probabilities"""
        predictions = []
        
        # Get last two symbols
        last_two = None
        if len(self.history) >= 2:
            last_two = (self.history[-2], self.history[-1])
        
        # Calculate REALISTIC probabilities
        probabilities = self.calculate_realistic_probabilities(last_two)
        
        # Calculate expected values
        ev_data = self.calculate_expected_value(probabilities)
        
        # Sort by probability (highest first)
        sorted_by_prob = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 3-4 most likely symbols
        predictions = [symbol for symbol, _ in sorted_by_prob[:4]]
        confidences = [probabilities[symbol] for symbol in predictions]
        
        # Overall confidence is probability of top prediction
        overall_confidence = max(confidences) if confidences else 0.3
        
        return predictions, confidences, overall_confidence, ev_data
    
    def calculate_safe_bets(self, predictions, confidences, ev_data, current_balance):
        """Calculate VERY SAFE bets - maximum 1% of bankroll per round"""
        bets = {}
        
        # MAXIMUM SAFETY RULES
        max_risk_per_round = 0.01  # 1% of bankroll max
        max_bet_per_symbol = 0.005  # 0.5% of bankroll max per symbol
        
        total_bankroll = current_balance
        
        for i, symbol in enumerate(predictions):
            if i >= 4:  # Only bet on top 4 predictions
                break
                
            data = ev_data.get(symbol)
            if not data:
                continue
                
            # Only bet if POSITIVE expected value
            if data['ev'] <= 0:
                continue
            
            # Calculate conservative bet
            base_bet = self.base_unit
            
            # Adjust for confidence
            confidence = confidences[i] if i < len(confidences) else 0.3
            adjusted_bet = base_bet * (1 + confidence)
            
            # Apply absolute limits
            max_allowed = total_bankroll * max_bet_per_symbol
            bet_amount = int(min(adjusted_bet, max_allowed, 50))  # Max $50 per symbol
            
            # Ensure minimum bet
            if bet_amount >= self.base_unit:
                bets[symbol] = bet_amount
        
        # Check total doesn't exceed limit
        total_bet = sum(bets.values())
        max_total = total_bankroll * max_risk_per_round
        
        if total_bet > max_total:
            # Scale down proportionally
            scale = max_total / total_bet
            for symbol in list(bets.keys()):
                bets[symbol] = int(bets[symbol] * scale)
                if bets[symbol] < self.base_unit:
                    del bets[symbol]
        
        return bets
    
    def update_learning(self, actual_symbol, current_round_symbols, bets_made):
        """Update learning safely"""
        # Update history
        self.history.append(actual_symbol)
        self.symbol_counts[actual_symbol] += 1
        
        # Update transition counts if we have enough history
        if len(self.history) >= 3:
            state = (self.history[-3], self.history[-2])
            next_symbol = self.history[-1]
            self.transition_counts[state][next_symbol] += 1
        
        # Track prediction accuracy
        if bets_made:
            was_correct = actual_symbol in bets_made
            profit = 0
            if was_correct:
                profit = (bets_made[actual_symbol] * self.MULTIPLIER[actual_symbol]) - sum(bets_made.values())
            else:
                profit = -sum(bets_made.values())
            
            self.prediction_log.append({
                'round': len(self.history),
                'predicted': list(bets_made.keys()),
                'actual': actual_symbol,
                'correct': was_correct,
                'profit': profit,
                'bets': bets_made.copy()
            })
            
            # Auto-adjust base unit based on performance
            if len(self.prediction_log) >= 5:
                recent_profits = [log['profit'] for log in self.prediction_log[-5:]]
                if all(p < 0 for p in recent_profits):  # Lost last 5 bets
                    self.base_unit = max(5, self.base_unit // 2)  # Cut bet size in half
                    print(Fore.YELLOW + f"[ADJUSTING] Reducing bet size to ${self.base_unit}")
                elif sum(recent_profits) > 100:  # Made good profit
                    self.base_unit = min(20, self.base_unit + 2)  # Increase slightly
                    print(Fore.GREEN + f"[ADJUSTING] Increasing bet size to ${self.base_unit}")
    
    def get_performance_stats(self):
        """Get detailed performance statistics"""
        if not self.prediction_log:
            return {
                'accuracy': 0,
                'total_profit': 0,
                'roi': 0,
                'total_invested': 0,
                'bets_made': 0
            }
        
        total_bets = len(self.prediction_log)
        correct_bets = sum(1 for log in self.prediction_log if log['correct'])
        total_profit = sum(log['profit'] for log in self.prediction_log)
        total_invested = sum(sum(log['bets'].values()) for log in self.prediction_log)
        
        roi = (total_profit / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'accuracy': correct_bets / total_bets,
            'total_profit': total_profit,
            'roi': roi,
            'total_invested': total_invested,
            'bets_made': total_bets
        }

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

# ===== SAFE MAIN PIPELINE =====
def run_optimized_pipeline():
    global current_round
    
    cumulative_profit = 0
    prev_bets = None
    predictor = OptimizedPredictor()
    
    print(Fore.GREEN + "="*70)
    print(Fore.GREEN + "🛡️  SAFE CONSERVATIVE PREDICTOR 🛡️")
    print(Fore.GREEN + f"Starting round: {current_round}")
    print(Fore.GREEN + f"Initial balance: {predictor.balance}")
    print(Fore.GREEN + f"Max risk per round: 1% of bankroll")
    print(Fore.GREEN + f"Base bet size: ${predictor.base_unit}")
    print(Fore.GREEN + "="*70 + "\n")
    
    try:
        while True:
            try:
                print(Fore.CYAN + f"\n[ROUND {current_round}]")
                print(Fore.CYAN + "─" * 50)
                
                # 1. Capture
                print(Fore.CYAN + "[1/5] Capturing screen...")
                screenshot_path = capture_screen_once()
                cropped_paths = crop_rois(screenshot_path)
                ribbon_path = cropped_paths["ribbon"]
                slot_paths = split_ribbon(ribbon_path)
                
                # 2. Detect
                print(Fore.CYAN + "[2/5] Detecting symbols...")
                detected = detect_icons_in_slots(slot_paths)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(Fore.CYAN + f"[RESULT] {' | '.join(detected)}")
                
                # 3. Log results
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp},{current_round},{','.join(detected)}\n")
                
                # 4. Calculate previous round profit
                if prev_bets:
                    profit, details = calculate_profit(detected, prev_bets)
                    cumulative_profit += profit
                    predictor.balance += profit
                    
                    # Update learning
                    predictor.update_learning(detected[0], detected, prev_bets)
                    
                    # Display result
                    if details["won"]:
                        roi = (details["profit"] / details["invested"] * 100) if details["invested"] > 0 else 0
                        print(Fore.GREEN + "━" * 50)
                        print(Fore.GREEN + f"✅ WIN! +{details['profit']} (ROI: {roi:.0f}%)")
                        print(Fore.GREEN + f"   Bet {details['winning_bet']} on {details['slot1']}")
                        print(Fore.GREEN + "━" * 50)
                    else:
                        print(Fore.YELLOW + "━" * 50)
                        print(Fore.YELLOW + f"⚠️  LOSS: {details['profit']}")
                        print(Fore.YELLOW + "━" * 50)
                    
                    # Show stats
                    if predictor.prediction_log:
                        total_bets = len(predictor.prediction_log)
                        correct = sum(1 for log in predictor.prediction_log if log['correct'])
                        accuracy = correct / total_bets if total_bets > 0 else 0
                        print(Fore.MAGENTA + f"[STATS] Accuracy: {accuracy:.1%} | Balance: {predictor.balance}")
                
                # 5. Make SAFE prediction
                print(Fore.CYAN + "[3/5] Calculating safe probabilities...")
                predictions, confidences, overall_confidence, ev_data = predictor.predict_safely(detected)
                
                # Show REALISTIC probabilities
                print(Fore.YELLOW + "[PROBABILITIES] Most likely:")
                sorted_probs = sorted([(s, d) for s, d in ev_data.items()], 
                                     key=lambda x: x[1]['probability'], reverse=True)
                for symbol, data in sorted_probs[:5]:
                    prob_pct = data['probability'] * 100
                    ev = data['ev']
                    if ev > 0:
                        color = Fore.GREEN
                        marker = "✅"
                    elif ev > -0.3:
                        color = Fore.YELLOW
                        marker = "⚠️"
                    else:
                        color = Fore.RED
                        marker = "❌"
                    
                    print(color + f"  {marker} {symbol}: {prob_pct:.1f}% chance, EV={ev:+.2f}")
                
                print(Fore.YELLOW + f"[PREDICTIONS] {predictions}")
                print(Fore.YELLOW + f"[CONFIDENCE] {overall_confidence:.0%}")
                
                # 6. Calculate SAFE bets
                print(Fore.CYAN + "[4/5] Calculating safe bets...")
                bets = predictor.calculate_safe_bets(predictions, confidences, ev_data, predictor.balance)
                
                if bets:
                    total_bet = sum(bets.values())
                    bankroll_pct = (total_bet / predictor.balance * 100) if predictor.balance > 0 else 0
                    
                    print(Fore.GREEN + f"💰 BETTING: ${total_bet} ({bankroll_pct:.1f}% of bankroll)")
                    
                    # Show bet distribution
                    print(Fore.CYAN + "  Bets placed:")
                    for symbol, bet in sorted(bets.items(), key=lambda x: x[1], reverse=True):
                        data = ev_data[symbol]
                        potential = (bet * data['payout']) - total_bet
                        print(Fore.CYAN + f"    {symbol}: ${bet} (Payout: {data['payout']}×)")
                else:
                    print(Fore.YELLOW + "  No safe bets - skipping this round")
                    bets = {}
                
                # 7. Prepare for next round
                prev_bets = bets
                current_round += 1
                
                # Save round number
                with open(ROUND_FILE, "w", encoding="utf-8") as f:
                    f.write(str(current_round))
                
                # 8. Wait for next round
                print(Fore.BLUE + f"\n[5/5] Next analysis in {INTERVAL}s...")
                print(Fore.BLUE + "─" * 50)
                
                # Countdown
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
        stats = predictor.get_performance_stats()
        
        print(Fore.RED + "\n🛑 SESSION STOPPED")
        print(Fore.MAGENTA + "="*70)
        print(Fore.MAGENTA + "SAFE PREDICTOR RESULTS:")
        print(Fore.MAGENTA + f"  Rounds observed: {len(predictor.history)}")
        print(Fore.MAGENTA + f"  Bets made: {stats['bets_made']}")
        print(Fore.MAGENTA + f"  Accuracy: {stats['accuracy']:.1%}")
        print(Fore.MAGENTA + f"  Total profit: ${stats['total_profit']:+}")
        print(Fore.MAGENTA + f"  ROI: {stats['roi']:.1f}%")
        print(Fore.MAGENTA + f"  Final balance: ${predictor.balance}")
        print(Fore.MAGENTA + f"  Final base bet: ${predictor.base_unit}")
        print(Fore.MAGENTA + "="*70)

# ===== TEST FUNCTION =====
def test_ev_calculation():
    """Test if the EV calculations make sense with your accuracy"""
    predictor = OptimizedPredictor()
    
    # Simulate your 66.7% accuracy
    test_probabilities = {
        'leg': 0.20, 'hotdog': 0.18, 'carrot': 0.18, 'tomato': 0.16,
        'ballon': 0.10, 'horse': 0.08, 'cycle': 0.06, 'car': 0.04
    }
    
    print(Fore.CYAN + "\n📊 EXPECTED VALUE CALCULATION TEST")
    print(Fore.CYAN + "─" * 50)
    
    for symbol, prob in test_probabilities.items():
        payout = predictor.MULTIPLIER[symbol]
        ev = (prob * payout) - 1
        kelly = (prob * payout - 1) / (payout - 1) if payout > 1 else 0
        
        if ev > 0:
            print(Fore.GREEN + f"✅ {symbol}: {prob:.0%} chance, {payout}× payout, EV={ev:+.2f}, Kelly={kelly:.3f}")
        else:
            print(Fore.RED + f"❌ {symbol}: {prob:.0%} chance, {payout}× payout, EV={ev:+.2f}")
    
    # Calculate expected profit per $100 bet
    expected_profit = sum(prob * (payout - 1) * 100 for symbol, prob in test_probabilities.items() 
                         for payout in [predictor.MULTIPLIER[symbol]])
    
    print(Fore.CYAN + "─" * 50)
    print(Fore.GREEN + f"Expected profit per $100 bet: ${expected_profit:.2f}")

# ===== RUN SELECTION =====
if __name__ == "__main__":
    print(Fore.RED + "⚠️  STOPPING CURRENT RISKY STRATEGY")
    print(Fore.GREEN + "Starting SAFE conservative strategy...\n")
    
    print(Fore.CYAN + "="*70)
    print(Fore.CYAN + "Select mode:")
    print(Fore.CYAN + "1. Run Safe Predictor (Live)")
    print(Fore.CYAN + "2. Test EV Calculations")
    print(Fore.CYAN + "="*70)
    
    choice = input(Fore.YELLOW + "Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        test_ev_calculation()
    else:
        run_optimized_pipeline()