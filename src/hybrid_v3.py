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

# ===== OPTIMIZED PREDICTOR WITH EXPECTED VALUE BETTING =====
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
        
        # Core data structures
        self.history = []  # All slot1 history
        self.full_rounds = []  # All 5 slots for pattern analysis
        self.actual_results = []
        
        # Statistical tracking
        self.symbol_counts = Counter()
        self.transition_counts = defaultdict(lambda: defaultdict(int))  # (prev1, prev2) -> next
        
        # Betting optimization
        self.balance = 10000
        self.base_unit = 20  # Reduced for better risk management
        self.min_observation = 5
        
        # Performance tracking
        self.prediction_log = []
        self.betting_log = []
        
        # Load existing data
        self.load_existing_data()
        
        print(Fore.GREEN + "[OPTIMIZED-PREDICTOR] Ready with EV-based betting!")
    
    def load_existing_data(self):
        """Load and analyze historical data"""
        log_file = Path(__file__).resolve().parent.parent / "data" / "logs" / "predictions.csv"
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                if 'slot1' in df.columns:
                    for _, row in df.tail(50).iterrows():
                        slot1 = row['slot1']
                        if slot1 in self.symbols:
                            self.history.append(slot1)
                            self.symbol_counts[slot1] += 1
                    
                    # Learn transitions from history
                    for i in range(len(self.history) - 2):
                        state = (self.history[i], self.history[i+1])
                        next_symbol = self.history[i+2]
                        self.transition_counts[state][next_symbol] += 1
                    
                    print(Fore.GREEN + f"[DATA] Loaded {len(self.history)} rounds")
                    
            except Exception as e:
                print(Fore.YELLOW + f"[WARNING] Could not load data: {e}")
    
    def calculate_probabilities(self, last_two_symbols):
        """Calculate exact probabilities for next symbol"""
        probabilities = {}
        
        # Method 1: Markov transitions (2nd order)
        if last_two_symbols and last_two_symbols in self.transition_counts:
            total = sum(self.transition_counts[last_two_symbols].values())
            for symbol, count in self.transition_counts[last_two_symbols].items():
                probabilities[symbol] = count / total
        
        # Method 2: Add overall frequency as fallback
        total_all = sum(self.symbol_counts.values())
        if total_all > 0:
            for symbol in self.symbols:
                base_prob = self.symbol_counts[symbol] / total_all
                if symbol in probabilities:
                    probabilities[symbol] = 0.7 * probabilities[symbol] + 0.3 * base_prob
                else:
                    probabilities[symbol] = base_prob
        
        # Ensure all symbols have at least minimal probability
        for symbol in self.symbols:
            if symbol not in probabilities or probabilities[symbol] < 0.01:
                probabilities[symbol] = 0.01
        
        # Normalize
        total = sum(probabilities.values())
        return {k: v/total for k, v in probabilities.items()}
    
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
    
    def predict_with_ev(self, current_round_symbols):
        """Make predictions with expected value calculations"""
        predictions = []
        probabilities = {}
        
        # Get last two symbols for Markov prediction
        last_two = None
        if len(self.history) >= 2:
            last_two = (self.history[-2], self.history[-1])
        
        # Calculate probabilities
        probabilities = self.calculate_probabilities(last_two)
        
        # Calculate expected values
        ev_data = self.calculate_expected_value(probabilities)
        
        # Sort by EV (highest first)
        sorted_by_ev = sorted(
            [(symbol, data) for symbol, data in ev_data.items()],
            key=lambda x: x[1]['ev'],
            reverse=True
        )
        
        # Get top predictions with positive EV
        predictions = []
        confidences = []
        
        for symbol, data in sorted_by_ev:
            if data['ev'] > 0:  # Only include positive EV bets
                predictions.append(symbol)
                confidences.append(data['probability'])
                if len(predictions) >= 4:  # Limit to 4 best bets
                    break
        
        # Fallback if no positive EV
        if not predictions:
            # Use most probable symbols
            sorted_by_prob = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            predictions = [s for s, _ in sorted_by_prob[:3]]
            confidences = [probabilities[s] for s in predictions]
        
        overall_confidence = max(confidences) if confidences else 0.5
        
        return predictions, confidences, overall_confidence, ev_data
    
    def calculate_optimal_bets(self, predictions, confidences, ev_data, current_balance):
        """Calculate optimal bet sizes using Kelly Criterion"""
        bets = {}
        total_bankroll = current_balance
        
        # Maximum percentage of bankroll to risk per round
        max_risk_per_round = 0.10  # 10% of bankroll
        
        for symbol in predictions:
            if symbol not in ev_data:
                continue
                
            data = ev_data[symbol]
            ev = data['ev']
            payout = data['payout']
            prob = data['probability']
            
            # Skip negative EV
            if ev <= 0:
                continue
            
            # Kelly Criterion formula: f* = (p*b - q) / b
            # Where p = probability of winning, b = payout (odds - 1), q = 1-p
            # For our case: b = payout - 1 (since we get payout times our bet back)
            b = payout - 1
            p = prob
            q = 1 - p
            
            # Full Kelly fraction
            kelly_fraction = (p * b - q) / b if b > 0 else 0
            
            # Use fractional Kelly (1/4) for safety
            fractional_kelly = kelly_fraction / 4
            
            # Calculate bet amount
            bet_amount = fractional_kelly * total_bankroll
            
            # Apply constraints
            min_bet = self.base_unit
            max_bet_single = total_bankroll * 0.05  # Max 5% on one symbol
            max_bet_total = total_bankroll * max_risk_per_round
            
            # Apply individual limits
            bet_amount = max(min_bet, min(bet_amount, max_bet_single))
            bet_amount = int(bet_amount)
            
            # Check if adding this bet exceeds total risk limit
            current_total = sum(bets.values()) + bet_amount
            if current_total > max_bet_total:
                # Scale down proportionally
                scale = max_bet_total / current_total
                bet_amount = int(bet_amount * scale)
            
            if bet_amount >= min_bet:
                bets[symbol] = bet_amount
        
        # Ensure we're not over-betting
        total_bet = sum(bets.values())
        if total_bet > max_bet_total:
            scale = max_bet_total / total_bet
            for symbol in list(bets.keys()):
                bets[symbol] = int(bets[symbol] * scale)
                if bets[symbol] < self.base_unit:
                    del bets[symbol]
        
        return bets
    
    def update_learning(self, actual_symbol, current_round_symbols, bets_made):
        """Update learning with new data"""
        # Update history
        self.history.append(actual_symbol)
        self.symbol_counts[actual_symbol] += 1
        self.full_rounds.append(current_round_symbols)
        
        # Update transition counts (for Markov model)
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
    
    def get_recommended_strategy(self):
        """Get betting strategy recommendations based on data"""
        recommendations = []
        
        # Analyze which symbols are most profitable
        symbol_profits = defaultdict(float)
        symbol_bets = defaultdict(int)
        
        for log in self.prediction_log:
            for symbol, bet in log['bets'].items():
                symbol_bets[symbol] += 1
                if log['actual'] == symbol:
                    symbol_profits[symbol] += (bet * self.MULTIPLIER[symbol]) - bet
                else:
                    symbol_profits[symbol] -= bet
        
        # Generate recommendations
        if symbol_bets:
            for symbol in self.symbols:
                if symbol_bets[symbol] > 0:
                    avg_profit = symbol_profits[symbol] / symbol_bets[symbol]
                    recommendations.append({
                        'symbol': symbol,
                        'avg_profit': avg_profit,
                        'times_bet': symbol_bets[symbol],
                        'payout': self.MULTIPLIER[symbol],
                        'recommendation': 'INCREASE' if avg_profit > 0 else 'DECREASE'
                    })
        
        return sorted(recommendations, key=lambda x: x['avg_profit'], reverse=True)

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

# ===== MAIN OPTIMIZED PIPELINE =====
def run_optimized_pipeline():
    global current_round
    
    cumulative_profit = 0
    prev_bets = None
    predictor = OptimizedPredictor()
    
    print(Fore.GREEN + "="*70)
    print(Fore.GREEN + "🎯 OPTIMIZED EV-BASED PREDICTOR 🎯")
    print(Fore.GREEN + f"Starting round: {current_round}")
    print(Fore.GREEN + f"Initial balance: {predictor.balance}")
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
                        print(Fore.GREEN + f"   Bet {details['winning_bet']} on {details['slot1']} (Payout: {MULTIPLIER[details['slot1']]}×)")
                        print(Fore.GREEN + "━" * 50)
                    else:
                        print(Fore.RED + "━" * 50)
                        print(Fore.RED + f"❌ LOSS: {details['profit']}")
                        print(Fore.RED + "━" * 50)
                    
                    # Show performance stats
                    stats = predictor.get_performance_stats()
                    print(Fore.MAGENTA + f"[PERFORMANCE] Accuracy: {stats['accuracy']:.1%} | ROI: {stats['roi']:.1f}%")
                    print(Fore.MAGENTA + f"[BALANCE] {predictor.balance} | Total Profit: {cumulative_profit:+}")
                else:
                    # Still update learning
                    predictor.update_learning(detected[0], detected, {})
                
                # 5. Make EV-based prediction
                print(Fore.CYAN + "[3/5] Calculating probabilities and EV...")
                predictions, confidences, overall_confidence, ev_data = predictor.predict_with_ev(detected)
                
                # Display EV analysis
                print(Fore.YELLOW + "[EV ANALYSIS] Top symbols:")
                for symbol in list(ev_data.keys())[:5]:
                    data = ev_data[symbol]
                    if data['ev'] > 0:
                        color = Fore.GREEN
                    elif data['ev'] > -0.2:
                        color = Fore.YELLOW
                    else:
                        color = Fore.RED
                    
                    print(color + f"  {symbol}: Prob={data['probability']:.1%}, Payout={data['payout']}×, EV={data['ev']:+.3f}, Kelly={data['kelly_fraction']:.3f}")
                
                print(Fore.YELLOW + f"[PREDICTIONS] {predictions}")
                print(Fore.YELLOW + f"[CONFIDENCE] {overall_confidence:.0%}")
                
                # 6. Calculate optimal bets
                print(Fore.CYAN + "[4/5] Calculating optimal bets (Kelly Criterion)...")
                bets = predictor.calculate_optimal_bets(predictions, confidences, ev_data, predictor.balance)
                
                if bets:
                    total_bet = sum(bets.values())
                    bankroll_pct = (total_bet / predictor.balance * 100) if predictor.balance > 0 else 0
                    
                    print(Fore.GREEN + f"💰 BETTING: ${total_bet} ({bankroll_pct:.1f}% of bankroll)")
                    
                    # Show bet distribution
                    print(Fore.CYAN + "  Distribution:")
                    for symbol, bet in sorted(bets.items(), key=lambda x: x[1], reverse=True):
                        data = ev_data[symbol]
                        potential = (bet * data['payout']) - total_bet
                        roi_potential = (potential / total_bet * 100) if total_bet > 0 else 0
                        
                        if data['ev'] > 0.1:
                            marker = "🎯"
                            color = Fore.GREEN
                        elif data['ev'] > 0:
                            marker = "💰"
                            color = Fore.YELLOW
                        else:
                            marker = "⚠️"
                            color = Fore.RED
                        
                        print(color + f"    {marker} {symbol}: ${bet} (EV={data['ev']:+.2f}, Pot. ROI={roi_potential:+.0f}%)")
                else:
                    print(Fore.YELLOW + "  No positive EV bets found - skipping this round")
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
        # Final results with detailed analysis
        stats = predictor.get_performance_stats()
        recommendations = predictor.get_recommended_strategy()
        
        print(Fore.RED + "\n🛑 SESSION ENDED")
        print(Fore.MAGENTA + "="*70)
        print(Fore.MAGENTA + "FINAL PERFORMANCE ANALYSIS:")
        print(Fore.MAGENTA + f"  Rounds observed: {len(predictor.history)}")
        print(Fore.MAGENTA + f"  Bets made: {stats['bets_made']}")
        print(Fore.MAGENTA + f"  Accuracy: {stats['accuracy']:.1%}")
        print(Fore.MAGENTA + f"  Total invested: ${stats['total_invested']}")
        print(Fore.MAGENTA + f"  Total profit: ${stats['total_profit']:+}")
        print(Fore.MAGENTA + f"  ROI: {stats['roi']:.1f}%")
        print(Fore.MAGENTA + f"  Final balance: ${predictor.balance}")
        
        if recommendations:
            print(Fore.MAGENTA + "\n📊 BETTING RECOMMENDATIONS:")
            for rec in recommendations[:5]:
                color = Fore.GREEN if rec['avg_profit'] > 0 else Fore.RED
                print(color + f"  {rec['symbol']}: ${rec['avg_profit']:+.2f} avg profit, {rec['times_bet']} bets → {rec['recommendation']}")
        
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
    print(Fore.CYAN + "="*70)
    print(Fore.CYAN + "Select mode:")
    print(Fore.CYAN + "1. Run Optimized Predictor (Live)")
    print(Fore.CYAN + "2. Test EV Calculations")
    print(Fore.CYAN + "="*70)
    
    choice = input(Fore.YELLOW + "Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        test_ev_calculation()
    else:
        run_optimized_pipeline()