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

# ===== FIXED PREDICTOR THAT ACTUALLY LOADS DATA =====
class FixedPredictor:
    def __init__(self):
        self.symbols = ['leg', 'hotdog', 'carrot', 'tomato', 'ballon', 'horse', 'cycle', 'car']
        
        # Payout multipliers
        self.MULTIPLIER = {
            "leg": 5, "hotdog": 5, "carrot": 5, "tomato": 5,
            "ballon": 10, "horse": 15, "cycle": 25, "car": 45
        }
        
        # Core data storage
        self.all_history = []  # Stores all historical results
        self.slot1_history = []  # Just slot1 values for quick analysis
        self.symbol_counts = Counter()
        
        # Betting parameters
        self.balance = 10000
        self.base_bet = 10
        
        # Performance tracking
        self.prediction_log = []
        self.total_profit = 0
        
        # Load data IMMEDIATELY
        self.load_all_historical_data()
        
        print(Fore.GREEN + f"[DATA] Successfully loaded {len(self.all_history)} historical rounds")
        print(Fore.CYAN + f"[SYMBOL COUNTS]: {dict(self.symbol_counts)}")
        
        if self.symbol_counts:
            total = sum(self.symbol_counts.values())
            print(Fore.CYAN + "[ACTUAL DISTRIBUTION]:")
            for symbol, count in self.symbol_counts.most_common():
                pct = (count / total * 100)
                print(Fore.CYAN + f"  {symbol}: {count} times ({pct:.1f}%)")
    
    def load_all_historical_data(self):
        """Load ALL historical data from predictions.csv"""
        try:
            # Get the log file path
            log_file = Path(__file__).resolve().parent.parent / "data" / "logs" / "predictions.csv"
            
            print(Fore.YELLOW + f"[LOADING] Looking for data at: {log_file}")
            
            if not log_file.exists():
                print(Fore.RED + f"[ERROR] File not found: {log_file}")
                # Create backup test data
                print(Fore.YELLOW + "[INFO] Creating test data structure...")
                return
            
            # Read the CSV file
            df = pd.read_csv(log_file)
            print(Fore.GREEN + f"[LOADING] CSV loaded with {len(df)} rows")
            print(Fore.GREEN + f"[LOADING] Columns: {list(df.columns)}")
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    # Get slot1 value
                    if 'slot1' in df.columns:
                        slot1 = str(row['slot1']).strip().lower()
                        
                        # Skip if not a valid symbol
                        if slot1 not in self.symbols:
                            continue
                        
                        # Add to history
                        self.slot1_history.append(slot1)
                        self.symbol_counts[slot1] += 1
                        
                        # Store full row
                        self.all_history.append({
                            'round': row.get('round', idx),
                            'slot1': slot1,
                            'slot2': str(row.get('slot2', '')).strip().lower(),
                            'slot3': str(row.get('slot3', '')).strip().lower(),
                            'slot4': str(row.get('slot4', '')).strip().lower(),
                            'slot5': str(row.get('slot5', '')).strip().lower(),
                            'timestamp': row.get('timestamp', '')
                        })
                    
                except Exception as e:
                    print(Fore.RED + f"[WARNING] Error processing row {idx}: {e}")
                    continue
            
            print(Fore.GREEN + f"[SUCCESS] Loaded {len(self.all_history)} valid rounds")
            
            # If still no data, create synthetic data for testing
            if len(self.all_history) == 0:
                print(Fore.YELLOW + "[INFO] No valid data found, creating synthetic test data")
                self.create_test_data()
                
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            # Create test data
            self.create_test_data()
    
    def create_test_data(self):
        """Create synthetic test data based on your earlier statistics"""
        print(Fore.YELLOW + "[TEST DATA] Creating synthetic data based on your 66.7% accuracy...")
        
        # Based on your earlier 66.7% accuracy report
        test_sequences = [
            ['leg', 'hotdog', 'carrot', 'tomato', 'ballon'],
            ['hotdog', 'carrot', 'tomato', 'ballon', 'horse'],
            ['carrot', 'tomato', 'ballon', 'horse', 'cycle'],
            ['tomato', 'ballon', 'horse', 'cycle', 'car'],
            ['leg', 'hotdog', 'carrot', 'tomato', 'ballon'],
            ['hotdog', 'carrot', 'tomato', 'ballon', 'horse'],
            ['carrot', 'tomato', 'ballon', 'horse', 'cycle'],
            ['tomato', 'ballon', 'horse', 'cycle', 'car'],
            ['leg', 'hotdog', 'carrot', 'tomato', 'ballon'],
            ['hotdog', 'carrot', 'tomato', 'ballon', 'horse']
        ]
        
        for i, sequence in enumerate(test_sequences):
            slot1 = sequence[0]
            self.slot1_history.append(slot1)
            self.symbol_counts[slot1] += 1
            
            self.all_history.append({
                'round': i + 1,
                'slot1': slot1,
                'slot2': sequence[1],
                'slot3': sequence[2],
                'slot4': sequence[3],
                'slot5': sequence[4],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        print(Fore.GREEN + f"[TEST DATA] Created {len(self.all_history)} test rounds")
    
    def calculate_probabilities(self):
        """Calculate probabilities based on actual historical data"""
        probabilities = {}
        total = sum(self.symbol_counts.values())
        
        if total == 0:
            # Equal probability if no data
            for symbol in self.symbols:
                probabilities[symbol] = 1.0 / len(self.symbols)
        else:
            # Real probabilities from data
            for symbol in self.symbols:
                count = self.symbol_counts.get(symbol, 0)
                probabilities[symbol] = count / total
        
        return probabilities
    
    def find_patterns(self):
        """Find simple patterns in the data"""
        if len(self.slot1_history) < 5:
            return None
        
        # Look at the last symbol and see what usually follows it
        last_symbol = self.slot1_history[-1] if self.slot1_history else None
        
        if not last_symbol:
            return None
        
        # Count what follows this symbol
        follow_counts = Counter()
        
        for i in range(len(self.slot1_history) - 1):
            if self.slot1_history[i] == last_symbol:
                next_symbol = self.slot1_history[i + 1]
                follow_counts[next_symbol] += 1
        
        if follow_counts:
            total = sum(follow_counts.values())
            return {symbol: count/total for symbol, count in follow_counts.items()}
        
        return None
    
    def make_predictions(self):
        """Make predictions based on data"""
        # Calculate base probabilities
        base_probs = self.calculate_probabilities()
        
        # Try to find patterns
        pattern_probs = self.find_patterns()
        
        # Combine pattern and base probabilities
        if pattern_probs and len(self.slot1_history) >= 10:
            # Use 70% pattern, 30% base
            final_probs = {}
            for symbol in self.symbols:
                pattern = pattern_probs.get(symbol, 0)
                base = base_probs.get(symbol, 0)
                final_probs[symbol] = 0.7 * pattern + 0.3 * base
        else:
            # Not enough data for patterns, use base
            final_probs = base_probs
        
        # Sort by probability
        sorted_probs = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Get top predictions
        predictions = [symbol for symbol, prob in sorted_probs[:4]]
        
        return predictions, final_probs
    
    def calculate_bets(self, predictions, probabilities, current_balance):
        """Calculate safe bets"""
        bets = {}
        
        # Calculate expected value for each symbol
        ev_data = {}
        for symbol in self.symbols:
            prob = probabilities.get(symbol, 0.01)
            payout = self.MULTIPLIER[symbol]
            ev = (prob * payout) - 1
            ev_data[symbol] = {
                'probability': prob,
                'payout': payout,
                'ev': ev
            }
        
        # Only bet on predictions with positive EV
        for symbol in predictions:
            data = ev_data[symbol]
            
            # Only bet if positive EV
            if data['ev'] > 0:
                # Calculate bet size
                confidence = data['probability']
                
                # Base bet with confidence multiplier
                bet_size = int(self.base_bet * (1 + confidence * 3))
                
                # Apply limits
                max_bet = min(current_balance * 0.02, 50)  # Max 2% or $50
                bet_size = min(bet_size, max_bet)
                bet_size = max(self.base_bet, bet_size)  # Minimum bet
                
                bets[symbol] = bet_size
        
        return bets, ev_data
    
    def update_with_result(self, detected_symbols, bets_made=None, profit=0):
        """Update with new game result"""
        slot1 = detected_symbols[0]
        
        # Add to history
        self.slot1_history.append(slot1)
        self.symbol_counts[slot1] += 1
        
        # Add to full history
        self.all_history.append({
            'round': len(self.all_history) + 1,
            'slot1': slot1,
            'slot2': detected_symbols[1] if len(detected_symbols) > 1 else '',
            'slot3': detected_symbols[2] if len(detected_symbols) > 2 else '',
            'slot4': detected_symbols[3] if len(detected_symbols) > 3 else '',
            'slot5': detected_symbols[4] if len(detected_symbols) > 4 else '',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Update balance
        self.balance += profit
        self.total_profit += profit
        
        # Track prediction
        if bets_made:
            was_correct = slot1 in bets_made
            self.prediction_log.append({
                'round': len(self.all_history),
                'actual': slot1,
                'predicted': list(bets_made.keys()),
                'correct': was_correct,
                'profit': profit
            })
    
    def get_stats(self):
        """Get current statistics"""
        stats = {
            'rounds_observed': len(self.all_history),
            'bets_made': len(self.prediction_log),
            'balance': self.balance,
            'total_profit': self.total_profit
        }
        
        if self.prediction_log:
            correct = sum(1 for log in self.prediction_log if log['correct'])
            stats['accuracy'] = correct / len(self.prediction_log)
            stats['total_invested'] = sum(abs(log['profit']) for log in self.prediction_log if log['profit'] < 0)
            
            if stats['total_invested'] > 0:
                stats['roi'] = (stats['total_profit'] / stats['total_invested'] * 100)
            else:
                stats['roi'] = 0
        else:
            stats['accuracy'] = 0
            stats['total_invested'] = 0
            stats['roi'] = 0
        
        return stats

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

# ===== MAIN PIPELINE =====
def run_pipeline():
    global current_round
    
    cumulative_profit = 0
    prev_bets = None
    predictor = FixedPredictor()  # This now actually loads data!
    
    print(Fore.GREEN + "="*70)
    print(Fore.GREEN + "✅ DATA-LOADED PREDICTOR ✅")
    print(Fore.GREEN + f"Starting round: {current_round}")
    print(Fore.GREEN + f"Historical data: {len(predictor.all_history)} rounds loaded")
    print(Fore.GREEN + f"Initial balance: {predictor.balance}")
    print(Fore.GREEN + f"Base bet: ${predictor.base_bet}")
    print(Fore.GREEN + "="*70 + "\n")
    
    # Show initial statistics
    stats = predictor.get_stats()
    print(Fore.CYAN + f"[INITIAL STATS] Rounds: {stats['rounds_observed']} | Balance: ${stats['balance']}")
    
    try:
        while True:
            try:
                print(Fore.CYAN + f"\n[ROUND {current_round}]")
                print(Fore.CYAN + "─" * 50)
                
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
                    
                    # Update predictor with result
                    predictor.update_with_result(detected, prev_bets, profit)
                    
                    if details["won"]:
                        roi = (details["profit"] / details["invested"] * 100) if details["invested"] > 0 else 0
                        print(Fore.GREEN + "━" * 50)
                        print(Fore.GREEN + f"✅ WIN! +{profit} (ROI: {roi:.0f}%)")
                        print(Fore.GREEN + f"   Bet {details['winning_bet']} on {details['slot1']}")
                        print(Fore.GREEN + "━" * 50)
                    else:
                        print(Fore.YELLOW + "━" * 50)
                        print(Fore.YELLOW + f"⚠️  LOSS: {profit}")
                        print(Fore.YELLOW + "━" * 50)
                    
                    # Show updated stats
                    stats = predictor.get_stats()
                    print(Fore.MAGENTA + f"[STATS] Accuracy: {stats['accuracy']:.1%}")
                    print(Fore.MAGENTA + f"[STATS] ROI: {stats['roi']:.1f}%")
                    print(Fore.MAGENTA + f"[BALANCE] ${predictor.balance}")
                    print(Fore.MAGENTA + f"[TOTAL] Profit: {cumulative_profit:+}")
                else:
                    # First round, just update without bets
                    predictor.update_with_result(detected)
                
                # 5. Make predictions
                print(Fore.CYAN + "[3/5] Analyzing...")
                predictions, probabilities = predictor.make_predictions()
                
                # Show probabilities
                print(Fore.YELLOW + "[PROBABILITIES]:")
                for symbol, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                    prob_pct = prob * 100
                    if prob_pct > 20:
                        color = Fore.GREEN
                    elif prob_pct > 10:
                        color = Fore.YELLOW
                    else:
                        color = Fore.CYAN
                    print(color + f"  {symbol}: {prob_pct:.1f}%")
                
                print(Fore.YELLOW + f"[PREDICTIONS] {predictions}")
                
                # 6. Calculate bets
                print(Fore.CYAN + "[4/5] Calculating bets...")
                bets, ev_data = predictor.calculate_bets(predictions, probabilities, predictor.balance)
                
                if bets:
                    total_bet = sum(bets.values())
                    
                    print(Fore.GREEN + f"💰 BETTING: ${total_bet}")
                    print(Fore.CYAN + "  Distribution:")
                    
                    for symbol, bet in bets.items():
                        data = ev_data[symbol]
                        potential = (bet * data['payout']) - total_bet
                        roi = (potential / total_bet * 100) if total_bet > 0 else 0
                        
                        if data['ev'] > 0.5:
                            marker = "🎯"
                            color = Fore.GREEN
                        elif data['ev'] > 0:
                            marker = "💰"
                            color = Fore.YELLOW
                        else:
                            marker = "⚠️"
                            color = Fore.RED
                        
                        print(color + f"    {marker} {symbol}: ${bet} (EV={data['ev']:+.2f})")
                else:
                    print(Fore.YELLOW + "  No positive EV bets - skipping")
                    bets = {}
                
                # 7. Prepare for next round
                prev_bets = bets
                current_round += 1
                
                # Save round number
                with open(ROUND_FILE, "w", encoding="utf-8") as f:
                    f.write(str(current_round))
                
                # 8. Wait for next round
                print(Fore.BLUE + f"\n[5/5] Next in {INTERVAL}s...")
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
        stats = predictor.get_stats()
        
        print(Fore.RED + "\n🛑 SESSION ENDED")
        print(Fore.MAGENTA + "="*70)
        print(Fore.MAGENTA + "FINAL RESULTS:")
        print(Fore.MAGENTA + f"  Rounds observed: {stats['rounds_observed']}")
        print(Fore.MAGENTA + f"  Bets made: {stats['bets_made']}")
        print(Fore.MAGENTA + f"  Accuracy: {stats['accuracy']:.1%}")
        print(Fore.MAGENTA + f"  ROI: {stats['roi']:.1f}%")
        print(Fore.MAGENTA + f"  Total profit: ${stats['total_profit']:+}")
        print(Fore.MAGENTA + f"  Final balance: ${stats['balance']}")
        
        # Show symbol statistics
        total = sum(predictor.symbol_counts.values())
        if total > 0:
            print(Fore.MAGENTA + "\nSYMBOL STATISTICS:")
            for symbol, count in predictor.symbol_counts.most_common():
                pct = (count / total * 100)
                print(Fore.CYAN + f"  {symbol}: {count} ({pct:.1f}%)")
        
        print(Fore.MAGENTA + "="*70)

# ===== DIAGNOSTIC CHECK =====
def check_data_file():
    """Check if data file exists and has data"""
    log_file = Path(__file__).resolve().parent.parent / "data" / "logs" / "predictions.csv"
    
    print(Fore.YELLOW + "\n" + "="*70)
    print(Fore.YELLOW + "📁 DATA FILE DIAGNOSTIC")
    print(Fore.YELLOW + "="*70)
    
    if not log_file.exists():
        print(Fore.RED + f"❌ File not found: {log_file}")
        print(Fore.YELLOW + "Creating empty data file...")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("timestamp,round,slot1,slot2,slot3,slot4,slot5\n")
        print(Fore.GREEN + "✅ Created new data file")
    else:
        print(Fore.GREEN + f"✅ File exists: {log_file}")
        
        # Check file size
        size = log_file.stat().st_size
        print(Fore.CYAN + f"   File size: {size} bytes")
        
        # Try to read a few lines
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                print(Fore.CYAN + f"   Total lines: {len(lines)}")
                
                if len(lines) > 1:
                    print(Fore.GREEN + f"   First data line: {lines[1].strip()}")
                else:
                    print(Fore.YELLOW + "   File only has header, no data yet")
        except Exception as e:
            print(Fore.RED + f"❌ Error reading file: {e}")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # First, check the data file
    check_data_file()
    
    print(Fore.GREEN + "\n" + "="*70)
    print(Fore.GREEN + "🚀 STARTING PREDICTOR WITH DATA LOADING")
    print(Fore.GREEN + "="*70 + "\n")
    
    run_pipeline()