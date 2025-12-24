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

# ===== AI/ML IMPORTS =====
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ===== Terminal Colors =====
from colorama import init, Fore, Style
init(autoreset=True)

# ===== IMPORT COMPONENTS =====
from src.capture.screen_capture import capture_screen_once
from src.capture.roi import crop_rois
from src.vision.ribbon_detector import split_ribbon
from src.vision.icon_detector import detect_icon

# ===== LSTM NEURAL NETWORK =====
class SlotLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SlotLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Additional layers for better learning
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out

# ===== DATASET FOR TRAINING =====
class SlotDataset(Dataset):
    def __init__(self, sequences, labels, sequence_length=10):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.FloatTensor(sequence), torch.LongTensor([label])

# ===== AI PREDICTOR WITH LSTM =====
class AIPredictor:
    def __init__(self):
        self.symbols = ['leg', 'hotdog', 'carrot', 'tomato', 'ballon', 'horse', 'cycle', 'car']
        self.food_symbols = ['leg', 'hotdog', 'carrot', 'tomato']
        self.toy_symbols = ['ballon', 'horse', 'cycle', 'car']
        
        # Payout multipliers
        self.MULTIPLIER = {
            "leg": 5, "hotdog": 5, "carrot": 5, "tomato": 5,
            "ballon": 10, "horse": 15, "cycle": 25, "car": 45
        }
        
        # Label encoder for symbols
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.symbols)
        
        # AI Model parameters
        self.input_size = 16  # 8 symbols one-hot encoded + 8 additional features
        self.hidden_size = 64
        self.num_layers = 2
        self.output_size = len(self.symbols)
        self.sequence_length = 10
        
        # Initialize AI model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SlotLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        ).to(self.device)
        
        # Optimizer and loss
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Data storage
        self.history = []
        self.training_data = []
        self.training_labels = []
        
        # Betting parameters
        self.balance = 10000
        self.base_total = 290  # 150+60+40+30+10 = 290
        self.min_bet = 10
        
        # Strategy weights
        self.food_weights = [150, 60, 40]
        self.toy_weights = [30, 10]
        
        # Performance tracking
        self.prediction_log = []
        
        # Load and train AI
        self.load_and_train()
        
        print(Fore.GREEN + "[AI-PREDICTOR] LSTM Neural Network Ready!")
        print(Fore.CYAN + f"[AI] Model: {self.input_size} → {self.hidden_size} → {self.output_size}")
        print(Fore.CYAN + f"[AI] Sequence length: {self.sequence_length}")
        print(Fore.CYAN + f"[AI] Device: {self.device}")
    
    def load_and_train(self):
        """Load historical data and train the AI"""
        try:
            # Try to load data
            log_file = Path(__file__).resolve().parent.parent / "data" / "logs" / "predictions.csv"
            
            if log_file.exists():
                df = pd.read_csv(log_file)
                
                # Extract slot1 history
                slot1_history = []
                for _, row in df.iterrows():
                    slot1 = str(row.get('slot1', '')).strip().lower()
                    if slot1 in self.symbols:
                        slot1_history.append(slot1)
                
                if len(slot1_history) >= 20:  # Need at least 20 rounds to train
                    self.history = slot1_history
                    print(Fore.GREEN + f"[AI] Loaded {len(self.history)} historical rounds")
                    
                    # Prepare training data
                    self.prepare_training_data()
                    
                    # Train the model
                    if len(self.training_data) > 0:
                        self.train_model(epochs=50)
                    else:
                        print(Fore.YELLOW + "[AI] Not enough data for training, using untrained model")
                else:
                    print(Fore.YELLOW + f"[AI] Only {len(slot1_history)} rounds found, need at least 20 for training")
                    # Create synthetic training data
                    self.create_synthetic_training_data()
            else:
                print(Fore.YELLOW + "[AI] No historical data file found")
                self.create_synthetic_training_data()
                
        except Exception as e:
            print(Fore.RED + f"[AI ERROR] {e}")
            import traceback
            traceback.print_exc()
            self.create_synthetic_training_data()
    
    def create_synthetic_training_data(self):
        """Create synthetic data for initial training"""
        print(Fore.YELLOW + "[AI] Creating synthetic training data...")
        
        # Create patterns that resemble your game data
        patterns = [
            ['leg', 'hotdog', 'carrot', 'tomato', 'ballon', 'horse', 'cycle', 'car'],
            ['hotdog', 'carrot', 'tomato', 'ballon', 'horse', 'cycle', 'car', 'leg'],
            ['carrot', 'tomato', 'ballon', 'horse', 'cycle', 'car', 'leg', 'hotdog'],
            ['tomato', 'ballon', 'horse', 'cycle', 'car', 'leg', 'hotdog', 'carrot'],
            ['ballon', 'horse', 'cycle', 'car', 'leg', 'hotdog', 'carrot', 'tomato']
        ]
        
        synthetic_history = []
        for pattern in patterns:
            synthetic_history.extend(pattern)
        
        # Repeat to get enough data
        synthetic_history = synthetic_history * 5
        self.history = synthetic_history[:100]  # Keep first 100
        
        print(Fore.GREEN + f"[AI] Created {len(self.history)} synthetic data points")
        self.prepare_training_data()
        self.train_model(epochs=30)
    
    def prepare_training_data(self):
        """Prepare sequences for LSTM training"""
        if len(self.history) < self.sequence_length + 1:
            print(Fore.YELLOW + f"[AI] Not enough data for sequences (need {self.sequence_length + 1}, have {len(self.history)})")
            return
        
        print(Fore.CYAN + f"[AI] Preparing training data from {len(self.history)} history points...")
        
        # Convert symbols to numerical labels
        encoded_history = self.label_encoder.transform(self.history)
        
        # Create sequences and labels
        self.training_data = []
        self.training_labels = []
        
        for i in range(len(encoded_history) - self.sequence_length):
            # Get sequence of sequence_length symbols
            sequence = encoded_history[i:i + self.sequence_length]
            
            # Get the next symbol as label
            label = encoded_history[i + self.sequence_length]
            
            # Convert to one-hot encoding with additional features
            sequence_features = self.encode_sequence_with_features(sequence)
            
            self.training_data.append(sequence_features)
            self.training_labels.append(label)
        
        print(Fore.GREEN + f"[AI] Prepared {len(self.training_data)} training sequences")
    
    def encode_sequence_with_features(self, sequence):
        """Encode sequence with additional features"""
        # Convert sequence indices to one-hot (8 dimensions)
        one_hot_encoded = np.zeros((self.sequence_length, len(self.symbols)))
        
        for i, symbol_idx in enumerate(sequence):
            one_hot_encoded[i, symbol_idx] = 1
        
        # Add additional features (8 more dimensions)
        additional_features = np.zeros((self.sequence_length, 8))
        
        # Feature 1: Position in sequence (normalized)
        additional_features[:, 0] = np.arange(self.sequence_length) / self.sequence_length
        
        # Feature 2: Is food symbol? (1 for food, 0 for toy)
        for i, symbol_idx in enumerate(sequence):
            symbol = self.label_encoder.inverse_transform([symbol_idx])[0]
            additional_features[i, 1] = 1 if symbol in self.food_symbols else 0
        
        # Feature 3: Payout multiplier (normalized)
        for i, symbol_idx in enumerate(sequence):
            symbol = self.label_encoder.inverse_transform([symbol_idx])[0]
            additional_features[i, 2] = self.MULTIPLIER[symbol] / 45  # Normalize by max payout
        
        # Feature 4: Streak of same symbol type
        streak = 1
        for i in range(self.sequence_length):
            if i > 0:
                prev_symbol_idx = sequence[i-1]
                curr_symbol_idx = sequence[i]
                prev_symbol = self.label_encoder.inverse_transform([prev_symbol_idx])[0]
                curr_symbol = self.label_encoder.inverse_transform([curr_symbol_idx])[0]
                
                prev_is_food = prev_symbol in self.food_symbols
                curr_is_food = curr_symbol in self.food_symbols
                
                if prev_is_food == curr_is_food:
                    streak += 1
                else:
                    streak = 1
            
            additional_features[i, 3] = streak / self.sequence_length  # Normalize
        
        # Combine features
        combined_features = np.concatenate([one_hot_encoded, additional_features], axis=1)
        
        return combined_features
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the LSTM model"""
        if len(self.training_data) == 0:
            print(Fore.YELLOW + "[AI] No training data available")
            return
        
        print(Fore.CYAN + f"[AI] Training LSTM for {epochs} epochs...")
        
        # Convert to numpy arrays
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = SlotDataset(X_train, y_train, self.sequence_length)
        val_dataset = SlotDataset(X_val, y_val, self.sequence_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).squeeze()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).squeeze()
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(Fore.CYAN + f"[AI Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.1f}%")
        
        print(Fore.GREEN + f"[AI] Training complete! Final validation accuracy: {val_accuracy:.1f}%")
    
    def predict_with_ai(self):
        """Use LSTM to predict next symbols with probabilities"""
        if len(self.history) < self.sequence_length:
            print(Fore.YELLOW + f"[AI] Not enough history ({len(self.history)}), need {self.sequence_length}")
            return self.predict_with_statistics()
        
        # Get last sequence_length symbols
        recent_history = self.history[-self.sequence_length:]
        
        # Encode the sequence
        encoded_history = self.label_encoder.transform(recent_history)
        sequence_features = self.encode_sequence_with_features(encoded_history)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_features).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = outputs.cpu().numpy()[0]
        
        # Get probabilities for each symbol
        symbol_probs = {}
        for i, symbol in enumerate(self.symbols):
            symbol_probs[symbol] = probabilities[i]
        
        # Sort symbols by probability
        sorted_symbols = sorted(symbol_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 3 food and top 2 toy symbols
        top_foods = []
        top_toys = []
        
        for symbol, prob in sorted_symbols:
            if symbol in self.food_symbols and len(top_foods) < 3:
                top_foods.append(symbol)
            elif symbol in self.toy_symbols and len(top_toys) < 2:
                top_toys.append(symbol)
            
            if len(top_foods) == 3 and len(top_toys) == 2:
                break
        
        # If we don't have enough, fill with fallbacks
        while len(top_foods) < 3:
            for symbol in self.food_symbols:
                if symbol not in top_foods:
                    top_foods.append(symbol)
                    break
        
        while len(top_toys) < 2:
            for symbol in self.toy_symbols:
                if symbol not in top_toys:
                    top_toys.append(symbol)
                    break
        
        return top_foods, top_toys, symbol_probs
    
    def predict_with_statistics(self):
        """Fallback to statistical prediction if AI isn't ready"""
        # Count symbol frequencies
        symbol_counts = Counter(self.history)
        
        # Get top 3 food by frequency
        food_counts = {s: symbol_counts.get(s, 0) for s in self.food_symbols}
        top_foods = sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_foods = [s for s, _ in top_foods]
        
        # Get top 2 toys by frequency
        toy_counts = {s: symbol_counts.get(s, 0) for s in self.toy_symbols}
        top_toys = sorted(toy_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        top_toys = [s for s, _ in top_toys]
        
        # Create dummy probabilities
        symbol_probs = {s: 0.1 for s in self.symbols}
        for symbol in top_foods + top_toys:
            symbol_probs[symbol] = 0.2
        
        return top_foods, top_toys, symbol_probs
    
    def calculate_bets(self, top_foods, top_toys, symbol_probs):
        """Calculate bets based on AI predictions"""
        bets = {}
        
        # Adjust based on confidence
        total_prob_food = sum(symbol_probs[s] for s in top_foods)
        total_prob_toy = sum(symbol_probs[s] for s in top_toys)
        
        # Scale bet amounts by confidence
        confidence_factor = (total_prob_food + total_prob_toy) / 5  # Normalize
        
        # Calculate food bets
        if len(top_foods) >= 3:
            # Main food
            main_food = top_foods[0]
            main_food_prob = symbol_probs[main_food]
            main_food_bet = int(self.food_weights[0] * confidence_factor * (1 + main_food_prob))
            bets[main_food] = max(self.min_bet, main_food_bet)
            
            # Backup food
            backup_food = top_foods[1]
            backup_food_prob = symbol_probs[backup_food]
            backup_food_bet = int(self.food_weights[1] * confidence_factor * (0.8 + backup_food_prob))
            bets[backup_food] = max(self.min_bet, backup_food_bet)
            
            # Failsafe food
            failsafe_food = top_foods[2]
            failsafe_food_prob = symbol_probs[failsafe_food]
            failsafe_food_bet = int(self.food_weights[2] * confidence_factor * (0.6 + failsafe_food_prob))
            bets[failsafe_food] = max(self.min_bet, failsafe_food_bet)
        
        # Calculate toy bets
        if len(top_toys) >= 2:
            # Main toy
            main_toy = top_toys[0]
            main_toy_prob = symbol_probs[main_toy]
            main_toy_bet = int(self.toy_weights[0] * confidence_factor * (0.9 + main_toy_prob))
            bets[main_toy] = max(self.min_bet, main_toy_bet)
            
            # Backup toy
            backup_toy = top_toys[1]
            backup_toy_prob = symbol_probs[backup_toy]
            backup_toy_bet = int(self.toy_weights[1] * confidence_factor * (0.7 + backup_toy_prob))
            bets[backup_toy] = max(self.min_bet, backup_toy_bet)
        
        # Ensure total doesn't exceed max
        total_bet = sum(bets.values())
        max_bet = min(self.base_total, int(self.balance * 0.05))  # Max 5% of balance
        
        if total_bet > max_bet:
            scale = max_bet / total_bet
            for symbol in list(bets.keys()):
                bets[symbol] = int(bets[symbol] * scale)
        
        return bets
    
    def update_history(self, new_symbol):
        """Update history and retrain periodically"""
        self.history.append(new_symbol)
        
        # Keep history manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        # Retrain AI every 50 new rounds
        if len(self.history) % 50 == 0 and len(self.history) >= 100:
            print(Fore.CYAN + "[AI] Retraining with new data...")
            self.prepare_training_data()
            if len(self.training_data) > 0:
                self.train_model(epochs=10)  # Quick retrain
    
    def update_balance(self, profit):
        """Update balance"""
        self.balance += profit
    
    def get_stats(self):
        """Get statistics"""
        stats = {
            'balance': self.balance,
            'history_length': len(self.history),
            'prediction_count': len(self.prediction_log)
        }
        
        if self.prediction_log:
            correct = sum(1 for log in self.prediction_log if log['correct'])
            stats['accuracy'] = correct / len(self.prediction_log)
            stats['total_profit'] = sum(log['profit'] for log in self.prediction_log)
        else:
            stats['accuracy'] = 0
            stats['total_profit'] = 0
        
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

# ===== AI PIPELINE =====
def run_ai_pipeline():
    global current_round
    
    cumulative_profit = 0
    prev_bets = None
    predictor = AIPredictor()
    
    print(Fore.GREEN + "="*70)
    print(Fore.GREEN + "🤖 LSTM AI PREDICTOR WITH NEURAL NETWORK")
    print(Fore.GREEN + f"Starting round: {current_round}")
    print(Fore.GREEN + f"Initial balance: ${predictor.balance}")
    print(Fore.GREEN + f"Base strategy: 3-Food {predictor.food_weights}, 2-Toy {predictor.toy_weights}")
    print(Fore.GREEN + f"AI Model: LSTM with {predictor.hidden_size} hidden units")
    print(Fore.GREEN + f"Training data: {len(predictor.history)} rounds")
    print(Fore.GREEN + "="*70 + "\n")
    
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
                    
                    # Update predictor
                    predictor.update_balance(profit)
                    predictor.update_history(detected[0])
                    
                    # Track prediction
                    was_correct = detected[0] in prev_bets
                    predictor.prediction_log.append({
                        'actual': detected[0],
                        'predicted': list(prev_bets.keys()),
                        'correct': was_correct,
                        'profit': profit,
                        'bets': prev_bets.copy()
                    })
                    
                    if details["won"]:
                        roi = (details["profit"] / details["invested"] * 100) if details["invested"] > 0 else 0
                        print(Fore.GREEN + "━" * 50)
                        print(Fore.GREEN + f"✅ WIN! +${profit} (ROI: {roi:.0f}%)")
                        print(Fore.GREEN + f"   Bet ${details['winning_bet']} on {details['slot1']}")
                        print(Fore.GREEN + "━" * 50)
                    else:
                        print(Fore.YELLOW + "━" * 50)
                        print(Fore.YELLOW + f"⚠️  LOSS: ${profit}")
                        print(Fore.YELLOW + "━" * 50)
                    
                    # Show stats
                    stats = predictor.get_stats()
                    print(Fore.MAGENTA + f"[ACCURACY] {stats['accuracy']:.1%}")
                    print(Fore.MAGENTA + f"[BALANCE] ${predictor.balance}")
                    print(Fore.MAGENTA + f"[NET PROFIT] ${cumulative_profit:+}")
                else:
                    # First round, just update history
                    predictor.update_history(detected[0])
                
                # 5. AI Prediction
                print(Fore.CYAN + "[3/5] AI Analyzing with LSTM...")
                top_foods, top_toys, symbol_probs = predictor.predict_with_ai()
                
                print(Fore.GREEN + f"[AI PREDICTION] Foods: {top_foods}")
                print(Fore.BLUE + f"[AI PREDICTION] Toys: {top_toys}")
                
                # Show AI confidence
                print(Fore.YELLOW + "[AI CONFIDENCE]:")
                for symbol in top_foods + top_toys:
                    prob = symbol_probs[symbol] * 100
                    if prob > 20:
                        color = Fore.GREEN
                    elif prob > 15:
                        color = Fore.YELLOW
                    else:
                        color = Fore.CYAN
                    print(color + f"  {symbol}: {prob:.1f}%")
                
                # 6. Calculate bets
                print(Fore.CYAN + "[4/5] AI Calculating optimal bets...")
                bets = predictor.calculate_bets(top_foods, top_toys, symbol_probs)
                
                if bets:
                    total_bet = sum(bets.values())
                    balance_pct = (total_bet / predictor.balance * 100) if predictor.balance > 0 else 0
                    
                    print(Fore.GREEN + f"💰 AI BETTING: ${total_bet} ({balance_pct:.1f}% of balance)")
                    
                    # Show bet breakdown
                    print(Fore.CYAN + "\n🥕 FOOD BETS:")
                    food_bets = {k:v for k,v in bets.items() if k in predictor.food_symbols}
                    for symbol, bet in sorted(food_bets.items(), key=lambda x: x[1], reverse=True):
                        multiplier = MULTIPLIER[symbol]
                        potential = (bet * multiplier) - total_bet
                        print(Fore.CYAN + f"  {symbol}: ${bet} ({multiplier}× → Net: ${potential:+})")
                    
                    print(Fore.BLUE + "\n🧸 TOY BETS:")
                    toy_bets = {k:v for k,v in bets.items() if k in predictor.toy_symbols}
                    for symbol, bet in sorted(toy_bets.items(), key=lambda x: x[1], reverse=True):
                        multiplier = MULTIPLIER[symbol]
                        potential = (bet * multiplier) - total_bet
                        print(Fore.BLUE + f"  {symbol}: ${bet} ({multiplier}× → Net: ${potential:+})")
                    
                    # Show coverage
                    food_coverage = sum(food_bets.values())
                    toy_coverage = sum(toy_bets.values())
                    print(Fore.MAGENTA + f"\n📊 COVERAGE: Food ${food_coverage} ({food_coverage/total_bet:.0%}), Toys ${toy_coverage} ({toy_coverage/total_bet:.0%})")
                else:
                    print(Fore.YELLOW + "  AI: No confident bets - skipping this round")
                    bets = {}
                
                # 7. Prepare for next round
                prev_bets = bets
                current_round += 1
                
                # Save round number
                with open(ROUND_FILE, "w", encoding="utf-8") as f:
                    f.write(str(current_round))
                
                # 8. Wait for next round
                print(Fore.BLUE + f"\n[⏱️] Next AI analysis in {INTERVAL}s...")
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
        
        print(Fore.RED + "\n🛑 AI SESSION ENDED")
        print(Fore.MAGENTA + "="*70)
        print(Fore.MAGENTA + "🤖 LSTM AI RESULTS")
        print(Fore.MAGENTA + "="*70)
        print(Fore.MAGENTA + f"  Total rounds: {stats['history_length']}")
        print(Fore.MAGENTA + f"  AI predictions: {stats['prediction_count']}")
        print(Fore.MAGENTA + f"  AI accuracy: {stats['accuracy']:.1%}")
        print(Fore.MAGENTA + f"  Total profit: ${stats['total_profit']:+}")
        print(Fore.MAGENTA + f"  Final balance: ${stats['balance']}")
        
        # Show AI model info
        print(Fore.MAGENTA + "\n🧠 AI MODEL INFO:")
        print(Fore.CYAN + f"  Type: LSTM Neural Network")
        print(Fore.CYAN + f"  Layers: {predictor.num_layers}")
        print(Fore.CYAN + f"  Hidden units: {predictor.hidden_size}")
        print(Fore.CYAN + f"  Sequence length: {predictor.sequence_length}")
        print(Fore.CYAN + f"  Training data: {len(predictor.history)} rounds")
        
        print(Fore.MAGENTA + "="*70)

# ===== RUN =====
if __name__ == "__main__":
    print(Fore.GREEN + "🤖 LAUNCHING LSTM AI PREDICTOR")
    print(Fore.GREEN + "="*70)
    
    # Check for PyTorch
    try:
        import torch
        print(Fore.GREEN + f"[AI] PyTorch version: {torch.__version__}")
        print(Fore.GREEN + f"[AI] CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print(Fore.RED + "[ERROR] PyTorch not installed!")
        print(Fore.YELLOW + "Install with: pip install torch torchvision")
        exit(1)
    
    run_ai_pipeline()