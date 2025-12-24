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

# ===== MODEL SAVING PATHS =====
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "slot_lstm_model.pth"
ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"
HISTORY_FILE = MODEL_DIR / "training_history.pkl"

# ===== LSTM NEURAL NETWORK =====
class SlotLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SlotLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out

# ===== DATASET FOR TRAINING =====
class SlotDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])

# ===== AI PREDICTOR WITH MODEL PERSISTENCE =====
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
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.symbols)
        
        # AI Model parameters
        self.input_size = 16  # 8 one-hot + 8 features
        self.hidden_size = 64
        self.num_layers = 2
        self.output_size = len(self.symbols)
        self.sequence_length = 10
        
        # Device (CPU/GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
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
        self.base_total = 290
        self.min_bet = 10
        
        # Strategy weights
        self.food_weights = [150, 60, 40]
        self.toy_weights = [30, 10]
        
        # Performance tracking
        self.prediction_log = []
        
        # ALWAYS load from CSV first to get ALL data
        print(Fore.CYAN + "[AI] Loading ALL data from CSV...")
        self.load_all_data_from_csv()
        
        # Then try to load existing model
        if self.load_model():
            print(Fore.GREEN + "[AI] Loaded existing model weights!")
            print(Fore.CYAN + f"[AI] Using {len(self.history)} symbols from CSV")
            print(Fore.CYAN + f"[AI] This corresponds to ~{len(self.history)//5} rounds")
        else:
            print(Fore.YELLOW + "[AI] No saved model found, training from scratch...")
            if len(self.history) >= 20:
                self.train_from_loaded_data()
    
    def debug_csv_structure(self, df):
        """Debug function to see CSV structure"""
        print(Fore.MAGENTA + "\n[DEBUG] CSV Structure:")
        print(Fore.MAGENTA + f"  Total rows: {len(df)}")
        print(Fore.MAGENTA + f"  Columns (after stripping): {list(df.columns)}")
        print(Fore.MAGENTA + f"  First few rows:")
        print(df.head(3).to_string())
        
        # Check what column names we have
        possible_round_columns = ['round', 'round_id', 'round_number', 'game_round', 'roundid']
        found_round_col = None
        for col in possible_round_columns:
            if col in df.columns:
                found_round_col = col
                break
        
        if found_round_col:
            print(Fore.MAGENTA + f"  Found round column: '{found_round_col}'")
        else:
            print(Fore.RED + f"  WARNING: No round column found! Available columns: {list(df.columns)}")
        
        # Also check first row values
        if len(df) > 0:
            print(Fore.MAGENTA + f"\n  Sample data from first row:")
            first_row = df.iloc[0]
            for col in ['slot1', 'slot2', 'slot3', 'slot4', 'slot5']:
                if col in df.columns:
                    value = first_row[col]
                    print(Fore.MAGENTA + f"    {col}: '{value}' (type: {type(value).__name__})")
    
    def load_all_data_from_csv(self):
        """Load ALL data from CSV, regardless of existing model"""
        try:
            log_file = Path(__file__).resolve().parent.parent / "data" / "logs" / "predictions.csv"
            
            if not log_file.exists():
                print(Fore.YELLOW + f"[AI] No CSV file found at: {log_file}")
                return
            
            print(Fore.CYAN + f"[AI] Reading CSV from: {log_file}")
            
            # Read CSV - strip whitespace from column names
            try:
                df = pd.read_csv(log_file)
                # STRIP WHITESPACE FROM COLUMN NAMES
                df.columns = df.columns.str.strip()
                print(Fore.GREEN + f"[AI] Fixed column names: {list(df.columns)}")
            except Exception as e:
                print(Fore.RED + f"[AI ERROR] Failed to read CSV: {e}")
                return
            
            # Debug: Show CSV structure
            self.debug_csv_structure(df)
            
            # Check for required columns (now without spaces)
            required_slot_columns = ['slot1', 'slot2', 'slot3', 'slot4', 'slot5']
            missing_slots = [col for col in required_slot_columns if col not in df.columns]
            
            if missing_slots:
                print(Fore.RED + f"[AI ERROR] Missing slot columns: {missing_slots}")
                print(Fore.RED + f"  Available columns: {list(df.columns)}")
                return
            
            # Extract ALL symbols from ALL slots
            all_symbols = []
            invalid_symbols = []
            rows_processed = 0
            
            for _, row in df.iterrows():
                rows_processed += 1
                for slot_col in required_slot_columns:
                    symbol = str(row[slot_col]).strip().lower()
                    
                    # Clean the symbol
                    symbol = self.clean_csv_symbol(symbol)
                    
                    if symbol:
                        all_symbols.append(symbol)
                    else:
                        # Track invalid symbols for debugging
                        original_symbol = str(row[slot_col])
                        if original_symbol not in ['', 'nan', 'None']:
                            invalid_symbols.append(original_symbol)
            
            print(Fore.GREEN + f"[AI] Processed {rows_processed} rows")
            print(Fore.GREEN + f"[AI] Successfully loaded {len(all_symbols)} symbols from CSV")
            print(Fore.GREEN + f"[AI] Estimated: {rows_processed} rounds × 5 slots = {rows_processed * 5} possible symbols")
            
            if invalid_symbols:
                print(Fore.YELLOW + f"[AI] Found {len(set(invalid_symbols))} unique invalid symbols")
                if set(invalid_symbols):
                    print(Fore.YELLOW + f"  Samples: {list(set(invalid_symbols))[:10]}")
            
            if len(all_symbols) == 0:
                print(Fore.RED + "[AI ERROR] No valid symbols found in CSV!")
                print(Fore.RED + "  Check if symbols in CSV match: " + ", ".join(self.symbols))
                # Let's see what's actually in the first row
                if len(df) > 0:
                    first_row = df.iloc[0]
                    print(Fore.RED + "  First row values:")
                    for col in required_slot_columns:
                        print(Fore.RED + f"    {col}: '{first_row[col]}'")
                return
            
            self.history = all_symbols
            
            # Show detailed stats
            symbol_counts = Counter(self.history)
            print(Fore.CYAN + "\n[AI] Symbol distribution from CSV:")
            print(Fore.CYAN + "-" * 40)
            total = len(self.history)
            
            # Show counts for ALL symbols (including 0 counts)
            for symbol in self.symbols:
                count = symbol_counts.get(symbol, 0)
                percentage = (count / total) * 100 if total > 0 else 0
                color = Fore.GREEN if count > 0 else Fore.RED
                print(f"{color}  {symbol:10s}: {count:4d} ({percentage:5.1f}%)")
            
            print(Fore.CYAN + "-" * 40)
            print(Fore.CYAN + f"  Total symbols: {total}")
            print(Fore.CYAN + f"  Estimated rounds: {total // 5}")
            
            # Show most common symbols
            most_common = symbol_counts.most_common(3)
            print(Fore.CYAN + f"\n[AI] Most common symbols:")
            for symbol, count in most_common:
                print(Fore.CYAN + f"  {symbol}: {count} times")
                
        except Exception as e:
            print(Fore.RED + f"[AI ERROR] Failed to load CSV: {e}")
            import traceback
            traceback.print_exc()
            self.history = []
    
    def clean_csv_symbol(self, symbol_str):
        """Clean and validate symbol from CSV"""
        # If it's already None or NaN
        if pd.isna(symbol_str):
            return None
        
        # Convert to string
        if not isinstance(symbol_str, str):
            symbol_str = str(symbol_str)
        
        # Strip and lowercase
        symbol = symbol_str.strip().lower()
        
        # Handle empty strings and common missing values
        if symbol in ['', 'nan', 'none', 'null', 'unknown', 'n/a', 'nan', 'na', 'undefined']:
            return None
        
        # Remove any extra quotes or whitespace
        symbol = symbol.replace('"', '').replace("'", "").strip()
        
        # Direct match
        if symbol in self.symbols:
            return symbol
        
        # Try to fix common typos (case-insensitive)
        corrections = {
            'balloon': 'ballon',
            'ballons': 'ballon',
            'carrots': 'carrot',
            'tomatoes': 'tomato',
            'hotdogs': 'hotdog',
            'legs': 'leg',
            'horses': 'horse',
            'cycles': 'cycle',
            'cars': 'car',
            'super_food': 'leg',  # Map rare symbols to regular ones
            'super_toy': 'ballon',
            'super food': 'leg',
            'super toy': 'ballon'
        }
        
        if symbol in corrections:
            return corrections[symbol]
        
        # If it's a rare symbol not in our main list, skip it
        if 'super' in symbol.lower():
            print(Fore.YELLOW + f"  Skipping rare symbol: {symbol}")
            return None
        
        # Try to see if it's close to one of our symbols
        for valid_symbol in self.symbols:
            if valid_symbol in symbol or symbol in valid_symbol:
                print(Fore.YELLOW + f"  Approximate match: '{symbol}' -> '{valid_symbol}'")
                return valid_symbol
        
        # If we get here, it's an unknown symbol
        print(Fore.RED + f"  Unknown symbol skipped: '{symbol}'")
        return None
    
    def train_from_loaded_data(self):
        """Train model with loaded CSV data"""
        if len(self.history) < self.sequence_length + 1:
            print(Fore.YELLOW + f"[AI] Need at least {self.sequence_length + 1} symbols, have {len(self.history)}")
            return
        
        print(Fore.CYAN + f"[AI] Training with {len(self.history)} symbols...")
        self.prepare_training_data()
        
        if len(self.training_data) > 0:
            print(Fore.CYAN + f"[AI] Generated {len(self.training_data)} training sequences")
            self.train_model(epochs=30)
            self.save_model()
        else:
            print(Fore.YELLOW + "[AI] Could not prepare training data")
    
    # ===== MODEL SAVING/LOADING =====
    def save_model(self):
        """Save complete model state to disk"""
        try:
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'output_size': self.output_size,
                'sequence_length': self.sequence_length,
                'training_loss_history': getattr(self, 'loss_history', [])
            }, MODEL_FILE)
            
            # Save label encoder
            with open(ENCODER_FILE, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save history and balance
            with open(HISTORY_FILE, 'wb') as f:
                pickle.dump({
                    'prediction_log': self.prediction_log,
                    'balance': self.balance
                }, f)
            
            print(Fore.GREEN + f"[AI] Model saved to {MODEL_FILE}")
            print(Fore.GREEN + f"[AI] Encoder saved to {ENCODER_FILE}")
            print(Fore.GREEN + f"[AI] History saved to {HISTORY_FILE}")
            return True
            
        except Exception as e:
            print(Fore.RED + f"[AI ERROR] Failed to save model: {e}")
            return False
    
    def load_model(self):
        """Load model from disk"""
        try:
            # Check if model files exist
            if not MODEL_FILE.exists():
                return False
            
            # Load PyTorch model
            checkpoint = torch.load(MODEL_FILE, map_location=self.device)
            
            # Recreate model architecture
            self.model = SlotLSTM(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                output_size=checkpoint['output_size']
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load other parameters
            self.input_size = checkpoint['input_size']
            self.hidden_size = checkpoint['hidden_size']
            self.num_layers = checkpoint['num_layers']
            self.output_size = checkpoint['output_size']
            self.sequence_length = checkpoint['sequence_length']
            self.loss_history = checkpoint.get('training_loss_history', [])
            
            # Load label encoder
            if ENCODER_FILE.exists():
                with open(ENCODER_FILE, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # Load balance and prediction log (but NOT history - we get that from CSV)
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE, 'rb') as f:
                    history_data = pickle.load(f)
                    self.prediction_log = history_data.get('prediction_log', [])
                    self.balance = history_data.get('balance', 10000)
                print(Fore.CYAN + f"[AI] Loaded balance: ${self.balance}")
                print(Fore.CYAN + f"[AI] Loaded {len(self.prediction_log)} prediction logs")
            
            return True
            
        except Exception as e:
            print(Fore.RED + f"[AI ERROR] Failed to load model: {e}")
            return False
    
    def prepare_training_data(self):
        """Prepare sequences for LSTM training"""
        if len(self.history) < self.sequence_length + 1:
            print(Fore.YELLOW + f"[AI] Need at least {self.sequence_length + 1} symbols for training")
            return
        
        # Convert symbols to numerical labels
        try:
            encoded_history = self.label_encoder.transform(self.history)
        except ValueError as e:
            print(Fore.RED + f"[AI ERROR] Label encoding failed: {e}")
            print(Fore.YELLOW + f"[AI] History contains: {set(self.history)}")
            return
        
        # Create sequences and labels
        self.training_data = []
        self.training_labels = []
        
        for i in range(len(encoded_history) - self.sequence_length):
            sequence = encoded_history[i:i + self.sequence_length]
            label = encoded_history[i + self.sequence_length]
            
            sequence_features = self.encode_sequence_with_features(sequence)
            
            self.training_data.append(sequence_features)
            self.training_labels.append(label)
        
        print(Fore.GREEN + f"[AI] Prepared {len(self.training_data)} training sequences")
    
    def encode_sequence_with_features(self, sequence):
        """Encode sequence with additional features"""
        # One-hot encoding (8 dimensions)
        one_hot_encoded = np.zeros((self.sequence_length, len(self.symbols)))
        for i, symbol_idx in enumerate(sequence):
            one_hot_encoded[i, symbol_idx] = 1
        
        # Additional features (8 dimensions)
        additional_features = np.zeros((self.sequence_length, 8))
        
        # Feature 1: Position in sequence
        additional_features[:, 0] = np.arange(self.sequence_length) / self.sequence_length
        
        # Feature 2: Is food symbol?
        for i, symbol_idx in enumerate(sequence):
            symbol = self.label_encoder.inverse_transform([symbol_idx])[0]
            additional_features[i, 1] = 1 if symbol in self.food_symbols else 0
        
        # Feature 3: Normalized payout
        for i, symbol_idx in enumerate(sequence):
            symbol = self.label_encoder.inverse_transform([symbol_idx])[0]
            additional_features[i, 2] = self.MULTIPLIER[symbol] / 45
        
        # Feature 4: Streak of same type
        streak = 1
        for i in range(self.sequence_length):
            if i > 0:
                prev_symbol = self.label_encoder.inverse_transform([sequence[i-1]])[0]
                curr_symbol = self.label_encoder.inverse_transform([sequence[i]])[0]
                prev_is_food = prev_symbol in self.food_symbols
                curr_is_food = curr_symbol in self.food_symbols
                
                if prev_is_food == curr_is_food:
                    streak += 1
                else:
                    streak = 1
            
            additional_features[i, 3] = streak / self.sequence_length
        
        # Combine features
        return np.concatenate([one_hot_encoded, additional_features], axis=1)
    
    def train_model(self, epochs=30, batch_size=32):
        """Train the LSTM model"""
        if len(self.training_data) == 0:
            print(Fore.YELLOW + "[AI] No training data")
            return
        
        print(Fore.CYAN + f"[AI] Training LSTM for {epochs} epochs with REAL data...")
        
        # Convert to numpy arrays
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Split data - IMPORTANT: Use time-based split for sequential data
        # We'll use the first 80% for training, last 20% for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = SlotDataset(X_train, y_train)
        val_dataset = SlotDataset(X_val, y_val)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        self.loss_history = []
        
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
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.loss_history.append(avg_train_loss)
            
            # Validation every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
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
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * correct / total
                
                print(Fore.CYAN + f"[AI Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.1f}%")
        
        print(Fore.GREEN + f"[AI] Training complete with REAL data!")
    
    # ===== PREDICTION FUNCTIONS =====
    def predict_with_ai(self):
        """Use LSTM to predict next symbols"""
        # If we don't have enough history, use simple statistics
        if len(self.history) < self.sequence_length:
            return self.predict_with_statistics()
        
        # Get last sequence_length symbols
        recent_history = self.history[-self.sequence_length:]
        
        # Encode sequence
        try:
            encoded_history = self.label_encoder.transform(recent_history)
        except ValueError:
            # If we encounter an unseen symbol, fall back to statistics
            return self.predict_with_statistics()
        
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
        
        # Get top 3 food and top 2 toy symbols
        top_foods = []
        top_toys = []
        
        for symbol, prob in sorted(symbol_probs.items(), key=lambda x: x[1], reverse=True):
            if symbol in self.food_symbols and len(top_foods) < 3:
                top_foods.append(symbol)
            elif symbol in self.toy_symbols and len(top_toys) < 2:
                top_toys.append(symbol)
            
            if len(top_foods) == 3 and len(top_toys) == 2:
                break
        
        # Fill missing predictions if needed
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
        """Fallback prediction using statistics from available history"""
        if not self.history:
            # If no history, return random predictions
            top_foods = random.sample(self.food_symbols, 3)
            top_toys = random.sample(self.toy_symbols, 2)
            symbol_probs = {s: 0.1 for s in self.symbols}
            return top_foods, top_toys, symbol_probs
        
        symbol_counts = Counter(self.history)
        
        food_counts = {s: symbol_counts.get(s, 0) for s in self.food_symbols}
        top_foods = sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_foods = [s for s, _ in top_foods]
        
        toy_counts = {s: symbol_counts.get(s, 0) for s in self.toy_symbols}
        top_toys = sorted(toy_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        top_toys = [s for s, _ in top_toys]
        
        # Calculate simple probabilities
        total = len(self.history)
        symbol_probs = {}
        for symbol in self.symbols:
            count = symbol_counts.get(symbol, 0)
            symbol_probs[symbol] = count / total if total > 0 else 0.1
        
        return top_foods, top_toys, symbol_probs
    
    def calculate_bets(self, top_foods, top_toys, symbol_probs):
        """Calculate bets based on AI predictions"""
        bets = {}
        
        # Confidence factor based on prediction probabilities
        total_prob = sum(symbol_probs[s] for s in top_foods + top_toys)
        confidence_factor = total_prob / 5
        
        # Calculate food bets
        if len(top_foods) >= 3:
            main_food = top_foods[0]
            main_prob = symbol_probs[main_food]
            main_bet = int(self.food_weights[0] * confidence_factor * (1 + main_prob))
            bets[main_food] = max(self.min_bet, min(main_bet, 200))
            
            backup_food = top_foods[1]
            backup_prob = symbol_probs[backup_food]
            backup_bet = int(self.food_weights[1] * confidence_factor * (0.8 + backup_prob))
            bets[backup_food] = max(self.min_bet, min(backup_bet, 100))
            
            failsafe_food = top_foods[2]
            failsafe_prob = symbol_probs[failsafe_food]
            failsafe_bet = int(self.food_weights[2] * confidence_factor * (0.6 + failsafe_prob))
            bets[failsafe_food] = max(self.min_bet, min(failsafe_bet, 80))
        
        # Calculate toy bets
        if len(top_toys) >= 2:
            main_toy = top_toys[0]
            main_toy_prob = symbol_probs[main_toy]
            main_toy_bet = int(self.toy_weights[0] * confidence_factor * (0.9 + main_toy_prob))
            bets[main_toy] = max(self.min_bet, min(main_toy_bet, 50))
            
            backup_toy = top_toys[1]
            backup_toy_prob = symbol_probs[backup_toy]
            backup_toy_bet = int(self.toy_weights[1] * confidence_factor * (0.7 + backup_toy_prob))
            bets[backup_toy] = max(self.min_bet, min(backup_toy_bet, 30))
        
        # Adjust total bet based on balance
        total_bet = sum(bets.values())
        max_bet = min(self.base_total, int(self.balance * 0.05))
        
        if total_bet > max_bet:
            scale = max_bet / total_bet
            for symbol in list(bets.keys()):
                bets[symbol] = int(bets[symbol] * scale)
        
        return bets
    
    def update_history(self, new_symbol, profit=0):
        """Update history and retrain periodically"""
        self.history.append(new_symbol)
        self.balance += profit
        
        # Keep history size manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        # Track prediction
        if hasattr(self, 'current_bets') and self.current_bets:
            was_correct = new_symbol in self.current_bets
            self.prediction_log.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'actual': new_symbol,
                'predicted': list(self.current_bets.keys()),
                'correct': was_correct,
                'profit': profit,
                'bets': self.current_bets.copy()
            })
        
        # Retrain and save every 50 rounds (only if we have enough data)
        if len(self.history) % 50 == 0 and len(self.history) >= 100:
            print(Fore.CYAN + "[AI] Retraining with new data...")
            self.prepare_training_data()
            if len(self.training_data) > 0:
                self.train_model(epochs=5)  # Quick retrain
                self.save_model()
    
    def get_stats(self):
        """Get statistics"""
        stats = {
            'balance': self.balance,
            'history_length': len(self.history),
            'prediction_count': len(self.prediction_log),
            'estimated_rounds': len(self.history) // 5
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
    print(Fore.GREEN + "🤖 LSTM AI PREDICTOR WITH REAL GAME DATA")
    print(Fore.GREEN + f"Starting round: {current_round}")
    print(Fore.GREEN + f"Initial balance: ${predictor.balance}")
    print(Fore.GREEN + f"Model files: {MODEL_DIR}")
    print(Fore.GREEN + "="*70 + "\n")
    
    # Show saved model info
    if MODEL_FILE.exists():
        file_size = MODEL_FILE.stat().st_size
        print(Fore.CYAN + f"[MODEL INFO] Saved model size: {file_size / 1024:.1f} KB")
    
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
                
                # 3. Log results - Use correct format matching your CSV
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp},{current_round},{','.join(detected)}\n")
                
                # 4. Calculate previous round profit
                if prev_bets:
                    profit, details = calculate_profit(detected, prev_bets)
                    cumulative_profit += profit
                    
                    # Update predictor
                    predictor.update_history(detected[0], profit)
                    
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
                    predictor.update_history(detected[0], 0)
                
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
                predictor.current_bets = bets  # Store for tracking
                
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
        # Final results and save model
        print(Fore.CYAN + "\n[AI] Saving model before exit...")
        predictor.save_model()
        
        stats = predictor.get_stats()
        
        print(Fore.RED + "\n🛑 AI SESSION ENDED")
        print(Fore.MAGENTA + "="*70)
        print(Fore.MAGENTA + "🤖 LSTM AI RESULTS (Trained on REAL Data)")
        print(Fore.MAGENTA + "="*70)
        print(Fore.MAGENTA + f"  Total symbols: {stats['history_length']}")
        print(Fore.MAGENTA + f"  Estimated rounds: {stats['estimated_rounds']}")
        print(Fore.MAGENTA + f"  AI predictions: {stats['prediction_count']}")
        print(Fore.MAGENTA + f"  AI accuracy: {stats['accuracy']:.1%}")
        print(Fore.MAGENTA + f"  Total profit: ${stats['total_profit']:+}")
        print(Fore.MAGENTA + f"  Final balance: ${stats['balance']}")
        
        # Show saved files
        print(Fore.MAGENTA + "\n💾 SAVED MODEL FILES:")
        for file in [MODEL_FILE, ENCODER_FILE, HISTORY_FILE]:
            if file.exists():
                size = file.stat().st_size
                print(Fore.CYAN + f"  {file.name}: {size / 1024:.1f} KB")
        
        print(Fore.MAGENTA + "="*70)

# ===== RUN =====
if __name__ == "__main__":
    print(Fore.GREEN + "🤖 LAUNCHING LSTM AI PREDICTOR WITH REAL GAME DATA")
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