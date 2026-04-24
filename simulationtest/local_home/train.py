"""
Local Training Script for Federated Learning
============================================
File: local_home/train.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from load_data import DataLoader
from model import TemperatureLSTM


class LocalTrainer:

    def __init__(self, home_id, sequence_length=16, learning_rate=0.0005):
        self.home_id = home_id
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        self.loader = DataLoader(data_dir='data', n_homes=10, n_days=7)
        self.model = TemperatureLSTM(
            input_dim=8, hidden_dim=8, sequence_length=sequence_length
        )
        self.model.compile_model(learning_rate=learning_rate)

    def create_sequences(self, X, y):
        n_samples = len(X)
        n_sequences = n_samples - self.sequence_length + 1
        X_seq = np.zeros((n_sequences, self.sequence_length, X.shape[1]))
        y_seq = np.zeros((n_sequences, 1))
        for i in range(n_sequences):
            X_seq[i] = X[i:i+self.sequence_length]
            y_seq[i] = y[i+self.sequence_length-1]
        return X_seq, y_seq

    def evaluate(self, X_seq, y_seq):
        y_pred = self.model.model.predict(X_seq, verbose=0)
        mse = np.mean((y_pred - y_seq)**2)
        mae = np.mean(np.abs(y_pred - y_seq))
        rmse = np.sqrt(mse)
        temp_range = y_seq.max() - y_seq.min()
        accuracy = (1 - mae / temp_range) * 100 if temp_range > 0 else 0
        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'accuracy': accuracy, 'temp_range': temp_range}

    def get_parameters(self):
        return self.model.get_parameters()
