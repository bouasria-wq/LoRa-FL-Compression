"""
Local Training Script for Federated Learning
============================================

Implements Equation 19: Local SGD Training
theta_k^(e+1) = theta_k^(e) - eta * gradient_L_k(theta_k^(e))

Metrics:
- Equation 2: Local Loss (MSE)
- Equation 3: Mean Squared Error
- Equation 4: Mean Absolute Error (primary)
- Equation 5: Root Mean Squared Error
- Equation 6: Accuracy Percentage

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
            input_dim=8,
            hidden_dim=8,
            sequence_length=sequence_length
        )
        self.model.compile_model(learning_rate=learning_rate)

    def create_sequences(self, X, y):
        """
        Equation 18: X_i = [x_{t-15}, ..., x_t] in R^{16x8}
        """
        n_samples = len(X)
        n_sequences = n_samples - self.sequence_length + 1

        X_seq = np.zeros((n_sequences, self.sequence_length, X.shape[1]))
        y_seq = np.zeros((n_sequences, 1))

        for i in range(n_sequences):
            X_seq[i] = X[i:i+self.sequence_length]
            y_seq[i] = y[i+self.sequence_length-1]

        return X_seq, y_seq

    def train(self, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """
        Equation 19: Local SGD
        """
        print("\n" + "="*60)
        print(f"Training Home {self.home_id:02d}")
        print("="*60)

        df = self.loader.load_home_data(self.home_id)
        X, y = self.loader.get_features_target(df)
        X_seq, y_seq = self.create_sequences(X, y)
        print(f"Created {len(X_seq)} sequences of length {self.sequence_length}")

        history = self.model.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

        metrics = self.evaluate(X_seq, y_seq)
        return history, metrics

    def evaluate(self, X_seq, y_seq):
        y_pred = self.model.model.predict(X_seq, verbose=0)

        # Equation 3: MSE
        mse = np.mean((y_pred - y_seq)**2)

        # Equation 4: MAE
        mae = np.mean(np.abs(y_pred - y_seq))

        # Equation 5: RMSE
        rmse = np.sqrt(mse)

        # Equation 6: Accuracy
        temp_range = y_seq.max() - y_seq.min()
        accuracy = (1 - mae / temp_range) * 100

        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
            'temp_range': temp_range
        }

        print("\n" + "="*60)
        print(f"Home {self.home_id:02d} - Final Metrics")
        print("="*60)
        print(f"Equation 3 (MSE):  {mse:.4f} degrees C^2")
        print(f"Equation 4 (MAE):  {mae:.4f} degrees C  [PRIMARY METRIC]")
        print(f"Equation 5 (RMSE): {rmse:.4f} degrees C")
        print(f"Equation 6 (Acc):  {accuracy:.2f}%")
        print(f"Temperature Range: {temp_range:.2f} degrees C")
        print("="*60)

        return metrics

    def get_parameters(self):
        return self.model.get_parameters()


if __name__ == "__main__":
    trainer = LocalTrainer(home_id=1, sequence_length=16, learning_rate=0.0005)
    history, metrics = trainer.train(epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    print("\nLocal training complete.")
