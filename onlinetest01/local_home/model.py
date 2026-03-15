"""
LSTM Model Architecture for Temperature Prediction
==================================================

Implements Equations 7-19 from the federated learning system.

LSTM Cell Operations:
- Equation 7:  Forget Gate
- Equation 8:  Input Gate
- Equation 9:  Candidate Cell State
- Equation 10: Cell State Update
- Equation 11: Output Gate
- Equation 12: Hidden State
- Equation 13: Output Prediction
- Equation 14: Sigmoid Activation

Parameter Count:
- Equation 15: Total parameters formula
- Equation 16: 553 parameters (H=8, d=8)
- Equation 17: 2212 bytes uncompressed

File: local_home/model.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class TemperatureLSTM:

    def __init__(self, input_dim=8, hidden_dim=8, sequence_length=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.model = self._build_model()
        self._verify_parameters()

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.input_dim)),
            layers.LSTM(
                units=self.hidden_dim,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=False,
                name='lstm_layer'
            ),
            layers.Dense(
                units=1,
                activation='linear',
                name='output_layer'
            )
        ], name='temperature_lstm')
        return model

    def _verify_parameters(self):
        total_params = self.model.count_params()
        expected_params = 4 * (self.hidden_dim * (self.input_dim + self.hidden_dim) + self.hidden_dim) + (self.hidden_dim + 1)
        print("\n" + "="*60)
        print("Model Parameter Verification")
        print("="*60)
        print(f"Equation 15: |theta| = 4(H(d + H) + H) + (H + 1)")
        print(f"Equation 16: |theta| = 4(8(8 + 8) + 8) + 9 = 553")
        print(f"Expected: {expected_params} parameters")
        print(f"Actual:   {total_params} parameters")
        if total_params == expected_params:
            print("Status: VERIFIED")
        else:
            print(f"Status: MISMATCH (difference: {total_params - expected_params})")
        uncompressed_size = total_params * 4
        print(f"Equation 17: Uncompressed size = {uncompressed_size} bytes")
        print("="*60)

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        print(f"Model compiled with learning_rate={learning_rate}")

    def summary(self):
        print("\n" + "="*60)
        print("LSTM Model Architecture")
        print("="*60)
        self.model.summary()
        print("="*60)

    def get_model(self):
        return self.model

    def get_parameters(self):
        params = []
        for layer in self.model.layers:
            weights = layer.get_weights()
            for w in weights:
                params.append(w.flatten())
        return np.concatenate(params)

    def set_parameters(self, params):
        idx = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            new_weights = []
            for w in weights:
                size = w.size
                new_w = params[idx:idx+size].reshape(w.shape)
                new_weights.append(new_w)
                idx += size
            layer.set_weights(new_weights)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        print(f"Weights saved to: {filepath}")

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        print(f"Weights loaded from: {filepath}")


if __name__ == "__main__":
    lstm = TemperatureLSTM(input_dim=8, hidden_dim=8, sequence_length=16)
    lstm.summary()
    lstm.compile_model(learning_rate=0.001)

    dummy_input = np.random.randn(10, 16, 8).astype(np.float32)
    predictions = lstm.model.predict(dummy_input, verbose=0)
    print(f"\nTest prediction shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0][0]:.2f} degrees C")

    params = lstm.get_parameters()
    print(f"Total parameters extracted: {len(params)}")
