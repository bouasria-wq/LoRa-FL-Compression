"""
LSTM Model Architecture for Temperature Prediction
==================================================
553 parameters (H=8, d=8)
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
        print(f"Model: {total_params} params (expected {expected_params}) — {'VERIFIED' if total_params == expected_params else 'MISMATCH'}")

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

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
