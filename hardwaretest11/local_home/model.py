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
        total = self.model.count_params()
        expected = 4*(hidden_dim*(input_dim+hidden_dim)+hidden_dim)+(hidden_dim+1)
        print(f"Model: {total} params (expected {expected}) — {'VERIFIED' if total==expected else 'MISMATCH'}")

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.input_dim)),
            layers.LSTM(units=self.hidden_dim, activation='tanh', recurrent_activation='sigmoid', return_sequences=False, name='lstm_layer'),
            layers.Dense(units=1, activation='linear', name='output_layer')
        ], name='temperature_lstm')
        return model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

    def get_parameters(self):
        return np.concatenate([w.flatten() for layer in self.model.layers for w in layer.get_weights()])

    def set_parameters(self, params):
        idx = 0
        for layer in self.model.layers:
            new_weights = []
            for w in layer.get_weights():
                new_weights.append(params[idx:idx+w.size].reshape(w.shape))
                idx += w.size
            layer.set_weights(new_weights)
