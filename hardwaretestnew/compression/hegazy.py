"""
Hegazy Aggregate Gaussian Compression - ME-CFL Version
File: compression/hegazy.py
"""
import numpy as np


class AggregateGaussianMechanism:

    def __init__(self, n_clients=10, sigma=0.1, seed=42):
        self.n_clients = n_clients
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.error_feedback = None
        self.prev_params = None
        self.local_shift = None
        self.zeta_i = 0.0
        self.gradient_history = []

    def decompose(self):
        return 1.0, 0.0

    def _sparsify(self, params, sparsity_ratio=0.1):
        k = max(1, int(len(params) * sparsity_ratio))
        indices = np.argsort(np.abs(params))[-k:]
        sparse = np.zeros_like(params)
        sparse[indices] = params[indices]
        return sparse, indices

    def _quantize(self, params, bits=8):
        if len(params) == 0:
            return params, 0.0, 1.0
        p_min, p_max = params.min(), params.max()
        scale = p_max - p_min
        if scale == 0:
            return np.zeros_like(params, dtype=np.int32), p_min, 1.0
        levels = (2 ** bits) - 1
        return np.round((params - p_min) / scale * levels).astype(np.int32), p_min, scale

    def _dequantize(self, quantized, p_min, scale, bits=8):
        return quantized.astype(np.float32) / ((2 ** bits) - 1) * scale + p_min

    def initialize_error_feedback(self, param_size):
        if self.error_feedback is None:
            self.error_feedback = np.zeros(param_size, dtype=np.float32)
            print(f"Error feedback initialized: {param_size} parameters")

    def initialize_local_shift(self, param_size):
        if self.local_shift is None:
            self.local_shift = np.zeros(param_size, dtype=np.float32)

    def apply_error_feedback(self, params):
        self.initialize_error_feedback(len(params))
        return params - self.error_feedback

    def update_error_feedback(self, original_params, compressed_params):
        self.initialize_error_feedback(len(original_params))
        self.error_feedback = self.error_feedback + (compressed_params - original_params)

    def measure_heterogeneous_variance(self, params):
        self.gradient_history.append(params.copy())
        if len(self.gradient_history) > 5:
            self.gradient_history.pop(0)
        if len(self.gradient_history) >= 2:
            stacked = np.stack(self.gradient_history)
            self.zeta_i = float(np.sqrt(np.mean(np.mean((stacked - stacked.mean(axis=0))**2, axis=0))))
        return self.zeta_i

    def encode_parameters(self, params, client_id, a, b):
        params_array = np.concatenate([p.flatten() for p in params])
        self.measure_heterogeneous_variance(params_array)
        corrected = self.apply_error_feedback(params_array)
        sparse, indices = self._sparsify(corrected)
        quantized, p_min, scale = self._quantize(sparse[indices])

        if self.prev_params is not None:
            prev = self.prev_params[indices] if len(self.prev_params) == len(sparse) else np.zeros_like(sparse[indices])
            delta = quantized.astype(np.int32) - self._quantize(prev)[0].astype(np.int32)
        else:
            delta = quantized.astype(np.int32)

        self.prev_params = sparse.copy()
        w = 2 * self.sigma * np.sqrt(3 * self.n_clients)
        dither = self.rng.uniform(-0.5, 0.5, size=len(delta))
        m_k = np.ceil(delta.astype(float) / (a * w) + dither).astype(np.int32)

        compressed = {
            'client_id': client_id, 'm_k': m_k, 'dither': dither,
            'indices': indices, 'p_min': p_min, 'scale': scale,
            'param_size': len(params_array), 'zeta_i': self.zeta_i, 'a': a, 'b': b
        }

        reconstructed = np.zeros_like(params_array)
        reconstructed[indices] = self._dequantize(quantized, p_min, scale)
        self.update_error_feedback(params_array, reconstructed)
        self.initialize_local_shift(len(params_array))
        self.local_shift = self.local_shift + 0.5 * (reconstructed - self.local_shift)

        return compressed

    def decode_parameters(self, compressed, a, b):
        w = 2 * self.sigma * np.sqrt(3 * self.n_clients)
        decoded = (a * w / self.n_clients) * (compressed['m_k'] - compressed['dither']) + b * self.sigma
        decoded = np.clip(decoded, -1e6, 1e6).astype(np.int32)
        dequantized = self._dequantize(decoded.astype(np.float32), compressed['p_min'], compressed['scale'])
        result = np.zeros(compressed['param_size'], dtype=np.float32)
        idx = compressed['indices']
        if len(idx) <= len(result):
            result[idx] = dequantized[:len(idx)]
        return result
