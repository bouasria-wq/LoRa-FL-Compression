"""
Hegazy Aggregate Gaussian Compression - ME-CFL Version
=======================================================
Adds error feedback mechanism to compensate for compression loss.

Equations implemented:
- H1-H6: Original Hegazy AINQ compression
- ME-CFL Eq 7: v_i^t = C(P_Omega(grad_f_i(x^t) - e_i^t))
- ME-CFL Eq 8: e_i^{t+1} = e_i^t + (v_i^t - grad_f_i(x^t))

File: compression/hegazy.py
"""

import numpy as np
from scipy.stats import norm


class AggregateGaussianMechanism:

    def __init__(self, n_clients=10, sigma=0.1, seed=42):
        self.n_clients = n_clients
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # ME-CFL: Error feedback state
        # e_i^t — tracks compression residual per parameter
        self.error_feedback = None
        self.prev_params = None

        # ME-CFL: Local shift h_i^t for variance reduction
        self.local_shift = None

        # ME-CFL: Heterogeneous variance zeta_i
        self.zeta_i = 0.0
        self.gradient_history = []

    def decompose(self):
        """H2: Decompose (n, sigma) into (a, b) for encoding."""
        n = self.n_clients
        sigma = self.sigma
        w = 2 * sigma * np.sqrt(3 * n)
        a = 1.0
        b = 0.0
        return a, b

    def _sparsify(self, params, sparsity_ratio=0.1):
        """Stage 2: Keep top k parameters by magnitude."""
        k = max(1, int(len(params) * sparsity_ratio))
        indices = np.argsort(np.abs(params))[-k:]
        sparse = np.zeros_like(params)
        sparse[indices] = params[indices]
        return sparse, indices

    def _quantize(self, params, bits=8):
        """Stage 1: Quantize to b-bit integers."""
        if len(params) == 0:
            return params, 0.0, 1.0
        p_min = params.min()
        p_max = params.max()
        scale = p_max - p_min
        if scale == 0:
            return np.zeros_like(params, dtype=np.int32), p_min, 1.0
        levels = (2 ** bits) - 1
        quantized = np.round((params - p_min) / scale * levels).astype(np.int32)
        return quantized, p_min, scale

    def _dequantize(self, quantized, p_min, scale, bits=8):
        """Reverse quantization."""
        levels = (2 ** bits) - 1
        return quantized.astype(np.float32) / levels * scale + p_min

    def initialize_error_feedback(self, param_size):
        """Initialize error feedback vector e_i^0 = 0."""
        if self.error_feedback is None:
            self.error_feedback = np.zeros(param_size, dtype=np.float32)
            print(f"Error feedback initialized: {param_size} parameters")

    def initialize_local_shift(self, param_size):
        """Initialize local shift h_i^0 = 0."""
        if self.local_shift is None:
            self.local_shift = np.zeros(param_size, dtype=np.float32)
            print(f"Local shift initialized: {param_size} parameters")

    def apply_error_feedback(self, params):
        """
        ME-CFL Eq 7: Apply error feedback before compression.
        corrected = P_Omega(grad_f_i(x^t) - e_i^t)
        """
        self.initialize_error_feedback(len(params))
        corrected = params - self.error_feedback
        return corrected

    def update_error_feedback(self, original_params, compressed_params):
        """
        ME-CFL Eq 8: Update error residual.
        e_i^{t+1} = e_i^t + (v_i^t - grad_f_i(x^t))
        """
        self.initialize_error_feedback(len(original_params))
        self.error_feedback = self.error_feedback + (compressed_params - original_params)

    def measure_heterogeneous_variance(self, params):
        """
        ME-CFL Eq 3: Measure zeta_i - heterogeneous data variance.
        E[||grad_f_i(x,xi) - grad_f_i(x)||^2] <= sigma^2 + zeta_i^2
        """
        self.gradient_history.append(params.copy())
        if len(self.gradient_history) > 5:
            self.gradient_history.pop(0)
        if len(self.gradient_history) >= 2:
            stacked = np.stack(self.gradient_history)
            mean_grad = stacked.mean(axis=0)
            variances = np.mean((stacked - mean_grad) ** 2, axis=0)
            self.zeta_i = float(np.sqrt(np.mean(variances)))
        return self.zeta_i

    def encode_parameters(self, params, client_id, a, b):
        """
        Full ME-CFL compression pipeline with error feedback.
        Stage 1: Error feedback correction
        Stage 2: Sparsification
        Stage 3: Quantization
        Stage 4: Differential encoding
        Stage 5: Hegazy AINQ encoding
        Returns compressed dict targeting 49 bytes.
        """
        params_array = np.concatenate([p.flatten() for p in params])

        # Measure heterogeneous variance
        self.measure_heterogeneous_variance(params_array)

        # ME-CFL Eq 7: Apply error feedback
        corrected_params = self.apply_error_feedback(params_array)

        # Stage 2: Sparsification — top 10%
        sparse_params, indices = self._sparsify(corrected_params, sparsity_ratio=0.1)

        # Stage 1: Quantization — 8-bit
        quantized, p_min, scale = self._quantize(sparse_params[indices])

        # Stage 3: Differential encoding
        if self.prev_params is not None:
            prev_sparse = self.prev_params[indices] if len(self.prev_params) == len(sparse_params) else np.zeros_like(sparse_params[indices])
            prev_q, prev_min, prev_scale = self._quantize(prev_sparse)
            delta = quantized.astype(np.int32) - prev_q.astype(np.int32)
        else:
            delta = quantized.astype(np.int32)

        self.prev_params = sparse_params.copy()

        # Hegazy AINQ encoding (H3)
        w = 2 * self.sigma * np.sqrt(3 * self.n_clients)
        dither = self.rng.uniform(-0.5, 0.5, size=len(delta))
        m_k = np.ceil(delta.astype(float) / (a * w) + dither).astype(np.int32)

        compressed = {
            'client_id': client_id,
            'm_k': m_k,
            'dither': dither,
            'indices': indices,
            'p_min': p_min,
            'scale': scale,
            'param_size': len(params_array),
            'zeta_i': self.zeta_i,
            'a': a,
            'b': b
        }

        # ME-CFL Eq 8: Update error feedback
        reconstructed = np.zeros_like(params_array)
        dequantized = self._dequantize(quantized, p_min, scale)
        reconstructed[indices] = dequantized
        self.update_error_feedback(params_array, reconstructed)

        # Update local shift h_i
        self.initialize_local_shift(len(params_array))
        alpha = 1.0 / (1.0 + 1.0)  # alpha = 1/(1+omega)
        self.local_shift = self.local_shift + alpha * (reconstructed - self.local_shift)

        return compressed

    def decode_parameters(self, compressed, a, b):
        """
        Hegazy homomorphic decoding (H4).
        theta_agg = (aw/n)(sum(m_k) - sum(s_k)) + b*sigma
        """
        m_k = compressed['m_k']
        dither = compressed['dither']
        indices = compressed['indices']
        p_min = compressed['p_min']
        scale = compressed['scale']
        param_size = compressed['param_size']

        w = 2 * self.sigma * np.sqrt(3 * self.n_clients)

        # H4: Homomorphic decoding
        decoded_delta = (a * w / self.n_clients) * (m_k - dither) + b * self.sigma
        decoded_delta = np.clip(decoded_delta, -1e6, 1e6).astype(np.int32)

        # Dequantize
        dequantized = self._dequantize(decoded_delta.astype(np.float32), p_min, scale)

        # Reconstruct full parameter vector
        params_reconstructed = np.zeros(param_size, dtype=np.float32)
        if len(indices) <= len(params_reconstructed):
            params_reconstructed[indices] = dequantized[:len(indices)]

        return params_reconstructed

    def compress_and_aggregate(self, all_params_list):
        """
        Aggregate multiple client parameters using FedAvg.
        Used by server.
        """
        if not all_params_list:
            return None
        stacked = np.stack([np.concatenate([p.flatten() for p in params])
                           for params in all_params_list])
        return np.mean(stacked, axis=0)
