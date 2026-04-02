import numpy as np
from gnuradio import gr

class hybrid_chirp_modulator(gr.sync_block):
    """
    Hybrid chirp modulation block (generalized version).
    Applies chirp rate modification to already modulated LoRa payload chirps.
    Automatically calculates samples_per_symbol based on SF, BW, and Fs.
    """

    def __init__(self, sf=7, bw=125000, fs=500000, preamble_chirps=12.25, mod_dict=None):
        gr.sync_block.__init__(
            self,
            name="hybrid_chirp_modulator_v3",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )

        # 参数设置
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.Tc = 2 ** sf / bw  # chirp duration
        self.samples_per_symbol = int(self.Tc * fs)
        self.N = self.samples_per_symbol

        self.preamble_samples = int(preamble_chirps * self.N)

        self.mod_dict = mod_dict or {
            0: {'ratio': 0.3, 'k1': 1.0, 'k2': 1.5},
            1: {'ratio': 0.5, 'k1': 1.1, 'k2': 1.6},
            2: {'ratio': 0.4, 'k1': 1.2, 'k2': 1.7},
        }

        self.symbol_ids = list(self.mod_dict.keys())
        self.sample_index = 0
        self.symbol_counter = 0

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        total_samples = len(in0)
        N = self.N

        # Pass-through preamble
        if self.sample_index < self.preamble_samples:
            pre_end = min(self.preamble_samples - self.sample_index, total_samples)
            out[:pre_end] = in0[:pre_end]
            self.sample_index += pre_end
            return pre_end

        i = self.preamble_samples - self.sample_index
        if i < 0:
            i = 0

        while i + N <= total_samples:
            chirp = in0[i:i + N]
            symbol_id = self.symbol_ids[self.symbol_counter % len(self.symbol_ids)]
            cfg = self.mod_dict[symbol_id]
            print(f"[{self.sample_index}] Modulating: ID={symbol_id}, ratio={cfg['ratio']}, k1={cfg['k1']}, k2={cfg['k2']}")
            mod = self.hybrid_modulate_on_payload(chirp, cfg['k1'], cfg['k2'], cfg['ratio'])
            out[i:i + N] = mod
            i += N
            self.sample_index += N
            self.symbol_counter += 1

        return i

    def hybrid_modulate_on_payload(self, chirp, k1, k2, alpha):
        """
        Modulate already modulated payload chirp by applying chirp rate change in phase.
        """
        N = len(chirp)
        split = int(alpha * N)
        t1 = np.arange(split)
        t2 = np.arange(N - split)
        t2_shifted = t2 + split

        # 相位修改部分是相对于标准 chirp 的增量（仍使用归一化方式）
        phi1 = 2 * np.pi * (k1 - 1.0) * (t1**2 / (2 * N) - 0.5 * t1)
        phi2 = 2 * np.pi * (k2 - 1.0) * (t2_shifted**2 / (2 * N) - 0.5 * t2_shifted)

        if len(phi2) > 0:
            phase_offset = phi1[-1] - phi2[0]
            phi2 += phase_offset

        phi = np.concatenate([phi1, phi2])
        return chirp * np.exp(1j * phi)
