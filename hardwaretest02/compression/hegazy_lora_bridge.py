import base64
import pickle
import struct
import numpy as np


class HegazyLoRaBridge:
    def __init__(self):
        pass

    def binary_to_lora_ascii(self, binary_data):
        encoded = base64.b64encode(binary_data).decode('ascii')
        chunks = [encoded[i:i+16] for i in range(0, len(encoded), 16)]
        return ','.join(chunks)

    def lora_ascii_to_binary(self, lora_ascii):
        encoded = lora_ascii.replace(',', '')
        return base64.b64decode(encoded)

    def write_lora_file(self, binary_data, output_file):
        lora_ascii = self.binary_to_lora_ascii(binary_data)
        with open(output_file, 'w') as f:
            f.write(lora_ascii)

    def pack_compressed(self, compressed: dict) -> bytes:
        """
        Pack compressed Hegazy dict into raw bytes for LoRa transmission.
        Replaces pickle — avoids Python metadata overhead.

        Format:
          - client_id:  1 byte  (uint8)
          - n_values:   1 byte  (uint8) — number of m_k/dither/indices values
          - p_min:      4 bytes (float32)
          - scale:      4 bytes (float32)
          - zeta_i:     4 bytes (float32)
          - a:          4 bytes (float32)
          - b:          4 bytes (float32)
          - m_k:        n_values * 2 bytes (int16 each)
          - indices:    n_values * 2 bytes (uint16 each)
          Total for n=55: 18 + 55*2 + 55*2 = 238 bytes
        """
        m_k     = np.array(compressed['m_k'],     dtype=np.int16)
        indices = np.array(compressed['indices'],  dtype=np.uint16)
        n       = len(m_k)

        # Fixed header: 18 bytes
        header = struct.pack(
            '>BBffff f',
            int(compressed['client_id']),  # 1 byte
            n,                              # 1 byte
            float(compressed['p_min']),     # 4 bytes
            float(compressed['scale']),     # 4 bytes
            float(compressed['zeta_i']),    # 4 bytes
            float(compressed['a']),         # 4 bytes
            float(compressed['b']),         # 4 bytes (note: space in fmt string ignored)
        )

        # Variable: m_k + indices
        m_k_bytes     = m_k.astype('>i2').tobytes()
        indices_bytes = indices.astype('>u2').tobytes()

        raw = header + m_k_bytes + indices_bytes
        print(f"[LoRaBridge] Packed compressed: {len(raw)} bytes "
              f"(vs pickle: ~1575 bytes)")
        return raw

    def unpack_compressed(self, raw: bytes) -> dict:
        """
        Unpack raw bytes back into Hegazy compressed dict.
        Inverse of pack_compressed().
        """
        # Fixed header: 18 bytes
        client_id, n, p_min, scale, zeta_i, a, b = struct.unpack(
            '>BBfffff', raw[:18]
        )

        # m_k: n * 2 bytes
        m_k_end = 18 + n * 2
        m_k = np.frombuffer(raw[18:m_k_end], dtype='>i2').astype(np.int32)

        # indices: n * 2 bytes
        indices_end = m_k_end + n * 2
        indices = np.frombuffer(raw[m_k_end:indices_end], dtype='>u2').astype(np.int64)

        return {
            'client_id':  int(client_id),
            'm_k':        m_k,
            'dither':     np.zeros(n, dtype=np.float64),  # dither not transmitted
            'indices':    indices,
            'p_min':      np.float32(p_min),
            'scale':      np.float32(scale),
            'param_size': 553,
            'zeta_i':     float(zeta_i),
            'a':          float(a),
            'b':          float(b),
        }
