import base64
import struct
import numpy as np


class HegazyLoRaBridge:
    def __init__(self):
        pass

    def pack_compressed(self, compressed: dict) -> bytes:
        m_k     = np.array(compressed['m_k'],    dtype=np.int16)
        indices = np.array(compressed['indices'], dtype=np.uint16)
        n       = len(m_k)

        header = struct.pack(
            '>BBfffff',
            int(compressed['client_id']),
            n,
            float(compressed['p_min']),
            float(compressed['scale']),
            float(compressed['zeta_i']),
            float(compressed['a']),
            float(compressed['b']),
        )

        m_k_bytes     = m_k.astype('>i2').tobytes()
        indices_bytes = indices.astype('>u2').tobytes()

        raw = header + m_k_bytes + indices_bytes
        print(f"[LoRaBridge] Packed compressed: {len(raw)} bytes")
        return raw

    def unpack_compressed(self, raw: bytes) -> dict:
        client_id, n, p_min, scale, zeta_i, a, b = struct.unpack(
            '>BBfffff', raw[:22]
        )

        m_k_end = 22 + n * 2
        m_k = np.frombuffer(raw[22:m_k_end], dtype='>i2').astype(np.int32)

        indices_end = m_k_end + n * 2
        indices = np.frombuffer(raw[m_k_end:indices_end], dtype='>u2').astype(np.int64)

        return {
            'client_id':  int(client_id),
            'm_k':        m_k,
            'dither':     np.zeros(n, dtype=np.float64),
            'indices':    indices,
            'p_min':      np.float32(p_min),
            'scale':      np.float32(scale),
            'param_size': 553,
            'zeta_i':     float(zeta_i),
            'a':          float(a),
            'b':          float(b),
        }

    def payload_to_hex_string(self, payload: bytes) -> str:
        return payload.hex()

    def hex_string_to_payload(self, hex_str: str) -> bytes:
        return bytes.fromhex(hex_str)
