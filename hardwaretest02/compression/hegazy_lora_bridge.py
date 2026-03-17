import base64
import pickle

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

    def write_lora_file_chunked(self, binary_data, output_file, chunk_size=40):
        chunks = []
        for i in range(0, len(binary_data), chunk_size):
            chunk = binary_data[i:i+chunk_size]
            encoded = base64.b64encode(chunk).decode('ascii')
            chunks.append(f"{i:04d}:{encoded}")
        lora_format = ','.join(chunks)
        with open(output_file, 'w') as f:
            f.write(lora_format)
        print(f"Written {len(chunks)} indexed chunks, {len(lora_format)} total chars")

    def read_lora_chunked(self, lora_ascii, expected_size):
        chunks = lora_ascii.split(',')
        data_dict = {}
        for chunk in chunks:
            if ':' not in chunk:
                continue
            try:
                idx_str, encoded = chunk.split(':', 1)
                idx = int(idx_str)
                data_dict[idx] = base64.b64decode(encoded)
            except:
                continue
        result = bytearray(expected_size)
        recovered = 0
        for idx, data in sorted(data_dict.items()):
            end = min(idx + len(data), expected_size)
            result[idx:end] = data[:end-idx]
            recovered += len(data)
        coverage = (recovered / expected_size) * 100
        print(f"Recovered {recovered}/{expected_size} bytes ({coverage:.1f}%)")
        return bytes(result), coverage

    def write_lora_file(self, binary_data, output_file):
        lora_ascii = self.binary_to_lora_ascii(binary_data)
        with open(output_file, 'w') as f:
            f.write(lora_ascii)
        print(f"Converted {len(binary_data)} bytes to {len(lora_ascii)} ASCII chars")
