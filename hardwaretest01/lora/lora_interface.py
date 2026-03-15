"""
LoRa Interface for GNU Radio Integration
=========================================

Implements Equations 26-34 through GNU Radio blocks.
SF=7, BW=125kHz, CR=4/5, Payload=49 bytes

File: lora/lora_interface.py
"""

import numpy as np
import os
import sys
import tempfile
from pathlib import Path

try:
    from gnuradio import gr, blocks, analog, channels
    from gnuradio import lora_sdr
    print("GNU Radio modules imported successfully")
except ImportError as e:
    print(f"Error importing GNU Radio modules: {e}")
    sys.exit(1)


class LoRaTransceiver:

    def __init__(self, sf=7, bw=125000, cr=2, payload_len=49):
        self.sf = sf
        self.bw = bw
        self.cr = cr
        self.payload_len = payload_len

        # Equations 26-28
        self.symbol_rate = bw / (2 ** sf)
        self.raw_bit_rate = sf * self.symbol_rate
        self.effective_data_rate = self.raw_bit_rate * (4/5)

        self.impl_head = False
        self.has_crc = True
        self.sync_word = 0x12
        self.tx_file = None
        self.rx_file = None

    def print_config(self):
        print("\n" + "="*60)
        print("LoRa Transceiver Configuration")
        print("="*60)
        print(f"Spreading Factor (SF): {self.sf}")
        print(f"Bandwidth (BW): {self.bw/1000:.1f} kHz")
        print(f"Coding Rate (CR): 4/{4+self.cr}")
        print(f"Payload Length: {self.payload_len} bytes")
        print(f"Equation 26 - Symbol Rate: {self.symbol_rate:.2f} symbols/sec")
        print(f"Equation 27 - Raw Bit Rate: {self.raw_bit_rate:.2f} bps")
        print(f"Equation 28 - Effective Rate: {self.effective_data_rate:.2f} bps")
        print("="*60)

    def transmit_packet(self, data_bytes, snr_db=10, channel_noise=True):
        if len(data_bytes) != self.payload_len:
            raise ValueError(f"Expected {self.payload_len} bytes, got {len(data_bytes)}")

        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as tx_f:
            self.tx_file = tx_f.name
            tx_f.write(data_bytes)

        self.rx_file = tempfile.mktemp(suffix='.bin')

        try:
            result = self._run_flowgraph(snr_db, channel_noise)
            return result
        finally:
            if os.path.exists(self.tx_file):
                os.remove(self.tx_file)
            if os.path.exists(self.rx_file):
                os.remove(self.rx_file)

    def _run_flowgraph(self, snr_db, channel_noise):
        tb = gr.top_block()

        src = blocks.file_source(gr.sizeof_char, self.tx_file, False)
        whitening = lora_sdr.whitening(False, True, ',', 'packet_len')
        add_header = lora_sdr.add_crc(self.has_crc)
        hamming_enc = lora_sdr.hamming_enc(self.cr, self.sf)
        interleaver = lora_sdr.interleaver(self.cr, self.sf, 0, self.bw)
        gray_encode = lora_sdr.gray_demap(self.sf)
        modulate = lora_sdr.modulate(self.sf, self.bw, self.bw, [0x12], 0, 8)

        if channel_noise:
            noise_voltage = 10**(-snr_db/20.0)
            channel = channels.channel_model(
                noise_voltage=noise_voltage,
                frequency_offset=0.0,
                epsilon=1.0,
                taps=[1.0],
                noise_seed=0
            )
        else:
            channel = blocks.multiply_const_cc(1.0)

        frame_sync = lora_sdr.frame_sync(868100000, self.bw, self.sf, self.impl_head, [0x12], 4, 8)
        fft_demod = lora_sdr.fft_demod(True, False)
        gray_decode = lora_sdr.gray_mapping(True)
        deinterleaver = lora_sdr.deinterleaver(True)
        hamming_dec = lora_sdr.hamming_dec(True)
        header_decoder = lora_sdr.header_decoder(self.impl_head, self.cr, self.payload_len, self.has_crc, 0, False)
        dewhitening = lora_sdr.dewhitening()
        crc_verif = lora_sdr.crc_verif(0, False)
        dst = blocks.file_sink(gr.sizeof_char, self.rx_file)

        tb.connect(src, whitening)
        tb.connect(whitening, add_header)
        tb.connect(add_header, hamming_enc)
        tb.connect(hamming_enc, interleaver)
        tb.connect(interleaver, gray_encode)
        tb.connect(gray_encode, modulate)
        tb.connect(modulate, channel)
        tb.connect(channel, frame_sync)
        tb.connect(frame_sync, fft_demod)
        tb.connect(fft_demod, gray_decode)
        tb.connect(gray_decode, deinterleaver)
        tb.connect(deinterleaver, hamming_dec)
        tb.connect(hamming_dec, header_decoder)
        tb.connect(header_decoder, dewhitening)
        tb.connect(dewhitening, crc_verif)
        tb.connect(crc_verif, dst)

        print("Transmitting LoRa packet...")
        tb.run()
        tb.wait()
        print("Transmission complete.")

        try:
            with open(self.rx_file, 'rb') as f:
                received_data = f.read()
            success = len(received_data) > 0
            with open(self.tx_file, 'rb') as f:
                tx_data = f.read()
            payload_match = received_data == tx_data
            return {
                'success': success,
                'received_data': received_data,
                'payload_match': payload_match,
                'tx_bytes': self.payload_len,
                'rx_bytes': len(received_data)
            }
        except FileNotFoundError:
            return {
                'success': False,
                'received_data': b'',
                'payload_match': False,
                'tx_bytes': self.payload_len,
                'rx_bytes': 0
            }


if __name__ == "__main__":
    lora = LoRaTransceiver(sf=7, bw=125000, cr=2, payload_len=49)
    lora.print_config()
    compressed_data = bytes([i % 256 for i in range(49)])
    result = lora.transmit_packet(compressed_data, snr_db=10, channel_noise=True)
    print(f"Success: {result['success']}")
    print(f"Payload match: {result['payload_match']}")
