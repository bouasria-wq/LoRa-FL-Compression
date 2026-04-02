#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lora_TX_usrp.py - LoRa TX with File Source + USRP B200
========================================================
Based on Chen's lora_TX.py from gr-lora_sdr.
The ONLY change from the original: Message Strobe replaced with
File Source connected to Whitening port 0, exactly as done in
tx_rx_simulation.grc from the same repo.

TX chain (LoRa PHY untouched):
  File Source -> Whitening -> Header -> Add CRC -> Hamming Enc
  -> Interleaver -> Gray Demap -> Modulate -> USRP Sink

Usage:
  python3 lora_TX_usrp.py --input tx_payload.txt --tx_serial 32BBAD0
"""

from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
import argparse
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import gnuradio.lora_sdr as lora_sdr


class lora_TX_usrp(gr.top_block):

    def __init__(self, input_file='tx_payload.txt', tx_serial='32BBAD0'):
        gr.top_block.__init__(self, "Lora TX USRP File Source", catch_exceptions=True)

        ##################################################
        # Variables — matched to lora_RX.py from Chen
        ##################################################
        self.sf          = sf          = 7
        self.bw          = bw          = 125000
        self.samp_rate   = samp_rate   = 500000
        self.impl_head   = impl_head   = False
        self.has_crc     = has_crc     = True
        self.cr          = cr          = 1
        self.center_freq = center_freq = 868.1e6
        self.TX_gain     = TX_gain     = 0

        ##################################################
        # Blocks
        ##################################################

        # USRP Sink
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join((f"serial={tx_serial}", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0, 1)),
            ),
            'frame_len',
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0.set_bandwidth(bw, 0)
        self.uhd_usrp_sink_0.set_gain(TX_gain, 0)

        # LoRa PHY chain — UNTOUCHED from Chen's lora_TX.py
        self.lora_sdr_whitening_0   = lora_sdr.whitening(False, False, ',', 'packet_len')
        self.lora_sdr_modulate_0    = lora_sdr.modulate(sf, samp_rate, bw, [8, 16], int(20 * 2**sf * samp_rate / bw), 8)
        self.lora_sdr_modulate_0.set_min_output_buffer(10000000)
        self.lora_sdr_interleaver_0 = lora_sdr.interleaver(cr, sf, 0, 125000)
        self.lora_sdr_header_0      = lora_sdr.header(impl_head, has_crc, cr)
        self.lora_sdr_hamming_enc_0 = lora_sdr.hamming_enc(cr, sf)
        self.lora_sdr_gray_demap_0  = lora_sdr.gray_demap(sf)
        self.lora_sdr_add_crc_0     = lora_sdr.add_crc(has_crc)

        # File Source — replaces Message Strobe (same as tx_rx_simulation.grc)
        self.blocks_file_source_0 = blocks.file_source(
            gr.sizeof_char * 1,
            input_file,
            False,  # repeat=False
            0,
            0
        )
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)

        ##################################################
        # Connections
        # File Source -> Whitening (port 0) — same as tx_rx_simulation.grc
        # Rest of chain untouched from Chen's lora_TX.py
        ##################################################
        self.connect((self.blocks_file_source_0,   0), (self.lora_sdr_whitening_0,   0))
        self.connect((self.lora_sdr_whitening_0,   0), (self.lora_sdr_header_0,      0))
        self.connect((self.lora_sdr_header_0,      0), (self.lora_sdr_add_crc_0,     0))
        self.connect((self.lora_sdr_add_crc_0,     0), (self.lora_sdr_hamming_enc_0, 0))
        self.connect((self.lora_sdr_hamming_enc_0, 0), (self.lora_sdr_interleaver_0, 0))
        self.connect((self.lora_sdr_interleaver_0, 0), (self.lora_sdr_gray_demap_0,  0))
        self.connect((self.lora_sdr_gray_demap_0,  0), (self.lora_sdr_modulate_0,    0))
        self.connect((self.lora_sdr_modulate_0,    0), (self.uhd_usrp_sink_0,        0))


def main(top_block_cls=lora_TX_usrp, options=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',     type=str, default='tx_payload.txt')
    parser.add_argument('--tx_serial', type=str, default='32BBAD0')
    args = parser.parse_args()

    tb = top_block_cls(input_file=args.input, tx_serial=args.tx_serial)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass

    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
