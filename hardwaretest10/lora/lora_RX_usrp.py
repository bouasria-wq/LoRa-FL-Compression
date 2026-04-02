#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lora_RX_usrp.py - LoRa RX with USRP B200
==========================================
Chen's lora_RX.py from gr-lora_sdr.
Only change: serial number and LoRa params as command line args.
The full LoRa RX PHY chain is untouched.

RX chain:
  USRP Source -> Frame Sync -> FFT Demod -> Gray Map -> Deinterleave
  -> Hamming Dec -> Header Dec -> Dewhiten -> CRC Verify

Usage:
  python3 lora_RX_usrp.py --rx_serial 32BBAD9
"""

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
import numpy as np


class lora_RX_usrp(gr.top_block):

    def __init__(self, rx_serial='32BBAD9'):
        gr.top_block.__init__(self, "Lora RX USRP", catch_exceptions=True)

        ##################################################
        # Variables — matched to TX
        ##################################################
        self.sf           = sf           = 7
        self.bw           = bw           = 125000
        self.samp_rate    = samp_rate    = 500000
        self.soft_decoding = soft_decoding = True
        self.pay_len      = pay_len      = 255
        self.impl_head    = impl_head    = False
        self.has_crc      = has_crc      = True
        self.cr           = cr           = 1
        self.center_freq  = center_freq  = 868.1e6

        ##################################################
        # Blocks — untouched from Chen's lora_RX.py
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join((f"serial={rx_serial}", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0, 1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_source_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_source_0.set_gain(100, 0)
        self.uhd_usrp_source_0.set_min_output_buffer(520)

        self.lora_sdr_header_decoder_0 = lora_sdr.header_decoder(impl_head, cr, pay_len, has_crc, False, True)
        self.lora_sdr_hamming_dec_0    = lora_sdr.hamming_dec(soft_decoding)
        self.lora_sdr_gray_mapping_0   = lora_sdr.gray_mapping(soft_decoding)
        self.lora_sdr_frame_sync_0     = lora_sdr.frame_sync(int(center_freq), bw, sf, impl_head, [18], int(samp_rate / bw), 8)
        self.lora_sdr_fft_demod_0      = lora_sdr.fft_demod(soft_decoding, True)
        self.lora_sdr_dewhitening_0    = lora_sdr.dewhitening()
        self.lora_sdr_deinterleaver_0  = lora_sdr.deinterleaver(soft_decoding)
        self.lora_sdr_crc_verif_0      = lora_sdr.crc_verif(1, False)

        ##################################################
        # Connections — untouched from Chen's lora_RX.py
        ##################################################
        self.msg_connect((self.lora_sdr_header_decoder_0, 'frame_info'), (self.lora_sdr_frame_sync_0, 'frame_info'))
        self.connect((self.lora_sdr_deinterleaver_0,  0), (self.lora_sdr_hamming_dec_0,     0))
        self.connect((self.lora_sdr_dewhitening_0,    0), (self.lora_sdr_crc_verif_0,       0))
        self.connect((self.lora_sdr_fft_demod_0,      0), (self.lora_sdr_gray_mapping_0,    0))
        self.connect((self.lora_sdr_frame_sync_0,     0), (self.lora_sdr_fft_demod_0,       0))
        self.connect((self.lora_sdr_gray_mapping_0,   0), (self.lora_sdr_deinterleaver_0,   0))
        self.connect((self.lora_sdr_hamming_dec_0,    0), (self.lora_sdr_header_decoder_0,  0))
        self.connect((self.lora_sdr_header_decoder_0, 0), (self.lora_sdr_dewhitening_0,     0))
        self.connect((self.uhd_usrp_source_0,         0), (self.lora_sdr_frame_sync_0,      0))


def main(top_block_cls=lora_RX_usrp, options=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rx_serial', type=str, default='32BBAD9')
    args = parser.parse_args()

    tb = top_block_cls(rx_serial=args.rx_serial)

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
