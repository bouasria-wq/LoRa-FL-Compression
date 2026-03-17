#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNU Radio LoRa Hardware Interface - hardwaretest02
===================================================
Real over-the-air LoRa TX/RX using 2 USRP B200s.
Matches master prompt specification exactly.

File: lora/gr_lora_hardware.py
"""

import time
import numpy as np
import pmt

from gnuradio import gr, blocks, uhd
import gnuradio.lora_sdr as lora_sdr

# ─────────────────────────────────────────────────────────────────────────────
# USRP serial numbers — replace with real values from `uhd_find_devices`
# ─────────────────────────────────────────────────────────────────────────────
USRP_HOME_SERIAL   = "serial=PLACEHOLDER_HOME"
USRP_SERVER_SERIAL = "serial=PLACEHOLDER_SERVER"

# ─────────────────────────────────────────────────────────────────────────────
# LoRa / RF parameters — from master prompt
# ─────────────────────────────────────────────────────────────────────────────
SF              = 7
BW              = 125000
CR              = 1
CENTER_FREQ     = 915000000
SAMP_RATE       = 1000000
TX_GAIN         = 15
RX_GAIN         = 15
PAYLOAD_LEN     = 49
HAS_CRC         = True
IMPL_HEAD       = False
SYNC_WORD       = [0x12]
LDRO_MODE       = 2
FRAME_ZERO_PADD = int(20 * 2**SF * SAMP_RATE / BW)


class LoRaTXFlowgraph(gr.top_block):
    def __init__(self, payload: bytes, usrp_serial: str):
        gr.top_block.__init__(self, "LoRa HW TX")

        self.uhd_usrp_sink = uhd.usrp_sink(
            usrp_serial,
            uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.uhd_usrp_sink.set_samp_rate(SAMP_RATE)
        self.uhd_usrp_sink.set_center_freq(CENTER_FREQ, 0)
        self.uhd_usrp_sink.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink.set_gain(TX_GAIN, 0)

        self.lora_tx = lora_sdr.lora_sdr_lora_tx(
            bw=BW,
            cr=CR,
            has_crc=HAS_CRC,
            impl_head=IMPL_HEAD,
            samp_rate=SAMP_RATE,
            sf=SF,
            ldro_mode=LDRO_MODE,
            frame_zero_padd=FRAME_ZERO_PADD,
            sync_word=SYNC_WORD
        )

        self.msg_strobe = blocks.message_strobe(
            pmt.init_u8vector(len(payload), list(payload)),
            500
        )

        self.msg_connect((self.msg_strobe, 'strobe'), (self.lora_tx, 'in'))
        self.connect((self.lora_tx, 0), (self.uhd_usrp_sink, 0))


class LoRaRXFlowgraph(gr.top_block):
    def __init__(self, usrp_serial: str):
        gr.top_block.__init__(self, "LoRa HW RX")

        self.uhd_usrp_source = uhd.usrp_source(
            usrp_serial,
            uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.uhd_usrp_source.set_samp_rate(SAMP_RATE)
        self.uhd_usrp_source.set_center_freq(CENTER_FREQ, 0)
        self.uhd_usrp_source.set_antenna('TX/RX', 0)
        self.uhd_usrp_source.set_gain(RX_GAIN, 0)
        self.uhd_usrp_source.set_min_output_buffer(
            int(np.ceil(SAMP_RATE / BW * (2**SF + 2)))
        )

        self.lora_rx = lora_sdr.lora_sdr_lora_rx(
            center_freq=int(CENTER_FREQ),
            bw=BW,
            cr=CR,
            has_crc=HAS_CRC,
            impl_head=IMPL_HEAD,
            pay_len=PAYLOAD_LEN,
            samp_rate=SAMP_RATE,
            sf=SF,
            sync_word=SYNC_WORD,
            ldro_mode=LDRO_MODE
        )

        self.msg_debug = blocks.message_debug()
        self.msg_connect((self.lora_rx, 'out'), (self.msg_debug, 'store'))
        self.connect((self.uhd_usrp_source, 0), (self.lora_rx, 0))

    def get_received_bytes(self):
        n = self.msg_debug.num_messages()
        if n > 0:
            msg = self.msg_debug.get_message(n - 1)
            return bytes(pmt.u8vector_elements(msg))
        return b''


class LoRaHardware:
    def __init__(self, tx_serial, rx_serial):
        self.tx_s = tx_serial
        self.rx_s = rx_serial

    def transmit(self, payload):
        tb = LoRaTXFlowgraph(payload, self.tx_s)
        tb.start()
        time.sleep(4.0)
        tb.stop()
        tb.wait()

    def receive(self, timeout=15):
        tb = LoRaRXFlowgraph(self.rx_s)
        tb.start()
        start = time.time()
        while time.time() - start < timeout:
            res = tb.get_received_bytes()
            if len(res) == PAYLOAD_LEN:
                tb.stop()
                tb.wait()
                return res
            time.sleep(0.1)
        tb.stop()
        tb.wait()
        return b''


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────
def get_home_radio() -> LoRaHardware:
    return LoRaHardware(tx_serial=USRP_HOME_SERIAL, rx_serial=USRP_HOME_SERIAL)

def get_server_radio() -> LoRaHardware:
    return LoRaHardware(tx_serial=USRP_SERVER_SERIAL, rx_serial=USRP_SERVER_SERIAL)
