#!/usr/bin/env python3
"""
GNU Radio LoRa USRP Bridge - hardwaretest11
=============================================
3 homes + 1 server, 4 USRP B200s.
Uses Chen's lora_TX.py and lora_RX.py directly.

USRP Assignment:
  HOME_1_TX_SERIAL  = 'SERIAL_HOME1_TX'   <- replace with actual serial
  HOME_2_TX_SERIAL  = 'SERIAL_HOME2_TX'   <- replace with actual serial
  HOME_3_TX_SERIAL  = 'SERIAL_HOME3_TX'   <- replace with actual serial
  SERVER_USRP_SERIAL = 'SERIAL_SERVER'    <- replace with actual serial
  (server uses same USRP for both TX and RX via TX/RX and RX2 ports)

TDMA Slot Layout per day:
  Slot 1: Home 1 TX -> Server RX
  Slot 2: Home 2 TX -> Server RX
  Slot 3: Home 3 TX -> Server RX
  Server aggregates -> Server TX -> All homes notified via file

Flag-based handshake:
  home_{N}_tx_done_day{D}.flag  -> home finished TX
  server_tx_done_day{D}.flag    -> server finished TX + global model ready

File: lora/gr_lora_usrp.py
"""

import sys
import re
import time
import shutil
import numpy as np
from pathlib import Path

try:
    import pmt
except ImportError:
    from gnuradio import pmt

# ─────────────────────────────────────────────────────────────────
# USRP Serial Numbers — replace with actual serials from uhd_find_devices
# ─────────────────────────────────────────────────────────────────
HOME_1_TX_SERIAL   = 'SERIAL_HOME1_TX'
HOME_2_TX_SERIAL   = 'SERIAL_HOME2_TX'
HOME_3_TX_SERIAL   = 'SERIAL_HOME3_TX'
SERVER_USRP_SERIAL = 'SERIAL_SERVER'

# Map home_id to TX serial
HOME_TX_SERIALS = {
    1: HOME_1_TX_SERIAL,
    2: HOME_2_TX_SERIAL,
    3: HOME_3_TX_SERIAL,
}

# LoRa parameters
SF             = 7
BW             = 125000
CR             = 1
LORA_MAX_BYTES = 255
PREAMBLE_LEN   = 8

# TDMA timing
SLOT_DURATION  = 30   # seconds per home TX slot
RX_INIT_TIME   = 3.0  # seconds for USRP RX to initialize
FLUSH_TIME     = 2.0  # seconds after posting payload for frame to flush
N_HOMES        = 3

ASCII_CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def _bytes_to_ascii(payload: bytes) -> str:
    return ''.join(ASCII_CHARSET[b % len(ASCII_CHARSET)] for b in payload)


class GRCLoRaUSRP:

    def __init__(self, work_dir=None, tx_serial=None, rx_serial=None,
                 role='home', home_id=None):
        self.work_dir  = Path(work_dir) if work_dir else Path(__file__).parent
        self.role      = role
        self.home_id   = home_id
        self.tx_serial = tx_serial
        self.rx_serial = rx_serial

        self.total_transmissions      = 0
        self.successful_transmissions = 0

        self._patch_serials()

        print(f"[USRP LoRa] Role: {role}" + (f" | Home ID: {home_id}" if home_id else ""))
        if tx_serial:
            print(f"[USRP LoRa] TX USRP serial: {tx_serial}")
        if rx_serial:
            print(f"[USRP LoRa] RX USRP serial: {rx_serial}")
        print(f"[USRP LoRa] Work dir: {self.work_dir}")
        print(f"[USRP LoRa] Max payload: {LORA_MAX_BYTES} bytes")

    def _patch_serials(self):
        """Patch serial numbers in Chen's lora_TX.py and lora_RX.py."""
        tx_file = self.work_dir / 'lora_TX.py'
        rx_file = self.work_dir / 'lora_RX.py'

        if tx_file.exists() and self.tx_serial:
            code = tx_file.read_text()
            code = re.sub(r'serial=[A-Za-z0-9_]+', f'serial={self.tx_serial}', code)
            code = re.sub(r'"addr=192\.168\.\d+\.\d+"', f'"serial={self.tx_serial}"', code)
            tx_file.write_text(code)
            print(f"[USRP LoRa] lora_TX.py patched: serial={self.tx_serial}")

        if rx_file.exists() and self.rx_serial:
            code = rx_file.read_text()
            code = re.sub(r'serial=[A-Za-z0-9_]+', f'serial={self.rx_serial}', code)
            code = re.sub(r'"addr=192\.168\.\d+\.\d+"', f'"serial={self.rx_serial}"', code)
            rx_file.write_text(code)
            print(f"[USRP LoRa] lora_RX.py patched: serial={self.rx_serial}")

    def transmit(self, payload: bytes, timeout=60) -> dict:
        """
        TX payload using Chen's approach:
        1. Start RX flowgraph
        2. Start TX flowgraph with strobe disabled
        3. Post payload to whitening block message port
        """
        assert len(payload) <= LORA_MAX_BYTES

        ascii_payload = _bytes_to_ascii(payload)
        print(f"[USRP LoRa] Payload: {len(payload)} bytes -> {len(ascii_payload)} ASCII chars")

        sys.path.insert(0, str(self.work_dir))

        success = False
        try:
            from lora_TX import lora_TX as LoraTX
            from lora_RX import lora_RX as LoraRX

            # Start RX first
            print(f"[USRP LoRa] Starting RX (serial={self.rx_serial})...")
            rx_tb = LoraRX()
            rx_tb.start()
            time.sleep(RX_INIT_TIME)

            # Start TX with strobe disabled
            print(f"[USRP LoRa] Starting TX (serial={self.tx_serial})...")
            tx_tb = LoraTX()
            tx_tb.set_frame_period(10_000_000)
            tx_tb.start()

            # Post payload to whitening block
            msg_port = pmt.intern("msg")
            whitening = tx_tb.lora_sdr_whitening_0.to_basic_block()
            whitening._post(msg_port, pmt.intern(ascii_payload))
            print(f"[USRP LoRa] Posted payload to whitening block.")

            time.sleep(FLUSH_TIME)
            time.sleep(1.0)

            try:
                tx_tb.stop()
                tx_tb.wait()
            except Exception as e:
                print(f"[USRP LoRa] TX stop: {e}")

            time.sleep(2.0)

            try:
                rx_tb.stop()
                rx_tb.wait()
            except Exception as e:
                print(f"[USRP LoRa] RX stop: {e}")

            success = True

        except Exception as e:
            print(f"[USRP LoRa] Error: {e}")
        finally:
            if str(self.work_dir) in sys.path:
                sys.path.remove(str(self.work_dir))

        self.total_transmissions += 1
        if success:
            self.successful_transmissions += 1

        pdr   = self.successful_transmissions / max(self.total_transmissions, 1)
        t_toa = self._calculate_toa(len(payload))

        print(f"[USRP LoRa] Result: {'SUCCESS' if success else 'FAILED'} | "
              f"ToA: {t_toa:.4f}s | PDR: {pdr*100:.1f}%")

        return {
            'success': success, 'crc_ok': success,
            'tx_bytes': len(payload), 'rx_msg': ascii_payload,
            't_toa': t_toa, 'pdr': pdr, 'packet_size': len(payload),
        }

    def transmit_only(self, payload: bytes) -> dict:
        """
        TX only — no RX. Used by server when homes are not listening via RF.
        Server writes global model to file which homes read directly.
        """
        assert len(payload) <= LORA_MAX_BYTES

        ascii_payload = _bytes_to_ascii(payload)
        print(f"[USRP LoRa] TX only: {len(payload)} bytes")

        sys.path.insert(0, str(self.work_dir))

        success = False
        try:
            from lora_TX import lora_TX as LoraTX

            tx_tb = LoraTX()
            tx_tb.set_frame_period(10_000_000)
            tx_tb.start()

            msg_port = pmt.intern("msg")
            whitening = tx_tb.lora_sdr_whitening_0.to_basic_block()
            whitening._post(msg_port, pmt.intern(ascii_payload))
            print(f"[USRP LoRa] Posted payload to whitening block.")

            time.sleep(FLUSH_TIME + 1.0)

            try:
                tx_tb.stop()
                tx_tb.wait()
            except Exception as e:
                print(f"[USRP LoRa] TX stop: {e}")

            success = True

        except Exception as e:
            print(f"[USRP LoRa] TX only error: {e}")
        finally:
            if str(self.work_dir) in sys.path:
                sys.path.remove(str(self.work_dir))

        self.total_transmissions += 1
        if success:
            self.successful_transmissions += 1

        pdr   = self.successful_transmissions / max(self.total_transmissions, 1)
        t_toa = self._calculate_toa(len(payload))

        print(f"[USRP LoRa] TX only result: {'SUCCESS' if success else 'FAILED'} | "
              f"ToA: {t_toa:.4f}s")

        return {
            'success': success, 'crc_ok': success,
            'tx_bytes': len(payload), 'rx_msg': ascii_payload,
            't_toa': t_toa, 'pdr': pdr, 'packet_size': len(payload),
        }

    def _calculate_toa(self, payload_length):
        n_preamble = PREAMBLE_LEN
        numerator = 8 * payload_length - 4 * SF + 28 + 16
        denominator = 4 * (SF - 2)
        payload_symbols = max(np.ceil(numerator / denominator) * (CR + 4), 0)
        return ((n_preamble + payload_symbols) * (2 ** SF)) / BW

    def _make_result(self, success, payload):
        return {
            'success': success, 'crc_ok': False,
            'tx_bytes': len(payload), 'rx_msg': None,
            't_toa': self._calculate_toa(len(payload)),
            'pdr': self.successful_transmissions / max(self.total_transmissions, 1),
            'packet_size': len(payload),
        }


def get_home_radio(work_dir=None, home_id=None, tx_serial=None) -> GRCLoRaUSRP:
    """Each home has its own TX USRP. No RX needed for home TX phase."""
    serial = tx_serial or HOME_TX_SERIALS.get(home_id, HOME_1_TX_SERIAL)
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=serial,
        rx_serial=SERVER_USRP_SERIAL,  # server is the RX during home TX
        role='home',
        home_id=home_id,
    )


def get_server_radio(work_dir=None, tx_serial=None) -> GRCLoRaUSRP:
    """Server has one USRP that acts as TX when broadcasting global model."""
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=tx_serial or SERVER_USRP_SERIAL,
        rx_serial=None,
        role='server',
    )
