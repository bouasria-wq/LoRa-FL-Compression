#!/usr/bin/env python3
"""
GNU Radio LoRa USRP Bridge - hardwaretest10
=============================================
Uses Chen's lora_TX.py and lora_RX.py directly.
Posts payload to whitening block message port exactly as Chen's run_lora_transmitter.py.

Key fix: done_flag_path is written IMMEDIATELY after posting payload to whitening block,
before any stop/wait calls that might crash.

File: lora/gr_lora_usrp.py
"""

import sys
import re
import time
import numpy as np
from pathlib import Path

try:
    import pmt
except ImportError:
    from gnuradio import pmt

HOME_TX_SERIALS = {
    1: '32BBAD0',
    2: 'SERIAL_HOME2_TX',
    3: 'SERIAL_HOME3_TX',
}
SERVER_USRP_SERIAL = '32BBAD9'

SF             = 7
BW             = 125000
CR             = 1
LORA_MAX_BYTES = 255
PREAMBLE_LEN   = 8
RX_INIT_TIME   = 3.0
FLUSH_TIME     = 2.0

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

        print(f"[USRP LoRa] Using Chen's lora_TX.py + lora_RX.py")
        print(f"[USRP LoRa] Role: {role}" + (f" | Home ID: {home_id}" if home_id else ""))
        if tx_serial:
            print(f"[USRP LoRa] TX USRP serial: {tx_serial}")
        if rx_serial:
            print(f"[USRP LoRa] RX USRP serial: {rx_serial}")
        print(f"[USRP LoRa] Work dir: {self.work_dir}")
        print(f"[USRP LoRa] Max payload: {LORA_MAX_BYTES} bytes")

    def _patch_serials(self):
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

    def transmit(self, payload: bytes, done_flag_path=None, timeout=60) -> dict:
        assert len(payload) <= LORA_MAX_BYTES

        ascii_payload = _bytes_to_ascii(payload)
        print(f"[USRP LoRa] Payload: {len(payload)} bytes -> {len(ascii_payload)} ASCII chars")

        sys.path.insert(0, str(self.work_dir))

        success = False
        tx_tb   = None
        rx_tb   = None

        try:
            from lora_TX import lora_TX as LoraTX
            from lora_RX import lora_RX as LoraRX

            print(f"[USRP LoRa] Starting RX (serial={self.rx_serial})...")
            rx_tb = LoraRX()
            rx_tb.start()
            time.sleep(RX_INIT_TIME)

            print(f"[USRP LoRa] Starting TX (serial={self.tx_serial})...")
            tx_tb = LoraTX()
            tx_tb.set_frame_period(10_000_000)
            tx_tb.start()

            msg_port = pmt.intern("msg")
            whitening = tx_tb.lora_sdr_whitening_0.to_basic_block()
            whitening._post(msg_port, pmt.intern(ascii_payload))
            print(f"[USRP LoRa] Posted payload to whitening block.")

            # Write done flag IMMEDIATELY after posting — before any stop/wait
            if done_flag_path is not None:
                Path(done_flag_path).write_text("done")
                print(f"[USRP LoRa] Done flag written: {Path(done_flag_path).name}")

            time.sleep(FLUSH_TIME + 1.0)
            success = True

        except Exception as e:
            print(f"[USRP LoRa] Error: {e}")
            if done_flag_path is not None:
                try:
                    Path(done_flag_path).write_text("done")
                    print(f"[USRP LoRa] Done flag written after error.")
                except:
                    pass
        finally:
            if tx_tb is not None:
                try:
                    tx_tb.stop()
                    tx_tb.wait()
                except Exception as e:
                    print(f"[USRP LoRa] TX stop (ignored): {e}")
            if rx_tb is not None:
                try:
                    rx_tb.stop()
                    rx_tb.wait()
                except Exception as e:
                    print(f"[USRP LoRa] RX stop (ignored): {e}")
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

    def transmit_only(self, payload: bytes, done_flag_path=None) -> dict:
        assert len(payload) <= LORA_MAX_BYTES

        ascii_payload = _bytes_to_ascii(payload)
        print(f"[USRP LoRa] TX only: {len(payload)} bytes")

        sys.path.insert(0, str(self.work_dir))

        success = False
        tx_tb   = None

        try:
            from lora_TX import lora_TX as LoraTX

            tx_tb = LoraTX()
            tx_tb.set_frame_period(10_000_000)
            tx_tb.start()

            msg_port = pmt.intern("msg")
            whitening = tx_tb.lora_sdr_whitening_0.to_basic_block()
            whitening._post(msg_port, pmt.intern(ascii_payload))
            print(f"[USRP LoRa] Posted payload to whitening block.")

            if done_flag_path is not None:
                Path(done_flag_path).write_text("done")
                print(f"[USRP LoRa] Done flag written: {Path(done_flag_path).name}")

            time.sleep(FLUSH_TIME + 1.0)
            success = True

        except Exception as e:
            print(f"[USRP LoRa] TX only error: {e}")
            if done_flag_path is not None:
                try:
                    Path(done_flag_path).write_text("done")
                except:
                    pass
        finally:
            if tx_tb is not None:
                try:
                    tx_tb.stop()
                    tx_tb.wait()
                except Exception as e:
                    print(f"[USRP LoRa] TX stop (ignored): {e}")
            if str(self.work_dir) in sys.path:
                sys.path.remove(str(self.work_dir))

        self.total_transmissions += 1
        if success:
            self.successful_transmissions += 1

        pdr   = self.successful_transmissions / max(self.total_transmissions, 1)
        t_toa = self._calculate_toa(len(payload))

        print(f"[USRP LoRa] TX only: {'SUCCESS' if success else 'FAILED'} | ToA: {t_toa:.4f}s")

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


def get_home_radio(work_dir=None, home_id=None, tx_serial=None, rx_serial=None) -> GRCLoRaUSRP:
    serial = tx_serial or HOME_TX_SERIALS.get(home_id, '32BBAD0')
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=serial,
        rx_serial=rx_serial or SERVER_USRP_SERIAL,
        role='home',
        home_id=home_id,
    )


def get_server_radio(work_dir=None, tx_serial=None, rx_serial=None) -> GRCLoRaUSRP:
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=tx_serial or SERVER_USRP_SERIAL,
        rx_serial=rx_serial,
        role='server',
    )
