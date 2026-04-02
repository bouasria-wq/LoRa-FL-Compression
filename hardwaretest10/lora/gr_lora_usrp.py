#!/usr/bin/env python3
"""
GNU Radio LoRa USRP Bridge - hardwaretest10
=============================================
Uses:
  - lora_TX_usrp.py: our TX flowgraph (file source -> LoRa PHY -> USRP)
  - lora_RX_usrp.py: Chen's RX flowgraph (USRP -> LoRa PHY -> print)

No patching. No grcc. Clean Python files called directly.
Flag-based handshake: home TX done -> server TX -> home done.

File: lora/gr_lora_usrp.py
"""

import sys
import time
import subprocess
import numpy as np
from pathlib import Path

# LoRa parameters
SF             = 7
BW             = 125000
CR             = 1
LORA_MAX_BYTES = 255
PREAMBLE_LEN   = 8

# USRP B200 serial numbers
DEFAULT_TX_SERIAL = '32BBAD0'
DEFAULT_RX_SERIAL = '32BBAD9'

# Timing
RX_INIT_TIME    = 8
TX_PROCESS_TIME = 15
RX_EXTRA_TIME   = 5

ASCII_CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def _bytes_to_ascii(payload: bytes) -> str:
    return ''.join(ASCII_CHARSET[b % len(ASCII_CHARSET)] for b in payload)


class GRCLoRaUSRP:

    def __init__(self, work_dir=None, tx_serial=None, rx_serial=None, role='home'):
        self.work_dir  = Path(work_dir) if work_dir else Path(__file__).parent
        self.role      = role
        self.tx_serial = tx_serial or DEFAULT_TX_SERIAL
        self.rx_serial = rx_serial or DEFAULT_RX_SERIAL

        self.tx_input_file = self.work_dir / 'tx_payload.txt'
        self.tx_script     = Path(__file__).parent / 'lora_TX_usrp.py'
        self.rx_script     = Path(__file__).parent / 'lora_RX_usrp.py'

        self.total_transmissions      = 0
        self.successful_transmissions = 0

        print(f"[USRP LoRa] TX flowgraph: {self.tx_script.name}")
        print(f"[USRP LoRa] RX flowgraph: {self.rx_script.name}")
        print(f"[USRP LoRa] Role: {role}")
        print(f"[USRP LoRa] TX USRP serial: {self.tx_serial}")
        print(f"[USRP LoRa] RX USRP serial: {self.rx_serial}")
        print(f"[USRP LoRa] Work dir: {self.work_dir}")
        print(f"[USRP LoRa] Max payload: {LORA_MAX_BYTES} bytes")

    def _write_payload(self, payload: bytes):
        ascii_payload = _bytes_to_ascii(payload)
        with open(self.tx_input_file, 'w') as f:
            f.write(ascii_payload + ',')
        print(f"[USRP LoRa] Wrote: {len(payload)} bytes -> {len(ascii_payload)} ASCII chars")

    def transmit(self, payload: bytes, timeout=60) -> dict:
        assert len(payload) <= LORA_MAX_BYTES

        self._write_payload(payload)

        if not self.tx_script.exists():
            print(f"[USRP LoRa] ERROR: {self.tx_script} not found")
            self.total_transmissions += 1
            return self._make_result(False, payload)

        if not self.rx_script.exists():
            print(f"[USRP LoRa] ERROR: {self.rx_script} not found")
            self.total_transmissions += 1
            return self._make_result(False, payload)

        print(f"[USRP LoRa] TX over the air (tx={self.tx_serial} rx={self.rx_serial})...")

        rx_msg = None
        crc_ok = False
        rx_proc = None
        tx_proc = None

        try:
            # Start RX FIRST
            print(f"[USRP LoRa] Starting RX (serial={self.rx_serial})...")
            rx_proc = subprocess.Popen(
                [sys.executable, str(self.rx_script),
                 '--rx_serial', self.rx_serial],
                cwd=str(self.work_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            print(f"[USRP LoRa] Waiting {RX_INIT_TIME}s for RX to initialize...")
            time.sleep(RX_INIT_TIME)

            # NOW start TX
            print(f"[USRP LoRa] Starting TX (serial={self.tx_serial})...")
            tx_proc = subprocess.Popen(
                [sys.executable, str(self.tx_script),
                 '--input', str(self.tx_input_file),
                 '--tx_serial', self.tx_serial],
                cwd=str(self.work_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            time.sleep(TX_PROCESS_TIME)

            # Stop TX
            try:
                tx_stdout, tx_stderr = tx_proc.communicate(input='\n', timeout=10)
                if tx_stdout:
                    for line in tx_stdout.strip().split('\n'):
                        if line.strip():
                            print(f"  [TX] {line.strip()}")
                if tx_stderr:
                    for line in tx_stderr.strip().split('\n'):
                        if line.strip() and '[INFO]' not in line and '[WARNING]' not in line:
                            print(f"  [TX ERR] {line.strip()}")
            except subprocess.TimeoutExpired:
                tx_proc.kill()
                tx_proc.wait()

            time.sleep(RX_EXTRA_TIME)

            # Stop RX
            try:
                rx_stdout, rx_stderr = rx_proc.communicate(input='\n', timeout=10)
            except subprocess.TimeoutExpired:
                rx_proc.kill()
                rx_stdout, rx_stderr = rx_proc.communicate()

            for output in [rx_stdout, rx_stderr]:
                if not output:
                    continue
                for line in output.strip().split('\n'):
                    if line.strip():
                        print(f"  [RX] {line.strip()}")
                    if 'rx msg:' in line:
                        rx_msg = line.split('rx msg:')[1].strip()
                    if 'CRC valid' in line:
                        crc_ok = True

        except Exception as e:
            print(f"[USRP LoRa] Error: {e}")
            for proc in [tx_proc, rx_proc]:
                if proc:
                    try:
                        proc.kill()
                        proc.wait()
                    except:
                        pass
            self.total_transmissions += 1
            return self._make_result(False, payload)

        self.total_transmissions += 1
        success = rx_msg is not None and crc_ok
        if success:
            self.successful_transmissions += 1

        pdr   = self.successful_transmissions / max(self.total_transmissions, 1)
        t_toa = self._calculate_toa(len(payload))

        print(f"[USRP LoRa] Result: {'SUCCESS' if success else 'FAILED'} | "
              f"CRC: {'OK' if crc_ok else 'FAIL'} | "
              f"ToA: {t_toa:.4f}s | PDR: {pdr*100:.1f}%")

        return {
            'success':     success,
            'crc_ok':      crc_ok,
            'tx_bytes':    len(payload),
            'rx_msg':      rx_msg,
            't_toa':       t_toa,
            'pdr':         pdr,
            'packet_size': len(payload),
        }

    def _calculate_toa(self, payload_length):
        n_preamble = PREAMBLE_LEN
        numerator = 8 * payload_length - 4 * SF + 28 + 16
        denominator = 4 * (SF - 2)
        payload_symbols = max(np.ceil(numerator / denominator) * (CR + 4), 0)
        n_symbols = n_preamble + payload_symbols
        return (n_symbols * (2 ** SF)) / BW

    def _make_result(self, success, payload):
        return {
            'success':     success,
            'crc_ok':      False,
            'tx_bytes':    len(payload),
            'rx_msg':      None,
            't_toa':       self._calculate_toa(len(payload)),
            'pdr':         self.successful_transmissions / max(self.total_transmissions, 1),
            'packet_size': len(payload),
        }


def get_home_radio(work_dir=None, tx_serial=None, rx_serial=None) -> GRCLoRaUSRP:
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=tx_serial or DEFAULT_TX_SERIAL,
        rx_serial=rx_serial or DEFAULT_RX_SERIAL,
        role='home'
    )


def get_server_radio(work_dir=None, tx_serial=None, rx_serial=None) -> GRCLoRaUSRP:
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=tx_serial or DEFAULT_RX_SERIAL,
        rx_serial=rx_serial or DEFAULT_TX_SERIAL,
        role='server'
    )
