#!/usr/bin/env python3
"""
GNU Radio LoRa via EPFL USRP Flowgraphs - hardwaretest04
==========================================================
Uses the REAL UNTOUCHED lora_TX.grc and lora_RX.grc from gr-lora_sdr (EPFL).
Zero modifications to the flowgraphs.

Only thing we do (the "envelope", not the "mailbox"):
  1. Write Hegazy compressed payload as ASCII to the input .txt file
     (comma-separated, same format as example_tx_source.txt)
  2. Compile the installed .grc with grcc
  3. Patch the file path to point to our tx_payload.txt
  4. Patch the USRP device args to use our serial numbers
  5. Run the compiled flowgraph
  6. Capture stdout for received messages (CRC verif prints them)

Hardware: Two USRP B200 radios connected via USB
  - TX USRP: sends LoRa signal over the air via antenna
  - RX USRP: receives LoRa signal from antenna

File: lora/gr_lora_usrp.py
"""

import os
import sys
import time
import subprocess
import re
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Paths to the REAL installed EPFL flowgraphs — NOT modified
# ─────────────────────────────────────────────────────────────────
INSTALLED_TX_GRC = Path('/usr/local/share/gnuradio/examples/lora_sdr/lora_TX.grc')
INSTALLED_RX_GRC = Path('/usr/local/share/gnuradio/examples/lora_sdr/lora_RX.grc')

# Fallback: combined TX+RX USRP flowgraph
INSTALLED_TXRX_GRC = Path('/usr/local/share/gnuradio/examples/lora_sdr/tx_rx_usrp.grc')

# LoRa parameters — must match the installed flowgraph
SF              = 7
BW              = 125000
CR              = 2       # as set in the installed flowgraph
LORA_MAX_BYTES  = 255
PREAMBLE_LEN    = 8
CENTER_FREQ     = 868100000   # 868.1 MHz
SAMP_RATE       = 500000

# USRP B200 serial numbers (from uhd_find_devices)
# These are YOUR actual devices from the lab
DEFAULT_TX_SERIAL = '32BBAD0'   # Home node transmitter
DEFAULT_RX_SERIAL = '32BBA90'   # Server receiver

# ASCII charset for payload encoding (same as hardwaretest03)
ASCII_CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def _bytes_to_ascii(payload: bytes) -> str:
    """Map binary bytes to printable ASCII chars for the whitening block."""
    return ''.join(ASCII_CHARSET[b % len(ASCII_CHARSET)] for b in payload)


class GRCLoRaUSRP:
    """
    Runs the REAL EPFL LoRa flowgraphs with actual USRP B200 hardware.
    TX and RX are separate flowgraphs on separate USRPs.
    We only write the input file and read the output. Nothing else.
    """

    def __init__(self, work_dir=None, tx_serial=None, rx_serial=None,
                 role='home'):
        if work_dir is None:
            self.work_dir = Path(__file__).parent
        else:
            self.work_dir = Path(work_dir)

        self.role = role  # 'home' or 'server'
        self.tx_serial = tx_serial or DEFAULT_TX_SERIAL
        self.rx_serial = rx_serial or DEFAULT_RX_SERIAL

        # Input file — the TX flowgraph's File Source reads from this
        self.tx_input_file = self.work_dir / 'tx_payload.txt'

        # Compiled Python from the REAL .grc files
        self.compiled_tx_py = self.work_dir / 'lora_TX.py'
        self.compiled_rx_py = self.work_dir / 'lora_RX.py'

        self.total_transmissions = 0
        self.successful_transmissions = 0
        self._tx_compiled = False
        self._rx_compiled = False

        # Check installed flowgraphs exist
        for grc, name in [(INSTALLED_TX_GRC, 'TX'), (INSTALLED_RX_GRC, 'RX')]:
            if not grc.exists():
                print(f"[USRP LoRa] WARNING: {name} flowgraph not found at {grc}")
            else:
                print(f"[USRP LoRa] Using EPFL {name} flowgraph: {grc}")

        print(f"[USRP LoRa] Role: {role}")
        print(f"[USRP LoRa] TX USRP serial: {self.tx_serial}")
        print(f"[USRP LoRa] RX USRP serial: {self.rx_serial}")
        print(f"[USRP LoRa] Frequency: {CENTER_FREQ/1e6:.1f} MHz")
        print(f"[USRP LoRa] Work dir: {self.work_dir}")
        print(f"[USRP LoRa] Max payload: {LORA_MAX_BYTES} bytes")

    def _compile_grc(self, grc_path, compiled_py, force=False):
        """Compile an installed .grc to Python using grcc."""
        if not force and compiled_py.exists():
            return True

        if not grc_path.exists():
            print(f"[USRP LoRa] ERROR: {grc_path} not found")
            return False

        print(f"[USRP LoRa] Compiling {grc_path.name} with grcc...")
        try:
            result = subprocess.run(
                ['grcc', '-o', str(self.work_dir), str(grc_path)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                print(f"[USRP LoRa] grcc error: {result.stderr[:500]}")
                return False

            if compiled_py.exists():
                print(f"[USRP LoRa] Compiled -> {compiled_py.name}")
                return True
            else:
                print(f"[USRP LoRa] grcc ran but {compiled_py.name} not found")
                return False

        except FileNotFoundError:
            print("[USRP LoRa] 'grcc' not found. Install gnuradio.")
            return False
        except subprocess.TimeoutExpired:
            print("[USRP LoRa] grcc timed out")
            return False

    def compile_tx(self, force=False):
        if self._tx_compiled and not force:
            return True
        ok = self._compile_grc(INSTALLED_TX_GRC, self.compiled_tx_py, force)
        if ok:
            self._tx_compiled = True
            self._patch_tx()
        return ok

    def compile_rx(self, force=False):
        if self._rx_compiled and not force:
            return True
        ok = self._compile_grc(INSTALLED_RX_GRC, self.compiled_rx_py, force)
        if ok:
            self._rx_compiled = True
            self._patch_rx()
        return ok

    def _patch_tx(self):
        """
        Patch the compiled TX flowgraph:
        1. Point File Source to our tx_payload.txt
        2. Set USRP device args to our TX serial
        """
        if not self.compiled_tx_py.exists():
            return

        with open(self.compiled_tx_py, 'r') as f:
            code = f.read()

        modified = False

        # Patch input file path
        if 'example_tx_source.txt' in code or 'jtappare' in code:
            code = re.sub(
                r"['\"][^'\"]*example_tx_source\.txt['\"]",
                f"'{str(self.tx_input_file)}'",
                code
            )
            modified = True

        # Patch USRP device args — replace IP addresses with USB serial
        # The EPFL flowgraph uses addr=192.168.10.3 for TX
        code = re.sub(
            r'"addr=192\.168\.\d+\.\d+"',
            f'"serial={self.tx_serial}"',
            code
        )
        # Also handle type=b200 format
        if 'serial=' not in code and 'uhd.usrp_sink' in code:
            code = re.sub(
                r'(uhd\.usrp_sink\([^)]*device_addr=)["\'][^"\']*["\']',
                f'\\1"serial={self.tx_serial}"',
                code
            )
            modified = True

        if modified or 'serial=' in code:
            with open(self.compiled_tx_py, 'w') as f:
                f.write(code)
            print(f"[USRP LoRa] TX patched: file={self.tx_input_file.name}, "
                  f"serial={self.tx_serial}")

    def _patch_rx(self):
        """
        Patch the compiled RX flowgraph:
        Set USRP device args to our RX serial
        """
        if not self.compiled_rx_py.exists():
            return

        with open(self.compiled_rx_py, 'r') as f:
            code = f.read()

        # Patch USRP device args — replace IP addresses with USB serial
        # The EPFL flowgraph uses addr=192.168.10.6 for RX
        code = re.sub(
            r'"addr=192\.168\.\d+\.\d+"',
            f'"serial={self.rx_serial}"',
            code
        )
        if 'serial=' not in code and 'uhd.usrp_source' in code:
            code = re.sub(
                r'(uhd\.usrp_source\([^)]*device_addr=)["\'][^"\']*["\']',
                f'\\1"serial={self.rx_serial}"',
                code
            )

        with open(self.compiled_rx_py, 'w') as f:
            f.write(code)
        print(f"[USRP LoRa] RX patched: serial={self.rx_serial}")

    def _write_payload(self, payload: bytes):
        """Write payload as ASCII chars, comma-separated (same as ht03)."""
        ascii_payload = _bytes_to_ascii(payload)
        with open(self.tx_input_file, 'w') as f:
            f.write(ascii_payload + ',')
        print(f"[USRP LoRa] Wrote payload: {len(payload)} bytes -> "
              f"{len(ascii_payload)} ASCII chars to {self.tx_input_file.name}")

    def transmit(self, payload: bytes, timeout=60) -> dict:
        """
        Transmit payload over the air using real USRP B200.

        1. Write payload to input file
        2. Compile TX flowgraph if needed
        3. Start RX flowgraph (listening)
        4. Start TX flowgraph (sends over the air)
        5. Wait for RX to receive and decode
        6. Parse stdout for CRC result
        """
        assert len(payload) <= LORA_MAX_BYTES, \
            f"Payload {len(payload)} bytes exceeds LoRa max {LORA_MAX_BYTES}"

        # Write payload
        self._write_payload(payload)

        # Compile both flowgraphs
        if not self.compile_tx():
            self.total_transmissions += 1
            return self._make_result(success=False, payload=payload)
        if not self.compile_rx():
            self.total_transmissions += 1
            return self._make_result(success=False, payload=payload)

        print(f"[USRP LoRa] TX over the air via USRP B200...")
        rx_msg = None
        crc_ok = False

        try:
            # Start RX first (it needs to be listening before TX sends)
            print(f"[USRP LoRa] Starting RX (serial={self.rx_serial})...")
            rx_proc = subprocess.Popen(
                [sys.executable, str(self.compiled_rx_py)],
                cwd=str(self.work_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Give RX time to initialize USRP and start listening
            time.sleep(5)

            # Start TX (sends the packet over the air)
            print(f"[USRP LoRa] Starting TX (serial={self.tx_serial})...")
            tx_proc = subprocess.Popen(
                [sys.executable, str(self.compiled_tx_py)],
                cwd=str(self.work_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for TX to finish transmitting
            # For SF=7, BW=125kHz, 242 bytes: ToA ~0.6s
            # Add time for USRP init and processing
            time.sleep(15)

            # Stop TX
            try:
                tx_stdout, tx_stderr = tx_proc.communicate(
                    input='\n', timeout=10
                )
                if tx_stdout:
                    for line in tx_stdout.strip().split('\n'):
                        if line.strip():
                            print(f"  [TX] {line.strip()}")
            except subprocess.TimeoutExpired:
                tx_proc.kill()
                tx_proc.wait()
                print("[USRP LoRa] TX timed out, killed")

            # Give RX a bit more time to process the received signal
            time.sleep(5)

            # Stop RX and collect output
            try:
                rx_stdout, rx_stderr = rx_proc.communicate(
                    input='\n', timeout=10
                )
            except subprocess.TimeoutExpired:
                rx_proc.kill()
                rx_stdout, rx_stderr = rx_proc.communicate()
                print("[USRP LoRa] RX timed out, killed")

            # Parse RX stdout for received message
            if rx_stdout:
                for line in rx_stdout.strip().split('\n'):
                    if line.strip():
                        print(f"  [RX] {line.strip()}")
                    if 'rx msg:' in line:
                        rx_msg = line.split('rx msg:')[1].strip()
                    if 'CRC valid' in line:
                        crc_ok = True

            # Also check RX stderr (some blocks print to stderr)
            if rx_stderr:
                for line in rx_stderr.strip().split('\n'):
                    if 'rx msg:' in line:
                        rx_msg = line.split('rx msg:')[1].strip()
                    if 'CRC valid' in line:
                        crc_ok = True

        except Exception as e:
            print(f"[USRP LoRa] Error: {e}")
            # Clean up any running processes
            for proc in [tx_proc, rx_proc]:
                try:
                    proc.kill()
                    proc.wait()
                except:
                    pass
            self.total_transmissions += 1
            return self._make_result(success=False, payload=payload)

        self.total_transmissions += 1
        success = rx_msg is not None and crc_ok

        if success:
            self.successful_transmissions += 1

        pdr = self.successful_transmissions / max(self.total_transmissions, 1)
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

    def receive(self, timeout=30) -> dict:
        """
        Listen for incoming LoRa packet using RX USRP.
        Used by server to receive from homes.
        """
        if not self.compile_rx():
            return self._make_result(success=False, payload=b'')

        print(f"[USRP LoRa] Listening on RX (serial={self.rx_serial})...")
        rx_msg = None
        crc_ok = False

        try:
            rx_proc = subprocess.Popen(
                [sys.executable, str(self.compiled_rx_py)],
                cwd=str(self.work_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for a packet to arrive
            time.sleep(timeout)

            # Stop RX
            try:
                rx_stdout, rx_stderr = rx_proc.communicate(
                    input='\n', timeout=10
                )
            except subprocess.TimeoutExpired:
                rx_proc.kill()
                rx_stdout, rx_stderr = rx_proc.communicate()

            # Parse output
            for output in [rx_stdout, rx_stderr]:
                if output:
                    for line in output.strip().split('\n'):
                        if 'rx msg:' in line:
                            rx_msg = line.split('rx msg:')[1].strip()
                        if 'CRC valid' in line:
                            crc_ok = True

        except Exception as e:
            print(f"[USRP LoRa] RX error: {e}")
            return self._make_result(success=False, payload=b'')

        success = rx_msg is not None and crc_ok
        if success:
            self.successful_transmissions += 1
            self.total_transmissions += 1

        print(f"[USRP LoRa] RX: {'GOT PACKET' if success else 'NO PACKET'} | "
              f"CRC: {'OK' if crc_ok else 'FAIL'}")

        return {
            'success':     success,
            'crc_ok':      crc_ok,
            'rx_msg':      rx_msg,
            'pdr':         self.successful_transmissions / max(self.total_transmissions, 1),
        }

    def _calculate_toa(self, payload_length):
        """LoRa time on air (Eq 29-30)."""
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
            'tx_bytes':    len(payload) if payload else 0,
            'rx_msg':      None,
            't_toa':       self._calculate_toa(len(payload)) if payload else 0,
            'pdr':         self.successful_transmissions / max(self.total_transmissions, 1),
            'packet_size': len(payload) if payload else 0,
        }


def get_home_radio(work_dir=None, tx_serial=None, rx_serial=None) -> GRCLoRaUSRP:
    """Home node: TX on one USRP, RX on the other."""
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=tx_serial or DEFAULT_TX_SERIAL,
        rx_serial=rx_serial or DEFAULT_RX_SERIAL,
        role='home'
    )


def get_server_radio(work_dir=None, tx_serial=None, rx_serial=None) -> GRCLoRaUSRP:
    """Server: TX on one USRP, RX on the other (swapped from home)."""
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=tx_serial or DEFAULT_RX_SERIAL,  # server TX = home's RX
        rx_serial=rx_serial or DEFAULT_TX_SERIAL,   # server RX = home's TX
        role='server'
    )
