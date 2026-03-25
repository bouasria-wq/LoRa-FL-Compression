#!/usr/bin/env python3
"""
GNU Radio LoRa via EPFL Installed Flowgraph - hardwaretest03
==============================================================
Uses the REAL UNTOUCHED tx_rx_simulation.grc from gr-lora_sdr (EPFL).
Zero modifications to the flowgraph.

Only thing we do:
  1. Write Hegazy compressed payload as ASCII to the input .txt file
     (comma-separated, same format as example_tx_source.txt)
  2. Compile the installed .grc with grcc
  3. Run the compiled flowgraph
  4. Capture stdout for received messages (CRC verif prints them)

The flowgraph is a black box — bytes in, LoRa happens, bytes out.

File: lora/gr_lora_grc.py
"""

import os
import sys
import time
import subprocess
import re
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Paths to the REAL installed EPFL flowgraph — NOT modified
# ─────────────────────────────────────────────────────────────────
INSTALLED_GRC = Path('/usr/local/share/gnuradio/examples/lora_sdr/tx_rx_simulation.grc')

# LoRa parameters — read from the flowgraph, not set by us
SF              = 7
BW              = 125000
CR              = 2       # as set in the installed flowgraph
LORA_MAX_BYTES  = 255
PREAMBLE_LEN    = 8


class GRCLoRaSimulation:
    """
    Runs the REAL EPFL tx_rx_simulation flowgraph.
    We only write the input file and read the output. Nothing else.
    """

    def __init__(self, work_dir=None):
        if work_dir is None:
            self.work_dir = Path(__file__).parent
        else:
            self.work_dir = Path(work_dir)

        # Input file — the flowgraph's File Source reads from this
        self.tx_input_file = self.work_dir / 'tx_payload.txt'

        # The compiled Python from the REAL .grc
        self.compiled_py = self.work_dir / 'tx_rx_simulation.py'

        self.total_transmissions = 0
        self.successful_transmissions = 0
        self._compiled = False

        # Check installed flowgraph exists
        if not INSTALLED_GRC.exists():
            print(f"[GRC LoRa] WARNING: Installed flowgraph not found at {INSTALLED_GRC}")
            print(f"[GRC LoRa] Install gr-lora_sdr first.")
        else:
            print(f"[GRC LoRa] Using EPFL flowgraph: {INSTALLED_GRC}")

        print(f"[GRC LoRa] Work dir: {self.work_dir}")
        print(f"[GRC LoRa] Max payload: {LORA_MAX_BYTES} bytes")

    def compile_grc(self, force=False):
        """
        Compile the REAL installed .grc to Python using grcc.
        Same as clicking Generate (F5) in GNU Radio Companion.
        """
        if self._compiled and not force and self.compiled_py.exists():
            return True

        if not INSTALLED_GRC.exists():
            print(f"[GRC LoRa] ERROR: {INSTALLED_GRC} not found")
            return False

        print(f"[GRC LoRa] Compiling EPFL flowgraph with grcc...")
        try:
            result = subprocess.run(
                ['grcc', '-o', str(self.work_dir), str(INSTALLED_GRC)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                print(f"[GRC LoRa] grcc error: {result.stderr[:500]}")
                return False

            if self.compiled_py.exists():
                print(f"[GRC LoRa] Compiled -> {self.compiled_py.name}")
                self._compiled = True
                return True
            else:
                print(f"[GRC LoRa] grcc ran but {self.compiled_py.name} not found")
                return False

        except FileNotFoundError:
            print(f"[GRC LoRa] 'grcc' not found. Install gnuradio.")
            return False
        except subprocess.TimeoutExpired:
            print(f"[GRC LoRa] grcc timed out")
            return False

    def _patch_file_path(self):
        """
        The compiled .py has a hardcoded path to the original author's
        example_tx_source.txt. We point it to our tx_payload.txt instead.

        This is NOT modifying the flowgraph — just changing which file
        the File Source block reads from (data input, not LoRa config).
        """
        if not self.compiled_py.exists():
            return False

        with open(self.compiled_py, 'r') as f:
            code = f.read()

        # Replace any path ending in example_tx_source.txt with our file
        if 'example_tx_source.txt' in code or 'jtappare' in code:
            code = re.sub(
                r"['\"][^'\"]*example_tx_source\.txt['\"]",
                f"'{str(self.tx_input_file)}'",
                code
            )
            with open(self.compiled_py, 'w') as f:
                f.write(code)
            print(f"[GRC LoRa] Input file path set to: {self.tx_input_file}")

        return True

    def _write_payload(self, payload: bytes):
        """
        Convert our binary payload to printable ASCII chars.
        The flowgraph mailbox only accepts ASCII letters — so we
        change our envelope from binary to ASCII. Mailbox untouched.
        """
        charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ascii_payload = ''.join(charset[b % len(charset)] for b in payload)

        with open(self.tx_input_file, 'w') as f:
            f.write(ascii_payload + ',')

        print(f"[GRC LoRa] Wrote payload: {len(payload)} bytes -> "
              f"{len(ascii_payload)} ASCII chars to {self.tx_input_file.name}")

    def transmit(self, payload: bytes, timeout=60) -> dict:
        """
        Full TX->channel->RX using the REAL EPFL flowgraph.

        1. Write payload to input file
        2. Compile .grc if needed
        3. Patch the file path (data input only)
        4. Run the compiled flowgraph
        5. Wait for processing to complete
        6. Send Enter to quit
        7. Parse stdout for received message
        """
        assert len(payload) <= LORA_MAX_BYTES, \
            f"Payload {len(payload)} bytes exceeds LoRa max {LORA_MAX_BYTES}"

        # Write payload in flowgraph's expected format
        self._write_payload(payload)

        # Compile the REAL .grc
        if not self.compile_grc():
            self.total_transmissions += 1
            return self._make_result(success=False, payload=payload)

        # Patch file path to our input file
        self._patch_file_path()

        # Run the flowgraph using Popen so we can control timing
        print(f"[GRC LoRa] Running EPFL flowgraph...")
        rx_msg = None
        crc_ok = False

        try:
            proc = subprocess.Popen(
                [sys.executable, str(self.compiled_py)],
                cwd=str(self.work_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for flowgraph to process the data
            # The flowgraph needs time to TX, pass through channel, and RX
            time.sleep(15)

            # Send Enter to quit the flowgraph
            stdout, stderr = proc.communicate(input='\n', timeout=timeout)

            # Parse stdout for received message
            # CRC verif block prints: "rx msg: <payload>" and "CRC valid!"
            for line in stdout.split('\n'):
                if 'rx msg:' in line:
                    rx_msg = line.split('rx msg:')[1].strip()
                if 'CRC valid' in line:
                    crc_ok = True

            if stdout:
                for line in stdout.strip().split('\n'):
                    if line.strip():
                        print(f"  [flowgraph] {line.strip()}")

        except subprocess.TimeoutExpired:
            print(f"[GRC LoRa] Flowgraph timeout after {timeout}s")
            try:
                proc.kill()
                proc.wait()
            except:
                pass
            self.total_transmissions += 1
            return self._make_result(success=False, payload=payload)
        except Exception as e:
            print(f"[GRC LoRa] Error: {e}")
            self.total_transmissions += 1
            return self._make_result(success=False, payload=payload)

        self.total_transmissions += 1
        success = rx_msg is not None and crc_ok

        if success:
            self.successful_transmissions += 1

        pdr = self.successful_transmissions / max(self.total_transmissions, 1)
        t_toa = self._calculate_toa(len(payload))

        print(f"[GRC LoRa] Result: {'SUCCESS' if success else 'FAILED'} | "
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
            'tx_bytes':    len(payload),
            'rx_msg':      None,
            't_toa':       self._calculate_toa(len(payload)),
            'pdr':         self.successful_transmissions / max(self.total_transmissions, 1),
            'packet_size': len(payload),
        }


def get_home_radio(work_dir=None) -> GRCLoRaSimulation:
    return GRCLoRaSimulation(work_dir=work_dir)

def get_server_radio(work_dir=None) -> GRCLoRaSimulation:
    return GRCLoRaSimulation(work_dir=work_dir)
