#!/usr/bin/env python3
"""
GNU Radio LoRa via EPFL USRP Flowgraphs - hardwaretest05
==========================================================
Uses the REAL UNTOUCHED lora_TX.grc and lora_RX.grc from gr-lora_sdr.

Hardware: 4x USRP B200
  - 3x Home TX USRPs (one per home, each with own serial)
  - 1x Server RX USRP (receives from all 3 homes)

TDMA Scheduling:
  Each home is assigned a time slot to avoid RF collisions.
  Home 1 transmits in slot 0, Home 2 in slot 1, Home 3 in slot 2.
  Server RX stays open for the entire window.

  Slot layout per day:
    [0s..30s]   Home 1 TX
    [30s..60s]  Home 2 TX
    [60s..90s]  Home 3 TX
    [90s..120s] Server aggregates + broadcasts global model
    [120s..]    All homes receive global model

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
# EPFL flowgraphs — NOT modified
# ─────────────────────────────────────────────────────────────────
INSTALLED_TX_GRC = Path('/usr/local/share/gnuradio/examples/lora_sdr/lora_TX.grc')
INSTALLED_RX_GRC = Path('/usr/local/share/gnuradio/examples/lora_sdr/lora_RX.grc')

# LoRa parameters — match the installed flowgraph
SF              = 7
BW              = 125000
CR              = 2
LORA_MAX_BYTES  = 255
PREAMBLE_LEN    = 8
CENTER_FREQ     = 868100000
SAMP_RATE       = 500000

# ─────────────────────────────────────────────────────────────────
# TDMA timing for 3 homes
# ─────────────────────────────────────────────────────────────────
N_HOMES             = 3
SLOT_DURATION       = 30    # seconds per home TX slot
USRP_INIT_TIME      = 5    # seconds for USRP to initialize
TX_PROCESS_TIME     = 15   # seconds for flowgraph to process packet
GUARD_TIME          = 5    # gap between slots to avoid overlap
SERVER_LISTEN_TIME  = SLOT_DURATION * N_HOMES + 10  # total RX window

# ASCII charset for payload encoding (same as hardwaretest03/04)
ASCII_CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def _bytes_to_ascii(payload: bytes) -> str:
    """Map binary bytes to printable ASCII chars for the whitening block."""
    return ''.join(ASCII_CHARSET[b % len(ASCII_CHARSET)] for b in payload)


def slot_start_time(home_id: int) -> float:
    """
    TDMA slot start offset for a given home.
    Home 1 -> 0s, Home 2 -> 30s, Home 3 -> 60s
    """
    return (home_id - 1) * SLOT_DURATION


class GRCLoRaUSRP:
    """
    Runs the REAL EPFL LoRa flowgraphs with actual USRP B200 hardware.
    Supports TDMA scheduling for multi-home operation.
    """

    def __init__(self, work_dir=None, tx_serial=None, rx_serial=None,
                 role='home', home_id=None):
        if work_dir is None:
            self.work_dir = Path(__file__).parent
        else:
            self.work_dir = Path(work_dir)

        self.role = role
        self.home_id = home_id
        self.tx_serial = tx_serial
        self.rx_serial = rx_serial

        # Files
        self.tx_input_file = self.work_dir / 'tx_payload.txt'
        self.compiled_tx_py = self.work_dir / 'lora_TX.py'
        self.compiled_rx_py = self.work_dir / 'lora_RX.py'

        # Stats
        self.total_transmissions = 0
        self.successful_transmissions = 0
        self._tx_compiled = False
        self._rx_compiled = False

        # Verify flowgraphs
        for grc, name in [(INSTALLED_TX_GRC, 'TX'), (INSTALLED_RX_GRC, 'RX')]:
            if not grc.exists():
                print(f"[USRP LoRa] WARNING: {name} flowgraph not found at {grc}")
            else:
                print(f"[USRP LoRa] Using EPFL {name}: {grc}")

        print(f"[USRP LoRa] Role: {role}" +
              (f" | Home ID: {home_id}" if home_id else ""))
        if tx_serial:
            print(f"[USRP LoRa] TX serial: {tx_serial}")
        if rx_serial:
            print(f"[USRP LoRa] RX serial: {rx_serial}")
        print(f"[USRP LoRa] Freq: {CENTER_FREQ/1e6:.1f} MHz | "
              f"SF={SF} BW={BW//1000}kHz CR={CR}")
        print(f"[USRP LoRa] TDMA slot: {SLOT_DURATION}s per home | "
              f"{N_HOMES} homes")
        print(f"[USRP LoRa] Max payload: {LORA_MAX_BYTES} bytes")

    # ─── Compile ──────────────────────────────────────────────────

    def _compile_grc(self, grc_path, compiled_py, force=False):
        if not force and compiled_py.exists():
            return True
        if not grc_path.exists():
            print(f"[USRP LoRa] ERROR: {grc_path} not found")
            return False

        print(f"[USRP LoRa] Compiling {grc_path.name}...")
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
            print(f"[USRP LoRa] {compiled_py.name} not found after grcc")
            return False
        except FileNotFoundError:
            print("[USRP LoRa] 'grcc' not found")
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

    # ─── Patch compiled flowgraphs ────────────────────────────────

    def _patch_tx(self):
        """Patch TX: input file path + USRP serial."""
        if not self.compiled_tx_py.exists():
            return
        with open(self.compiled_tx_py, 'r') as f:
            code = f.read()

        # Patch input file
        code = re.sub(
            r"['\"][^'\"]*example_tx_source\.txt['\"]",
            f"'{str(self.tx_input_file)}'",
            code
        )
        # Patch USRP addr -> serial
        if self.tx_serial:
            code = re.sub(
                r'"addr=192\.168\.\d+\.\d+"',
                f'"serial={self.tx_serial}"',
                code
            )
        with open(self.compiled_tx_py, 'w') as f:
            f.write(code)
        print(f"[USRP LoRa] TX patched: serial={self.tx_serial}")

    def _patch_rx(self):
        """Patch RX: USRP serial."""
        if not self.compiled_rx_py.exists():
            return
        with open(self.compiled_rx_py, 'r') as f:
            code = f.read()

        if self.rx_serial:
            code = re.sub(
                r'"addr=192\.168\.\d+\.\d+"',
                f'"serial={self.rx_serial}"',
                code
            )
        with open(self.compiled_rx_py, 'w') as f:
            f.write(code)
        print(f"[USRP LoRa] RX patched: serial={self.rx_serial}")

    # ─── Payload ──────────────────────────────────────────────────

    def _write_payload(self, payload: bytes):
        ascii_payload = _bytes_to_ascii(payload)
        with open(self.tx_input_file, 'w') as f:
            f.write(ascii_payload + ',')
        print(f"[USRP LoRa] Wrote: {len(payload)} bytes -> "
              f"{len(ascii_payload)} ASCII chars")

    # ─── Home TX with TDMA slot ───────────────────────────────────

    def transmit_tdma(self, payload: bytes, day_start_time: float,
                      timeout=60) -> dict:
        """
        Transmit in this home's assigned TDMA slot.

        1. Wait until our slot starts
        2. TX the packet
        3. Return result

        day_start_time: time.time() when the day round began
                        (all homes must agree on this via file flag)
        """
        assert len(payload) <= LORA_MAX_BYTES, \
            f"Payload {len(payload)}B > max {LORA_MAX_BYTES}B"
        assert self.home_id is not None, "home_id required for TDMA"

        # Write payload
        self._write_payload(payload)

        # Compile TX
        if not self.compile_tx():
            self.total_transmissions += 1
            return self._make_result(False, payload)

        # Wait for our TDMA slot
        my_slot = slot_start_time(self.home_id)
        elapsed = time.time() - day_start_time
        wait = my_slot - elapsed
        if wait > 0:
            print(f"[USRP LoRa] Home {self.home_id:02d}: waiting {wait:.1f}s "
                  f"for TDMA slot {self.home_id} "
                  f"(starts at +{my_slot}s)")
            time.sleep(wait)
        else:
            print(f"[USRP LoRa] Home {self.home_id:02d}: TDMA slot "
                  f"{self.home_id} is now")

        # TX the packet
        print(f"[USRP LoRa] Home {self.home_id:02d}: TX over the air "
              f"(serial={self.tx_serial})...")

        rx_msg = None
        crc_ok = False

        try:
            tx_proc = subprocess.Popen(
                [sys.executable, str(self.compiled_tx_py)],
                cwd=str(self.work_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            time.sleep(TX_PROCESS_TIME)

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

        except Exception as e:
            print(f"[USRP LoRa] TX error: {e}")
            self.total_transmissions += 1
            return self._make_result(False, payload)

        # Home TX doesn't know if server received — we report success
        # based on TX completing without error. Server confirms via
        # global model broadcast.
        self.total_transmissions += 1
        self.successful_transmissions += 1
        t_toa = self._calculate_toa(len(payload))
        pdr = self.successful_transmissions / max(self.total_transmissions, 1)

        print(f"[USRP LoRa] Home {self.home_id:02d}: TX complete | "
              f"ToA: {t_toa:.4f}s | PDR: {pdr*100:.1f}%")

        return {
            'success': True,
            'crc_ok': True,  # assumed — server will verify
            'tx_bytes': len(payload),
            'rx_msg': None,
            't_toa': t_toa,
            'pdr': pdr,
            'packet_size': len(payload),
        }

    # ─── Server RX: listen for all homes ──────────────────────────

    def receive_all_homes(self, n_homes=3, timeout=None) -> list:
        """
        Server listens for packets from all homes.
        Keeps RX open for the full TDMA window.
        Returns list of {home_slot, rx_msg, crc_ok} dicts.

        timeout: total listen time (default: all slots + buffer)
        """
        if timeout is None:
            timeout = SERVER_LISTEN_TIME

        if not self.compile_rx():
            return []

        print(f"[USRP LoRa] Server RX: listening for {n_homes} homes "
              f"({timeout}s window, serial={self.rx_serial})...")

        received_packets = []

        try:
            rx_proc = subprocess.Popen(
                [sys.executable, str(self.compiled_rx_py)],
                cwd=str(self.work_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Keep RX open for the full TDMA window
            time.sleep(timeout)

            # Stop RX
            try:
                rx_stdout, rx_stderr = rx_proc.communicate(
                    input='\n', timeout=10
                )
            except subprocess.TimeoutExpired:
                rx_proc.kill()
                rx_stdout, rx_stderr = rx_proc.communicate()

            # Parse ALL received packets from stdout
            # Each packet produces: "rx msg: ..." and "CRC valid!"
            # Multiple packets = multiple occurrences
            current_msg = None
            current_crc = False

            for output in [rx_stdout, rx_stderr]:
                if not output:
                    continue
                for line in output.strip().split('\n'):
                    if line.strip():
                        print(f"  [RX] {line.strip()}")

                    if 'rx msg:' in line:
                        # If we had a previous message, save it
                        if current_msg is not None:
                            received_packets.append({
                                'rx_msg': current_msg,
                                'crc_ok': current_crc,
                                'packet_num': len(received_packets) + 1,
                            })
                        current_msg = line.split('rx msg:')[1].strip()
                        current_crc = False

                    if 'CRC valid' in line and current_msg is not None:
                        current_crc = True

            # Don't forget the last packet
            if current_msg is not None:
                received_packets.append({
                    'rx_msg': current_msg,
                    'crc_ok': current_crc,
                    'packet_num': len(received_packets) + 1,
                })

        except Exception as e:
            print(f"[USRP LoRa] Server RX error: {e}")

        print(f"[USRP LoRa] Server RX: received {len(received_packets)}/{n_homes} "
              f"packets")
        for i, pkt in enumerate(received_packets):
            print(f"  Packet {i+1}: CRC={'OK' if pkt['crc_ok'] else 'FAIL'} | "
                  f"len={len(pkt['rx_msg']) if pkt['rx_msg'] else 0}")

        return received_packets

    # ─── Server TX: broadcast global model ────────────────────────

    def broadcast(self, payload: bytes, timeout=60) -> dict:
        """
        Server broadcasts global model to all homes.
        All homes should be listening when this runs.
        """
        assert len(payload) <= LORA_MAX_BYTES

        self._write_payload(payload)
        if not self.compile_tx():
            return self._make_result(False, payload)

        print(f"[USRP LoRa] Server TX broadcast "
              f"(serial={self.tx_serial})...")

        try:
            tx_proc = subprocess.Popen(
                [sys.executable, str(self.compiled_tx_py)],
                cwd=str(self.work_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            time.sleep(TX_PROCESS_TIME)

            try:
                tx_stdout, _ = tx_proc.communicate(input='\n', timeout=10)
                if tx_stdout:
                    for line in tx_stdout.strip().split('\n'):
                        if line.strip():
                            print(f"  [TX] {line.strip()}")
            except subprocess.TimeoutExpired:
                tx_proc.kill()
                tx_proc.wait()

        except Exception as e:
            print(f"[USRP LoRa] Broadcast error: {e}")
            return self._make_result(False, payload)

        t_toa = self._calculate_toa(len(payload))
        print(f"[USRP LoRa] Broadcast complete | ToA: {t_toa:.4f}s")

        return {
            'success': True, 'crc_ok': True,
            'tx_bytes': len(payload), 'rx_msg': None,
            't_toa': t_toa, 'pdr': 1.0,
            'packet_size': len(payload),
        }

    # ─── Home RX: listen for server broadcast ─────────────────────

    def receive_broadcast(self, timeout=30) -> dict:
        """Home listens for server's global model broadcast."""
        if not self.compile_rx():
            return {'success': False, 'rx_msg': None, 'crc_ok': False}

        print(f"[USRP LoRa] Home {self.home_id:02d}: listening for "
              f"server broadcast...")

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

            time.sleep(timeout)

            try:
                rx_stdout, rx_stderr = rx_proc.communicate(
                    input='\n', timeout=10
                )
            except subprocess.TimeoutExpired:
                rx_proc.kill()
                rx_stdout, rx_stderr = rx_proc.communicate()

            for output in [rx_stdout, rx_stderr]:
                if not output:
                    continue
                for line in output.strip().split('\n'):
                    if 'rx msg:' in line:
                        rx_msg = line.split('rx msg:')[1].strip()
                    if 'CRC valid' in line:
                        crc_ok = True

        except Exception as e:
            print(f"[USRP LoRa] Home RX error: {e}")

        success = rx_msg is not None and crc_ok
        print(f"[USRP LoRa] Home {self.home_id:02d}: "
              f"{'GOT broadcast' if success else 'NO broadcast'}")

        return {'success': success, 'rx_msg': rx_msg, 'crc_ok': crc_ok}

    # ─── Helpers ──────────────────────────────────────────────────

    def _calculate_toa(self, payload_length):
        n_preamble = PREAMBLE_LEN
        numerator = 8 * payload_length - 4 * SF + 28 + 16
        denominator = 4 * (SF - 2)
        payload_symbols = max(np.ceil(numerator / denominator) * (CR + 4), 0)
        n_symbols = n_preamble + payload_symbols
        return (n_symbols * (2 ** SF)) / BW

    def _make_result(self, success, payload):
        return {
            'success': success, 'crc_ok': False,
            'tx_bytes': len(payload) if payload else 0,
            'rx_msg': None,
            't_toa': self._calculate_toa(len(payload)) if payload else 0,
            'pdr': self.successful_transmissions / max(self.total_transmissions, 1),
            'packet_size': len(payload) if payload else 0,
        }


# ─── Factory functions ────────────────────────────────────────────

def get_home_radio(work_dir=None, home_id=None,
                   tx_serial=None, rx_serial=None) -> GRCLoRaUSRP:
    """
    Home node radio. Each home has its own TX USRP.
    RX USRP is optional (used to receive server broadcast).
    """
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=tx_serial,
        rx_serial=rx_serial,
        role='home',
        home_id=home_id,
    )


def get_server_radio(work_dir=None,
                     tx_serial=None, rx_serial=None) -> GRCLoRaUSRP:
    """
    Server radio. Has its own RX USRP to listen to all homes.
    TX USRP is used to broadcast global model.
    """
    return GRCLoRaUSRP(
        work_dir=work_dir,
        tx_serial=tx_serial,
        rx_serial=rx_serial,
        role='server',
    )
