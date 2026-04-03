#!/usr/bin/env python3
"""
GNU Radio LoRa USRP Bridge - hardwaretestnew
=============================================
Home TX serial:   32BBAD0
Server TX serial: 32BBAD0  (same USRP, different time slot - never clash)
Home RX serial:   32BBAD9
Server RX serial: 32BBAD9  (same USRP, different time slot - never clash)

Flow per day:
  Phase 1: Server RX starts (32BBAD9 listening)
           Home TX starts   (32BBAD0 transmits compressed model)
           Home TX done     -> Home stops TX
           Server receives  -> Server stops RX
  Phase 2: Server aggregates + compresses global model
  Phase 3: Home RX starts   (32BBAD9 listening)
           Server TX starts (32BBAD0 transmits global model)
           Server TX done   -> Server stops TX
           Home receives    -> Home stops RX
  Phase 4: Home updates local model -> repeat
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

# Serial numbers
HOME_TX_SERIAL   = '32BBAD0'
SERVER_TX_SERIAL = '32BBAD0'   # same USRP as home TX, different time slot
HOME_RX_SERIAL   = '32BBAD9'
SERVER_RX_SERIAL = '32BBAD9'   # same USRP as server TX, different time slot

SF           = 7
BW           = 125000
CR           = 1
LORA_MAX     = 255
PREAMBLE_LEN = 8
RX_INIT_TIME = 3.0
FLUSH_TIME   = 3.0

ASCII_CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def bytes_to_ascii(payload: bytes) -> str:
    return ''.join(ASCII_CHARSET[b % len(ASCII_CHARSET)] for b in payload)


def _patch(filepath, serial):
    if not filepath.exists():
        return
    code = filepath.read_text()
    code = re.sub(r'serial=[A-Za-z0-9_]+', f'serial={serial}', code)
    filepath.write_text(code)
    print(f"[LoRa] Patched {filepath.name} -> serial={serial}")


def _calculate_toa(payload_length):
    numerator = 8 * payload_length - 4 * SF + 28 + 16
    denominator = 4 * (SF - 2)
    payload_symbols = max(np.ceil(numerator / denominator) * (CR + 4), 0)
    return ((PREAMBLE_LEN + payload_symbols) * (2 ** SF)) / BW


def _stop(tb, name):
    if tb is None:
        return
    try:
        tb.stop()
        tb.wait()
        print(f"[LoRa] {name} stopped.")
    except Exception as e:
        print(f"[LoRa] {name} stop (ignored): {e}")


def tx(work_dir, tx_serial, payload: bytes, done_flag_path=None):
    """
    Transmit payload via LoRa.
    Stops TX fully before writing done flag.
    """
    assert len(payload) <= LORA_MAX
    work_dir = Path(work_dir)
    ascii_payload = bytes_to_ascii(payload)

    _patch(work_dir / 'lora_TX.py', tx_serial)

    sys.path.insert(0, str(work_dir))
    tb = None
    success = False
    try:
        from lora_TX import lora_TX as LoraTX
        tb = LoraTX()
        tb.set_frame_period(10_000_000)
        tb.start()
        msg_port = pmt.intern("msg")
        whitening = tb.lora_sdr_whitening_0.to_basic_block()
        whitening._post(msg_port, pmt.intern(ascii_payload))
        print(f"[LoRa] TX posted {len(payload)} bytes -> {tx_serial}")
        time.sleep(FLUSH_TIME)
        success = True
    except Exception as e:
        print(f"[LoRa] TX error: {e}")
    finally:
        _stop(tb, "TX")
        if str(work_dir) in sys.path:
            sys.path.remove(str(work_dir))

    # Write done flag AFTER TX fully stopped
    if done_flag_path is not None:
        Path(done_flag_path).write_text("done")
        print(f"[LoRa] Done flag: {Path(done_flag_path).name}")

    t_toa = _calculate_toa(len(payload))
    print(f"[LoRa] TX {'OK' if success else 'FAILED'} | ToA={t_toa:.3f}s")
    return success, t_toa


def rx(work_dir, rx_serial, duration=10.0):
    """
    Start RX and listen for duration seconds.
    Returns after duration whether or not a packet was received
    (CRC verification is printed by the flowgraph itself).
    """
    work_dir = Path(work_dir)
    _patch(work_dir / 'lora_RX.py', rx_serial)

    sys.path.insert(0, str(work_dir))
    tb = None
    try:
        from lora_RX import lora_RX as LoraRX
        tb = LoraRX()
        tb.start()
        print(f"[LoRa] RX started -> {rx_serial} | listening {duration}s")
        time.sleep(RX_INIT_TIME + duration)
    except Exception as e:
        print(f"[LoRa] RX error: {e}")
    finally:
        _stop(tb, "RX")
        if str(work_dir) in sys.path:
            sys.path.remove(str(work_dir))
    print(f"[LoRa] RX stopped.")
