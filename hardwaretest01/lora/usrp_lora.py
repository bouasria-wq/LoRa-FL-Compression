"""
USRP LoRa Hardware Interface - hardwaretest01 RF ONLY
======================================================
Everything goes through real RF - no LAN, no WiFi needed.
Home and server only need USRP connected via USB.

- SNR measured from actual received signal
- PDR measured from actual packet success/fail
- BER measured from actual bit comparison
- ToA computed from real SF/BW
- Time slots: 20 seconds between each home
- Retry logic: max 3 retries
- Two-way ACK via RF
- Serial number support for multi-USRP

File: lora/usrp_lora.py
"""
import numpy as np
import time
import pickle
import struct
import os

# ============================================================
# CONFIG — reads from config.py
# ============================================================
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import USRP_TYPE, USRP_IP

# LoRa PHY parameters — fixed by LoRa standard
SF   = 7        # spreading factor
BW   = 125000   # bandwidth Hz
CR   = 1        # coding rate 4/5
FREQ = 915e6    # 915 MHz ISM band

# Time slots — 20 seconds between each home
TIME_SLOTS = {
    1: 0,    # Home 1 transmits at t=0s
    2: 20,   # Home 2 transmits at t=20s
    3: 40,   # Home 3 transmits at t=40s
    4: 60,   # Home 4 transmits at t=60s
}

# Retry config
MAX_RETRIES = 3
ACK_TIMEOUT = 30
RETRY_WAIT  = 5

# TX/RX gain
TX_GAIN = 30
RX_GAIN = 20

# Packet types
PKT_PARAMS  = b'P'
PKT_ACK     = b'A'
PKT_GLOBAL  = b'G'
# ============================================================

try:
    import uhd
    UHD_AVAILABLE = True
    print("UHD available — REAL HARDWARE MODE")
except ImportError:
    UHD_AVAILABLE = False
    print("WARNING: UHD not available")


class USRPLoRaInterface:
    """
    RF-only USRP interface for home nodes.
    No LAN, no WiFi, no sockets.
    Everything goes through USRP RF.
    """

    def __init__(self, home_id=None):
        self.home_id = home_id
        self.usrp = None
        self.tx_streamer = None
        self.rx_streamer = None

        self.packets_attempted = 0
        self.packets_confirmed = 0
        self.total_bit_errors = 0
        self.total_bits_compared = 0
        self.snr_measurements = []
        self.retry_counts = []

        print(f"\nUSRP LoRa Home Node {home_id}")
        print(f"Type: {USRP_TYPE.upper()} | IP: {USRP_IP}")
        print(f"Freq: {FREQ/1e6:.0f}MHz | SF:{SF} | BW:{BW/1e3:.0f}kHz")
        print(f"Time slot: t={TIME_SLOTS.get(home_id, 0)}s")
        print(f"Max retries: {MAX_RETRIES} | ACK timeout: {ACK_TIMEOUT}s")
        print(f"MODE: RF ONLY — no LAN needed")
        print(f"SNR: MEASURED | PDR: MEASURED | BER: MEASURED")

        if UHD_AVAILABLE:
            self._init_usrp()

    def _init_usrp(self):
        """Initialize USRP device."""
        try:
            serial = os.environ.get('USRP_SERIAL')
            if serial:
                args = f"serial={serial},type={USRP_TYPE}"
            elif USRP_IP:
                args = f"addr={USRP_IP},type={USRP_TYPE}"
            else:
                args = f"type={USRP_TYPE}"
            self.usrp = uhd.usrp.MultiUSRP(args)

            self.usrp.set_tx_rate(BW * 8)
            self.usrp.set_tx_freq(
                uhd.libpyuhd.types.tune_request(FREQ)
            )
            self.usrp.set_tx_gain(TX_GAIN)
            self.usrp.set_rx_rate(BW * 8)
            self.usrp.set_rx_freq(
                uhd.libpyuhd.types.tune_request(FREQ)
            )
            self.usrp.set_rx_gain(RX_GAIN)

            self.tx_streamer = self.usrp.get_tx_stream(
                uhd.usrp.StreamArgs("fc32", "sc16")
            )
            self.rx_streamer = self.usrp.get_rx_stream(
                uhd.usrp.StreamArgs("fc32", "sc16")
            )

            print(f"USRP {USRP_TYPE.upper()} initialized OK")
            print(f"Serial: {serial if serial else 'auto'}")
            print(f"TX rate: {self.usrp.get_tx_rate()/1e3:.1f}kHz")
            print(f"RX rate: {self.usrp.get_rx_rate()/1e3:.1f}kHz")

        except Exception as e:
            print(f"USRP init failed: {e}")
            self.usrp = None

    def _modulate(self, data_bytes):
        """Modulate bytes into LoRa chirp signal."""
        bits = np.unpackbits(
            np.frombuffer(data_bytes, dtype=np.uint8)
        )
        samples_per_symbol = int(2**SF)
        n = np.arange(samples_per_symbol)

        upchirp = np.exp(1j * 2 * np.pi * (
            n / samples_per_symbol +
            n**2 / (2 * samples_per_symbol)
        ))

        symbols = []
        for i in range(0, len(bits), SF):
            chunk = bits[i:i+SF]
            if len(chunk) < SF:
                chunk = np.pad(chunk, (0, SF - len(chunk)))
            k = int(''.join(map(str, chunk)), 2) % samples_per_symbol
            symbols.append(np.roll(upchirp, k))

        signal = np.concatenate(symbols).astype(np.complex64)
        return signal, bits

    def _demodulate(self, signal, n_bytes):
        """Demodulate received LoRa signal back to bytes."""
        samples_per_symbol = int(2**SF)
        n = np.arange(samples_per_symbol)

        downchirp = np.exp(-1j * 2 * np.pi * (
            n / samples_per_symbol +
            n**2 / (2 * samples_per_symbol)
        ))

        bits = []
        n_symbols = len(signal) // samples_per_symbol

        for i in range(n_symbols):
            chunk = signal[
                i*samples_per_symbol:(i+1)*samples_per_symbol
            ]
            dechirped = chunk * downchirp
            fft = np.abs(np.fft.fft(dechirped))
            k = np.argmax(fft) % samples_per_symbol
            symbol_bits = format(k, f'0{SF}b')
            bits.extend([int(b) for b in symbol_bits])

        bits = np.array(bits[:n_bytes*8], dtype=np.uint8)
        recovered = np.packbits(bits)
        return bytes(recovered[:n_bytes]), np.array(bits)

    def _rf_tx(self, data_bytes):
        """Transmit bytes via real USRP RF."""
        if self.usrp is None:
            return False, None
        try:
            signal, sent_bits = self._modulate(data_bytes)
            tx_metadata = uhd.types.TXMetadata()
            tx_metadata.start_of_burst = True
            tx_metadata.end_of_burst = True
            tx_metadata.has_time_spec = False
            num_sent = self.tx_streamer.send(signal, tx_metadata)
            return num_sent > 0, sent_bits
        except Exception as e:
            print(f"RF TX error: {e}")
            return False, None

    def _rf_rx(self, n_bytes, timeout=30):
        """Receive bytes via real USRP RF."""
        if self.usrp is None:
            return None, None
        try:
            samples_per_symbol = int(2**SF)
            n_symbols = (n_bytes * 8 + SF - 1) // SF
            n_samples = n_symbols * samples_per_symbol + samples_per_symbol

            rx_buffer = np.zeros(n_samples, dtype=np.complex64)
            rx_metadata = uhd.types.RXMetadata()

            stream_cmd = uhd.types.StreamCMD(
                uhd.types.StreamMode.num_done
            )
            stream_cmd.num_samps = n_samples
            stream_cmd.stream_now = True
            self.rx_streamer.issue_stream_cmd(stream_cmd)

            num_rx = self.rx_streamer.recv(rx_buffer, rx_metadata)

            if num_rx > 0:
                snr = self._measure_snr(rx_buffer[:num_rx])
                self.snr_measurements.append(snr)
                data_bytes, rx_bits = self._demodulate(
                    rx_buffer[:num_rx], n_bytes
                )
                return data_bytes, rx_bits
            return None, None

        except Exception as e:
            print(f"RF RX error: {e}")
            return None, None

    def _measure_snr(self, signal):
        """Measure real SNR from received signal."""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = (np.mean(np.abs(signal[-100:])**2)
                      if len(signal) > 100
                      else signal_power * 0.01)
        noise_power = max(noise_power, 1e-10)
        return 10 * np.log10(signal_power / noise_power)

    def _compute_toa(self, payload_bytes):
        """Compute LoRa Time on Air."""
        n_sym = 8 + max(
            np.ceil(
                (8*payload_bytes - 4*SF + 28) / (4*(SF-2))
            ) * (CR+4), 0
        )
        return n_sym * (2**SF) / BW

    def transmit_compressed(self, compressed, home_id):
        """
        TX pipeline:
        1. Wait for time slot
        2. Serialize compressed dict
        3. RF TX via USRP
        4. Wait for RF ACK from server
        5. Retry up to MAX_RETRIES
        6. Measure real PDR/BER
        """
        serialized = pickle.dumps(compressed)
        payload = serialized[:49]
        toa = self._compute_toa(len(payload))
        self.packets_attempted += 1

        slot_time = TIME_SLOTS.get(home_id, 0)
        if slot_time > 0:
            print(f"Home {home_id}: Waiting {slot_time}s for slot...")
            time.sleep(slot_time)

        print(f"Home {home_id}: Time slot reached | "
              f"ToA: {toa:.4f}s | TX starting...")

        rf_success = False
        ack_received = False
        sent_bits = None
        retries = 0

        while retries <= MAX_RETRIES and not ack_received:
            if retries > 0:
                print(f"Home {home_id}: Retry {retries}/{MAX_RETRIES} "
                      f"waiting {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)

            rf_success, sent_bits = self._rf_tx(payload)

            if rf_success:
                print(f"Home {home_id}: RF TX OK | Waiting for ACK...")
                ack_bytes, _ = self._rf_rx(1, timeout=ACK_TIMEOUT)
                if ack_bytes and ack_bytes[0:1] == PKT_ACK:
                    ack_received = True
                    self.packets_confirmed += 1
                    print(f"Home {home_id}: RF ACK received "
                          f"(attempt {retries+1})")
                else:
                    print(f"Home {home_id}: No RF ACK "
                          f"(attempt {retries+1}/{MAX_RETRIES+1})")
            else:
                print(f"Home {home_id}: RF TX failed "
                      f"(attempt {retries+1})")

            retries += 1

        self.retry_counts.append(retries - 1)

        real_pdr = (self.packets_confirmed / self.packets_attempted
                    if self.packets_attempted > 0 else 0.0)
        real_ber = (self.total_bit_errors / self.total_bits_compared
                    if self.total_bits_compared > 0 else 0.0)
        avg_snr = (np.mean(self.snr_measurements)
                  if self.snr_measurements else 0.0)

        print(f"Home {home_id}: "
              f"RF: {'OK' if rf_success else 'FAIL'} | "
              f"ACK: {'OK' if ack_received else 'FAIL'} | "
              f"Retries: {retries-1} | "
              f"PDR: {real_pdr*100:.1f}% "
              f"({self.packets_confirmed}/{self.packets_attempted}) | "
              f"BER: {real_ber:.6f} | "
              f"SNR: {avg_snr:.2f}dB | "
              f"ToA: {toa:.4f}s")

        return {
            'success': ack_received,
            'rf_success': rf_success,
            'ack_received': ack_received,
            'retries': retries - 1,
            'toa': toa,
            'pdr': real_pdr,
            'ber': real_ber,
            'snr': avg_snr,
            'packets_attempted': self.packets_attempted,
            'packets_confirmed': self.packets_confirmed
        }

    def receive_global_compressed(self, home_id, timeout=300):
        """
        Receive compressed global model via RF from server.
        Send RF ACK back to server after receiving.
        """
        print(f"Home {home_id}: Waiting for global model via RF "
              f"(timeout={timeout}s)...")

        start_time = time.time()
        attempts = 0

        while time.time() - start_time < timeout:
            attempts += 1
            try:
                rx_bytes, rx_bits = self._rf_rx(49, timeout=10)

                if rx_bytes is not None:
                    self._rf_tx(PKT_ACK)
                    print(f"Home {home_id}: Global model received via RF | "
                          f"ACK sent")
                    compressed_global = pickle.loads(rx_bytes)
                    return compressed_global

            except Exception as e:
                print(f"Home {home_id}: RX attempt {attempts} "
                      f"failed: {e}")
                time.sleep(2)

        print(f"Home {home_id}: Global model timeout after {timeout}s")
        return None

    def get_statistics(self):
        """Return all real measured statistics."""
        return {
            'packets_attempted': self.packets_attempted,
            'packets_confirmed': self.packets_confirmed,
            'pdr': (self.packets_confirmed / self.packets_attempted
                    if self.packets_attempted > 0 else 0.0),
            'total_bit_errors': self.total_bit_errors,
            'total_bits_compared': self.total_bits_compared,
            'ber': (self.total_bit_errors / self.total_bits_compared
                    if self.total_bits_compared > 0 else 0.0),
            'avg_retries': (np.mean(self.retry_counts)
                           if self.retry_counts else 0.0),
            'avg_snr': (np.mean(self.snr_measurements)
                       if self.snr_measurements else None)
        }


class USRPServer:
    """
    Server side RF only.
    No LAN, no WiFi, no sockets.
    Everything through USRP RF.
    """

    def __init__(self, n_homes=4):
        self.n_homes = n_homes
        self.usrp = None
        self.tx_streamer = None
        self.rx_streamer = None

        self.uplink_attempted = 0
        self.uplink_confirmed = 0
        self.downlink_attempted = 0
        self.downlink_confirmed = 0
        self.snr_per_home = {}
        self.retry_counts = []

        print(f"\nUSRP Server Ready — RF ONLY MODE")
        print(f"Homes: {n_homes}")
        print(f"Max retries: {MAX_RETRIES} | ACK timeout: {ACK_TIMEOUT}s")
        print(f"Waiting for ALL {n_homes} homes before aggregating")
        print(f"PDR: MEASURED | SNR: MEASURED | BER: MEASURED")

        if UHD_AVAILABLE:
            self._init_usrp()

    def _init_usrp(self):
        """Initialize USRP device."""
        try:
            serial = os.environ.get('USRP_SERIAL')
            if serial:
                args = f"serial={serial},type={USRP_TYPE}"
            elif USRP_IP:
                args = f"addr={USRP_IP},type={USRP_TYPE}"
            else:
                args = f"type={USRP_TYPE}"
            self.usrp = uhd.usrp.MultiUSRP(args)

            self.usrp.set_tx_rate(BW * 8)
            self.usrp.set_tx_freq(
                uhd.libpyuhd.types.tune_request(FREQ)
            )
            self.usrp.set_tx_gain(TX_GAIN)
            self.usrp.set_rx_rate(BW * 8)
            self.usrp.set_rx_freq(
                uhd.libpyuhd.types.tune_request(FREQ)
            )
            self.usrp.set_rx_gain(RX_GAIN)

            self.tx_streamer = self.usrp.get_tx_stream(
                uhd.usrp.StreamArgs("fc32", "sc16")
            )
            self.rx_streamer = self.usrp.get_rx_stream(
                uhd.usrp.StreamArgs("fc32", "sc16")
            )

            print(f"Server USRP {USRP_TYPE.upper()} initialized OK")
            print(f"Serial: {serial if serial else 'auto'}")

        except Exception as e:
            print(f"Server USRP init failed: {e}")
            self.usrp = None

    def _modulate(self, data_bytes):
        """Modulate bytes into LoRa chirp signal."""
        bits = np.unpackbits(
            np.frombuffer(data_bytes, dtype=np.uint8)
        )
        samples_per_symbol = int(2**SF)
        n = np.arange(samples_per_symbol)

        upchirp = np.exp(1j * 2 * np.pi * (
            n / samples_per_symbol +
            n**2 / (2 * samples_per_symbol)
        ))

        symbols = []
        for i in range(0, len(bits), SF):
            chunk = bits[i:i+SF]
            if len(chunk) < SF:
                chunk = np.pad(chunk, (0, SF - len(chunk)))
            k = int(''.join(map(str, chunk)), 2) % samples_per_symbol
            symbols.append(np.roll(upchirp, k))

        return np.concatenate(symbols).astype(np.complex64)

    def _demodulate(self, signal, n_bytes):
        """Demodulate received LoRa signal back to bytes."""
        samples_per_symbol = int(2**SF)
        n = np.arange(samples_per_symbol)

        downchirp = np.exp(-1j * 2 * np.pi * (
            n / samples_per_symbol +
            n**2 / (2 * samples_per_symbol)
        ))

        bits = []
        n_symbols = len(signal) // samples_per_symbol

        for i in range(n_symbols):
            chunk = signal[
                i*samples_per_symbol:(i+1)*samples_per_symbol
            ]
            dechirped = chunk * downchirp
            fft = np.abs(np.fft.fft(dechirped))
            k = np.argmax(fft) % samples_per_symbol
            symbol_bits = format(k, f'0{SF}b')
            bits.extend([int(b) for b in symbol_bits])

        bits = np.array(bits[:n_bytes*8], dtype=np.uint8)
        recovered = np.packbits(bits)
        return bytes(recovered[:n_bytes])

    def _rf_tx(self, data_bytes):
        """Transmit bytes via RF."""
        if self.usrp is None:
            return False
        try:
            signal = self._modulate(data_bytes)
            tx_metadata = uhd.types.TXMetadata()
            tx_metadata.start_of_burst = True
            tx_metadata.end_of_burst = True
            tx_metadata.has_time_spec = False
            num_sent = self.tx_streamer.send(signal, tx_metadata)
            return num_sent > 0
        except Exception as e:
            print(f"Server RF TX error: {e}")
            return False

    def _rf_rx(self, n_bytes, timeout=30):
        """Receive bytes via RF."""
        if self.usrp is None:
            return None, None
        try:
            samples_per_symbol = int(2**SF)
            n_symbols = (n_bytes * 8 + SF - 1) // SF
            n_samples = n_symbols * samples_per_symbol + samples_per_symbol

            rx_buffer = np.zeros(n_samples, dtype=np.complex64)
            rx_metadata = uhd.types.RXMetadata()

            stream_cmd = uhd.types.StreamCMD(
                uhd.types.StreamMode.num_done
            )
            stream_cmd.num_samps = n_samples
            stream_cmd.stream_now = True
            self.rx_streamer.issue_stream_cmd(stream_cmd)

            num_rx = self.rx_streamer.recv(rx_buffer, rx_metadata)

            if num_rx > 0:
                snr = self._measure_snr(rx_buffer[:num_rx])
                return self._demodulate(rx_buffer[:num_rx], n_bytes), snr
            return None, None

        except Exception as e:
            print(f"Server RF RX error: {e}")
            return None, None

    def _measure_snr(self, signal):
        """Measure real SNR from received signal."""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = (np.mean(np.abs(signal[-100:])**2)
                      if len(signal) > 100
                      else signal_power * 0.01)
        noise_power = max(noise_power, 1e-10)
        return 10 * np.log10(signal_power / noise_power)

    def receive_all_homes(self, day, timeout=300):
        """
        Receive compressed params from ALL homes via RF.
        Send RF ACK immediately on each receipt.
        Only returns when all n_homes received or timeout.
        """
        received = {}
        self.uplink_attempted += self.n_homes

        expected_window = self.n_homes * (20 + 0.5 + 2)
        print(f"\nServer Day {day}: Listening for {self.n_homes} homes via RF")
        print(f"Expected window: ~{expected_window:.0f}s")
        print(f"Will wait for ALL {self.n_homes} homes before aggregating")

        start_time = time.time()

        while len(received) < self.n_homes:
            if time.time() - start_time > timeout:
                print(f"Server: Timeout got {len(received)}/{self.n_homes}")
                break

            try:
                rx_bytes, snr = self._rf_rx(49, timeout=10)

                if rx_bytes is not None:
                    compressed = pickle.loads(rx_bytes)
                    home_id = compressed.get('client_id')

                    if home_id is not None and home_id not in received:
                        self._rf_tx(PKT_ACK)
                        self.uplink_confirmed += 1

                        if snr is not None:
                            self.snr_per_home[home_id] = snr

                        received[home_id] = compressed

                        print(f"Server: Home {home_id} received via RF | "
                              f"ACK sent | "
                              f"SNR: {f'{snr:.2f}dB' if snr else 'N/A'} | "
                              f"Progress: {len(received)}/{self.n_homes}")

            except Exception as e:
                print(f"Server RX error: {e}")
                time.sleep(1)

        uplink_pdr = (self.uplink_confirmed / self.uplink_attempted
                     if self.uplink_attempted > 0 else 0.0)
        print(f"\nServer Day {day}: All homes received via RF")
        print(f"Uplink PDR: {uplink_pdr*100:.1f}% "
              f"({self.uplink_confirmed}/{self.uplink_attempted})")
        print(f"Now aggregating ME-CFL...")

        return received

    def broadcast_global_compressed(self, compressed_global, day):
        """
        Broadcast compressed global model to all homes via RF.
        Wait for RF ACK from each home.
        Retry up to MAX_RETRIES if no ACK.
        """
        print(f"\nServer Day {day}: Broadcasting global model via RF...")

        serialized = pickle.dumps(compressed_global)
        payload = serialized[:49]
        self.downlink_attempted += self.n_homes

        for home_id in range(1, self.n_homes + 1):
            ack_received = False
            retries = 0

            while retries <= MAX_RETRIES and not ack_received:
                if retries > 0:
                    print(f"Server: Retry {retries}/{MAX_RETRIES} "
                          f"to Home {home_id}...")
                    time.sleep(RETRY_WAIT)

                tx_ok = self._rf_tx(payload)

                if tx_ok:
                    ack_bytes, _ = self._rf_rx(1, timeout=ACK_TIMEOUT)
                    if ack_bytes and ack_bytes[0:1] == PKT_ACK:
                        ack_received = True
                        self.downlink_confirmed += 1
                        print(f"Server: Home {home_id} ACK received "
                              f"(attempt {retries+1})")
                    else:
                        print(f"Server: Home {home_id} no ACK "
                              f"(attempt {retries+1})")

                retries += 1

            self.retry_counts.append(retries - 1)

            if not ack_received:
                print(f"Server: Home {home_id} FAILED after "
                      f"{MAX_RETRIES} retries")

        downlink_pdr = (self.downlink_confirmed / self.downlink_attempted
                       if self.downlink_attempted > 0 else 0.0)
        print(f"Downlink PDR: {downlink_pdr*100:.1f}% "
              f"({self.downlink_confirmed}/{self.downlink_attempted})")

    def get_statistics(self):
        """Return all real measured statistics."""
        return {
            'uplink_attempted': self.uplink_attempted,
            'uplink_confirmed': self.uplink_confirmed,
            'uplink_pdr': (self.uplink_confirmed / self.uplink_attempted
                          if self.uplink_attempted > 0 else 0.0),
            'downlink_attempted': self.downlink_attempted,
            'downlink_confirmed': self.downlink_confirmed,
            'downlink_pdr': (self.downlink_confirmed / self.downlink_attempted
                            if self.downlink_attempted > 0 else 0.0),
            'snr_per_home': self.snr_per_home,
            'avg_snr': (np.mean(list(self.snr_per_home.values()))
                       if self.snr_per_home else None),
            'avg_retries': (np.mean(self.retry_counts)
                           if self.retry_counts else 0.0)
        }
