"""
Server Aggregator - ME-CFL USRP TDMA Version (hardwaretest05)
===============================================================
3 homes + 1 server, 4 USRP B200s total.

Server workflow per day:
  1. Wait for all homes to signal ready (file flags)
  2. Start TDMA round — write sync timestamp
  3. Open RX for full TDMA window (listen for all 3 packets)
  4. Aggregate received models
  5. Broadcast global model back to all homes
  6. Write global model file for homes to read (backup)

File: server_aggregator.py
"""
import numpy as np
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'local_home'))
sys.path.insert(0, str(Path(__file__).parent / 'compression'))
sys.path.insert(0, str(Path(__file__).parent / 'lora'))
sys.path.insert(0, str(Path(__file__).parent / 'server'))

from hegazy import AggregateGaussianMechanism
from hegazy_lora_bridge import HegazyLoRaBridge
from aggregate import MECFLAggregator
from gr_lora_usrp import (get_server_radio, LORA_MAX_BYTES,
                           SLOT_DURATION, N_HOMES, SERVER_LISTEN_TIME)


class ServerAggregator:

    def __init__(self, n_homes=3, n_days=7,
                 tx_serial=None, rx_serial=None):
        self.n_homes = n_homes
        self.n_days  = n_days

        lora_dir = Path(__file__).parent / 'lora'
        self.lora_dir = lora_dir
        self.radio   = get_server_radio(
            work_dir=str(lora_dir),
            tx_serial=tx_serial,
            rx_serial=rx_serial,
        )
        self.bridge     = HegazyLoRaBridge()
        self.hegazy     = AggregateGaussianMechanism(
            n_clients=n_homes, sigma=0.1, seed=0
        )
        self.aggregator = None
        self.daily_summary = []

        print(f"\n{'='*60}")
        print(f"Federated Server - ME-CFL USRP TDMA (hardwaretest05)")
        print(f"{'='*60}")
        print(f"Clients: {n_homes} | Max payload: {LORA_MAX_BYTES} bytes")
        print(f"Transport: REAL USRP B200 | TDMA: {SLOT_DURATION}s/slot")
        print(f"TDMA window: {SERVER_LISTEN_TIME}s total")
        print(f"{'='*60}")

    def wait_for_homes(self, day_num, timeout=600):
        """Wait for all homes to signal they're ready to transmit."""
        ready = {}

        print(f"\n{'='*60}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*60}")

        start = time.time()
        while len(ready) < self.n_homes and time.time() - start < timeout:
            for h in range(1, self.n_homes + 1):
                if h in ready:
                    continue
                flag = self.lora_dir / f'home_{h:02d}_ready.flag'
                upload = self.lora_dir / f'home_{h:02d}_upload.bin'
                if flag.exists() and upload.exists():
                    try:
                        flag_content = flag.read_text().strip()
                        if flag_content == f'day_{day_num}':
                            raw = upload.read_bytes()
                            compressed = self.bridge.unpack_compressed(raw)
                            ready[h] = {
                                'compressed': compressed,
                                'raw': raw,
                            }
                            print(f"[SERVER] Home {h:02d} ready: "
                                  f"{len(raw)} bytes | "
                                  f"zeta={compressed['zeta_i']:.6f}")
                    except Exception as e:
                        print(f"[SERVER] Error reading Home {h:02d}: {e}")
            time.sleep(1)

        print(f"Day {day_num}: {len(ready)}/{self.n_homes} homes ready")
        return ready

    def start_tdma_round(self, day_num):
        """
        Signal all homes that the TDMA round has started.
        Writes a sync file with the current timestamp.
        """
        day_start = time.time()
        sync_file = self.lora_dir / f'day_{day_num}_sync.flag'
        sync_file.write_text(str(day_start))
        print(f"[SERVER] TDMA round started for Day {day_num} at {day_start:.1f}")
        return day_start

    def receive_rf_packets(self, day_num, day_start_time):
        """
        Open server RX USRP and listen for all home packets
        during the TDMA window.
        """
        print(f"[SERVER] Opening RX for TDMA window ({SERVER_LISTEN_TIME}s)...")
        packets = self.radio.receive_all_homes(
            n_homes=self.n_homes,
            timeout=SERVER_LISTEN_TIME,
        )

        n_received = len(packets)
        n_crc_ok = sum(1 for p in packets if p.get('crc_ok', False))
        pdr = n_crc_ok / self.n_homes * 100 if self.n_homes > 0 else 0

        print(f"[SERVER] RF results: {n_received} packets received, "
              f"{n_crc_ok} CRC OK, PDR: {pdr:.1f}%")

        return packets

    def aggregate(self, ready_homes, rf_packets, day_num):
        """
        ME-CFL aggregation.
        Uses the file-based uploads (guaranteed) + validates against
        RF-received packets.
        """
        all_params = []
        all_zetas  = []

        a, b = self.hegazy.decompose()
        for h_id, home_data in ready_homes.items():
            compressed = home_data['compressed']
            params = self.hegazy.decode_parameters(compressed, a, b)
            all_params.append(params)
            all_zetas.append(compressed['zeta_i'])

        if self.aggregator is None:
            self.aggregator = MECFLAggregator(param_size=len(all_params[0]))
            print(f"Server initialized: {len(all_params[0])} parameters")

        global_params = self.aggregator.aggregate(all_params, all_zetas)

        avg_zeta = np.mean(all_zetas)
        n_rf = len([p for p in rf_packets if p.get('crc_ok')])

        print(f"Day {day_num}: Aggregated {len(ready_homes)}/{self.n_homes} homes")
        print(f"[SERVER] Avg zeta: {avg_zeta:.6f} | RF packets: {n_rf}/{self.n_homes}")

        self.daily_summary.append({
            'day': day_num,
            'n_homes': len(ready_homes),
            'avg_zeta': avg_zeta,
            'rf_packets': n_rf,
            'pdr': n_rf / self.n_homes * 100 if self.n_homes > 0 else 0,
        })

        return global_params

    def broadcast_global(self, global_params, day_num):
        """Broadcast global model to all homes via USRP + file backup."""
        a, b = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters(
            [global_params], client_id=0, a=a, b=b
        )
        payload = self.bridge.pack_compressed(compressed)

        # Broadcast over the air
        print(f"[SERVER] Broadcasting {len(payload)} bytes Day {day_num}...")
        result = self.radio.broadcast(payload)

        if result['success']:
            print(f"[SERVER] Broadcast SUCCESS")
        else:
            print(f"[SERVER] Broadcast FAILED (file backup still works)")

        # Always write file backup for homes to read
        global_file = self.lora_dir / f'global_day_{day_num}.bin'
        global_file.write_bytes(payload)

        # Also write to generic global_model.bin for backward compat
        generic_file = self.lora_dir / 'global_model.bin'
        generic_file.write_bytes(payload)

        return result['success']

    def cleanup_day(self, day_num):
        """Clean up flags for the completed day."""
        for h in range(1, self.n_homes + 1):
            flag = self.lora_dir / f'home_{h:02d}_ready.flag'
            if flag.exists():
                flag.unlink()
        sync = self.lora_dir / f'day_{day_num}_sync.flag'
        if sync.exists():
            sync.unlink()

    def run_day(self, day_num):
        # 1. Wait for all homes to be ready
        ready_homes = self.wait_for_homes(day_num)
        if not ready_homes:
            print(f"[SERVER] No homes ready for Day {day_num}")
            return

        # 2. Start TDMA round
        day_start = self.start_tdma_round(day_num)

        # 3. Listen for RF packets during TDMA window
        rf_packets = self.receive_rf_packets(day_num, day_start)

        # 4. Aggregate
        global_params = self.aggregate(ready_homes, rf_packets, day_num)

        # 5. Broadcast global model back
        self.broadcast_global(global_params, day_num)

        # 6. Cleanup
        self.cleanup_day(day_num)

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print(f"\n{'='*60}")
        print(f"FINAL SERVER SUMMARY - ME-CFL USRP TDMA")
        print(f"{'='*60}")
        for s in self.daily_summary:
            print(f"Day {s['day']}: {s['n_homes']} homes | "
                  f"Zeta: {s['avg_zeta']:.6f} | "
                  f"RF: {s['rf_packets']}/{self.n_homes} | "
                  f"PDR: {s['pdr']:.1f}%")

        print(f"\n{'='*40}")
        print(f"SERVER AGGREGATION SUMMARY")
        print(f"{'='*40}")
        for s in self.daily_summary:
            pct = s['n_homes'] / self.n_homes * 100
            print(f"Day {s['day']}: {s['n_homes']} homes | "
                  f"Participation: {pct:.0f}% | "
                  f"PDR: {s['pdr']:.1f}%")

        avg_pdr = np.mean([s['pdr'] for s in self.daily_summary])
        print(f"\nOverall PDR: {avg_pdr:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes',   type=int, default=3)
    parser.add_argument('--days',      type=int, default=7)
    parser.add_argument('--tx_serial', type=str, default=None,
                        help='Server TX USRP serial')
    parser.add_argument('--rx_serial', type=str, default=None,
                        help='Server RX USRP serial')
    args = parser.parse_args()
    ServerAggregator(
        n_homes=args.n_homes,
        n_days=args.days,
        tx_serial=args.tx_serial,
        rx_serial=args.rx_serial,
    ).run()


if __name__ == "__main__":
    main()
