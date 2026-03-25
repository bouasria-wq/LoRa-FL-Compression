"""
Server Aggregator - ME-CFL USRP Version (hardwaretest04)
==========================================================
Same as hardwaretest03 server but uses REAL USRP B200 hardware
for broadcasting global model to homes.

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
from gr_lora_usrp import get_server_radio, LORA_MAX_BYTES


class ServerAggregator:

    def __init__(self, n_homes=1, n_days=7, tx_serial=None, rx_serial=None):
        self.n_homes = n_homes
        self.n_days  = n_days

        lora_dir = Path(__file__).parent / 'lora'
        self.radio      = get_server_radio(
            work_dir=str(lora_dir),
            tx_serial=tx_serial,
            rx_serial=rx_serial
        )
        self.bridge     = HegazyLoRaBridge()
        self.hegazy     = AggregateGaussianMechanism(
            n_clients=n_homes, sigma=0.1, seed=0
        )
        self.aggregator = None

        self.daily_summary = []

        print(f"\n{'='*50}")
        print(f"Federated Server - ME-CFL USRP VERSION (hardwaretest04)")
        print(f"{'='*50}")
        print(f"Clients: {n_homes} | Max payload: {LORA_MAX_BYTES} bytes")
        print(f"Transport: REAL USRP B200 hardware (over the air)")
        print(f"{'='*50}")

    def wait_for_homes(self, day_num, timeout=300):
        """Wait for all homes to upload their compressed models."""
        lora_dir = Path(__file__).parent / 'lora'
        ready = {}

        print(f"\n{'='*50}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*50}")

        start = time.time()
        while len(ready) < self.n_homes and time.time() - start < timeout:
            for h in range(1, self.n_homes + 1):
                if h in ready:
                    continue
                flag = lora_dir / f'home_{h:02d}_ready.flag'
                upload = lora_dir / f'home_{h:02d}_upload.bin'
                if flag.exists() and upload.exists():
                    try:
                        flag_content = flag.read_text().strip()
                        if flag_content == f'day_{day_num}':
                            raw = upload.read_bytes()
                            compressed = self.bridge.unpack_compressed(raw)
                            ready[h] = compressed
                            print(f"[SERVER] Received Home {h:02d}: "
                                  f"{len(raw)} bytes | "
                                  f"zeta={compressed['zeta_i']:.6f}")
                    except Exception as e:
                        print(f"[SERVER] Error reading Home {h:02d}: {e}")
            time.sleep(1)

        print(f"Day {day_num}: {len(ready)} homes ready")
        return ready

    def aggregate(self, ready_homes, day_num):
        """ME-CFL aggregation with variance reduction."""
        all_params = []
        all_zetas  = []

        a, b = self.hegazy.decompose()
        for h_id, compressed in ready_homes.items():
            params = self.hegazy.decode_parameters(compressed, a, b)
            all_params.append(params)
            all_zetas.append(compressed['zeta_i'])

        if self.aggregator is None:
            self.aggregator = MECFLAggregator(param_size=len(all_params[0]))
            print(f"Server initialized: {len(all_params[0])} parameters")

        global_params = self.aggregator.aggregate(all_params, all_zetas)

        print(f"Day {day_num}: Aggregated {len(ready_homes)}/{self.n_homes} homes")
        avg_zeta = np.mean(all_zetas)
        print(f"[SERVER] Avg zeta: {avg_zeta:.6f}")

        self.daily_summary.append({
            'day': day_num,
            'n_homes': len(ready_homes),
            'avg_zeta': avg_zeta
        })

        return global_params

    def broadcast_global(self, global_params, day_num):
        """Broadcast global model to all homes via USRP B200."""
        a, b = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters(
            [global_params], client_id=0, a=a, b=b
        )
        payload = self.bridge.pack_compressed(compressed)

        print(f"[SERVER] Broadcasting {len(payload)} bytes Day {day_num} via USRP B200...")
        result = self.radio.transmit(payload)

        if result['success']:
            print(f"[SERVER] Broadcast SUCCESS")
        else:
            print(f"[SERVER] Broadcast FAILED")

        # Also write to file for homes to read (backup if RF fails)
        global_file = Path(__file__).parent / 'lora' / 'global_model.bin'
        global_file.write_bytes(payload)

        return result['success']

    def cleanup_day(self, day_num):
        """Clean up flags for the day."""
        lora_dir = Path(__file__).parent / 'lora'
        for h in range(1, self.n_homes + 1):
            flag = lora_dir / f'home_{h:02d}_ready.flag'
            if flag.exists():
                flag.unlink()

    def run_day(self, day_num):
        ready_homes = self.wait_for_homes(day_num)

        if not ready_homes:
            print(f"[SERVER] No homes ready for Day {day_num}")
            return

        global_params = self.aggregate(ready_homes, day_num)
        self.broadcast_global(global_params, day_num)
        self.cleanup_day(day_num)

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print(f"\n{'='*50}")
        print(f"FINAL SERVER SUMMARY - ME-CFL USRP")
        print(f"{'='*50}")
        for s in self.daily_summary:
            print(f"Day {s['day']}: {s['n_homes']} homes | "
                  f"Zeta: {s['avg_zeta']:.6f}")

        print(f"\n{'='*40}")
        print(f"SERVER AGGREGATION SUMMARY")
        print(f"{'='*40}")
        for s in self.daily_summary:
            pct = s['n_homes'] / self.n_homes * 100
            print(f"Day {s['day']}: {s['n_homes']} homes | "
                  f"Participation: {pct:.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes', type=int, default=1)
    parser.add_argument('--days',    type=int, default=7)
    parser.add_argument('--tx_serial', type=str, default=None,
                        help='USRP B200 serial for server TX')
    parser.add_argument('--rx_serial', type=str, default=None,
                        help='USRP B200 serial for server RX')
    args = parser.parse_args()
    ServerAggregator(
        n_homes=args.n_homes,
        n_days=args.days,
        tx_serial=args.tx_serial,
        rx_serial=args.rx_serial,
    ).run()


if __name__ == "__main__":
    main()
