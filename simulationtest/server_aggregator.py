"""
Federated Server Aggregator - ME-CFL GRC Version (hardwaretest03)
==================================================================
Transport: GNU Radio Companion flowgraph (simulation)
Both directions use Hegazy compression.

File: server_aggregator.py
"""
import numpy as np
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'compression'))
sys.path.insert(0, str(Path(__file__).parent / 'server'))
sys.path.insert(0, str(Path(__file__).parent / 'lora'))

from hegazy import AggregateGaussianMechanism
from hegazy_lora_bridge import HegazyLoRaBridge
from aggregate import FederatedServer
from gr_lora_grc import get_server_radio, LORA_MAX_BYTES


class ServerAggregator:

    def __init__(self, n_homes=1, n_days=7):
        self.n_homes = n_homes
        self.n_days  = n_days

        self.lora_dir    = Path(__file__).parent / 'lora'
        self.lora_bridge = HegazyLoRaBridge()
        self.server      = FederatedServer(
            n_clients=n_homes, alpha=0.25, beta=0.9, eta=0.01
        )
        self.hegazy = AggregateGaussianMechanism(
            n_clients=n_homes, sigma=0.1, seed=0
        )
        self.radio = get_server_radio(work_dir=str(self.lora_dir))
        self.daily_results = []

        print("\n" + "="*50)
        print("Federated Server - ME-CFL GRC VERSION (hardwaretest03)")
        print("="*50)
        print(f"Clients: {n_homes} | Max payload: {LORA_MAX_BYTES} bytes")
        print(f"Transport: GNU Radio Companion flowgraph (simulation)")
        print("="*50)

    def wait_for_homes(self, day_num, timeout=300):
        """Wait for homes to submit their compressed models."""
        ready_homes = set()
        start = time.time()

        while len(ready_homes) < self.n_homes:
            if time.time() - start > timeout:
                print(f"Timeout! Got {len(ready_homes)}/{self.n_homes}")
                break
            for home_id in range(1, self.n_homes + 1):
                if home_id not in ready_homes:
                    flag = self.lora_dir / f'home_{home_id:02d}_ready.flag'
                    if flag.exists():
                        try:
                            content = flag.read_text().strip()
                            if content == f'day_{day_num}':
                                ready_homes.add(home_id)
                        except:
                            pass
            time.sleep(1)

        return ready_homes

    def receive_from_home(self, home_id):
        """Read compressed model uploaded by home node."""
        upload_file = self.lora_dir / f'home_{home_id:02d}_upload.bin'
        if not upload_file.exists():
            return None, 0.0

        try:
            raw = upload_file.read_bytes()
            if len(raw) >= 22:
                compressed = self.lora_bridge.unpack_compressed(raw)
                decoded    = self.hegazy.decode_parameters(
                    compressed, compressed['a'], compressed['b']
                )
                print(f"[SERVER] Received Home {home_id:02d}: "
                      f"{len(raw)} bytes | zeta={compressed['zeta_i']:.6f}")
                return decoded, compressed['zeta_i']
        except Exception as e:
            print(f"[SERVER] Error reading Home {home_id:02d}: {e}")

        return None, 0.0

    def broadcast_global_model(self, global_flat, day_num):
        """Compress and write global model for homes to read."""
        a, b       = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters([global_flat], 0, a, b)
        payload    = self.lora_bridge.pack_compressed(compressed)

        # Write to file for homes to read
        global_file = self.lora_dir / 'global_model.bin'
        global_file.write_bytes(payload)

        # Also transmit through GRC flowgraph (simulation)
        print(f"[SERVER] Broadcasting {len(payload)} bytes Day {day_num} via GRC...")
        result = self.radio.transmit(payload)
        print(f"[SERVER] Broadcast {'SUCCESS' if result['success'] else 'FAILED'}")

    def cleanup_day(self, day_num):
        """Clean up flag and upload files."""
        time.sleep(2)
        for home_id in range(1, self.n_homes + 1):
            for suffix in ['_ready.flag', '_upload.bin']:
                f = self.lora_dir / f'home_{home_id:02d}{suffix}'
                try:
                    f.unlink()
                except:
                    pass
        # Remove global model file
        try:
            (self.lora_dir / 'global_model.bin').unlink()
        except:
            pass

    def run_day(self, day_num):
        print(f"\n{'='*50}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*50}")

        ready_homes = self.wait_for_homes(day_num)
        print(f"Day {day_num}: {len(ready_homes)} homes ready")

        if not ready_homes:
            return

        client_params_dict = {}
        zeta_values        = {}

        for home_id in ready_homes:
            decoded, zeta_i = self.receive_from_home(home_id)
            if decoded is not None:
                client_params_dict[home_id] = decoded
                zeta_values[home_id]        = zeta_i

        if not client_params_dict:
            return

        # ME-CFL aggregation
        global_flat = self.server.aggregate_round(client_params_dict, day_num)

        if zeta_values:
            print(f"[SERVER] Avg zeta: {np.mean(list(zeta_values.values())):.6f}")

        # Broadcast back
        self.broadcast_global_model(global_flat, day_num)

        self.daily_results.append({
            'day':            day_num,
            'n_participants': len(client_params_dict),
            'avg_zeta':       np.mean(list(zeta_values.values())) if zeta_values else 0.0
        })

        self.cleanup_day(day_num)

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print("\n" + "="*50)
        print("FINAL SERVER SUMMARY - ME-CFL GRC")
        print("="*50)
        for r in self.daily_results:
            print(f"Day {r['day']}: {r['n_participants']} homes | "
                  f"Zeta: {r['avg_zeta']:.6f}")
        self.server.get_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes', type=int, default=1)
    parser.add_argument('--days',    type=int, default=7)
    args = parser.parse_args()
    ServerAggregator(n_homes=args.n_homes, n_days=args.days).run()


if __name__ == "__main__":
    main()
