"""
Federated Server Aggregator - ME-CFL Hardware Version
======================================================
Transport: Real LoRa over RF via USRP B200
Both directions use Hegazy compression — single LoRa packet each way.

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
from gr_lora_hardware import get_server_radio, LORA_MAX_BYTES


class ServerAggregator:

    def __init__(self, n_homes=1, n_days=7):
        self.n_homes = n_homes
        self.n_days  = n_days

        self.lora_bridge = HegazyLoRaBridge()
        self.server = FederatedServer(
            n_clients=n_homes,
            alpha=0.25,
            beta=0.9,
            eta=0.01
        )
        self.hegazy = AggregateGaussianMechanism(
            n_clients=n_homes, sigma=0.1, seed=0
        )
        self.radio = get_server_radio()
        self.daily_results = []

        print("\n" + "="*50)
        print("Federated Server - ME-CFL HARDWARE")
        print("="*50)
        print(f"Clients: {n_homes} | Max payload: {LORA_MAX_BYTES} bytes")
        print(f"Transport: Real LoRa over RF via USRP B200")
        print(f"Compression: Hegazy both directions")
        print("="*50)

    def receive_from_home(self, home_id, day_num, timeout=180):
        """Receive Hegazy compressed model from home via RF RX."""
        print(f"[SERVER] Waiting for Home {home_id:02d} Day {day_num}...")
        raw = self.radio.receive(timeout=timeout)

        if len(raw) >= 18:
            try:
                compressed = self.lora_bridge.unpack_compressed(raw)
                decoded    = self.hegazy.decode_parameters(
                    compressed,
                    compressed['a'],
                    compressed['b']
                )
                print(f"[SERVER] Received from Home {home_id:02d}. "
                      f"zeta_i={compressed['zeta_i']:.6f}")
                return decoded, compressed['zeta_i']
            except Exception as e:
                print(f"[SERVER] Decode error from Home {home_id:02d}: {e}")
        else:
            print(f"[SERVER] No data from Home {home_id:02d} (timeout).")
        return None, 0.0

    def broadcast_global_model(self, global_flat, day_num):
        """Compress global model with Hegazy and broadcast via RF TX."""
        print(f"[SERVER] Compressing global model Day {day_num}...")

        a, b       = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters([global_flat], 0, a, b)
        payload    = self.lora_bridge.pack_compressed(compressed)

        assert len(payload) <= LORA_MAX_BYTES, \
            f"Global model payload {len(payload)} bytes exceeds LoRa max {LORA_MAX_BYTES}"

        print(f"[SERVER] Broadcasting {len(payload)} bytes Day {day_num}...")
        self.radio.transmit(payload)
        print(f"[SERVER] Broadcast complete Day {day_num}.")

    def run_day(self, day_num):
        print(f"\n{'='*50}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*50}")

        client_params_dict = {}
        zeta_values        = {}

        decoded, zeta_i = self.receive_from_home(home_id=1, day_num=day_num)

        if decoded is not None:
            client_params_dict[1] = decoded
            zeta_values[1]        = zeta_i

        if not client_params_dict:
            print(f"[SERVER] No updates received Day {day_num}!")
            return

        # ME-CFL aggregation
        global_flat = self.server.aggregate_round(client_params_dict, day_num)

        if zeta_values:
            print(f"[SERVER] Avg zeta: {np.mean(list(zeta_values.values())):.6f}")

        # Compress and broadcast back to home
        self.broadcast_global_model(global_flat, day_num)

        self.daily_results.append({
            'day':            day_num,
            'n_participants': len(client_params_dict),
            'avg_zeta':       np.mean(list(zeta_values.values())) if zeta_values else 0.0
        })

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print("\n" + "="*40)
        print("FINAL SERVER SUMMARY - ME-CFL HARDWARE")
        print("="*40)
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
