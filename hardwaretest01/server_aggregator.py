"""
Federated Server Aggregator - ME-CFL Hardware Version RF ONLY
=============================================================
Everything goes through RF only.
No LAN, no WiFi, no sockets needed.

File: server_aggregator.py
"""
import numpy as np
import sys
import time
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'compression'))
sys.path.insert(0, str(Path(__file__).parent / 'server'))
sys.path.insert(0, str(Path(__file__).parent / 'lora'))

from hegazy import AggregateGaussianMechanism
from aggregate import FederatedServer
from usrp_lora import USRPServer


class ServerAggregator:

    def __init__(self, n_homes=4, n_days=7):
        self.n_homes = n_homes
        self.n_days = n_days

        self.usrp_server = USRPServer(n_homes=n_homes)
        self.server = FederatedServer(
            n_clients=n_homes,
            alpha=0.25,
            beta=0.9,
            eta=0.01
        )
        self.hegazy = AggregateGaussianMechanism(
            n_clients=n_homes,
            sigma=0.1
        )
        self.daily_results = []

        print("\n" + "="*50)
        print("Federated Server - ME-CFL HARDWARE RF ONLY")
        print("="*50)
        print(f"Parameters: 553")
        print(f"Alpha: {self.server.alpha} | "
              f"Beta: {self.server.beta} | "
              f"Eta: {self.server.eta}")
        print(f"Homes: {n_homes} | Days: {n_days}")
        print(f"Mode: REAL USRP RF ONLY — no LAN needed")
        print("="*50)

    def run_day(self, day_num):
        print(f"\n{'='*50}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*50}")

        # Receive compressed params from all homes via RF
        received = self.usrp_server.receive_all_homes(day_num)

        if not received:
            print(f"Day {day_num}: No homes received!")
            return

        print(f"Day {day_num}: {len(received)}/{self.n_homes} homes received")

        # Decode compressed params from each home
        client_params_dict = {}
        zeta_values = {}

        for home_id, compressed in received.items():
            try:
                a = compressed.get('a', 1.0)
                b = compressed.get('b', 0.0)
                params_flat = self.hegazy.decode_parameters(
                    compressed, a, b
                )
                client_params_dict[home_id] = params_flat
                zeta_values[home_id] = compressed.get('zeta_i', 0.0)

                print(f"Home {home_id}: "
                      f"Decoded {len(params_flat)} params | "
                      f"Zeta: {compressed.get('zeta_i', 0.0):.6f}")

            except Exception as e:
                print(f"Error decoding Home {home_id}: {e}")

        if not client_params_dict:
            print("No valid params decoded!")
            return

        # ME-CFL aggregation
        global_flat = self.server.aggregate_round(
            client_params_dict, day_num
        )

        if zeta_values:
            avg_zeta = np.mean(list(zeta_values.values()))
            print(f"Avg heterogeneous variance zeta: {avg_zeta:.6f}")

        # Compress global model for RF broadcast
        a_g, b_g = self.hegazy.decompose()

        # Create a fake params list structure for encode
        compressed_global = {
            'm_k': np.zeros(1, dtype=np.int32),
            'dither': np.zeros(1),
            'indices': np.arange(len(global_flat)),
            'p_min': float(global_flat.min()),
            'scale': float(global_flat.max() - global_flat.min()),
            'param_size': len(global_flat),
            'zeta_i': avg_zeta if zeta_values else 0.0,
            'a': a_g,
            'b': b_g,
            'params_flat': global_flat.tolist()
        }

        # Broadcast compressed global model via RF
        self.usrp_server.broadcast_global_compressed(
            compressed_global, day_num
        )
        print(f"Global Broadcast Complete for Day {day_num}.")

        self.daily_results.append({
            'day': day_num,
            'n_participants': len(received),
            'avg_zeta': avg_zeta if zeta_values else 0.0
        })

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print("\n" + "="*40)
        print("FINAL SERVER SUMMARY - ME-CFL HARDWARE RF ONLY")
        print("="*40)
        for r in self.daily_results:
            print(f"Day {r['day']}: "
                  f"Aggregated {r['n_participants']} homes | "
                  f"Avg Zeta: {r['avg_zeta']:.6f}")
        self.server.get_summary()

        # Print final RF statistics
        stats = self.usrp_server.get_statistics()
        print("\n" + "="*40)
        print("RF COMMUNICATION STATISTICS")
        print("="*40)
        print(f"Uplink PDR: {stats['uplink_pdr']*100:.1f}%")
        print(f"Downlink PDR: {stats['downlink_pdr']*100:.1f}%")
        if stats['avg_snr']:
            print(f"Avg SNR: {stats['avg_snr']:.2f}dB")
        print(f"Avg Retries: {stats['avg_retries']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes', type=int, default=4)
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--serial', type=str, default=None)
    args = parser.parse_args()
    
    if args.serial:
        import os
        os.environ['USRP_SERIAL'] = args.serial
    
    ServerAggregator(n_homes=args.n_homes, n_days=args.days).run()
