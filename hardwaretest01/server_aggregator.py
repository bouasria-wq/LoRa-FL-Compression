"""
Federated Server Aggregator - ME-CFL Hardware Version
======================================================
Variance-reduced aggregation with global shift h^t
and momentum-based model updates.
Uses real LAN sockets instead of file system.

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
        self.hegazy = AggregateGaussianMechanism(n_clients=n_homes, sigma=0.1)
        self.daily_results = []

        print("\n" + "="*50)
        print("Federated Server - ME-CFL HARDWARE VERSION")
        print("="*50)
        print(f"Parameters: 553")
        print(f"Alpha: {self.server.alpha} | Beta: {self.server.beta} | Eta: {self.server.eta}")
        print(f"Homes: {n_homes} | Days: {n_days}")
        print(f"Mode: REAL USRP HARDWARE + LAN")
        print("="*50)

    def run_day(self, day_num):
        print(f"\n{'='*50}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*50}")

        # Receive from all homes via LAN
        received = self.usrp_server.receive_all_homes(day_num)

        if not received:
            print(f"Day {day_num}: No homes received!")
            return

        print(f"Day {day_num}: {len(received)}/{self.n_homes} homes received")

        # Extract params from received packets
        client_params_dict = {}
        zeta_values = {}

        for home_id, packet in received.items():
            try:
                payload = packet.get('payload')
                if payload is not None:
                    data = pickle.loads(payload)
                    params = data.get('params')
                    if params is not None:
                        flat = np.concatenate([p.flatten() for p in params])
                        client_params_dict[home_id] = flat
                        zeta_values[home_id] = data.get('zeta_i', 0.0)
                        print(f"Home {home_id}: "
                              f"{'RF+LAN' if packet.get('rf_success') else 'LAN only'} | "
                              f"Zeta: {data.get('zeta_i', 0.0):.6f}")
            except Exception as e:
                print(f"Error parsing Home {home_id}: {e}")

        if not client_params_dict:
            print("No valid params received!")
            return

        # ME-CFL aggregation
        global_flat = self.server.aggregate_round(client_params_dict, day_num)

        if zeta_values:
            avg_zeta = np.mean(list(zeta_values.values()))
            print(f"Avg heterogeneous variance zeta: {avg_zeta:.6f}")

        # Broadcast global model back to all homes via LAN
        self.usrp_server.broadcast_global_model(global_flat, day_num)
        print(f"Global Broadcast Complete for Day {day_num}.")

        self.daily_results.append({
            'day': day_num,
            'n_participants': len(received),
            'avg_zeta': np.mean(list(zeta_values.values())) if zeta_values else 0.0
        })

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print("\n" + "="*40)
        print("FINAL SERVER SUMMARY - ME-CFL HARDWARE")
        print("="*40)
        for r in self.daily_results:
            print(f"Day {r['day']}: "
                  f"Aggregated {r['n_participants']} homes | "
                  f"Avg Zeta: {r['avg_zeta']:.6f}")
        self.server.get_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes', type=int, default=4)
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()
    ServerAggregator(n_homes=args.n_homes, n_days=args.days).run()


if __name__ == "__main__":
    main()
