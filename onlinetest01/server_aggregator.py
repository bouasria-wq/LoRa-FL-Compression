"""
Central Server Aggregator
=========================
Federated Averaging with High Convergence Mode.
"""
import numpy as np
import sys
import time
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'compression'))
sys.path.insert(0, str(Path(__file__).parent / 'lora'))
sys.path.insert(0, str(Path(__file__).parent / 'server'))

from hegazy_lora_bridge import HegazyLoRaBridge
from aggregate import FederatedServer


class ServerAggregator:
    def __init__(self, n_homes=10, n_days=7):
        self.n_homes = n_homes
        self.n_days = n_days
        self.aggregator = FederatedServer(n_params=553, alpha=0.25, n_clients=n_homes)
        self.lora_bridge = HegazyLoRaBridge()
        self.data_dir = Path(__file__).parent / 'lora'
        self.global_model_file = self.data_dir / 'global_model_broadcast.txt'

        print("\n" + "="*60 + "\nFEDERATED SERVER: HIGH CONVERGENCE MODE ACTIVE\n" + "="*60)

        for f in self.data_dir.glob("*.flag"):
            f.unlink()
        if self.global_model_file.exists():
            self.global_model_file.unlink()

        self.history = []

    def run_day(self, day_num):
        print(f"\n--- SERVER DAY {day_num} ---")
        ready_homes = set()
        start_time = time.time()

        while len(ready_homes) < self.n_homes and (time.time() - start_time < 600):
            for h_id in range(1, self.n_homes + 1):
                rf = self.data_dir / f'home_{h_id:02d}_ready.flag'
                if rf.exists():
                    try:
                        with open(rf, 'r') as f:
                            if f.read().strip() == f"day_{day_num}":
                                ready_homes.add(h_id)
                    except:
                        pass
            time.sleep(2)

        received_params = []
        for h_id in ready_homes:
            tf = self.data_dir / f'home_{h_id:02d}_tx.txt'
            try:
                with open(tf, 'r') as f:
                    data = pickle.loads(self.lora_bridge.lora_ascii_to_binary(f.read()))
                    if data.get('day') == day_num:
                        received_params.append(data['params'])
            except:
                pass

        if received_params:
            print(f"Aggregating {len(received_params)} nodes...")
            global_p = self.aggregator.aggregate_round(received_params)
            self.lora_bridge.write_lora_file(
                pickle.dumps({'day': day_num, 'params': global_p}),
                str(self.global_model_file)
            )
            print(f"Global Broadcast Complete for Day {day_num}.")
            self.history.append({'day': day_num, 'clients': len(received_params)})

        time.sleep(5)
        for h_id in range(1, self.n_homes + 1):
            flag = self.data_dir / f'home_{h_id:02d}_ready.flag'
            if flag.exists():
                flag.unlink()

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)
        print("\n" + "="*40 + "\nFINAL SERVER SUMMARY\n" + "="*40)
        for s in self.history:
            print(f"Day {s['day']}: Aggregated {s['clients']} homes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes', type=int, default=10)
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()
    ServerAggregator(n_homes=args.n_homes, n_days=args.days).run()


if __name__ == "__main__":
    main()
