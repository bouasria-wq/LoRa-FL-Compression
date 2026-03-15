"""
Federated Server Aggregator - ME-CFL Version
=============================================
Variance-reduced aggregation with global shift h^t
and momentum-based model updates.

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

from hegazy import AggregateGaussianMechanism
from hegazy_lora_bridge import HegazyLoRaBridge
from aggregate import FederatedServer


class ServerAggregator:

    def __init__(self, n_homes=10, n_days=7):
        self.n_homes = n_homes
        self.n_days = n_days

        self.data_dir = Path(__file__).parent / 'lora'
        self.data_dir.mkdir(exist_ok=True)

        self.lora_bridge = HegazyLoRaBridge()
        self.server = FederatedServer(
            n_clients=n_homes,
            alpha=0.25,
            beta=0.9,
            eta=0.01
        )
        self.hegazy = AggregateGaussianMechanism(n_clients=n_homes, sigma=0.1)

        self.daily_results = []

        print("\n" + "="*50)
        print("Federated Server Initialized - ME-CFL VERSION")
        print("="*50)
        print(f"Parameters: 553")
        print(f"Aggregation rate alpha: {self.server.alpha}")
        print(f"Momentum beta: {self.server.beta}")
        print(f"Learning rate eta: {self.server.eta}")
        print(f"Number of clients: {n_homes}")
        print("="*50)
        print("\nFEDERATED SERVER: ME-CFL MODE ACTIVE")
        print("Error Feedback + Variance Reduction + Momentum")
        print("="*50)

    def wait_for_homes(self, day_num, timeout=600):
        """Wait for all homes to submit updates."""
        ready_homes = set()
        start_time = time.time()

        while len(ready_homes) < self.n_homes:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"Timeout! Got {len(ready_homes)}/{self.n_homes} homes")
                break

            for home_id in range(1, self.n_homes + 1):
                if home_id not in ready_homes:
                    flag_file = self.data_dir / f'home_{home_id:02d}_ready.flag'
                    if flag_file.exists():
                        try:
                            with open(flag_file, 'r') as f:
                                content = f.read().strip()
                            if content == f'day_{day_num}':
                                ready_homes.add(home_id)
                        except:
                            pass
            time.sleep(2)

        return ready_homes

    def read_home_params(self, home_id):
        """Read parameters from home tx file."""
        tx_file = self.data_dir / f'home_{home_id:02d}_tx.txt'
        try:
            with open(tx_file, 'r') as f:
                data = pickle.loads(self.lora_bridge.lora_ascii_to_binary(f.read()))
            return data
        except Exception as e:
            print(f"Error reading home {home_id}: {e}")
            return None

    def broadcast_global_model(self, global_params, day_num):
        """Broadcast global model to all homes."""
        broadcast_file = self.data_dir / 'global_model_broadcast.txt'
        params_list = []
        idx = 0
        # Reconstruct as list of arrays matching LSTM structure
        shapes = [(4, 8, 8), (4, 8), (8, 1), (1,)]  # LSTM weights shapes approx
        for shape in shapes:
            size = np.prod(shape)
            if idx + size <= len(global_params):
                params_list.append(global_params[idx:idx+size].reshape(shape))
                idx += size

        if idx < len(global_params):
            params_list.append(global_params[idx:])

        broadcast_data = {'day': day_num, 'params': params_list}
        self.lora_bridge.write_lora_file(
            pickle.dumps(broadcast_data),
            str(broadcast_file)
        )

    def cleanup_day(self, day_num, ready_homes):
        """Clean up flag files after aggregation."""
        time.sleep(5)
        for home_id in ready_homes:
            flag_file = self.data_dir / f'home_{home_id:02d}_ready.flag'
            try:
                flag_file.unlink()
            except:
                pass

    def run_day(self, day_num):
        print(f"\n{'='*50}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*50}")

        # Wait for homes
        ready_homes = self.wait_for_homes(day_num)
        print(f"Day {day_num}: {len(ready_homes)} homes ready")

        if not ready_homes:
            return

        # Read params from all ready homes
        client_params_dict = {}
        zeta_values = {}

        for home_id in ready_homes:
            data = self.read_home_params(home_id)
            if data is not None:
                params = data.get('params')
                if params is not None:
                    flat = np.concatenate([p.flatten() for p in params])
                    client_params_dict[home_id] = flat
                    zeta_values[home_id] = data.get('zeta_i', 0.0)

        if not client_params_dict:
            return

        # ME-CFL aggregation
        global_flat = self.server.aggregate_round(client_params_dict, day_num)

        # Print zeta values
        if zeta_values:
            avg_zeta = np.mean(list(zeta_values.values()))
            print(f"Avg heterogeneous variance zeta: {avg_zeta:.6f}")

        # Broadcast global model
        self.broadcast_global_model(global_flat, day_num)
        print(f"Global Broadcast Complete for Day {day_num}.")

        self.daily_results.append({
            'day': day_num,
            'n_participants': len(ready_homes),
            'avg_zeta': np.mean(list(zeta_values.values())) if zeta_values else 0.0
        })

        self.cleanup_day(day_num, ready_homes)

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print("\n" + "="*40)
        print("FINAL SERVER SUMMARY - ME-CFL")
        print("="*40)
        for r in self.daily_results:
            print(f"Day {r['day']}: Aggregated {r['n_participants']} homes | "
                  f"Avg Zeta: {r['avg_zeta']:.6f}")
        self.server.get_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes', type=int, default=10)
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()
    ServerAggregator(n_homes=args.n_homes, n_days=args.days).run()


if __name__ == "__main__":
    main()
