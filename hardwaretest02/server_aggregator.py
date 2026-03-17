"""
Federated Server Aggregator - ME-CFL Hardware Version
======================================================
Variance-reduced aggregation with global shift h^t
and momentum-based model updates.
Transport: Real LoRa over RF via USRP B200

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
from hegazy_lora_bridge import HegazyLoRaBridge
from aggregate import FederatedServer
from gr_lora_hardware import get_server_radio, PAYLOAD_LEN


class ServerAggregator:

    def __init__(self, n_homes=10, n_days=7):
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
            n_clients=n_homes,
            sigma=0.1
        )

        # Hardware radio
        self.radio = get_server_radio()

        self.daily_results = []

        print("\n" + "="*50)
        print("Federated Server Initialized - ME-CFL HARDWARE")
        print("="*50)
        print(f"Parameters: 553")
        print(f"Aggregation rate alpha: {self.server.alpha}")
        print(f"Momentum beta: {self.server.beta}")
        print(f"Learning rate eta: {self.server.eta}")
        print(f"Number of clients: {n_homes}")
        print(f"Transport: Real LoRa over RF via USRP B200")
        print("="*50)

    def receive_from_home(self, home_id, day_num, timeout=180):
        """
        Receive compressed model from home node via real RF RX.
        Uses gr_lora_hardware.LoRaHardware.receive()
        """
        print(f"[SERVER] Waiting for Home {home_id:02d} Day {day_num}...")

        raw = self.radio.receive(timeout=timeout)

        if len(raw) == PAYLOAD_LEN:
            try:
                data = pickle.loads(
                    self.lora_bridge.lora_ascii_to_binary(
                        raw.decode('latin-1')
                    )
                )
                print(f"[SERVER] Received from Home {home_id:02d}.")
                return data
            except Exception as e:
                print(f"[SERVER] Failed to decode from Home {home_id:02d}: {e}")
        else:
            print(f"[SERVER] No data received from Home {home_id:02d} (timeout).")

        return None

    def broadcast_global_model(self, global_params, day_num):
        """
        Broadcast global model to home node via real RF TX.
        Uses gr_lora_hardware.LoRaHardware.transmit()
        """
        print(f"[SERVER] Broadcasting global model for Day {day_num}...")

        broadcast_data = {'day': day_num, 'params': global_params}
        serialized = pickle.dumps(broadcast_data)
        ascii_str  = self.lora_bridge.binary_to_lora_ascii(serialized)

        # Encode to bytes and trim to PAYLOAD_LEN
        payload = ascii_str.encode('latin-1')[:PAYLOAD_LEN]
        # Pad if needed
        payload = payload.ljust(PAYLOAD_LEN, b'\x00')

        self.radio.transmit(payload)
        print(f"[SERVER] Global model broadcast complete for Day {day_num}.")

    def run_day(self, day_num):
        print(f"\n{'='*50}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*50}")

        client_params_dict = {}
        zeta_values        = {}

        # Since we only have 1 home USRP, receive once per day
        data = self.receive_from_home(home_id=1, day_num=day_num)

        if data is not None:
            params = data.get('params')
            if params is not None:
                flat = np.concatenate([p.flatten() for p in params])
                client_params_dict[1] = flat
                zeta_values[1]        = data.get('zeta_i', 0.0)

        if not client_params_dict:
            print(f"[SERVER] No client updates received for Day {day_num}!")
            return

        # ME-CFL aggregation
        global_flat = self.server.aggregate_round(client_params_dict, day_num)

        if zeta_values:
            avg_zeta = np.mean(list(zeta_values.values()))
            print(f"[SERVER] Avg heterogeneous variance zeta: {avg_zeta:.6f}")

        # Broadcast global model back to home via RF
        self.broadcast_global_model(global_flat, day_num)

        self.daily_results.append({
            'day':           day_num,
            'n_participants': len(client_params_dict),
            'avg_zeta':      np.mean(list(zeta_values.values())) if zeta_values else 0.0
        })

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print("\n" + "="*40)
        print("FINAL SERVER SUMMARY - ME-CFL HARDWARE")
        print("="*40)
        for r in self.daily_results:
            print(f"Day {r['day']}: Aggregated {r['n_participants']} homes | "
                  f"Avg Zeta: {r['avg_zeta']:.6f}")
        self.server.get_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes', type=int, default=1)
    parser.add_argument('--days',    type=int, default=7)
    args = parser.parse_args()

    ServerAggregator(
        n_homes=args.n_homes,
        n_days=args.days
    ).run()


if __name__ == "__main__":
    main()
