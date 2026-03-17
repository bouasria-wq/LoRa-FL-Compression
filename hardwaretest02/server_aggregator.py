"""
Federated Server Aggregator - ME-CFL Hardware Version
======================================================
Transport: Real LoRa over RF via USRP B200
Payload: 238 bytes struct.pack (Hegazy compressed)

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
from gr_lora_hardware import get_server_radio, PAYLOAD_LEN


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
            n_clients=n_homes, sigma=0.1
        )
        self.radio = get_server_radio()

        self.daily_results = []

        print("\n" + "="*50)
        print("Federated Server - ME-CFL HARDWARE")
        print("="*50)
        print(f"Clients: {n_homes} | Payload: {PAYLOAD_LEN} bytes (struct.pack)")
        print(f"Transport: Real LoRa over RF via USRP B200")
        print("="*50)

    def receive_from_home(self, home_id, day_num, timeout=180):
        """Receive struct.packed compressed model from home via RF RX."""
        print(f"[SERVER] Waiting for Home {home_id:02d} Day {day_num}...")
        raw = self.radio.receive(timeout=timeout)

        if len(raw) == PAYLOAD_LEN:
            try:
                data = self.lora_bridge.unpack_compressed(raw)
                print(f"[SERVER] Received from Home {home_id:02d}.")
                return data
            except Exception as e:
                print(f"[SERVER] Decode error from Home {home_id:02d}: {e}")
        else:
            print(f"[SERVER] No data from Home {home_id:02d} (timeout).")
        return None

    def broadcast_global_model(self, global_params, day_num):
        """Broadcast global model to home via RF TX using struct.pack."""
        print(f"[SERVER] Broadcasting global model Day {day_num}...")

        # Pack global params for transmission
        broadcast = {
            'client_id':  0,
            'm_k':        (global_params[:55] * 1000).astype(np.int16),
            'dither':     np.zeros(55),
            'indices':    np.arange(55, dtype=np.uint16),
            'p_min':      np.float32(global_params.min()),
            'scale':      np.float32(global_params.max() - global_params.min()),
            'param_size': 553,
            'zeta_i':     0.0,
            'a':          float(day_num),   # encode day in 'a' field
            'b':          0.0,
        }

        payload = self.lora_bridge.pack_compressed(broadcast)
        self.radio.transmit(payload)
        print(f"[SERVER] Broadcast complete Day {day_num}.")

    def run_day(self, day_num):
        print(f"\n{'='*50}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*50}")

        client_params_dict = {}
        zeta_values        = {}

        data = self.receive_from_home(home_id=1, day_num=day_num)

        if data is not None:
            # Decode parameters from compressed dict
            decoded = self.hegazy.decode_parameters(data, data['a'], data['b'])
            client_params_dict[1] = decoded
            zeta_values[1]        = data.get('zeta_i', 0.0)

        if not client_params_dict:
            print(f"[SERVER] No updates received Day {day_num}!")
            return

        # ME-CFL aggregation
        global_flat = self.server.aggregate_round(client_params_dict, day_num)

        if zeta_values:
            print(f"[SERVER] Avg zeta: {np.mean(list(zeta_values.values())):.6f}")

        # Broadcast back to home
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
