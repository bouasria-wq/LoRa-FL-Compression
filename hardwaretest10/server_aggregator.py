"""
Server Aggregator - ME-CFL USRP hardwaretest10
================================================
Uses lora_TX_usrp.py (file source) + lora_RX_usrp.py (Chen's RX).
Flag-based handshake for clean TX/RX role switching.

Server workflow per day:
  1. Wait for homes ready flag
  2. Wait for home TX done flag (home finished TX, USRP free)
  3. Aggregate global model
  4. SERVER = TX: broadcast global model
  5. Write server_tx_done flag + global_model_day{N}.bin
  6. Cleanup

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
from aggregate import FederatedServer
from gr_lora_usrp import get_server_radio, LORA_MAX_BYTES


class ServerAggregator:

    def __init__(self, n_homes=1, n_days=7, tx_serial=None, rx_serial=None):
        self.n_homes  = n_homes
        self.n_days   = n_days
        self.lora_dir = Path(__file__).parent / 'lora'

        # Server gets its own work dir to avoid file conflicts with home
        self.server_lora_dir = Path(__file__).parent / 'lora_server'
        self.server_lora_dir.mkdir(exist_ok=True)

        self.radio  = get_server_radio(
            work_dir=str(self.server_lora_dir),
            tx_serial=tx_serial,
            rx_serial=rx_serial
        )
        self.bridge     = HegazyLoRaBridge()
        self.hegazy     = AggregateGaussianMechanism(n_clients=n_homes, sigma=0.1, seed=0)
        self.aggregator = FederatedServer(n_clients=n_homes, alpha=0.25, beta=0.9, eta=0.01)
        self.daily_summary = []

        print(f"\n{'='*50}")
        print(f"Federated Server - ME-CFL USRP hardwaretest10")
        print(f"{'='*50}")
        print(f"Clients: {n_homes} | Max payload: {LORA_MAX_BYTES} bytes")
        print(f"Transport: REAL USRP B200 | TX: file source flowgraph")
        print(f"{'='*50}")

    def wait_for_homes_ready(self, day_num, timeout=600):
        ready = {}
        print(f"\n{'='*50}")
        print(f"--- SERVER DAY {day_num} ---")
        print(f"{'='*50}")

        start = time.time()
        while len(ready) < self.n_homes and time.time() - start < timeout:
            for h in range(1, self.n_homes + 1):
                if h in ready:
                    continue
                flag   = self.lora_dir / f'home_{h:02d}_ready.flag'
                upload = self.lora_dir / f'home_{h:02d}_upload.bin'
                if flag.exists() and upload.exists():
                    try:
                        if flag.read_text().strip() == f'day_{day_num}':
                            raw        = upload.read_bytes()
                            compressed = self.bridge.unpack_compressed(raw)
                            ready[h]   = compressed
                            print(f"[SERVER] Home {h:02d} ready: "
                                  f"{len(raw)} bytes | zeta={compressed['zeta_i']:.6f}")
                    except Exception as e:
                        print(f"[SERVER] Error reading Home {h:02d}: {e}")
            time.sleep(1)

        print(f"Day {day_num}: {len(ready)}/{self.n_homes} homes ready")
        return ready

    def wait_for_home_tx_done(self, day_num, timeout=300):
        flag = self.lora_dir / f'home_tx_done_day{day_num}.flag'
        print(f"[SERVER] Waiting for home TX done (Day {day_num})...")
        start = time.time()
        while time.time() - start < timeout:
            if flag.exists():
                print(f"[SERVER] Home TX done — USRP free for server TX.")
                return True
            time.sleep(1)
        print(f"[SERVER] Timeout waiting for home TX done.")
        return False

    def aggregate(self, ready_homes, day_num):
        all_params = []
        all_zetas  = []

        a, b = self.hegazy.decompose()
        for h_id, compressed in ready_homes.items():
            params = self.hegazy.decode_parameters(compressed, a, b)
            all_params.append(params)
            all_zetas.append(compressed['zeta_i'])

        global_params = self.aggregator.aggregate_round(
            dict(enumerate(all_params)), day_num
        )

        avg_zeta = np.mean(all_zetas)
        print(f"[SERVER] Avg zeta: {avg_zeta:.6f}")

        self.daily_summary.append({
            'day': day_num, 'n_homes': len(ready_homes), 'avg_zeta': avg_zeta
        })
        return global_params

    def broadcast_global(self, global_params, day_num):
        """SERVER = TX: broadcast global model to homes."""
        a, b = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters([global_params], client_id=0, a=a, b=b)
        payload = self.bridge.pack_compressed(compressed)

        print(f"[SERVER] Broadcasting {len(payload)} bytes Day {day_num} via USRP B200...")
        result = self.radio.transmit(payload)

        if result['success']:
            print(f"[SERVER] Broadcast SUCCESS")
        else:
            print(f"[SERVER] Broadcast FAILED")

        # Write day-specific global model file for home to read
        global_file = self.lora_dir / f'global_model_day{day_num}.bin'
        global_file.write_bytes(payload)

        # Signal home that server TX is done
        (self.lora_dir / f'server_tx_done_day{day_num}.flag').write_text("done")
        print(f"[SERVER] Server TX done flag written for Day {day_num}.")

        return result['success']

    def cleanup_day(self, day_num):
        for h in range(1, self.n_homes + 1):
            for fname in [
                f'home_{h:02d}_ready.flag',
                f'home_{h:02d}_upload.bin',
                f'home_tx_done_day{day_num}.flag',
            ]:
                f = self.lora_dir / fname
                if f.exists():
                    try:
                        f.unlink()
                    except:
                        pass

        f = self.lora_dir / f'server_tx_done_day{day_num}.flag'
        if f.exists():
            try:
                f.unlink()
            except:
                pass

        # Clean up previous day's global model
        if day_num > 1:
            old = self.lora_dir / f'global_model_day{day_num-1}.bin'
            if old.exists():
                try:
                    old.unlink()
                except:
                    pass

    def run_day(self, day_num):
        # 1. Wait for homes ready
        ready_homes = self.wait_for_homes_ready(day_num)
        if not ready_homes:
            print(f"[SERVER] No homes ready for Day {day_num}")
            return

        # 2. Wait for home TX done
        self.wait_for_home_tx_done(day_num)

        # 3. Aggregate
        global_params = self.aggregate(ready_homes, day_num)

        # 4. Broadcast global model
        self.broadcast_global(global_params, day_num)

        # 5. Cleanup
        self.cleanup_day(day_num)

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print(f"\n{'='*50}")
        print(f"FINAL SERVER SUMMARY - hardwaretest10")
        print(f"{'='*50}")
        for s in self.daily_summary:
            print(f"Day {s['day']}: {s['n_homes']} homes | Zeta: {s['avg_zeta']:.6f}")

        self.aggregator.get_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes',   type=int, default=1)
    parser.add_argument('--days',      type=int, default=7)
    parser.add_argument('--tx_serial', type=str, default=None)
    parser.add_argument('--rx_serial', type=str, default=None)
    args = parser.parse_args()
    ServerAggregator(
        n_homes=args.n_homes, n_days=args.days,
        tx_serial=args.tx_serial, rx_serial=args.rx_serial,
    ).run()


if __name__ == "__main__":
    main()
