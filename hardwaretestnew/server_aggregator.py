"""
Server Aggregator - ME-CFL hardwaretestnew
============================================
Flow per day:
  1. Wait for home ready flag
  2. Start RX (32BBAD9) to receive home's compressed model
  3. Home TX arrives -> server receives
  4. Wait for home_tx_done flag
  5. Stop RX
  6. Aggregate + compress global model
  7. Write global_model file
  8. Start TX (32BBAD0) to send global model to home
  9. Write server_tx_done flag
  10. Repeat
"""
import numpy as np
import sys
import time
import argparse
import shutil
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'compression'))
sys.path.insert(0, str(Path(__file__).parent / 'lora'))
sys.path.insert(0, str(Path(__file__).parent / 'server'))

from hegazy import AggregateGaussianMechanism
from hegazy_lora_bridge import HegazyLoRaBridge
from aggregate import FederatedServer
from gr_lora_usrp import tx, rx, SERVER_TX_SERIAL, SERVER_RX_SERIAL

LORA_DIR        = Path(__file__).parent / 'lora'
SERVER_LORA_DIR = Path(__file__).parent / 'lora_server'


def wait_flag(flag_path, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        if Path(flag_path).exists():
            return True
        time.sleep(1)
    return False


class ServerAggregator:
    def __init__(self, n_homes=1, n_days=7):
        self.n_homes = n_homes
        self.n_days  = n_days

        SERVER_LORA_DIR.mkdir(exist_ok=True)
        for fname in ['lora_TX.py', 'lora_RX.py', 'lora_TX_epy_block_0_0.py']:
            src = LORA_DIR / fname
            dst = SERVER_LORA_DIR / fname
            if src.exists():
                shutil.copy(src, dst)

        self.bridge     = HegazyLoRaBridge()
        self.hegazy     = AggregateGaussianMechanism(n_clients=n_homes, sigma=0.1, seed=0)
        self.aggregator = FederatedServer(n_clients=n_homes, alpha=0.25, beta=0.9, eta=0.01)
        self.summary    = []

        print(f"\n{'='*50}")
        print(f"Server - ME-CFL hardwaretestnew")
        print(f"{'='*50}")
        print(f"Homes: {n_homes} | RX: {SERVER_RX_SERIAL} | TX: {SERVER_TX_SERIAL}")
        print(f"{'='*50}")

    def wait_homes_ready(self, day_num, timeout=600):
        ready = {}
        print(f"\n{'='*50}\n--- SERVER DAY {day_num} ---\n{'='*50}")
        start = time.time()
        while len(ready) < self.n_homes and time.time() - start < timeout:
            for h in range(1, self.n_homes + 1):
                if h in ready:
                    continue
                flag   = LORA_DIR / f'home_{h:02d}_ready.flag'
                upload = LORA_DIR / f'home_{h:02d}_upload.bin'
                if flag.exists() and upload.exists():
                    try:
                        if flag.read_text().strip() == f'day_{day_num}':
                            raw        = upload.read_bytes()
                            compressed = self.bridge.unpack_compressed(raw)
                            ready[h]   = compressed
                            print(f"[SERVER] Home {h:02d} ready: {len(raw)} bytes")
                    except Exception as e:
                        print(f"[SERVER] Read error Home {h:02d}: {e}")
            time.sleep(1)
        print(f"[SERVER] {len(ready)}/{self.n_homes} homes ready")
        return ready

    def aggregate(self, ready_homes, day_num):
        a, b = self.hegazy.decompose()
        all_params, all_zetas = [], []
        for h_id, compressed in ready_homes.items():
            all_params.append(self.hegazy.decode_parameters(compressed, a, b))
            all_zetas.append(compressed['zeta_i'])
        global_params = self.aggregator.aggregate_round(
            dict(enumerate(all_params)), day_num)
        avg_zeta = np.mean(all_zetas)
        print(f"[SERVER] Aggregated | Avg zeta: {avg_zeta:.6f}")
        self.summary.append({'day': day_num, 'n': len(ready_homes), 'zeta': avg_zeta})
        return global_params

    def run_day(self, day_num):
        # 1. Wait for homes ready
        ready_homes = self.wait_homes_ready(day_num)
        if not ready_homes:
            return

        # 2. Start RX to receive home TX
        print(f"[SERVER] Phase 1: Starting RX ({SERVER_RX_SERIAL})...")
        rx_thread = threading.Thread(
            target=rx,
            args=(str(SERVER_LORA_DIR), SERVER_RX_SERIAL),
            kwargs={'duration': 15.0},
            daemon=True
        )
        rx_thread.start()

        # 3. Wait for home TX done
        print(f"[SERVER] Waiting for home TX done...")
        if not wait_flag(LORA_DIR / f'home_01_tx_done_day{day_num}.flag'):
            print(f"[SERVER] Timeout waiting for home TX.")

        # Wait for RX thread to finish
        rx_thread.join(timeout=20)

        # 4. Aggregate
        global_params = self.aggregate(ready_homes, day_num)

        # 5. Write global model file
        a, b       = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters([global_params], client_id=0, a=a, b=b)
        payload    = self.bridge.pack_compressed(compressed)
        global_file = LORA_DIR / f'global_model_day{day_num}.bin'
        global_file.write_bytes(payload)
        print(f"[SERVER] Global model written: {len(payload)} bytes")

        # 6. TX global model to home
        done_flag = LORA_DIR / f'server_tx_done_day{day_num}.flag'
        print(f"[SERVER] Phase 2: TX global model ({SERVER_TX_SERIAL})...")
        success, t_toa = tx(str(SERVER_LORA_DIR), SERVER_TX_SERIAL, payload,
                            done_flag_path=str(done_flag))
        print(f"[SERVER] TX {'OK' if success else 'FAILED'} | ToA={t_toa:.3f}s")

        # 7. Cleanup old files
        self._cleanup(day_num)

    def _cleanup(self, day_num):
        for h in range(1, self.n_homes + 1):
            for fname in [f'home_{h:02d}_ready.flag',
                          f'home_{h:02d}_upload.bin',
                          f'home_{h:02d}_tx_done_day{day_num}.flag']:
                f = LORA_DIR / fname
                if f.exists():
                    try: f.unlink()
                    except: pass
        if day_num > 1:
            old = LORA_DIR / f'global_model_day{day_num-1}.bin'
            if old.exists():
                try: old.unlink()
                except: pass

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)
        print(f"\n{'='*50}\nFINAL SERVER SUMMARY\n{'='*50}")
        for s in self.summary:
            print(f"Day {s['day']}: {s['n']} homes | Zeta: {s['zeta']:.6f}")
        self.aggregator.get_summary()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_homes', type=int, default=1)
    p.add_argument('--days',    type=int, default=7)
    args = p.parse_args()
    ServerAggregator(args.n_homes, args.days).run()


if __name__ == '__main__':
    main()
