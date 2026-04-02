"""
Server Aggregator - ME-CFL USRP hardwaretest11
================================================
3 homes + 1 server, 4 USRP B200s.
TDMA: server signals each home's slot, listens during each slot,
aggregates after all 3 homes done, broadcasts global model.

TDMA per day:
  1. Wait for all 3 homes ready
  2. Signal Home 1 slot -> wait for Home 1 TX done
  3. Signal Home 2 slot -> wait for Home 2 TX done
  4. Signal Home 3 slot -> wait for Home 3 TX done
  5. Aggregate all 3 models
  6. Broadcast global model (TX only)
  7. Write global_model_day{N}.bin + server_tx_done flag
  8. Cleanup

File: server_aggregator.py
"""
import numpy as np
import sys
import time
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'compression'))
sys.path.insert(0, str(Path(__file__).parent / 'lora'))
sys.path.insert(0, str(Path(__file__).parent / 'server'))

from hegazy import AggregateGaussianMechanism
from hegazy_lora_bridge import HegazyLoRaBridge
from aggregate import FederatedServer
from gr_lora_usrp import get_server_radio, LORA_MAX_BYTES, SLOT_DURATION, N_HOMES


class ServerAggregator:

    def __init__(self, n_homes=3, n_days=7, tx_serial=None):
        self.n_homes  = n_homes
        self.n_days   = n_days
        self.lora_dir = Path(__file__).parent / 'lora'

        # Server gets its own lora dir
        self.server_lora_dir = Path(__file__).parent / 'lora_server'
        self.server_lora_dir.mkdir(exist_ok=True)

        # Copy Chen's TX file to server lora dir
        for fname in ['lora_TX.py', 'lora_RX.py', 'lora_TX_epy_block_0_0.py']:
            src = self.lora_dir / fname
            dst = self.server_lora_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)

        self.radio      = get_server_radio(work_dir=str(self.server_lora_dir),
                                           tx_serial=tx_serial)
        self.bridge     = HegazyLoRaBridge()
        self.hegazy     = AggregateGaussianMechanism(n_clients=n_homes, sigma=0.1, seed=0)
        self.aggregator = FederatedServer(n_clients=n_homes, alpha=0.25, beta=0.9, eta=0.01)
        self.daily_summary = []

        print(f"\n{'='*60}")
        print(f"Federated Server - ME-CFL USRP hardwaretest11")
        print(f"{'='*60}")
        print(f"Clients: {n_homes} | Transport: REAL USRP B200")
        print(f"TDMA: {SLOT_DURATION}s per home slot | {n_homes} homes")
        print(f"{'='*60}")

    def wait_for_all_homes_ready(self, day_num, timeout=600):
        """Wait for all homes to signal they are trained and compressed."""
        ready = {}
        print(f"\n{'='*60}\n--- SERVER DAY {day_num} ---\n{'='*60}")
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

    def run_tdma_slot(self, home_id, day_num, timeout=120):
        """Signal home to TX in its slot, wait for it to finish."""
        # Signal this home's slot
        slot_flag = self.lora_dir / f'tdma_slot_{home_id}_day{day_num}.flag'
        slot_flag.write_text("go")
        print(f"[SERVER] TDMA slot {home_id} started (Day {day_num})")

        # Wait for home TX done
        tx_done_flag = self.lora_dir / f'home_{home_id:02d}_tx_done_day{day_num}.flag'
        start = time.time()
        while time.time() - start < timeout:
            if tx_done_flag.exists():
                print(f"[SERVER] Home {home_id:02d} TX done.")
                return True
            time.sleep(1)
        print(f"[SERVER] Timeout waiting for Home {home_id:02d} TX done.")
        return False

    def aggregate(self, ready_homes, day_num):
        a, b = self.hegazy.decompose()
        all_params = []
        all_zetas  = []
        for h_id, compressed in ready_homes.items():
            all_params.append(self.hegazy.decode_parameters(compressed, a, b))
            all_zetas.append(compressed['zeta_i'])

        global_params = self.aggregator.aggregate_round(
            dict(enumerate(all_params)), day_num)

        avg_zeta = np.mean(all_zetas)
        n_rf = len(ready_homes)
        pdr  = n_rf / self.n_homes * 100
        print(f"[SERVER] Avg zeta: {avg_zeta:.6f} | Homes received: {n_rf}/{self.n_homes} | PDR: {pdr:.1f}%")

        self.daily_summary.append({
            'day': day_num, 'n_homes': n_rf,
            'avg_zeta': avg_zeta, 'pdr': pdr
        })
        return global_params

    def broadcast_global(self, global_params, day_num):
        """Server TX: broadcast global model. Homes read from file."""
        a, b       = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters([global_params], client_id=0, a=a, b=b)
        payload    = self.bridge.pack_compressed(compressed)

        print(f"[SERVER] Broadcasting {len(payload)} bytes Day {day_num}...")
        result = self.radio.transmit_only(payload)
        print(f"[SERVER] Broadcast {'SUCCESS' if result['success'] else 'FAILED'}")

        # Write global model file for all homes to read
        global_file = self.lora_dir / f'global_model_day{day_num}.bin'
        global_file.write_bytes(payload)
        print(f"[SERVER] Global model written: {global_file.name}")

    def cleanup_day(self, day_num):
        for h in range(1, self.n_homes + 1):
            for fname in [
                f'home_{h:02d}_ready.flag',
                f'home_{h:02d}_upload.bin',
                f'home_{h:02d}_tx_done_day{day_num}.flag',
                f'tdma_slot_{h}_day{day_num}.flag',
            ]:
                f = self.lora_dir / fname
                if f.exists():
                    try: f.unlink()
                    except: pass

        if day_num > 1:
            old = self.lora_dir / f'global_model_day{day_num-1}.bin'
            if old.exists():
                try: old.unlink()
                except: pass

    def run_day(self, day_num):
        # 1. Wait for all homes ready
        ready_homes = self.wait_for_all_homes_ready(day_num)
        if not ready_homes:
            print(f"[SERVER] No homes ready for Day {day_num}")
            return

        # 2. TDMA: signal each home slot one by one
        for home_id in range(1, self.n_homes + 1):
            if home_id in ready_homes:
                print(f"\n[SERVER] === TDMA SLOT {home_id} ===")
                self.run_tdma_slot(home_id, day_num)
            else:
                print(f"[SERVER] Home {home_id:02d} not ready — skipping slot")

        # 3. Aggregate
        global_params = self.aggregate(ready_homes, day_num)

        # 4. Broadcast global model
        self.broadcast_global(global_params, day_num)

        # 5. Cleanup
        self.cleanup_day(day_num)

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print(f"\n{'='*60}")
        print(f"FINAL SERVER SUMMARY - hardwaretest11")
        print(f"{'='*60}")
        for s in self.daily_summary:
            print(f"Day {s['day']}: {s['n_homes']} homes | "
                  f"Zeta: {s['avg_zeta']:.6f} | PDR: {s['pdr']:.1f}%")

        self.aggregator.get_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_homes',   type=int, default=3)
    parser.add_argument('--days',      type=int, default=7)
    parser.add_argument('--tx_serial', type=str, default=None,
                        help='Server USRP serial (overrides default)')
    args = parser.parse_args()
    ServerAggregator(n_homes=args.n_homes, n_days=args.days,
                     tx_serial=args.tx_serial).run()


if __name__ == "__main__":
    main()
