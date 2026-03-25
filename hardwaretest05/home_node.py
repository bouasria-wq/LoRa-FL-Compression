"""
Individual Home Node - ME-CFL USRP TDMA Version (hardwaretest05)
=================================================================
3 homes + 1 server, 4 USRP B200s total.
Each home transmits in its assigned TDMA time slot to avoid collisions.

Slot layout per day:
  Home 1 TX:  0s  .. 30s
  Home 2 TX:  30s .. 60s
  Home 3 TX:  60s .. 90s
  Server aggregates + broadcasts: 90s .. 120s
  All homes receive global model:  120s+

File: home_node.py
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

from load_data import DataLoader
from train import LocalTrainer
from hegazy import AggregateGaussianMechanism
from hegazy_lora_bridge import HegazyLoRaBridge
from gr_lora_usrp import get_home_radio, LORA_MAX_BYTES, SLOT_DURATION, N_HOMES


class HomeNode:

    def __init__(self, home_id, n_days=7, epochs_per_day=100,
                 tx_serial=None, rx_serial=None):
        self.home_id        = home_id
        self.n_days         = n_days
        self.epochs_per_day = epochs_per_day
        self.samples_per_day = 96

        print(f"\nHOME {self.home_id:02d} - ME-CFL USRP TDMA (hardwaretest05)")
        print(f"Error feedback: ON | Variance reduction: ON | Momentum: ON")
        print(f"Transport: REAL USRP B200 | TDMA slot: {home_id}/{N_HOMES}")
        print(f"Slot window: {(home_id-1)*SLOT_DURATION}s - {home_id*SLOT_DURATION}s")
        print(f"Max payload: {LORA_MAX_BYTES} bytes")

        self.trainer = LocalTrainer(
            home_id=home_id,
            sequence_length=16,
            learning_rate=0.0005
        )
        self.data_loader = DataLoader(
            data_dir='data',
            n_homes=10,
            n_days=n_days
        )
        self.df_full = self.data_loader.load_home_data(home_id)

        lora_dir = Path(__file__).parent / 'lora'
        self.radio = get_home_radio(
            work_dir=str(lora_dir),
            home_id=home_id,
            tx_serial=tx_serial,
            rx_serial=rx_serial,
        )
        self.lora_bridge = HegazyLoRaBridge()
        self.hegazy      = AggregateGaussianMechanism(
            n_clients=10, sigma=0.1, seed=home_id
        )

        # ME-CFL Eq 10: Momentum
        self.momentum    = None
        self.beta        = 0.9
        self.eta         = 0.01
        self.prev_global = None
        self.daily_metrics = []

    def get_cumulative_data(self, day_num):
        df = self.df_full.iloc[0:day_num * self.samples_per_day].copy()
        for col in ['T_indoor', 'T_outdoor']:
            if col in df.columns:
                df[col] = np.clip((df[col] + 50.0) / 100.0, 0, 1)
        return self.data_loader.get_features_target(df)

    def apply_momentum_update(self, local_params, global_flat):
        local_flat = np.concatenate([p.flatten() for p in local_params])
        g_t = local_flat - global_flat
        if self.momentum is None:
            self.momentum = np.zeros_like(g_t)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * g_t
        return local_flat - self.eta * self.momentum

    def train_on_day(self, day_num):
        print(f"\n--- HOME {self.home_id:02d} | DAY {day_num} | "
              f"{self.epochs_per_day} EPOCHS | ME-CFL USRP TDMA ---")

        X_cum, y_cum = self.get_cumulative_data(day_num)
        X_seq, y_seq = self.trainer.create_sequences(X_cum, y_cum)

        self.trainer.model.model.fit(
            X_seq, y_seq,
            epochs=self.epochs_per_day,
            batch_size=16,
            validation_split=0.1,
            shuffle=True,
            verbose=0
        )

        metrics    = self.trainer.evaluate(X_seq, y_seq)
        actual_mae = metrics['mae'] * 100.0
        temp_range = y_seq.max() - y_seq.min()
        accuracy   = (1 - metrics['mae'] / temp_range) * 100 if temp_range > 0 else 0.0

        params      = self.trainer.get_parameters()
        params_flat = np.concatenate([p.flatten() for p in params])
        zeta_i      = self.hegazy.measure_heterogeneous_variance(params_flat)

        print(f"Result: MAE {actual_mae:.4f}°C | "
              f"Accuracy: {accuracy:.2f}% | Zeta_i: {zeta_i:.6f}")

        self.daily_metrics.append({
            'day': day_num, 'mae': actual_mae,
            'accuracy': accuracy, 'zeta_i': zeta_i
        })
        return params

    def wait_for_day_sync(self, day_num, timeout=300):
        """
        Wait for the server to signal that the day round has started.
        Returns the day_start_time for TDMA synchronization.
        """
        sync_file = Path(__file__).parent / 'lora' / f'day_{day_num}_sync.flag'
        print(f"[HOME {self.home_id:02d}] Waiting for Day {day_num} sync...")

        start = time.time()
        while time.time() - start < timeout:
            if sync_file.exists():
                try:
                    day_start = float(sync_file.read_text().strip())
                    print(f"[HOME {self.home_id:02d}] Day {day_num} sync received")
                    return day_start
                except:
                    pass
            time.sleep(0.5)

        print(f"[HOME {self.home_id:02d}] Sync timeout — using current time")
        return time.time()

    def wait_for_global_model(self, day_num, timeout=120):
        """Wait for server to write global model file."""
        global_file = Path(__file__).parent / 'lora' / f'global_day_{day_num}.bin'
        print(f"[HOME {self.home_id:02d}] Waiting for global model (Day {day_num})...")

        start = time.time()
        while time.time() - start < timeout:
            if global_file.exists():
                try:
                    raw = global_file.read_bytes()
                    if len(raw) >= 22:
                        compressed = self.lora_bridge.unpack_compressed(raw)
                        global_flat = self.hegazy.decode_parameters(
                            compressed, compressed['a'], compressed['b']
                        )
                        print(f"[HOME {self.home_id:02d}] Global model received.")
                        return global_flat
                except Exception as e:
                    print(f"[HOME {self.home_id:02d}] Read error: {e}")
            time.sleep(2)

        print(f"[HOME {self.home_id:02d}] Timeout waiting for global model.")
        return None

    def run_day(self, day_num):
        # 1. Train locally
        params = self.train_on_day(day_num)

        # 2. Compress with Hegazy
        a, b = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters(params, self.home_id, a, b)
        payload = self.lora_bridge.pack_compressed(compressed)
        print(f"[HOME {self.home_id:02d}] Compressed: {len(payload)} bytes")

        # 3. Write upload file (server reads this)
        home_upload = Path(__file__).parent / 'lora' / f'home_{self.home_id:02d}_upload.bin'
        home_upload.write_bytes(payload)

        # 4. Signal ready
        ready_file = Path(__file__).parent / 'lora' / f'home_{self.home_id:02d}_ready.flag'
        ready_file.write_text(f"day_{day_num}")

        # 5. Wait for day sync (server starts TDMA round)
        day_start = self.wait_for_day_sync(day_num)

        # 6. Transmit in our TDMA slot
        print(f"[HOME {self.home_id:02d}] TX in TDMA slot {self.home_id}...")
        tx_result = self.radio.transmit_tdma(payload, day_start)
        print(f"[HOME {self.home_id:02d}] TX {'SUCCESS' if tx_result['success'] else 'FAILED'} | "
              f"ToA: {tx_result['t_toa']:.4f}s | PDR: {tx_result['pdr']*100:.1f}%")

        # 7. Wait for global model from server
        global_flat = self.wait_for_global_model(day_num)

        if global_flat is not None:
            if self.prev_global is not None:
                updated_flat = self.apply_momentum_update(params, global_flat)
                self.trainer.model.set_parameters(updated_flat)
                print(f"[HOME {self.home_id:02d}] Momentum update (beta={self.beta})")
            else:
                self.trainer.model.set_parameters(global_flat)
            self.prev_global = global_flat

        time.sleep(1)

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print(f"\n{'='*50}")
        print(f"FINAL HOME {self.home_id:02d} SUMMARY - ME-CFL USRP TDMA")
        print(f"{'='*50}")
        for m in self.daily_metrics:
            print(f"Day {m['day']}: MAE {m['mae']:.4f}°C | "
                  f"Acc {m['accuracy']:.2f}% | Zeta_i {m['zeta_i']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_id',   type=int, required=True)
    parser.add_argument('--days',      type=int, default=7)
    parser.add_argument('--epochs',    type=int, default=100)
    parser.add_argument('--tx_serial', type=str, default=None)
    parser.add_argument('--rx_serial', type=str, default=None)
    args = parser.parse_args()
    HomeNode(
        home_id=args.home_id,
        n_days=args.days,
        epochs_per_day=args.epochs,
        tx_serial=args.tx_serial,
        rx_serial=args.rx_serial,
    ).run()


if __name__ == "__main__":
    main()
