"""
Home Node - ME-CFL hardwaretestnew
====================================
Flow per day:
  1. Train locally
  2. Compress
  3. TX compressed model to server (32BBAD0 -> server RX 32BBAD9)
  4. Write home_tx_done flag
  5. Wait for server_tx_done flag
  6. Start RX (32BBAD9) to receive global model
  7. Read global model from file
  8. Apply momentum update
  9. Repeat
"""
import numpy as np
import sys
import time
import argparse
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'local_home'))
sys.path.insert(0, str(Path(__file__).parent / 'compression'))
sys.path.insert(0, str(Path(__file__).parent / 'lora'))
sys.path.insert(0, str(Path(__file__).parent / 'server'))

from load_data import DataLoader
from train import LocalTrainer
from hegazy import AggregateGaussianMechanism
from hegazy_lora_bridge import HegazyLoRaBridge
from gr_lora_usrp import tx, rx, HOME_TX_SERIAL, HOME_RX_SERIAL, bytes_to_ascii

LORA_DIR = Path(__file__).parent / 'lora'


def wait_flag(flag_path, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        if Path(flag_path).exists():
            return True
        time.sleep(1)
    return False


class HomeNode:
    def __init__(self, home_id, n_days=7, epochs=100):
        self.home_id  = home_id
        self.n_days   = n_days
        self.epochs   = epochs
        self.samples_per_day = 96

        print(f"\nHOME {home_id:02d} - ME-CFL hardwaretestnew")
        print(f"TX USRP: {HOME_TX_SERIAL} | RX USRP: {HOME_RX_SERIAL}")

        self.trainer     = LocalTrainer(home_id=home_id, sequence_length=16, learning_rate=0.0005)
        self.data_loader = DataLoader(data_dir='data', n_homes=10, n_days=n_days)
        self.df_full     = self.data_loader.load_home_data(home_id)
        self.bridge      = HegazyLoRaBridge()
        self.hegazy      = AggregateGaussianMechanism(n_clients=10, sigma=0.1, seed=home_id)
        self.momentum    = None
        self.beta        = 0.9
        self.eta         = 0.01
        self.prev_global = None
        self.metrics     = []

    def get_data(self, day_num):
        df = self.df_full.iloc[0:day_num * self.samples_per_day].copy()
        for col in ['T_indoor', 'T_outdoor']:
            if col in df.columns:
                df[col] = np.clip((df[col] + 50.0) / 100.0, 0, 1)
        return self.data_loader.get_features_target(df)

    def train(self, day_num):
        print(f"\n--- HOME {self.home_id:02d} | DAY {day_num} | {self.epochs} EPOCHS ---")
        X, y = self.get_data(day_num)
        Xs, ys = self.trainer.create_sequences(X, y)
        self.trainer.model.model.fit(Xs, ys, epochs=self.epochs,
            batch_size=16, validation_split=0.1, shuffle=True, verbose=0)
        m   = self.trainer.evaluate(Xs, ys)
        mae = m['mae'] * 100.0
        acc = m['accuracy']
        params = self.trainer.get_parameters()
        zeta   = self.hegazy.measure_heterogeneous_variance(
            np.concatenate([p.flatten() for p in params]))
        print(f"MAE {mae:.4f}C | Acc {acc:.2f}% | Zeta {zeta:.6f}")
        self.metrics.append({'day': day_num, 'mae': mae, 'acc': acc, 'zeta': zeta})
        return params

    def momentum_update(self, local_params, global_flat):
        local_flat = np.concatenate([p.flatten() for p in local_params])
        g_t = local_flat - global_flat
        if self.momentum is None:
            self.momentum = np.zeros_like(g_t)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * g_t
        return local_flat - self.eta * self.momentum

    def run_day(self, day_num):
        # 1. Train
        params = self.train(day_num)

        # 2. Compress
        a, b       = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters(params, self.home_id, a, b)
        payload    = self.bridge.pack_compressed(compressed)
        print(f"[HOME] Compressed: {len(payload)} bytes")

        # Write upload file
        (LORA_DIR / f'home_{self.home_id:02d}_upload.bin').write_bytes(payload)
        (LORA_DIR / f'home_{self.home_id:02d}_ready.flag').write_text(f"day_{day_num}")

        # 3. TX compressed model to server
        # Server is already listening (started RX when it saw ready flag)
        done_flag = LORA_DIR / f'home_{self.home_id:02d}_tx_done_day{day_num}.flag'
        print(f"[HOME] Phase 1: TX compressed model...")
        success, t_toa = tx(str(LORA_DIR), HOME_TX_SERIAL, payload,
                            done_flag_path=str(done_flag))
        print(f"[HOME] TX {'OK' if success else 'FAILED'} | ToA={t_toa:.3f}s")

        # 4. Wait for server to aggregate and TX global model
        print(f"[HOME] Waiting for server TX done...")
        if not wait_flag(LORA_DIR / f'server_tx_done_day{day_num}.flag'):
            print(f"[HOME] Timeout waiting for server.")
            return

        # 5. Start RX to receive global model at same time as reading file
        print(f"[HOME] Phase 2: Switching to RX mode...")
        rx_thread = threading.Thread(
            target=rx,
            args=(str(LORA_DIR), HOME_RX_SERIAL),
            kwargs={'duration': 8.0},
            daemon=True
        )
        rx_thread.start()

        # 6. Read global model from file
        global_file = LORA_DIR / f'global_model_day{day_num}.bin'
        global_flat = None
        start = time.time()
        while time.time() - start < 30:
            if global_file.exists():
                try:
                    raw        = global_file.read_bytes()
                    compressed = self.bridge.unpack_compressed(raw)
                    global_flat = self.hegazy.decode_parameters(
                        compressed, compressed['a'], compressed['b'])
                    print(f"[HOME] Global model received.")
                    break
                except Exception as e:
                    print(f"[HOME] Read error: {e}")
            time.sleep(1)

        rx_thread.join(timeout=15)

        # 7. Apply momentum update
        if global_flat is not None:
            if self.prev_global is not None:
                updated = self.momentum_update(params, global_flat)
                self.trainer.model.set_parameters(updated)
                print(f"[HOME] Momentum update applied.")
            else:
                self.trainer.model.set_parameters(global_flat)
            self.prev_global = global_flat

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)
        print(f"\n{'='*50}\nFINAL HOME {self.home_id:02d} SUMMARY\n{'='*50}")
        for m in self.metrics:
            print(f"Day {m['day']}: MAE {m['mae']:.4f}C | Acc {m['acc']:.2f}% | Zeta {m['zeta']:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--home_id', type=int, required=True)
    p.add_argument('--days',    type=int, default=7)
    p.add_argument('--epochs',  type=int, default=100)
    args = p.parse_args()
    HomeNode(args.home_id, args.days, args.epochs).run()


if __name__ == '__main__':
    main()
