"""
Individual Home Node - ME-CFL Hardware Version
================================================
Implements:
- Error feedback (ME-CFL Eq 7-8)
- Local shift tracking (ME-CFL Eq 9)
- Momentum updates (ME-CFL Eq 10)
- Heterogeneous variance measurement (ME-CFL Eq 3)
- Real GNU Radio LoRa TX/RX via USRP B200

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
from model import TemperatureLSTM
from train import LocalTrainer
from hegazy import AggregateGaussianMechanism
from hegazy_lora_bridge import HegazyLoRaBridge
from gr_lora_hardware import get_home_radio, PAYLOAD_LEN


class HomeNode:

    def __init__(self, home_id, n_days=7, epochs_per_day=100):
        self.home_id = home_id
        self.n_days = n_days
        self.epochs_per_day = epochs_per_day
        self.samples_per_day = 96

        print(f"\nHOME {self.home_id:02d} - ME-CFL HARDWARE VERSION")
        print(f"Error feedback: ON | Variance reduction: ON | Momentum: ON")
        print(f"Transport: Real LoRa over RF via USRP B200")
        print(f"Payload size: {PAYLOAD_LEN} bytes (struct.pack)")

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

        self.radio       = get_home_radio()
        self.lora_bridge = HegazyLoRaBridge()
        self.hegazy      = AggregateGaussianMechanism(
            n_clients=10, sigma=0.1, seed=home_id
        )

        # ME-CFL Eq 10: Momentum state
        self.momentum    = None
        self.beta        = 0.9
        self.eta         = 0.01
        self.prev_global = None

        self.daily_metrics = []

    def get_cumulative_data(self, day_num):
        df_cumulative = self.df_full.iloc[0:day_num * self.samples_per_day].copy()
        for col in ['T_indoor', 'T_outdoor']:
            if col in df_cumulative.columns:
                df_cumulative[col] = np.clip(
                    (df_cumulative[col] + 50.0) / 100.0, 0, 1
                )
        return self.data_loader.get_features_target(df_cumulative)

    def flatten_global(self, global_params):
        if isinstance(global_params, list):
            return np.concatenate([np.array(p).flatten() for p in global_params])
        elif isinstance(global_params, np.ndarray):
            return global_params.flatten()
        return np.array(global_params).flatten()

    def apply_momentum_update(self, local_params, global_flat):
        """ME-CFL Eq 10: Momentum update in Hilbert space."""
        local_flat = np.concatenate([p.flatten() for p in local_params])
        g_t = local_flat - global_flat
        if self.momentum is None:
            self.momentum = np.zeros_like(g_t)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * g_t
        return local_flat - self.eta * self.momentum

    def train_on_day(self, day_num):
        print(f"\n--- HOME {self.home_id:02d} | DAY {day_num} | "
              f"{self.epochs_per_day} EPOCHS | ME-CFL ---")

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
              f"Accuracy: {accuracy:.2f}% | "
              f"Zeta_i: {zeta_i:.6f}")

        self.daily_metrics.append({
            'day': day_num, 'mae': actual_mae,
            'accuracy': accuracy, 'zeta_i': zeta_i
        })
        return params

    def transmit_via_lora(self, payload: bytes) -> bool:
        """Transmit struct.packed compressed model over RF via USRP B200."""
        assert len(payload) == PAYLOAD_LEN, \
            f"Expected {PAYLOAD_LEN} bytes, got {len(payload)}"
        print(f"[HOME {self.home_id:02d}] TX {PAYLOAD_LEN} bytes over RF...")
        self.radio.transmit(payload)
        print(f"[HOME {self.home_id:02d}] TX complete.")
        return True

    def wait_for_global_model(self, day_num, timeout=180):
        """Receive global model from server via RF RX."""
        print(f"[HOME {self.home_id:02d}] Waiting for global model (Day {day_num})...")
        raw = self.radio.receive(timeout=timeout)

        if len(raw) == PAYLOAD_LEN:
            try:
                data = self.lora_bridge.unpack_compressed(raw)
                if data.get('day') == day_num:
                    print(f"[HOME {self.home_id:02d}] Global model received Day {day_num}.")
                    return data['params']
            except Exception as e:
                print(f"[HOME {self.home_id:02d}] Decode error: {e}")
        else:
            print(f"[HOME {self.home_id:02d}] No global model received (timeout).")
        return None

    def run_day(self, day_num):
        params = self.train_on_day(day_num)

        # ME-CFL Eq 7-8: Compress with error feedback
        a, b       = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters(params, self.home_id, a, b)

        # Pack using struct instead of pickle — achieves real 238-byte payload
        payload = self.lora_bridge.pack_compressed(compressed)

        # Transmit over RF
        self.transmit_via_lora(payload)

        # Wait for global model from server
        global_params = self.wait_for_global_model(day_num)

        if global_params is not None:
            global_flat = self.flatten_global(global_params)
            if self.prev_global is not None:
                updated_flat = self.apply_momentum_update(params, global_flat)
                self.trainer.model.set_parameters(updated_flat)
                print(f"[HOME {self.home_id:02d}] Momentum update applied (beta={self.beta})")
            else:
                self.trainer.model.set_parameters(global_flat)
            self.prev_global = global_flat

        time.sleep(2)

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)

        print("\n" + "="*40)
        print(f"FINAL HOME {self.home_id:02d} SUMMARY - ME-CFL HARDWARE")
        print("="*40)
        for m in self.daily_metrics:
            print(f"Day {m['day']}: MAE {m['mae']:.4f}°C | "
                  f"Acc {m['accuracy']:.2f}% | "
                  f"Zeta_i {m['zeta_i']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_id', type=int, required=True)
    parser.add_argument('--days',    type=int, default=7)
    parser.add_argument('--epochs',  type=int, default=100)
    args = parser.parse_args()
    HomeNode(
        home_id=args.home_id,
        n_days=args.days,
        epochs_per_day=args.epochs
    ).run()


if __name__ == "__main__":
    main()
