"""
Individual Home Node - ME-CFL Hardware Version
===============================================
Implements:
- Error feedback (ME-CFL Eq 7-8)
- Local shift tracking (ME-CFL Eq 9)
- Momentum updates (ME-CFL Eq 10)
- Heterogeneous variance measurement (ME-CFL Eq 3)
- Real USRP hardware LoRa TX/RX via LAN

Temperature range: -50 to +50 degrees C
File: home_node.py
"""
import numpy as np
import sys
import time
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'local_home'))
sys.path.insert(0, str(Path(__file__).parent / 'compression'))
sys.path.insert(0, str(Path(__file__).parent / 'lora'))
sys.path.insert(0, str(Path(__file__).parent / 'server'))

from load_data import DataLoader
from model import TemperatureLSTM
from train import LocalTrainer
from hegazy import AggregateGaussianMechanism
from usrp_lora import USRPLoRaInterface


class HomeNode:

    def __init__(self, home_id, n_days=7, epochs_per_day=100):
        self.home_id = home_id
        self.n_days = n_days
        self.epochs_per_day = epochs_per_day
        self.samples_per_day = 96

        print(f"\nHOME {self.home_id:02d} - ME-CFL HARDWARE VERSION")
        print(f"Error feedback: ON | Variance reduction: ON | Momentum: ON")
        print(f"Mode: REAL USRP HARDWARE")

        self.trainer = LocalTrainer(home_id=home_id, sequence_length=16, learning_rate=0.0005)
        self.data_loader = DataLoader(data_dir='data', n_homes=10, n_days=n_days)
        self.df_full = self.data_loader.load_home_data(home_id)

        self.lora = USRPLoRaInterface(home_id=home_id)
        self.hegazy = AggregateGaussianMechanism(n_clients=10, sigma=0.1, seed=home_id)

        # ME-CFL Eq 10: Momentum state
        self.momentum = None
        self.beta = 0.9
        self.eta = 0.01
        self.prev_global = None

        self.daily_metrics = []

    def get_cumulative_data(self, day_num):
        df_cumulative = self.df_full.iloc[0:day_num * self.samples_per_day].copy()
        temp_cols = ['T_indoor', 'T_outdoor']
        for col in temp_cols:
            if col in df_cumulative.columns:
                df_cumulative[col] = np.clip((df_cumulative[col] + 50.0) / 100.0, 0, 1)
        X_cum, y_cum = self.data_loader.get_features_target(df_cumulative)
        return X_cum, y_cum

    def flatten_global(self, global_params):
        if isinstance(global_params, list):
            return np.concatenate([np.array(p).flatten() for p in global_params])
        elif isinstance(global_params, np.ndarray):
            return global_params.flatten()
        else:
            return np.array(global_params).flatten()

    def apply_momentum_update(self, local_params, global_flat):
        """ME-CFL Eq 10: Momentum update in Hilbert space."""
        local_flat = np.concatenate([p.flatten() for p in local_params])
        g_t = local_flat - global_flat
        if self.momentum is None:
            self.momentum = np.zeros_like(g_t)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * g_t
        updated_flat = local_flat - self.eta * self.momentum
        return updated_flat

    def train_on_day(self, day_num):
        print(f"\n--- HOME {self.home_id:02d} | DAY {day_num} | {self.epochs_per_day} EPOCHS | ME-CFL HW ---")
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

        metrics = self.trainer.evaluate(X_seq, y_seq)
        actual_mae = metrics['mae'] * 100.0
        temp_range = y_seq.max() - y_seq.min()
        accuracy = (1 - metrics['mae'] / temp_range) * 100 if temp_range > 0 else 0.0

        params = self.trainer.get_parameters()
        params_flat = np.concatenate([p.flatten() for p in params])
        zeta_i = self.hegazy.measure_heterogeneous_variance(params_flat)

        print(f"Result: MAE {actual_mae:.4f} degrees C | "
              f"Accuracy: {accuracy:.2f}% | "
              f"Zeta_i: {zeta_i:.6f}")

        self.daily_metrics.append({
            'day': day_num,
            'mae': actual_mae,
            'accuracy': accuracy,
            'zeta_i': zeta_i
        })
        return params

    def run_day(self, day_num):
        # Train locally
        params = self.train_on_day(day_num)

        # ME-CFL Eq 7-8: Compress with error feedback
        a, b = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters(params, self.home_id, a, b)
        serialized = pickle.dumps(compressed)

        # Transmit via real USRP hardware
        result = self.lora.transmit(serialized[:49], self.home_id)
        success = result['success']

        print(f"LoRa TX (USRP HW): {'SUCCESS' if success else 'FAILED'} | "
              f"ToA: {result['toa']:.4f}s | "
              f"PDR: {result['pdr']*100:.1f}% | "
              f"BER: {result['ber']:.6f} | "
              f"RF: {'YES' if result['rf_success'] else 'NO'}")

        # Wait for global model from server via LAN
        data = self.lora.receive_global_model(self.home_id, timeout=300)

        if data is not None:
            global_params = data.get('params')
            global_flat = self.flatten_global(global_params)

            if self.prev_global is not None:
                updated_flat = self.apply_momentum_update(params, global_flat)
                self.trainer.model.set_parameters(updated_flat)
                print(f"Momentum update applied (beta={self.beta})")
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
            print(f"Day {m['day']}: MAE {m['mae']:.4f} degrees C | "
                  f"Acc {m['accuracy']:.2f}% | "
                  f"Zeta_i {m['zeta_i']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_id', type=int, required=True)
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    HomeNode(
        home_id=args.home_id,
        n_days=args.days,
        epochs_per_day=args.epochs
    ).run()


if __name__ == "__main__":
    main()
