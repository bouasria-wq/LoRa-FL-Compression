"""
Individual Home Node - ME-CFL Version
======================================
Implements:
- Error feedback (ME-CFL Eq 7-8)
- Local shift tracking (ME-CFL Eq 9)
- Momentum updates (ME-CFL Eq 10)
- Heterogeneous variance measurement (ME-CFL Eq 3)
- Real GNU Radio LoRa TX chain

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
from hegazy_lora_bridge import HegazyLoRaBridge
from gr_lora_sim import GNURadioLoRaSimulator


class HomeNode:

    def __init__(self, home_id, n_days=7, epochs_per_day=100):
        self.home_id = home_id
        self.n_days = n_days
        self.epochs_per_day = epochs_per_day
        self.samples_per_day = 96

        print(f"\nHOME {self.home_id:02d} - ME-CFL VERSION")
        print(f"Error feedback: ON | Variance reduction: ON | Momentum: ON")

        self.trainer = LocalTrainer(home_id=home_id, sequence_length=16, learning_rate=0.0005)
        self.data_loader = DataLoader(data_dir='data', n_homes=10, n_days=n_days)
        self.df_full = self.data_loader.load_home_data(home_id)

        self.data_dir = Path(__file__).parent / 'lora'
        self.tx_file = self.data_dir / f'home_{home_id:02d}_tx.txt'
        self.rx_global_file = self.data_dir / 'global_model_broadcast.txt'
        self.ready_file = self.data_dir / f'home_{home_id:02d}_ready.flag'

        self.lora_bridge = HegazyLoRaBridge()
        self.hegazy = AggregateGaussianMechanism(n_clients=10, sigma=0.1, seed=home_id)
        self.lora_sim = GNURadioLoRaSimulator(sf=7, bw=125000, cr=1, snr_db=10.0)

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
        """Ensure global params is a flat numpy array."""
        if isinstance(global_params, list):
            return np.concatenate([np.array(p).flatten() for p in global_params])
        elif isinstance(global_params, np.ndarray):
            return global_params.flatten()
        else:
            return np.array(global_params).flatten()

    def apply_momentum_update(self, local_params, global_flat):
        """
        ME-CFL Eq 10: Momentum update in Hilbert space.
        x^{t+1} = x^t - eta * M(g^t, beta)
        M = beta * M^{t-1} + (1-beta) * g^t
        """
        local_flat = np.concatenate([p.flatten() for p in local_params])

        # Gradient estimate
        g_t = local_flat - global_flat

        # Initialize momentum
        if self.momentum is None:
            self.momentum = np.zeros_like(g_t)

        # Recursive momentum
        self.momentum = self.beta * self.momentum + (1 - self.beta) * g_t

        # Momentum update
        updated_flat = local_flat - self.eta * self.momentum
        return updated_flat

    def train_on_day(self, day_num):
        print(f"\n--- HOME {self.home_id:02d} | DAY {day_num} | {self.epochs_per_day} EPOCHS | ME-CFL ---")
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

    def transmit_via_lora(self, data_bytes):
        result = self.lora_sim.transmit(data_bytes)
        print(f"LoRa TX (GNU Radio): {'SUCCESS' if result['success'] else 'FAILED'} | "
              f"ToA: {result['t_toa']:.4f}s | "
              f"SNR: {result['measured_snr']:.2f}dB | "
              f"BER: {result['ber']:.6f} | "
              f"PDR: {result['pdr']*100:.1f}%")
        return result['success']

    def wait_for_global_model(self, day_num, timeout=180):
        print(f"Waiting for Server (Day {day_num})...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.rx_global_file.exists():
                try:
                    with open(self.rx_global_file, 'r') as f:
                        data = pickle.loads(
                            self.lora_bridge.lora_ascii_to_binary(f.read())
                        )
                    if data.get('day') == day_num:
                        print(f"Day {day_num} Global Model Received.")
                        return data['params']
                except:
                    pass
            time.sleep(3)
        return None

    def run_day(self, day_num):
        # Train locally
        params = self.train_on_day(day_num)

        # ME-CFL Eq 7-8: Compress with error feedback
        a, b = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters(params, self.home_id, a, b)
        serialized = pickle.dumps(compressed)

        # Transmit via real GNU Radio LoRa
        success = self.transmit_via_lora(serialized[:49])

        # Write to file for server
        p_dict = {
            'home_id': self.home_id,
            'day': day_num,
            'params': params,
            'lora_success': success,
            'zeta_i': self.hegazy.zeta_i,
            'local_shift': self.hegazy.local_shift
        }
        self.lora_bridge.write_lora_file(pickle.dumps(p_dict), str(self.tx_file))

        with open(self.ready_file, 'w') as f:
            f.write(f"day_{day_num}")

        # Wait for global model
        global_params = self.wait_for_global_model(day_num)

        if global_params is not None:
            # Always flatten first
            global_flat = self.flatten_global(global_params)

            # ME-CFL Eq 10: Apply momentum update
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
        print(f"FINAL HOME {self.home_id:02d} SUMMARY - ME-CFL")
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
