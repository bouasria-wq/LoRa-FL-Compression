"""
Individual Home Node
====================
With real GNU Radio LoRa PHY layer - Option B.
Temperature range: -50 to +50 degrees C
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

        print(f"\nHOME {self.home_id:02d} - STARTING (-50 to +50 degrees C RANGE)")

        self.trainer = LocalTrainer(home_id=home_id, sequence_length=16, learning_rate=0.0005)
        self.data_loader = DataLoader(data_dir='data', n_homes=10, n_days=n_days)
        self.df_full = self.data_loader.load_home_data(home_id)

        self.data_dir = Path(__file__).parent / 'lora'
        self.tx_file = self.data_dir / f'home_{home_id:02d}_tx.txt'
        self.rx_global_file = self.data_dir / 'global_model_broadcast.txt'
        self.ready_file = self.data_dir / f'home_{home_id:02d}_ready.flag'

        self.lora_bridge = HegazyLoRaBridge()
        self.hegazy = AggregateGaussianMechanism(n_clients=1, sigma=0.1, seed=home_id)
        self.lora_sim = GNURadioLoRaSimulator(sf=7, bw=125000, cr=1, snr_db=10.0)

        self.daily_metrics = []

    def get_cumulative_data(self, day_num):
        df_cumulative = self.df_full.iloc[0:day_num * self.samples_per_day].copy()
        temp_cols = ['T_indoor', 'T_outdoor']
        for col in temp_cols:
            if col in df_cumulative.columns:
                df_cumulative[col] = np.clip((df_cumulative[col] + 50.0) / 100.0, 0, 1)
        X_cum, y_cum = self.data_loader.get_features_target(df_cumulative)
        return X_cum, y_cum

    def train_on_day(self, day_num):
        print(f"\n--- HOME {self.home_id:02d} | DAY {day_num} | {self.epochs_per_day} EPOCHS ---")
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
        if temp_range > 0:
            accuracy = (1 - metrics['mae'] / temp_range) * 100
        else:
            accuracy = 0.0

        print(f"Result: MAE {actual_mae:.4f} degrees C | Accuracy: {accuracy:.2f}%")
        self.daily_metrics.append({
            'day': day_num,
            'mae': actual_mae,
            'accuracy': accuracy
        })
        return self.trainer.get_parameters()

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
                        data = pickle.loads(self.lora_bridge.lora_ascii_to_binary(f.read()))
                    if data.get('day') == day_num:
                        print(f"Day {day_num} Global Model Received.")
                        return data['params']
                except:
                    pass
            time.sleep(3)
        return None

    def run_day(self, day_num):
        params = self.train_on_day(day_num)

        a, b = self.hegazy.decompose()
        compressed = self.hegazy.encode_parameters(params, 0, a, b)
        serialized = pickle.dumps(compressed)

        success = self.transmit_via_lora(serialized[:49])

        p_dict = {'home_id': self.home_id, 'day': day_num, 'params': params, 'lora_success': success}
        self.lora_bridge.write_lora_file(pickle.dumps(p_dict), str(self.tx_file))

        with open(self.ready_file, 'w') as f:
            f.write(f"day_{day_num}")

        global_params = self.wait_for_global_model(day_num)
        if global_params is not None:
            self.trainer.model.set_parameters(global_params)
        time.sleep(2)

    def run(self):
        for d in range(1, self.n_days + 1):
            self.run_day(d)
        print("\n" + "="*30 + "\nFINAL HOME SUMMARY\n" + "="*30)
        for m in self.daily_metrics:
            print(f"Day {m['day']}: MAE {m['mae']:.4f} degrees C | Acc {m['accuracy']:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_id', type=int, required=True)
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    HomeNode(home_id=args.home_id, n_days=args.days, epochs_per_day=args.epochs).run()


if __name__ == "__main__":
    main()
