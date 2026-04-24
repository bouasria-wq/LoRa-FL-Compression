"""
TEST 02 — LoRa Uncompressed FedAvg Baseline
============================================
This is Case 2 in Figure 1.
Uncompressed float32 parameters sent over LoRa (2212 bytes, 9 packets).
Plain FedAvg aggregation — no Hegazy compression, no momentum.

This is the BASELINE to compare against LoRa-FedComp (hardwaretest03).

What this test measures:
  - FL prediction error (MAE, MSE, RMSE) without compression
  - Confirms that uncompressed FL converges correctly
  - Provides the Case 2 data for Figure 1

How it differs from hardwaretest03 (Case 1):
  - No Hegazy compression — raw float32 params
  - No error feedback accumulator
  - No momentum on server
  - 2212B payload = 9 LoRa packets vs 1 packet
  - Server uses plain FedAvg mean

Run from ~/LoRa_Fl_Compression.proj/:
  python3 run_test02.py --n_homes 10 --days 7 --epochs 100

Results saved to: results/test02/results.csv

References:
  McMahan et al. (2017) FedAvg. AISTATS.
  Semtech AN1200.22 Eq.12-13 (ToA for 9 x 255B packets)
"""

import argparse
import numpy as np
import csv
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
HT03_DIR     = PROJECT_ROOT / 'hardwaretest03'

sys.path.insert(0, str(HT03_DIR / 'local_home'))
sys.path.insert(0, str(HT03_DIR / 'server'))

from load_data import DataLoader
from train     import LocalTrainer


# ═══════════════════════════════════════════════════════════════════
# PLAIN FEDAVG SERVER — no momentum, no variance reduction
# McMahan et al. (2017) Algorithm 1
# ═══════════════════════════════════════════════════════════════════
class FedAvgServer:
    def __init__(self, n_clients):
        self.n_clients    = n_clients
        self.global_model = None
        self.round        = 0

    def aggregate(self, client_params_list):
        """Plain weighted average. w_global = (1/n) sum(w_i)."""
        stacked           = np.stack(client_params_list, axis=0)
        self.global_model = stacked.mean(axis=0)
        self.round       += 1
        return self.global_model


# ═══════════════════════════════════════════════════════════════════
# LORA UNCOMPRESSED TRANSMISSION MODEL
# 2212 bytes raw float32 = 9 x 255B LoRa packets
# ToA per packet = 5808ms total (AN1200.22, SF7, BW125k, CR4/6)
# P_round = 0.95^9 = 0.6302 per home per round
# ═══════════════════════════════════════════════════════════════════
LORA_MAX_PKT  = 255
PARAM_BYTES   = 553 * 4   # 2212 bytes — raw float32
N_PKTS        = int(np.ceil(PARAM_BYTES / LORA_MAX_PKT))   # 9 packets
P_PKT_SUCCESS = 0.95       # per-packet PDR, SF7 BW125k SNR=-7dB
P_ROUND       = P_PKT_SUCCESS ** N_PKTS   # 0.6302

print(f"LoRa uncompressed: {PARAM_BYTES}B = {N_PKTS} packets")
print(f"P_round = {P_PKT_SUCCESS}^{N_PKTS} = {P_ROUND:.4f}")


def simulate_lora_round(rng):
    """Returns True if all N_PKTS packets succeed."""
    return bool((rng.random(N_PKTS) < P_PKT_SUCCESS).all())


# ═══════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════
def run_test02(n_homes=10, n_days=7, epochs=100):
    print("=" * 60)
    print("TEST 02 — LoRa Uncompressed FedAvg (Case 2 baseline)")
    print(f"  Homes: {n_homes}  Days: {n_days}  Epochs/day: {epochs}")
    print(f"  Payload: {PARAM_BYTES}B ({N_PKTS} pkts)  P_round={P_ROUND:.4f}")
    print("=" * 60)

    data_loader = DataLoader(
        data_dir=str(HT03_DIR / 'data'),
        n_homes=n_homes,
        n_days=n_days
    )
    samples_per_day = 96

    # Initialise one LSTM trainer per home
    trainers = {}
    rngs     = {}
    for h in range(1, n_homes + 1):
        trainers[h] = LocalTrainer(
            home_id=h,
            sequence_length=16,
            learning_rate=0.0005
        )
        rngs[h] = np.random.default_rng(seed=h)
        print(f"  Home {h:02d} initialised")

    server        = FedAvgServer(n_clients=n_homes)
    global_params = None
    results       = []
    failed_rounds = {h: 0 for h in range(1, n_homes + 1)}

    for day in range(1, n_days + 1):
        print(f"\n  --- Day {day} ---")
        day_params = {}
        day_mae    = {}
        day_acc    = {}

        for h in range(1, n_homes + 1):
            # Load cumulative data up to this day
            df     = data_loader.load_home_data(h)
            df_day = df.iloc[0:day * samples_per_day].copy()
            for col in ['T_indoor', 'T_outdoor']:
                if col in df_day.columns:
                    df_day[col] = np.clip(
                        (df_day[col] + 50.0) / 100.0, 0, 1)

            X, y         = data_loader.get_features_target(df_day)
            X_seq, y_seq = trainers[h].create_sequences(X, y)

            # Local training — standard, no proximal term
            trainers[h].model.model.fit(
                X_seq, y_seq,
                epochs=epochs,
                batch_size=16,
                validation_split=0.1,
                shuffle=True,
                verbose=0
            )

            # Evaluate
            metrics    = trainers[h].evaluate(X_seq, y_seq)
            mae        = metrics['mae'] * 100.0
            temp_range = y_seq.max() - y_seq.min()
            acc        = ((1 - metrics['mae'] / temp_range) * 100
                          if temp_range > 0 else 0)

            # Get raw float32 params (no compression)
            params      = trainers[h].get_parameters()
            params_flat = np.concatenate([p.flatten() for p in params])

            # Simulate LoRa 9-packet transmission
            tx_ok = simulate_lora_round(rngs[h])

            if tx_ok:
                day_params[h] = params_flat
                status        = f'TX OK | {PARAM_BYTES}B ({N_PKTS} pkts)'
            else:
                failed_rounds[h] += 1
                status = f'TX FAIL (round {failed_rounds[h]} failed)'

            day_mae[h] = mae
            day_acc[h] = acc

            print(f"    Home {h:02d} | MAE {mae:.3f}°C | "
                  f"Acc {acc:.2f}% | {status}")

        # Server FedAvg aggregation — only from homes that succeeded
        if day_params:
            global_params = server.aggregate(list(day_params.values()))
            print(f"  Day {day}: Aggregated "
                  f"{len(day_params)}/{n_homes} homes "
                  f"(plain FedAvg, no momentum)")

            # Distribute global model — plain set, no momentum
            for h in range(1, n_homes + 1):
                trainers[h].model.set_parameters(global_params)

        # Save results
        for h in range(1, n_homes + 1):
            results.append({
                'scenario' : 'Case2_LoRa_Uncompressed_FedAvg',
                'day'      : day,
                'home'     : h,
                'mae_C'    : day_mae[h],
                'accuracy' : day_acc[h],
                'tx_ok'    : h in day_params,
                'failed_rounds': failed_rounds[h],
                'payload_B': PARAM_BYTES,
                'n_pkts'   : N_PKTS,
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test 02 — LoRa Uncompressed FedAvg Baseline')
    parser.add_argument('--n_homes', type=int, default=10)
    parser.add_argument('--days',    type=int, default=7)
    parser.add_argument('--epochs',  type=int, default=100)
    args = parser.parse_args()

    results = run_test02(args.n_homes, args.days, args.epochs)

    out_dir = Path('results/test02')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'results.csv'

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    np.save(str(out_dir / 'results.npy'), results)
    print(f"\nTest 02 complete. Results saved to {out_csv}")

    # Summary
    print("\nDay 7 summary:")
    day7 = [r for r in results if r['day'] == 7]
    maes = [r['mae_C'] for r in day7]
    print(f"  Mean MAE: {np.mean(maes):.3f}°C ± {np.std(maes):.3f}°C")
    print(f"  Min MAE:  {np.min(maes):.3f}°C")
    print(f"  Max MAE:  {np.max(maes):.3f}°C")


if __name__ == '__main__':
    main()
