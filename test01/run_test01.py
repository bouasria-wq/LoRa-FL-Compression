#!/usr/bin/env python3
"""
TEST 01 — Convergence Efficiency
==================================
Compares ME-CFL with Hegazy compression (242 bytes/round)
against Baseline FedAvg with NO compression (2212 bytes/round).

Metrics:
  - Accuracy vs Cumulative KB sent
  - Accuracy vs Cumulative Energy (Joules)
  - MAE over days

Presentation:
  - Dual-axis line graph (Accuracy + Energy vs Rounds)
  - Accuracy vs Cumulative KB (communication cost)
  - Summary table

LoRa Energy Model:
  - TX power: 50 mW (typical LoRa at 14 dBm)
  - ToA for 242 bytes @ SF7 BW125k: ~0.61s → 1 packet
  - ToA for 2212 bytes: needs 9+ packets × 0.61s each = ~5.5s total
  - Energy = Power × ToA

Usage:
  cd ~/LoRa_Fl_Compression.proj
  python3 test01/run_test01.py

  (Requires hardwaretest03/ with data/ folder present)

File: test01/run_test01.py
"""

import numpy as np
import sys
import time
import os
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
HT03_DIR     = PROJECT_ROOT / 'hardwaretest03'

sys.path.insert(0, str(HT03_DIR / 'local_home'))
sys.path.insert(0, str(HT03_DIR / 'compression'))
sys.path.insert(0, str(HT03_DIR / 'server'))

from load_data import DataLoader
from train import LocalTrainer
from hegazy import AggregateGaussianMechanism
from hegazy_lora_bridge import HegazyLoRaBridge
from aggregate import FederatedServer

# ─── LoRa energy model ───────────────────────────────────────────
TX_POWER_W        = 0.050       # 50 mW TX power (14 dBm typical LoRa)
RX_POWER_W        = 0.015       # 15 mW RX power
IDLE_POWER_W      = 0.000002    # 2 µW sleep
SF                = 7
BW                = 125000
CR                = 2
PREAMBLE_LEN      = 8
LORA_MAX_PAYLOAD  = 255         # bytes per packet


def calculate_toa(payload_bytes):
    """LoRa time on air for a single packet."""
    n_preamble = PREAMBLE_LEN
    numerator = 8 * payload_bytes - 4 * SF + 28 + 16
    denominator = 4 * (SF - 2)
    payload_symbols = max(np.ceil(numerator / denominator) * (CR + 4), 0)
    n_symbols = n_preamble + payload_symbols
    return (n_symbols * (2 ** SF)) / BW


def packets_needed(total_bytes):
    """How many LoRa packets needed to send total_bytes."""
    return int(np.ceil(total_bytes / LORA_MAX_PAYLOAD))


def energy_per_round(payload_bytes):
    """Total TX energy to send payload_bytes over LoRa."""
    n_packets = packets_needed(payload_bytes)
    toa_per_pkt = calculate_toa(min(payload_bytes, LORA_MAX_PAYLOAD))
    total_toa = n_packets * toa_per_pkt
    return TX_POWER_W * total_toa  # Joules


# ─── Simulation engine ───────────────────────────────────────────

class FLSimulator:
    """
    Runs federated learning with configurable compression.
    """

    def __init__(self, n_homes=10, n_days=7, epochs=100,
                 use_compression=True, use_error_feedback=True,
                 use_momentum=True, label="ME-CFL"):
        self.n_homes = n_homes
        self.n_days  = n_days
        self.epochs  = epochs
        self.use_compression    = use_compression
        self.use_error_feedback = use_error_feedback
        self.use_momentum       = use_momentum
        self.label = label

        self.data_loader = DataLoader(
            data_dir=str(HT03_DIR / 'data'),
            n_homes=n_homes,
            n_days=n_days
        )
        self.bridge = HegazyLoRaBridge()
        self.samples_per_day = 96

    def run(self):
        """Run full FL simulation, return per-day metrics."""
        print(f"\n{'='*60}")
        print(f"RUNNING: {self.label}")
        print(f"{'='*60}")
        print(f"  Homes: {self.n_homes} | Days: {self.n_days} | "
              f"Epochs: {self.epochs}")
        print(f"  Compression: {'ON' if self.use_compression else 'OFF'}")
        print(f"  Error feedback: {'ON' if self.use_error_feedback else 'OFF'}")
        print(f"  Momentum: {'ON' if self.use_momentum else 'OFF'}")

        # Initialize homes
        homes = []
        for h_id in range(1, self.n_homes + 1):
            trainer = LocalTrainer(
                home_id=h_id,
                sequence_length=16,
                learning_rate=0.0005
            )
            df = self.data_loader.load_home_data(h_id)
            hegazy = AggregateGaussianMechanism(
                n_clients=self.n_homes, sigma=0.1, seed=h_id
            )
            homes.append({
                'id': h_id,
                'trainer': trainer,
                'df': df,
                'hegazy': hegazy,
                'momentum': None,
                'prev_global': None,
            })

        aggregator = None
        results = []

        for day in range(1, self.n_days + 1):
            day_start = time.time()
            day_params = []
            day_zetas  = []
            day_maes   = []
            day_accs   = []

            # Each home trains
            for home in homes:
                h_id = home['id']
                df = home['df']
                trainer = home['trainer']

                # Get cumulative data
                df_day = df.iloc[0:day * self.samples_per_day].copy()
                for col in ['T_indoor', 'T_outdoor']:
                    if col in df_day.columns:
                        df_day[col] = np.clip((df_day[col] + 50.0) / 100.0, 0, 1)
                X, y = self.data_loader.get_features_target(df_day)
                X_seq, y_seq = trainer.create_sequences(X, y)

                # Train
                trainer.model.model.fit(
                    X_seq, y_seq,
                    epochs=self.epochs,
                    batch_size=16,
                    validation_split=0.1,
                    shuffle=True,
                    verbose=0
                )

                # Evaluate
                metrics = trainer.evaluate(X_seq, y_seq)
                mae = metrics['mae'] * 100.0
                temp_range = y_seq.max() - y_seq.min()
                acc = (1 - metrics['mae'] / temp_range) * 100 if temp_range > 0 else 0

                params = trainer.get_parameters()
                params_flat = np.concatenate([p.flatten() for p in params])

                zeta = home['hegazy'].measure_heterogeneous_variance(params_flat)

                day_params.append(params)
                day_zetas.append(zeta)
                day_maes.append(mae)
                day_accs.append(acc)

            # Compress + measure payload size
            if self.use_compression:
                # Hegazy compression: 242 bytes per home
                payload_bytes_per_home = 242
            else:
                # Raw: 553 params × 4 bytes = 2212 bytes
                payload_bytes_per_home = 553 * 4  # float32

            total_bytes_this_round = payload_bytes_per_home * self.n_homes
            total_packets = sum(
                packets_needed(payload_bytes_per_home)
                for _ in range(self.n_homes)
            )
            total_energy = sum(
                energy_per_round(payload_bytes_per_home)
                for _ in range(self.n_homes)
            )
            total_toa = sum(
                packets_needed(payload_bytes_per_home) *
                calculate_toa(min(payload_bytes_per_home, LORA_MAX_PAYLOAD))
                for _ in range(self.n_homes)
            )

            # Aggregate
            all_flat = [np.concatenate([p.flatten() for p in params])
                        for params in day_params]
            if aggregator is None:
                aggregator = FederatedServer(n_clients=10)

            client_dict = {i+1: all_flat[i] for i in range(len(all_flat))}
            global_params = aggregator.aggregate_round(client_dict, day)

            # Distribute global model back to homes
            for home in homes:
                params = home['trainer'].get_parameters()
                if self.use_momentum and home['prev_global'] is not None:
                    local_flat = np.concatenate([p.flatten() for p in params])
                    g_t = local_flat - global_params
                    if home['momentum'] is None:
                        home['momentum'] = np.zeros_like(g_t)
                    home['momentum'] = 0.9 * home['momentum'] + 0.1 * g_t
                    updated = local_flat - 0.01 * home['momentum']
                    home['trainer'].model.set_parameters(updated)
                else:
                    home['trainer'].model.set_parameters(global_params)
                home['prev_global'] = global_params.copy()

            avg_mae = np.mean(day_maes)
            avg_acc = np.mean(day_accs)
            avg_zeta = np.mean(day_zetas)
            elapsed = time.time() - day_start

            results.append({
                'day': day,
                'avg_mae': avg_mae,
                'avg_acc': avg_acc,
                'avg_zeta': avg_zeta,
                'bytes_per_home': payload_bytes_per_home,
                'total_bytes': total_bytes_this_round,
                'total_packets': total_packets,
                'total_energy_j': total_energy,
                'total_toa_s': total_toa,
                'cumulative_kb': sum(r['total_bytes'] for r in results) / 1024 + total_bytes_this_round / 1024,
                'cumulative_energy_j': sum(r['total_energy_j'] for r in results) + total_energy,
                'home_accs': day_accs,
                'home_maes': day_maes,
            })

            print(f"  Day {day}: MAE {avg_mae:.4f}°C | Acc {avg_acc:.2f}% | "
                  f"Bytes: {total_bytes_this_round} | "
                  f"Packets: {total_packets} | "
                  f"Energy: {total_energy*1000:.1f}mJ | "
                  f"({elapsed:.1f}s)")

        return results


def generate_plots(mecfl_results, baseline_results, output_dir):
    """Generate IEEE-quality comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # IEEE style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'figure.figsize': (8, 5),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'axes.spines.top': False,
        'axes.spines.right': True,  # needed for dual axis
    })

    os.makedirs(output_dir, exist_ok=True)

    days = [r['day'] for r in mecfl_results]

    # ─── Plot 1: Dual-Axis — Accuracy + Energy vs Days ──────────
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left Y: Accuracy
    ax1.set_xlabel('Communication Round (Day)')
    ax1.set_ylabel('Average Accuracy (%)', color='#1a5276')
    l1, = ax1.plot(days, [r['avg_acc'] for r in mecfl_results],
                   'o-', color='#1a5276', label='ME-CFL Accuracy')
    l2, = ax1.plot(days, [r['avg_acc'] for r in baseline_results],
                   's--', color='#7fb3d8', label='Baseline Accuracy')
    ax1.set_ylim([75, 101])
    ax1.tick_params(axis='y', labelcolor='#1a5276')

    # Right Y: Cumulative Energy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Energy (mJ)', color='#c0392b')
    l3, = ax2.plot(days, [r['cumulative_energy_j']*1000 for r in mecfl_results],
                   '^-', color='#c0392b', label='ME-CFL Energy')
    l4, = ax2.plot(days, [r['cumulative_energy_j']*1000 for r in baseline_results],
                   'v--', color='#e6a4a4', label='Baseline Energy')
    ax2.tick_params(axis='y', labelcolor='#c0392b')

    # Combined legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)

    plt.title('Convergence Efficiency: Accuracy vs Energy Cost')
    fig.tight_layout()
    plt.savefig(f'{output_dir}/fig1_dual_axis_accuracy_energy.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_dual_axis_accuracy_energy.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_dual_axis_accuracy_energy.pdf")

    # ─── Plot 2: Accuracy vs Cumulative KB ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot([r['cumulative_kb'] for r in mecfl_results],
            [r['avg_acc'] for r in mecfl_results],
            'o-', color='#1a5276', linewidth=2.5,
            label=f'ME-CFL (242 B/home/round)')
    ax.plot([r['cumulative_kb'] for r in baseline_results],
            [r['avg_acc'] for r in baseline_results],
            's--', color='#e67e22', linewidth=2.5,
            label=f'Baseline (2212 B/home/round)')

    # Target line
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.7,
               label='95% Target')

    ax.set_xlabel('Cumulative Data Transmitted (KB)')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Communication Cost: Accuracy vs Data Transferred')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([75, 101])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_accuracy_vs_kb.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_accuracy_vs_kb.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_accuracy_vs_kb.pdf")

    # ─── Plot 3: MAE Convergence ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(days, [r['avg_mae'] for r in mecfl_results],
            'o-', color='#1a5276', linewidth=2.5,
            label='ME-CFL (Compressed)')
    ax.plot(days, [r['avg_mae'] for r in baseline_results],
            's--', color='#e67e22', linewidth=2.5,
            label='Baseline (Uncompressed)')

    ax.set_xlabel('Communication Round (Day)')
    ax.set_ylabel('Average MAE (°C)')
    ax.set_title('Temperature Prediction Error Over Time')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_mae_convergence.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_mae_convergence.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_mae_convergence.pdf")

    # ─── Plot 4: Per-Round Cost Comparison (Bar Chart) ───────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bytes per round
    methods = ['ME-CFL\n(Compressed)', 'Baseline\n(Uncompressed)']
    bytes_vals = [mecfl_results[0]['total_bytes'],
                  baseline_results[0]['total_bytes']]
    colors = ['#1a5276', '#e67e22']

    bars = ax1.bar(methods, bytes_vals, color=colors, width=0.5,
                   edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('Bytes per Round (all homes)')
    ax1.set_title('Data Cost per Round')
    for bar, val in zip(bars, bytes_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,} B', ha='center', fontsize=11, fontweight='bold')

    reduction = (1 - bytes_vals[0]/bytes_vals[1]) * 100
    ax1.text(0.5, 0.5, f'{reduction:.0f}%\nreduction',
             transform=ax1.transAxes, ha='center', fontsize=14,
             color='#27ae60', fontweight='bold')

    # Packets per round
    packets_vals = [mecfl_results[0]['total_packets'],
                    baseline_results[0]['total_packets']]

    bars = ax2.bar(methods, packets_vals, color=colors, width=0.5,
                   edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('LoRa Packets per Round')
    ax2.set_title('Packet Cost per Round')
    for bar, val in zip(bars, packets_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_cost_comparison.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig4_cost_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_cost_comparison.pdf")


def print_summary(mecfl_results, baseline_results):
    """Print summary comparison table."""
    print(f"\n{'='*70}")
    print(f"TEST 01 RESULTS — CONVERGENCE EFFICIENCY")
    print(f"{'='*70}")

    print(f"\n{'Metric':<35} {'ME-CFL':<20} {'Baseline':<20}")
    print(f"{'-'*70}")

    # Bytes per home per round
    m_bytes = mecfl_results[0]['bytes_per_home']
    b_bytes = baseline_results[0]['bytes_per_home']
    print(f"{'Bytes/home/round':<35} {m_bytes:<20} {b_bytes:<20}")

    # Packets per home per round
    m_pkts = packets_needed(m_bytes)
    b_pkts = packets_needed(b_bytes)
    print(f"{'Packets/home/round':<35} {m_pkts:<20} {b_pkts:<20}")

    # Total KB over 7 days
    m_kb = mecfl_results[-1]['cumulative_kb']
    b_kb = baseline_results[-1]['cumulative_kb']
    print(f"{'Total KB (7 days)':<35} {m_kb:<20.1f} {b_kb:<20.1f}")

    # Total energy
    m_ej = mecfl_results[-1]['cumulative_energy_j'] * 1000
    b_ej = baseline_results[-1]['cumulative_energy_j'] * 1000
    print(f"{'Total energy (mJ, 7 days)':<35} {m_ej:<20.1f} {b_ej:<20.1f}")

    # Final accuracy
    m_acc = mecfl_results[-1]['avg_acc']
    b_acc = baseline_results[-1]['avg_acc']
    print(f"{'Final accuracy (%)':<35} {m_acc:<20.2f} {b_acc:<20.2f}")

    # Final MAE
    m_mae = mecfl_results[-1]['avg_mae']
    b_mae = baseline_results[-1]['avg_mae']
    print(f"{'Final MAE (°C)':<35} {m_mae:<20.4f} {b_mae:<20.4f}")

    # Rounds to 95%
    m_95 = next((r['day'] for r in mecfl_results if r['avg_acc'] >= 95), 'N/A')
    b_95 = next((r['day'] for r in baseline_results if r['avg_acc'] >= 95), 'N/A')
    print(f"{'Rounds to 95% accuracy':<35} {str(m_95):<20} {str(b_95):<20}")

    print(f"\n{'IMPROVEMENTS':<35}")
    print(f"{'-'*70}")
    print(f"{'Data reduction':<35} {(1 - m_bytes/b_bytes)*100:.1f}%")
    print(f"{'Packet reduction':<35} {(1 - m_pkts/b_pkts)*100:.1f}%")
    print(f"{'Energy reduction':<35} {(1 - m_ej/b_ej)*100:.1f}%")
    print(f"{'KB reduction (7 days)':<35} {(1 - m_kb/b_kb)*100:.1f}%")


def main():
    print("="*60)
    print("TEST 01 — CONVERGENCE EFFICIENCY")
    print("ME-CFL (242 B) vs Baseline FedAvg (2212 B)")
    print("="*60)

    output_dir = str(Path(__file__).parent / 'results')

    # Run ME-CFL (compressed, error feedback, momentum)
    mecfl_sim = FLSimulator(
        n_homes=10, n_days=7, epochs=100,
        use_compression=True,
        use_error_feedback=True,
        use_momentum=True,
        label="ME-CFL (Hegazy Compressed, 242 B/round)"
    )
    mecfl_results = mecfl_sim.run()

    # Run Baseline FedAvg (no compression, no error feedback, no momentum)
    baseline_sim = FLSimulator(
        n_homes=10, n_days=7, epochs=100,
        use_compression=False,
        use_error_feedback=False,
        use_momentum=False,
        label="Baseline FedAvg (Uncompressed, 2212 B/round)"
    )
    baseline_results = baseline_sim.run()

    # Print comparison
    print_summary(mecfl_results, baseline_results)

    # Generate plots
    print(f"\nGenerating IEEE-quality plots...")
    generate_plots(mecfl_results, baseline_results, output_dir)

    print(f"\nAll results saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
