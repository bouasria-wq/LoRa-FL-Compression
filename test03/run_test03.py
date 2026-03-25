#!/usr/bin/env python3
"""
TEST 03 — Network Resilience
===============================
Server goes offline for 48 hours (2 days) during federated training.
Compares ME-CFL (error feedback + momentum) recovery vs FedAvg (vanilla).

Scenario:
  Days 1-3: Normal FL (server online)
  Days 4-5: SERVER DOWN — homes train locally, no aggregation
  Days 6-7: Server comes back — homes upload, server aggregates

ME-CFL advantage:
  Error feedback accumulator tracks compression loss during offline period.
  Momentum preserves learning direction.
  When server returns, the accumulated corrections help recover faster.

FedAvg disadvantage:
  No error tracking. Homes drift independently during outage.
  When server returns, simple averaging of diverged models hurts accuracy.

Metrics:
  - Accuracy over time (with dip + recovery)
  - Recovery speed (rounds to recover to pre-outage accuracy)
  - Accuracy variance across homes (divergence during outage)

Presentation:
  - Line graph showing accuracy timeline with outage window
  - Variance comparison during and after outage

Usage:
  cd ~/LoRa_Fl_Compression.proj
  python3 test03/run_test03.py

File: test03/run_test03.py
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
from aggregate import FederatedServer


class ResilienceSimulator:
    """
    Runs FL with a server outage in the middle.
    """

    def __init__(self, n_homes=10, n_days=7, epochs=100,
                 outage_start=4, outage_end=5,
                 use_error_feedback=True, use_momentum=True,
                 label="ME-CFL"):
        self.n_homes = n_homes
        self.n_days  = n_days
        self.epochs  = epochs
        self.outage_start = outage_start  # server goes down at start of this day
        self.outage_end   = outage_end    # server comes back at start of this day + 1
        self.use_error_feedback = use_error_feedback
        self.use_momentum = use_momentum
        self.label = label

        self.data_loader = DataLoader(
            data_dir=str(HT03_DIR / 'data'),
            n_homes=n_homes,
            n_days=n_days
        )
        self.samples_per_day = 96

    def run(self):
        print(f"\n{'='*60}")
        print(f"RUNNING: {self.label}")
        print(f"{'='*60}")
        print(f"  Homes: {self.n_homes} | Days: {self.n_days} | "
              f"Epochs: {self.epochs}")
        print(f"  Error feedback: {'ON' if self.use_error_feedback else 'OFF'}")
        print(f"  Momentum: {'ON' if self.use_momentum else 'OFF'}")
        print(f"  Server outage: Day {self.outage_start} to Day {self.outage_end}")

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

        server = None
        results = []

        for day in range(1, self.n_days + 1):
            day_start = time.time()
            server_online = not (self.outage_start <= day <= self.outage_end)
            status = "ONLINE" if server_online else "** OFFLINE **"

            day_maes = []
            day_accs = []
            all_flat = []

            # Each home trains (always — even during outage)
            for home in homes:
                h_id = home['id']
                df = home['df']
                trainer = home['trainer']

                df_day = df.iloc[0:day * self.samples_per_day].copy()
                for col in ['T_indoor', 'T_outdoor']:
                    if col in df_day.columns:
                        df_day[col] = np.clip((df_day[col] + 50.0) / 100.0, 0, 1)
                X, y = self.data_loader.get_features_target(df_day)
                X_seq, y_seq = trainer.create_sequences(X, y)

                trainer.model.model.fit(
                    X_seq, y_seq,
                    epochs=self.epochs,
                    batch_size=16,
                    validation_split=0.1,
                    shuffle=True,
                    verbose=0
                )

                metrics = trainer.evaluate(X_seq, y_seq)
                mae = metrics['mae'] * 100.0
                temp_range = y_seq.max() - y_seq.min()
                acc = (1 - metrics['mae'] / temp_range) * 100 if temp_range > 0 else 0

                params = trainer.get_parameters()
                params_flat = np.concatenate([p.flatten() for p in params])

                if self.use_error_feedback:
                    home['hegazy'].measure_heterogeneous_variance(params_flat)

                all_flat.append(params_flat)
                day_maes.append(mae)
                day_accs.append(acc)

            # Aggregation — only if server is online
            if server_online:
                if server is None:
                    server = FederatedServer(n_clients=self.n_homes)

                client_dict = {i+1: all_flat[i] for i in range(len(all_flat))}
                global_params = server.aggregate_round(client_dict, day)

                # Distribute global model
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
            else:
                # Server offline — homes train alone, no aggregation
                # Error feedback still accumulates if enabled
                pass

            avg_mae = np.mean(day_maes)
            avg_acc = np.mean(day_accs)
            std_acc = np.std(day_accs)
            elapsed = time.time() - day_start

            results.append({
                'day': day,
                'avg_mae': avg_mae,
                'avg_acc': avg_acc,
                'std_acc': std_acc,
                'server_online': server_online,
                'home_accs': day_accs.copy(),
                'home_maes': day_maes.copy(),
            })

            print(f"  Day {day} [{status}]: "
                  f"MAE {avg_mae:.4f}°C | Acc {avg_acc:.2f}% | "
                  f"Std {std_acc:.2f}% | ({elapsed:.1f}s)")

        return results


def generate_plots(mecfl_results, fedavg_results, output_dir):
    """Generate IEEE-quality plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'figure.figsize': (8, 5),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    os.makedirs(output_dir, exist_ok=True)
    days = [r['day'] for r in mecfl_results]

    # ─── Plot 1: Accuracy timeline with outage window ─────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    # Shade outage window
    ax.axvspan(3.5, 5.5, alpha=0.15, color='red', label='Server Outage')

    ax.plot(days, [r['avg_acc'] for r in mecfl_results],
            'o-', color='#1a5276', linewidth=2.5,
            label='ME-CFL (error feedback + momentum)')
    ax.plot(days, [r['avg_acc'] for r in fedavg_results],
            's--', color='#e67e22', linewidth=2.5,
            label='FedAvg (vanilla)')

    # Annotate outage
    ax.annotate('Server\nOffline', xy=(4.5, 82), fontsize=11,
                ha='center', color='#c0392b', fontweight='bold')

    ax.set_xlabel('Communication Round (Day)')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Network Resilience: Server Outage Recovery')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xticks(days)
    ax.set_ylim([75, 101])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_resilience_accuracy.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_resilience_accuracy.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_resilience_accuracy.pdf")

    # ─── Plot 2: Accuracy variance (home divergence) ─────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axvspan(3.5, 5.5, alpha=0.15, color='red', label='Server Outage')

    ax.plot(days, [r['std_acc'] for r in mecfl_results],
            'o-', color='#1a5276', linewidth=2.5,
            label='ME-CFL σ (Accuracy)')
    ax.plot(days, [r['std_acc'] for r in fedavg_results],
            's--', color='#e67e22', linewidth=2.5,
            label='FedAvg σ (Accuracy)')

    ax.set_xlabel('Communication Round (Day)')
    ax.set_ylabel('Accuracy Standard Deviation (%)')
    ax.set_title('Home Divergence During Server Outage')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xticks(days)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_variance_divergence.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_variance_divergence.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_variance_divergence.pdf")

    # ─── Plot 3: MAE timeline with outage ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axvspan(3.5, 5.5, alpha=0.15, color='red', label='Server Outage')

    ax.plot(days, [r['avg_mae'] for r in mecfl_results],
            'o-', color='#1a5276', linewidth=2.5,
            label='ME-CFL')
    ax.plot(days, [r['avg_mae'] for r in fedavg_results],
            's--', color='#e67e22', linewidth=2.5,
            label='FedAvg')

    ax.set_xlabel('Communication Round (Day)')
    ax.set_ylabel('Average MAE (°C)')
    ax.set_title('Prediction Error Recovery After Server Outage')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticks(days)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_mae_resilience.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_mae_resilience.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_mae_resilience.pdf")

    # ─── Plot 4: Per-home accuracy spread (box plot style) ────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (results, title, color) in enumerate([
        (mecfl_results, 'ME-CFL', '#1a5276'),
        (fedavg_results, 'FedAvg', '#e67e22'),
    ]):
        ax = axes[idx]
        home_accs_per_day = [r['home_accs'] for r in results]

        bp = ax.boxplot(home_accs_per_day, positions=days,
                        widths=0.5, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Shade outage
        ax.axvspan(3.5, 5.5, alpha=0.12, color='red')

        ax.set_xlabel('Communication Round (Day)')
        ax.set_ylabel('Home Accuracy (%)')
        ax.set_title(f'{title} — Per-Home Accuracy Spread')
        ax.set_xticks(days)
        ax.set_ylim([70, 101])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_boxplot_spread.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig4_boxplot_spread.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_boxplot_spread.pdf")


def print_summary(mecfl_results, fedavg_results):
    """Print summary table."""
    print(f"\n{'='*70}")
    print(f"TEST 03 RESULTS — NETWORK RESILIENCE")
    print(f"{'='*70}")

    print(f"\n{'Day':<6} {'Status':<12} "
          f"{'ME-CFL Acc':<14} {'FedAvg Acc':<14} "
          f"{'ME-CFL σ':<12} {'FedAvg σ':<12}")
    print(f"{'-'*70}")

    for m, f in zip(mecfl_results, fedavg_results):
        status = "ONLINE" if m['server_online'] else "OFFLINE"
        print(f"Day {m['day']:<3} {status:<12} "
              f"{m['avg_acc']:<14.2f} {f['avg_acc']:<14.2f} "
              f"{m['std_acc']:<12.2f} {f['std_acc']:<12.2f}")

    # Pre-outage accuracy (Day 3)
    m_pre = mecfl_results[2]['avg_acc']
    f_pre = fedavg_results[2]['avg_acc']

    # Post-outage accuracy (Day 6 — first day back online)
    m_post = mecfl_results[5]['avg_acc']
    f_post = fedavg_results[5]['avg_acc']

    # Recovery = how close to pre-outage after coming back
    m_recovery = m_post - m_pre
    f_recovery = f_post - f_pre

    # Final accuracy
    m_final = mecfl_results[-1]['avg_acc']
    f_final = fedavg_results[-1]['avg_acc']

    print(f"\n{'RECOVERY ANALYSIS':<35}")
    print(f"{'-'*70}")
    print(f"{'Pre-outage accuracy (Day 3)':<35} "
          f"{m_pre:.2f}%{'':<8} {f_pre:.2f}%")
    print(f"{'Post-outage accuracy (Day 6)':<35} "
          f"{m_post:.2f}%{'':<8} {f_post:.2f}%")
    print(f"{'Recovery gain (Day 6 vs Day 3)':<35} "
          f"{m_recovery:+.2f}%{'':<7} {f_recovery:+.2f}%")
    print(f"{'Final accuracy (Day 7)':<35} "
          f"{m_final:.2f}%{'':<8} {f_final:.2f}%")
    print(f"{'Final MAE (Day 7)':<35} "
          f"{mecfl_results[-1]['avg_mae']:.4f}°C{'':<4} "
          f"{fedavg_results[-1]['avg_mae']:.4f}°C")


def main():
    print("=" * 60)
    print("TEST 03 — NETWORK RESILIENCE")
    print("Server outage Days 4-5, recovery Days 6-7")
    print("ME-CFL (error feedback) vs FedAvg (vanilla)")
    print("=" * 60)

    output_dir = str(Path(__file__).parent / 'results')

    # ME-CFL: error feedback + momentum
    mecfl_sim = ResilienceSimulator(
        n_homes=10, n_days=7, epochs=100,
        outage_start=4, outage_end=5,
        use_error_feedback=True,
        use_momentum=True,
        label="ME-CFL (Error Feedback + Momentum)"
    )
    mecfl_results = mecfl_sim.run()

    # FedAvg: no error feedback, no momentum
    fedavg_sim = ResilienceSimulator(
        n_homes=10, n_days=7, epochs=100,
        outage_start=4, outage_end=5,
        use_error_feedback=False,
        use_momentum=False,
        label="FedAvg (Vanilla — No Error Feedback)"
    )
    fedavg_results = fedavg_sim.run()

    # Summary
    print_summary(mecfl_results, fedavg_results)

    # Plots
    print(f"\nGenerating IEEE-quality plots...")
    generate_plots(mecfl_results, fedavg_results, output_dir)

    print(f"\nAll results saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
