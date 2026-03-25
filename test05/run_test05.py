#!/usr/bin/env python3
"""
TEST 05 — System Scalability (Channel Occupancy & Collisions)
================================================================
Projects collision probability as the number of homes scales
from 10 to 100 using the Aloha collision model.

Key insight:
  Collision probability depends on how long each node occupies the channel.
  LoRa+Compression: 1 packet × 0.61s = 0.61s per home
  Raw LoRa:         9 packets × 0.61s = 5.49s per home
  WiFi:             fast TX but needs AP — fails at range/walls

  Pure Aloha collision: P(collision) = 1 - e^(-2GT)
  where G = arrival rate, T = packet duration

  More homes + longer TX = exponential collision increase.

Also models:
  - FCC 915 MHz dwell time compliance
  - WiFi 2.4 GHz wall penetration loss
  - Total channel occupancy per hour

Metrics:
  - Collision probability vs number of homes
  - Channel occupancy percentage
  - Dwell time compliance check
  - WiFi range degradation through walls

Usage:
  cd ~/LoRa_Fl_Compression.proj
  python3 test05/run_test05.py

File: test05/run_test05.py
"""

import numpy as np
import os
from pathlib import Path

# ─── LoRa parameters ─────────────────────────────────────────────
LORA_SF         = 7
LORA_BW         = 125000
LORA_CR         = 2
LORA_PREAMBLE   = 8
LORA_MAX_PAYLOAD = 255
COMPRESSED_BYTES = 242
RAW_BYTES        = 2212
FCC_DWELL_LIMIT  = 0.400  # 400ms FCC dwell time limit (915 MHz)

# ─── WiFi parameters ─────────────────────────────────────────────
WIFI_TX_TIME_S  = 0.005   # ~5ms for 242 bytes at 1 Mbps effective
WIFI_OVERHEAD_S = 3.0     # connection setup + idle + ACK
WIFI_WALL_LOSS_DB = 6     # ~6 dB loss per wall at 2.4 GHz
WIFI_MAX_RANGE_M  = 50    # indoor range with clear line of sight

# ─── LoRa range ──────────────────────────────────────────────────
LORA_WALL_LOSS_DB = 2     # ~2 dB loss per wall at 915 MHz
LORA_MAX_RANGE_M  = 2000  # suburban range


def lora_toa(payload_bytes):
    """LoRa time on air for one packet."""
    n_preamble = LORA_PREAMBLE
    numerator = 8 * payload_bytes - 4 * LORA_SF + 28 + 16
    denominator = 4 * (LORA_SF - 2)
    payload_symbols = max(np.ceil(numerator / denominator) * (LORA_CR + 4), 0)
    n_symbols = n_preamble + payload_symbols
    return (n_symbols * (2 ** LORA_SF)) / LORA_BW


def packets_needed(total_bytes, max_payload):
    return int(np.ceil(total_bytes / max_payload))


def aloha_collision_prob(n_homes, tx_time_s, window_s=3600):
    """
    Pure Aloha collision probability.
    G = total offered load = n_homes * tx_time / window
    P(success for a single TX) = e^(-2G)
    P(at least one collision in the network) = 1 - (e^(-2G))^n_homes
    """
    G = n_homes * tx_time_s / window_s
    p_success_single = np.exp(-2 * G)
    p_no_collision_network = p_success_single ** n_homes
    p_collision = 1 - p_no_collision_network
    return p_collision, G, p_success_single


def channel_occupancy(n_homes, tx_time_s, window_s=3600):
    """What fraction of the hour is the channel busy."""
    return (n_homes * tx_time_s) / window_s * 100  # percentage


def wifi_range_through_walls(n_walls):
    """WiFi effective range after passing through walls."""
    # Free space path loss model simplified
    # Each wall adds ~6 dB loss at 2.4 GHz
    # Range roughly halves every 6 dB
    loss_factor = 10 ** (n_walls * WIFI_WALL_LOSS_DB / 20)
    return WIFI_MAX_RANGE_M / loss_factor


def lora_range_through_walls(n_walls):
    """LoRa effective range after passing through walls."""
    loss_factor = 10 ** (n_walls * LORA_WALL_LOSS_DB / 20)
    return LORA_MAX_RANGE_M / loss_factor


def run_analysis():
    """Run full scalability analysis."""
    # TX times
    comp_toa = lora_toa(COMPRESSED_BYTES)
    raw_n_pkts = packets_needed(RAW_BYTES, LORA_MAX_PAYLOAD)
    raw_toa = raw_n_pkts * lora_toa(LORA_MAX_PAYLOAD)

    print(f"\n{'='*60}")
    print(f"TX TIME PER HOME")
    print(f"{'='*60}")
    print(f"  LoRa+Compression: {comp_toa*1000:.1f} ms "
          f"({packets_needed(COMPRESSED_BYTES, LORA_MAX_PAYLOAD)} packet)")
    print(f"  LoRa Raw:         {raw_toa*1000:.1f} ms "
          f"({raw_n_pkts} packets)")
    print(f"  WiFi:             {(WIFI_TX_TIME_S + WIFI_OVERHEAD_S)*1000:.1f} ms "
          f"(incl. setup)")
    print(f"  FCC dwell limit:  {FCC_DWELL_LIMIT*1000:.0f} ms")

    # FCC compliance check
    print(f"\n{'='*60}")
    print(f"FCC 915 MHz DWELL TIME COMPLIANCE")
    print(f"{'='*60}")
    print(f"  LoRa+Comp ({comp_toa*1000:.0f}ms): "
          f"{'COMPLIANT' if comp_toa <= FCC_DWELL_LIMIT else 'VIOLATION'}")
    print(f"  LoRa Raw per-pkt ({lora_toa(LORA_MAX_PAYLOAD)*1000:.0f}ms): "
          f"{'COMPLIANT' if lora_toa(LORA_MAX_PAYLOAD) <= FCC_DWELL_LIMIT else 'VIOLATION'}")
    print(f"  LoRa Raw total ({raw_toa*1000:.0f}ms): "
          f"Needs frequency hopping for {raw_n_pkts} packets")

    # Scalability sweep
    home_counts = [5, 10, 15, 20, 30, 40, 50, 75, 100]
    results = []

    print(f"\n{'='*60}")
    print(f"COLLISION PROBABILITY (Pure Aloha, 1-hour window)")
    print(f"{'='*60}")
    print(f"\n{'Homes':>6} | {'Comp Collision':>15} | {'Raw Collision':>15} | "
          f"{'Comp Occupancy':>15} | {'Raw Occupancy':>15}")
    print(f"{'-'*75}")

    for n in home_counts:
        p_comp, g_comp, _ = aloha_collision_prob(n, comp_toa)
        p_raw, g_raw, _   = aloha_collision_prob(n, raw_toa)
        occ_comp = channel_occupancy(n, comp_toa)
        occ_raw  = channel_occupancy(n, raw_toa)

        results.append({
            'n_homes': n,
            'p_collision_comp': p_comp * 100,
            'p_collision_raw': p_raw * 100,
            'occupancy_comp': occ_comp,
            'occupancy_raw': occ_raw,
            'g_comp': g_comp,
            'g_raw': g_raw,
        })

        print(f"{n:>6} | {p_comp*100:>14.1f}% | {p_raw*100:>14.1f}% | "
              f"{occ_comp:>14.2f}% | {occ_raw:>14.2f}%")

    # Wall penetration
    print(f"\n{'='*60}")
    print(f"RANGE THROUGH WALLS")
    print(f"{'='*60}")
    print(f"\n{'Walls':>6} | {'WiFi Range (m)':>15} | {'LoRa Range (m)':>15}")
    print(f"{'-'*42}")

    wall_results = []
    for n_walls in range(0, 6):
        wifi_r = wifi_range_through_walls(n_walls)
        lora_r = lora_range_through_walls(n_walls)
        wall_results.append({
            'walls': n_walls,
            'wifi_range': wifi_r,
            'lora_range': lora_r,
        })
        print(f"{n_walls:>6} | {wifi_r:>14.1f}m | {lora_r:>14.1f}m")

    return results, wall_results, comp_toa, raw_toa


def generate_plots(results, wall_results, comp_toa, raw_toa, output_dir):
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

    # ─── Plot 1: Collision probability vs homes ──────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    homes = [r['n_homes'] for r in results]
    p_comp = [r['p_collision_comp'] for r in results]
    p_raw  = [r['p_collision_raw'] for r in results]

    ax.plot(homes, p_comp, 'o-', color='#1a5276',
            label=f'LoRa+Compression ({comp_toa*1000:.0f}ms/home)')
    ax.plot(homes, p_raw, 's--', color='#e67e22',
            label=f'LoRa Raw ({raw_toa*1000:.0f}ms/home)')

    # Mark where raw LoRa hits 50% collision
    for r in results:
        if r['p_collision_raw'] >= 50:
            ax.axvline(x=r['n_homes'], color='#e67e22', linestyle=':',
                       alpha=0.5)
            ax.annotate(f'Raw: 50%+ collision\nat {r["n_homes"]} homes',
                        xy=(r['n_homes'], 50),
                        xytext=(r['n_homes'] + 5, 55),
                        fontsize=9, color='#e67e22',
                        arrowprops=dict(arrowstyle='->', color='#e67e22'))
            break

    ax.set_xlabel('Number of Homes')
    ax.set_ylabel('Collision Probability (%)')
    ax.set_title('System Scalability: Collision Risk vs Network Size')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_collision_vs_homes.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_collision_vs_homes.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_collision_vs_homes.pdf")

    # ─── Plot 2: Channel occupancy vs homes ──────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    occ_comp = [r['occupancy_comp'] for r in results]
    occ_raw  = [r['occupancy_raw'] for r in results]

    ax.plot(homes, occ_comp, 'o-', color='#1a5276',
            label='LoRa+Compression')
    ax.plot(homes, occ_raw, 's--', color='#e67e22',
            label='LoRa Raw')

    # FCC duty cycle reference (1% typical)
    ax.axhline(y=1, color='red', linestyle=':', alpha=0.7,
               label='1% Duty Cycle Guideline')

    ax.set_xlabel('Number of Homes')
    ax.set_ylabel('Channel Occupancy (%)')
    ax.set_title('Channel Utilization vs Network Size (per hour)')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_channel_occupancy.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_channel_occupancy.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_channel_occupancy.pdf")

    # ─── Plot 3: Range through walls ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    walls = [r['walls'] for r in wall_results]
    wifi_r = [r['wifi_range'] for r in wall_results]
    lora_r = [r['lora_range'] for r in wall_results]

    ax.plot(walls, lora_r, 'o-', color='#1a5276', linewidth=2.5,
            label='LoRa (915 MHz)')
    ax.plot(walls, wifi_r, 's--', color='#e67e22', linewidth=2.5,
            label='WiFi (2.4 GHz)')

    # Mark typical home scenario (2-3 walls)
    ax.axvspan(1.5, 3.5, alpha=0.1, color='gray')
    ax.annotate('Typical\nhome', xy=(2.5, max(lora_r)*0.5),
                fontsize=10, ha='center', color='gray')

    ax.set_xlabel('Number of Walls')
    ax.set_ylabel('Effective Range (m)')
    ax.set_title('Signal Penetration: LoRa vs WiFi Through Walls')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_range_vs_walls.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_range_vs_walls.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_range_vs_walls.pdf")

    # ─── Plot 4: Dwell time comparison ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ['LoRa+Comp\n(1 pkt)', 'LoRa Raw\n(per pkt)', 'LoRa Raw\n(total 9 pkts)']
    times_ms = [comp_toa * 1000,
                lora_toa(LORA_MAX_PAYLOAD) * 1000,
                raw_toa * 1000]
    colors = ['#1a5276', '#e67e22', '#c0392b']

    bars = ax.bar(methods, times_ms, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.5)

    # FCC limit line
    ax.axhline(y=FCC_DWELL_LIMIT * 1000, color='red', linestyle='--',
               linewidth=2, label=f'FCC Dwell Limit ({FCC_DWELL_LIMIT*1000:.0f}ms)')

    for bar, val in zip(bars, times_ms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}ms', ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('Time on Air (ms)')
    ax.set_title('FCC 915 MHz Dwell Time Compliance')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_dwell_time.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig4_dwell_time.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_dwell_time.pdf")


def print_summary(results):
    """Print final summary."""
    print(f"\n{'='*70}")
    print(f"TEST 05 RESULTS — SYSTEM SCALABILITY")
    print(f"{'='*70}")

    # Find where raw hits 50% collision
    raw_50 = next((r['n_homes'] for r in results
                    if r['p_collision_raw'] >= 50), '>100')
    comp_50 = next((r['n_homes'] for r in results
                     if r['p_collision_comp'] >= 50), '>100')

    print(f"\n{'HEADLINE NUMBERS'}")
    print(f"{'-'*70}")
    print(f"Raw LoRa hits 50% collision at:      {raw_50} homes")
    print(f"Compressed LoRa hits 50% collision at: {comp_50} homes")
    print(f"At 100 homes:")
    r100 = results[-1]
    print(f"  Compressed collision: {r100['p_collision_comp']:.1f}%")
    print(f"  Raw collision:        {r100['p_collision_raw']:.1f}%")
    print(f"  Compressed occupancy: {r100['occupancy_comp']:.2f}%")
    print(f"  Raw occupancy:        {r100['occupancy_raw']:.2f}%")


def main():
    print("=" * 60)
    print("TEST 05 — SYSTEM SCALABILITY")
    print("Collision probability & channel occupancy (Aloha model)")
    print("LoRa+Compression vs Raw LoRa vs WiFi")
    print("=" * 60)

    output_dir = str(Path(__file__).parent / 'results')

    results, wall_results, comp_toa, raw_toa = run_analysis()
    print_summary(results)

    print(f"\nGenerating IEEE-quality plots...")
    generate_plots(results, wall_results, comp_toa, raw_toa, output_dir)

    print(f"\nAll results saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
