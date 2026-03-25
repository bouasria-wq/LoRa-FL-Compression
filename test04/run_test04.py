#!/usr/bin/env python3
"""
TEST 04 — Energy / Battery Life
==================================
Compares LoRa+Compression vs WiFi (ESP32) vs Zigbee for
federated learning model transmission.

Uses real datasheet power values to calculate:
  - Energy per transmission round
  - Daily energy consumption
  - Projected battery life (2000mAh coin cell / AA battery)
  - Cost per node

Key insight:
  LoRa TX is slow but ultra-low power in sleep.
  WiFi TX is fast but idles at high power.
  With Hegazy compression (242 bytes), LoRa needs only 1 short TX.
  WiFi wastes energy on connection setup + idle listening.

Datasheet sources:
  - LoRa SX1276: TX 50mA @ 14dBm, RX 12mA, Sleep 0.2µA
  - WiFi ESP32:  TX 240mA, RX 95mA, Idle 20mA, Sleep 10µA
  - Zigbee CC2530: TX 34mA, RX 24mA, Sleep 1µA, max payload 104B

Usage:
  cd ~/LoRa_Fl_Compression.proj
  python3 test04/run_test04.py

File: test04/run_test04.py
"""

import numpy as np
import os
from pathlib import Path

# ─── Battery ──────────────────────────────────────────────────────
BATTERY_MAH  = 2000    # typical AA or 18650 cell
VOLTAGE      = 3.3     # V
BATTERY_WH   = BATTERY_MAH * VOLTAGE / 1000  # Wh
BATTERY_J    = BATTERY_WH * 3600              # Joules

# ─── FL parameters ────────────────────────────────────────────────
COMPRESSED_BYTES = 242
RAW_BYTES        = 2212
ROUNDS_PER_DAY   = 1    # one FL round per day
LORA_MAX_PAYLOAD = 255
ZIGBEE_MAX_PAYLOAD = 104

# ─── LoRa SX1276 (datasheet values) ──────────────────────────────
LORA_TX_MA      = 50      # mA at 14 dBm
LORA_RX_MA      = 12      # mA
LORA_SLEEP_MA   = 0.0002  # 0.2 µA
LORA_SF         = 7
LORA_BW         = 125000
LORA_CR         = 2
LORA_PREAMBLE   = 8
LORA_MODULE_COST = 5      # USD (SX1276 module)

# ─── WiFi ESP32 (datasheet values) ───────────────────────────────
WIFI_TX_MA      = 240     # mA
WIFI_RX_MA      = 95      # mA
WIFI_IDLE_MA    = 20      # mA (connected, idle)
WIFI_SLEEP_MA   = 0.01    # 10 µA deep sleep
WIFI_CONNECT_S  = 3.0     # seconds to connect to AP
WIFI_TX_RATE    = 1e6     # ~1 Mbps effective throughput (TCP overhead)
WIFI_MODULE_COST = 15     # USD (ESP32 module)

# ─── Zigbee CC2530 (datasheet values) ────────────────────────────
ZB_TX_MA        = 34      # mA
ZB_RX_MA        = 24      # mA
ZB_SLEEP_MA     = 0.001   # 1 µA
ZB_TX_RATE      = 250000  # 250 kbps
ZB_MODULE_COST  = 8       # USD


def lora_toa(payload_bytes):
    """LoRa time on air."""
    n_preamble = LORA_PREAMBLE
    numerator = 8 * payload_bytes - 4 * LORA_SF + 28 + 16
    denominator = 4 * (LORA_SF - 2)
    payload_symbols = max(np.ceil(numerator / denominator) * (LORA_CR + 4), 0)
    n_symbols = n_preamble + payload_symbols
    return (n_symbols * (2 ** LORA_SF)) / LORA_BW


def packets_needed(total_bytes, max_payload):
    return int(np.ceil(total_bytes / max_payload))


def calculate_lora_energy(payload_bytes):
    """Energy for one LoRa transmission (compressed)."""
    n_pkts = packets_needed(payload_bytes, LORA_MAX_PAYLOAD)
    toa = n_pkts * lora_toa(min(payload_bytes, LORA_MAX_PAYLOAD))
    tx_energy = LORA_TX_MA * VOLTAGE * toa / 1000   # mJ -> J? No, keep mJ
    # TX energy in mJ
    tx_mj = LORA_TX_MA * VOLTAGE * toa  # mA * V * s = mW*s = mJ
    # RX energy (listening for server broadcast — similar duration)
    rx_mj = LORA_RX_MA * VOLTAGE * toa
    return {
        'tx_time_s': toa,
        'tx_energy_mj': tx_mj,
        'rx_energy_mj': rx_mj,
        'total_active_mj': tx_mj + rx_mj,
        'n_packets': n_pkts,
    }


def calculate_wifi_energy(payload_bytes):
    """Energy for one WiFi transmission."""
    # Connection setup
    connect_mj = WIFI_TX_MA * VOLTAGE * WIFI_CONNECT_S

    # TX time (very fast)
    tx_time = payload_bytes * 8 / WIFI_TX_RATE
    tx_mj = WIFI_TX_MA * VOLTAGE * tx_time

    # Idle time waiting for server response (~2 seconds)
    idle_time = 2.0
    idle_mj = WIFI_IDLE_MA * VOLTAGE * idle_time

    # RX time (receive global model)
    rx_time = payload_bytes * 8 / WIFI_TX_RATE
    rx_mj = WIFI_RX_MA * VOLTAGE * rx_time

    total_active_time = WIFI_CONNECT_S + tx_time + idle_time + rx_time

    return {
        'tx_time_s': total_active_time,
        'connect_mj': connect_mj,
        'tx_energy_mj': tx_mj,
        'idle_mj': idle_mj,
        'rx_energy_mj': rx_mj,
        'total_active_mj': connect_mj + tx_mj + idle_mj + rx_mj,
        'n_packets': 1,
    }


def calculate_zigbee_energy(payload_bytes):
    """Energy for one Zigbee transmission."""
    n_pkts = packets_needed(payload_bytes, ZIGBEE_MAX_PAYLOAD)
    tx_time_per_pkt = ZIGBEE_MAX_PAYLOAD * 8 / ZB_TX_RATE
    total_tx_time = n_pkts * tx_time_per_pkt

    tx_mj = ZB_TX_MA * VOLTAGE * total_tx_time
    rx_mj = ZB_RX_MA * VOLTAGE * total_tx_time  # listening for ACKs

    return {
        'tx_time_s': total_tx_time * 2,  # TX + RX
        'tx_energy_mj': tx_mj,
        'rx_energy_mj': rx_mj,
        'total_active_mj': tx_mj + rx_mj,
        'n_packets': n_pkts,
    }


def calculate_daily_energy(active_mj, sleep_ma):
    """Total daily energy: one active round + 24h sleep."""
    seconds_per_day = 86400
    # Active time is negligible compared to 24h
    # Sleep energy for the rest of the day
    sleep_mj = sleep_ma * VOLTAGE * seconds_per_day  # mA * V * s = mJ
    return active_mj + sleep_mj


def battery_life_days(daily_mj):
    """How many days until battery dies."""
    battery_mj = BATTERY_MAH * VOLTAGE * 3600 / 1000  # mAh * V * 3600s / 1000 = mJ... 
    # Actually: mAh * V = mWh, * 3.6 = mJ... let me be careful
    # 2000 mAh * 3.3V = 6600 mWh = 6.6 Wh
    # 6.6 Wh * 3600 s/h * 1000 mJ/J = 23,760,000 mJ
    battery_mj = BATTERY_MAH * VOLTAGE * 3600  # = 23,760,000 mJ
    return battery_mj / daily_mj


def run_analysis():
    """Run full energy analysis for all three radios."""
    print(f"\n{'='*60}")
    print(f"ENERGY ANALYSIS — PER ROUND (Compressed: {COMPRESSED_BYTES}B)")
    print(f"{'='*60}")

    lora = calculate_lora_energy(COMPRESSED_BYTES)
    wifi = calculate_wifi_energy(COMPRESSED_BYTES)
    zigbee = calculate_zigbee_energy(COMPRESSED_BYTES)

    # Also calculate raw (uncompressed) LoRa for comparison
    lora_raw = calculate_lora_energy(RAW_BYTES)

    results = {
        'LoRa+Compression': {
            **lora, 'sleep_ma': LORA_SLEEP_MA,
            'cost': LORA_MODULE_COST, 'payload': COMPRESSED_BYTES,
        },
        'LoRa (Raw)': {
            **lora_raw, 'sleep_ma': LORA_SLEEP_MA,
            'cost': LORA_MODULE_COST, 'payload': RAW_BYTES,
        },
        'WiFi (ESP32)': {
            **wifi, 'sleep_ma': WIFI_SLEEP_MA,
            'cost': WIFI_MODULE_COST, 'payload': COMPRESSED_BYTES,
        },
        'Zigbee': {
            **zigbee, 'sleep_ma': ZB_SLEEP_MA,
            'cost': ZB_MODULE_COST, 'payload': COMPRESSED_BYTES,
        },
    }

    # Print per-round
    print(f"\n{'Radio':<22} {'Packets':<10} {'TX Time':<12} "
          f"{'Active mJ':<12} {'Sleep µA':<10}")
    print(f"{'-'*66}")
    for name, r in results.items():
        print(f"{name:<22} {r['n_packets']:<10} "
              f"{r['tx_time_s']:<12.4f} "
              f"{r['total_active_mj']:<12.2f} "
              f"{r['sleep_ma']*1000:<10.1f}")

    # Daily energy
    print(f"\n{'='*60}")
    print(f"DAILY ENERGY (1 round/day + 24h sleep)")
    print(f"{'='*60}")

    daily = {}
    for name, r in results.items():
        d = calculate_daily_energy(r['total_active_mj'], r['sleep_ma'])
        daily[name] = d
        life = battery_life_days(d)
        print(f"{name:<22} {d:>10.1f} mJ/day | "
              f"Battery: {life:>8.0f} days ({life/365:.1f} years)")

    # Cost analysis for 10-home deployment
    print(f"\n{'='*60}")
    print(f"10-HOME DEPLOYMENT COST")
    print(f"{'='*60}")
    for name, r in results.items():
        total_cost = r['cost'] * 10
        print(f"{name:<22} ${r['cost']}/node × 10 = ${total_cost}")

    return results, daily


def generate_plots(results, daily, output_dir):
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
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    os.makedirs(output_dir, exist_ok=True)

    names = list(results.keys())
    colors = ['#1a5276', '#5dade2', '#e67e22', '#27ae60']

    # ─── Plot 1: Energy per round (grouped bar) ──────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(names))
    active_mj = [results[n]['total_active_mj'] for n in names]

    bars = ax.bar(x, active_mj, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, active_mj):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f} mJ', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Radio Technology')
    ax.set_ylabel('Energy per FL Round (mJ)')
    ax.set_title('Active Energy Cost per Federated Learning Round')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_energy_per_round.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_energy_per_round.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_energy_per_round.pdf")

    # ─── Plot 2: Battery life (bar chart) ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    life_days = [battery_life_days(daily[n]) for n in names]
    life_years = [d / 365 for d in life_days]

    bars = ax.bar(x, life_years, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.5)

    for bar, val_y, val_d in zip(bars, life_years, life_days):
        if val_y >= 1:
            label = f'{val_y:.1f} yr'
        else:
            label = f'{val_d:.0f} days'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                label, ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Radio Technology')
    ax.set_ylabel('Projected Battery Life (Years)')
    ax.set_title(f'Battery Life with {BATTERY_MAH}mAh Battery (1 FL Round/Day)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_battery_life.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_battery_life.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_battery_life.pdf")

    # ─── Plot 3: Cost vs Battery Life (scatter) ──────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, name in enumerate(names):
        cost = results[name]['cost']
        life = battery_life_days(daily[name]) / 365
        ax.scatter(cost, life, color=colors[i], s=200, zorder=5,
                   edgecolors='white', linewidth=2)
        ax.annotate(name, xy=(cost, life),
                    xytext=(cost + 0.5, life + 0.15),
                    fontsize=10, fontweight='bold', color=colors[i])

    ax.set_xlabel('Module Cost per Node (USD)')
    ax.set_ylabel('Projected Battery Life (Years)')
    ax.set_title('Cost vs Battery Life — Radio Technology Comparison')

    # Ideal region
    ax.annotate('IDEAL\n(low cost, long life)',
                xy=(3, max(life_years) * 0.9),
                fontsize=10, color='gray', fontstyle='italic', ha='center')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_cost_vs_battery.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_cost_vs_battery.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_cost_vs_battery.pdf")

    # ─── Plot 4: Comparison table as figure ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    col_labels = ['Metric', 'LoRa+Comp', 'LoRa Raw', 'WiFi', 'Zigbee']
    row_data = [
        ['Payload (B)', f'{COMPRESSED_BYTES}', f'{RAW_BYTES}',
         f'{COMPRESSED_BYTES}', f'{COMPRESSED_BYTES}'],
        ['Packets/round', str(results[names[0]]['n_packets']),
         str(results[names[1]]['n_packets']),
         str(results[names[2]]['n_packets']),
         str(results[names[3]]['n_packets'])],
        ['Active energy (mJ)',
         f"{results[names[0]]['total_active_mj']:.1f}",
         f"{results[names[1]]['total_active_mj']:.1f}",
         f"{results[names[2]]['total_active_mj']:.1f}",
         f"{results[names[3]]['total_active_mj']:.1f}"],
        ['Battery life',
         f"{battery_life_days(daily[names[0]])/365:.1f} yr",
         f"{battery_life_days(daily[names[1]])/365:.1f} yr",
         f"{battery_life_days(daily[names[2]])/365:.1f} yr",
         f"{battery_life_days(daily[names[3]])/365:.1f} yr"],
        ['Module cost',
         f"${results[names[0]]['cost']}",
         f"${results[names[1]]['cost']}",
         f"${results[names[2]]['cost']}",
         f"${results[names[3]]['cost']}"],
    ]

    table = ax.table(cellText=row_data, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1a5276')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight LoRa+Comp column
    for i in range(1, len(row_data) + 1):
        table[i, 1].set_facecolor('#d4e6f1')

    ax.set_title('Radio Technology Comparison Summary', fontsize=13,
                 fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_comparison_table.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig4_comparison_table.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_comparison_table.pdf")


def print_summary(results, daily):
    """Print final summary."""
    names = list(results.keys())

    print(f"\n{'='*70}")
    print(f"TEST 04 RESULTS — ENERGY / BATTERY LIFE")
    print(f"{'='*70}")

    lora_life = battery_life_days(daily[names[0]])
    wifi_life = battery_life_days(daily[names[2]])
    zb_life   = battery_life_days(daily[names[3]])

    print(f"\n{'HEADLINE NUMBERS'}")
    print(f"{'-'*70}")
    print(f"LoRa + Compression:  {lora_life/365:.1f} years on {BATTERY_MAH}mAh battery")
    print(f"WiFi (ESP32):        {wifi_life:.0f} days ({wifi_life/30:.1f} months)")
    print(f"Zigbee:              {zb_life/365:.1f} years")
    print(f"\nLoRa advantage over WiFi: {lora_life/wifi_life:.0f}× longer battery life")
    print(f"LoRa cost advantage: ${results[names[2]]['cost'] - results[names[0]]['cost']}/node cheaper")


def main():
    print("=" * 60)
    print("TEST 04 — ENERGY / BATTERY LIFE")
    print("LoRa+Compression vs WiFi vs Zigbee")
    print("=" * 60)

    output_dir = str(Path(__file__).parent / 'results')

    results, daily = run_analysis()
    print_summary(results, daily)

    print(f"\nGenerating IEEE-quality plots...")
    generate_plots(results, daily, output_dir)

    print(f"\nAll results saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
