#!/usr/bin/env python3
"""
TEST 02 v2 — Packet Reliability (SNR Floor)
=============================================
Sweeps SNR on the REAL EPFL tx_rx_simulation.grc flowgraph
and measures actual PDR at each point.

Then applies (PDR_single)^9 to show multi-packet baseline failure rate.

Usage:
  cd ~/LoRa_Fl_Compression.proj
  python3 test02v2/run_test02.py
"""

import numpy as np
import os
import sys
import re
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
HT03_DIR     = PROJECT_ROOT / 'hardwaretest03'
LORA_DIR     = HT03_DIR / 'lora'

INSTALLED_GRC = Path('/usr/local/share/gnuradio/examples/lora_sdr/tx_rx_simulation.grc')
COMPILED_PY   = LORA_DIR / 'tx_rx_simulation.py'
TX_PAYLOAD    = LORA_DIR / 'tx_payload.txt'

# Focused sweep range — skip the dead zone below -10
SNR_VALUES = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 2, 5, 10]

N_TRIALS = 10  # per SNR point

COMPRESSED_BYTES = 242
RAW_BYTES        = 2212
LORA_MAX_PAYLOAD = 255
PACKETS_RAW      = int(np.ceil(RAW_BYTES / LORA_MAX_PAYLOAD))  # 9

ASCII_CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def compile_flowgraph():
    if not INSTALLED_GRC.exists():
        print(f"ERROR: {INSTALLED_GRC} not found")
        sys.exit(1)

    print(f"[TEST02] Compiling EPFL flowgraph...")
    result = subprocess.run(
        ['grcc', '-o', str(LORA_DIR), str(INSTALLED_GRC)],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"grcc error: {result.stderr[:500]}")
        sys.exit(1)

    if not COMPILED_PY.exists():
        print(f"ERROR: {COMPILED_PY.name} not found after grcc")
        sys.exit(1)

    with open(COMPILED_PY, 'r') as f:
        code = f.read()

    # Patch input file path
    code = re.sub(
        r"['\"][^'\"]*example_tx_source\.txt['\"]",
        f"'{str(TX_PAYLOAD)}'",
        code
    )
    code = re.sub(
        r"['\"][^'\"]*jtappare[^'\"]*['\"]",
        f"'{str(TX_PAYLOAD)}'",
        code
    )

    with open(COMPILED_PY, 'w') as f:
        f.write(code)

    print(f"[TEST02] Compiled and patched -> {COMPILED_PY.name}")


def write_test_payload():
    rng = np.random.default_rng(42)
    payload_bytes = rng.integers(0, 256, size=COMPRESSED_BYTES, dtype=np.uint8)
    ascii_payload = ''.join(ASCII_CHARSET[b % len(ASCII_CHARSET)] for b in payload_bytes)
    with open(TX_PAYLOAD, 'w') as f:
        f.write(ascii_payload + ',')
    print(f"[TEST02] Test payload: {COMPRESSED_BYTES} bytes -> {len(ascii_payload)} chars")


def patch_snr(snr_db):
    with open(COMPILED_PY, 'r') as f:
        code = f.read()
    code = re.sub(
        r'self\.SNRdB\s*=\s*SNRdB\s*=\s*[-\d.]+',
        f'self.SNRdB = SNRdB = {snr_db}',
        code
    )
    with open(COMPILED_PY, 'w') as f:
        f.write(code)


def run_single_trial(snr_db, trial_num):
    try:
        proc = subprocess.Popen(
            [sys.executable, str(COMPILED_PY)],
            cwd=str(LORA_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(15)

        try:
            stdout_bytes, stderr_bytes = proc.communicate(input=b'\n', timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_bytes, stderr_bytes = proc.communicate()

        # Decode with error handling
        stdout = stdout_bytes.decode('utf-8', errors='replace')
        stderr = stderr_bytes.decode('utf-8', errors='replace')

        crc_ok = False
        rx_found = False

        for line in stdout.split('\n'):
            if 'rx msg:' in line:
                rx_found = True
            if 'CRC valid' in line:
                crc_ok = True

        return crc_ok, rx_found

    except Exception as e:
        print(f"    Trial {trial_num} error: {e}")
        return False, False


def run_snr_sweep():
    results = []

    print(f"\n{'='*60}")
    print(f"SNR SWEEP: {SNR_VALUES[0]}dB to {SNR_VALUES[-1]}dB")
    print(f"Trials per point: {N_TRIALS}")
    print(f"Payload: {COMPRESSED_BYTES} bytes | Baseline: {PACKETS_RAW} packets")
    print(f"{'='*60}")

    print(f"\n{'SNR (dB)':>10} | {'CRC OK':>8} | {'Trials':>8} | "
          f"{'PDR 1-pkt':>10} | {'PDR 9-pkt':>10}")
    print(f"{'-'*55}")

    for snr in SNR_VALUES:
        patch_snr(snr)
        crc_ok_count = 0

        for trial in range(N_TRIALS):
            ok, _ = run_single_trial(snr, trial + 1)
            if ok:
                crc_ok_count += 1
            sys.stdout.write(f"\r  SNR={snr:>4}dB: trial {trial+1}/{N_TRIALS} "
                           f"({crc_ok_count} OK so far)   ")
            sys.stdout.flush()

        pdr_single = crc_ok_count / N_TRIALS
        pdr_multi = pdr_single ** PACKETS_RAW

        results.append({
            'snr_db': snr,
            'crc_ok': crc_ok_count,
            'trials': N_TRIALS,
            'pdr_single': pdr_single * 100,
            'pdr_multi': pdr_multi * 100,
        })

        print(f"\r{snr:>10} | {crc_ok_count:>8} | {N_TRIALS:>8} | "
              f"{pdr_single*100:>9.1f}% | {pdr_multi*100:>9.1f}%")

    return results


def generate_plots(results, output_dir):
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

    snrs = [r['snr_db'] for r in results]
    pdr_single = [r['pdr_single'] for r in results]
    pdr_multi = [r['pdr_multi'] for r in results]

    # Plot 1: PDR vs SNR
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(snrs, pdr_single, 'o-', color='#1a5276', markersize=6,
            label=f'ME-CFL (1 packet, {COMPRESSED_BYTES}B)')
    ax.plot(snrs, pdr_multi, 's--', color='#e67e22', markersize=6,
            label=f'Baseline ({PACKETS_RAW} packets, {RAW_BYTES}B)')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Packet Delivery Ratio (%)')
    ax.set_title('Measured PDR vs SNR (EPFL gr-lora_sdr Flowgraph)')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([-5, 105])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_pdr_vs_snr.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_pdr_vs_snr.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_pdr_vs_snr.pdf")

    # Plot 2: Reliability gap
    fig, ax = plt.subplots(figsize=(9, 5))

    gap = [s - m for s, m in zip(pdr_single, pdr_multi)]
    ax.fill_between(snrs, 0, gap, alpha=0.3, color='#27ae60',
                     label='Reliability advantage (ME-CFL)')
    ax.plot(snrs, gap, 'o-', color='#27ae60', markersize=5)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('PDR Advantage (percentage points)')
    ax.set_title('Reliability Gain: Single-Packet vs Multi-Packet')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_reliability_gap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_reliability_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_reliability_gap.pdf")

    # Plot 3: Bar chart at key SNR values
    fig, ax = plt.subplots(figsize=(9, 5))

    key_snrs = [s for s in snrs if s in [-8, -6, -5, -4, -2, 0, 5]]
    if not key_snrs:
        key_snrs = snrs[::2]  # every other point

    comp_vals = []
    raw_vals = []
    for snr in key_snrs:
        r = next((x for x in results if x['snr_db'] == snr), None)
        if r:
            comp_vals.append(r['pdr_single'])
            raw_vals.append(r['pdr_multi'])

    x = np.arange(len(key_snrs))
    width = 0.35

    bars1 = ax.bar(x - width/2, comp_vals, width,
                   label='ME-CFL (1 packet)', color='#1a5276',
                   edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, raw_vals, width,
                   label=f'Baseline ({PACKETS_RAW} packets)', color='#e67e22',
                   edgecolor='white', linewidth=1.5)

    for bar in bars1:
        if bar.get_height() > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{bar.get_height():.0f}%', ha='center', fontsize=9,
                    fontweight='bold', color='#1a5276')
    for bar in bars2:
        if bar.get_height() > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{bar.get_height():.0f}%', ha='center', fontsize=9,
                    fontweight='bold', color='#e67e22')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Round Success Rate (%)')
    ax.set_title('Success Rate at Key SNR Levels (Measured)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s} dB' for s in key_snrs])
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim([0, 115])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_bar_snr.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_bar_snr.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_bar_snr.pdf")


def print_summary(results):
    print(f"\n{'='*70}")
    print(f"TEST 02 v2 RESULTS — PACKET RELIABILITY (MEASURED)")
    print(f"{'='*70}")

    print(f"\n{'SNR (dB)':>10} | {'1-pkt PDR':>12} | {'9-pkt PDR':>12} | "
          f"{'Advantage':>12}")
    print(f"{'-'*50}")

    for r in results:
        gap = r['pdr_single'] - r['pdr_multi']
        print(f"{r['snr_db']:>10} | {r['pdr_single']:>11.1f}% | "
              f"{r['pdr_multi']:>11.1f}% | {gap:>11.1f}pp")

    # Find cliff
    cliff = None
    for i, r in enumerate(results):
        if r['pdr_single'] > 0 and i > 0 and results[i-1]['pdr_single'] == 0:
            cliff = r['snr_db']
            break

    # Find where single > 80% but multi < 50%
    sweet_spot = None
    for r in results:
        if r['pdr_single'] >= 80 and r['pdr_multi'] < 50:
            sweet_spot = r
            break

    print(f"\n{'HEADLINE NUMBERS'}")
    print(f"{'-'*70}")
    if cliff:
        print(f"PDR cliff (first successful packet): SNR = {cliff} dB")
    if sweet_spot:
        print(f"Sweet spot (SNR = {sweet_spot['snr_db']} dB):")
        print(f"  1-packet: {sweet_spot['pdr_single']:.1f}% success")
        print(f"  9-packet: {sweet_spot['pdr_multi']:.1f}% success")
        print(f"  Advantage: {sweet_spot['pdr_single'] - sweet_spot['pdr_multi']:.1f} pp")

    # At best SNR
    best = results[-1]
    print(f"At {best['snr_db']} dB (clear channel):")
    print(f"  1-packet: {best['pdr_single']:.1f}%")
    print(f"  9-packet: {best['pdr_multi']:.1f}%")


def main():
    print("=" * 60)
    print("TEST 02 v2 — PACKET RELIABILITY (MEASURED SNR SWEEP)")
    print("Using REAL EPFL gr-lora_sdr flowgraph")
    print("=" * 60)

    output_dir = str(Path(__file__).parent / 'results')

    compile_flowgraph()
    write_test_payload()
    results = run_snr_sweep()
    print_summary(results)

    print(f"\nGenerating IEEE-quality plots...")
    generate_plots(results, output_dir)

    print(f"\nAll results saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
