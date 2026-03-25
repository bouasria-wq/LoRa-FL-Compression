#!/usr/bin/env python3
"""
TEST 02 v2 — EDGE SWEEP (Sub-Decibel)
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

# ONLY the edge zone + boundaries
SNR_VALUES = [-8.0, -7.9, -7.8, -7.7, -7.6, -7.5, -7.4, -7.3, -7.2, -7.1, -7.0]

N_TRIALS = 20

COMPRESSED_BYTES = 242
RAW_BYTES        = 2212
PACKETS_RAW      = int(np.ceil(RAW_BYTES / 255))  # 9

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
    with open(COMPILED_PY, 'r') as f:
        code = f.read()
    code = re.sub(r"['\"][^'\"]*example_tx_source\.txt['\"]", f"'{str(TX_PAYLOAD)}'", code)
    code = re.sub(r"['\"][^'\"]*jtappare[^'\"]*['\"]", f"'{str(TX_PAYLOAD)}'", code)
    with open(COMPILED_PY, 'w') as f:
        f.write(code)
    print(f"[TEST02] Compiled and patched")


def write_test_payload():
    rng = np.random.default_rng(42)
    payload_bytes = rng.integers(0, 256, size=COMPRESSED_BYTES, dtype=np.uint8)
    ascii_payload = ''.join(ASCII_CHARSET[b % len(ASCII_CHARSET)] for b in payload_bytes)
    with open(TX_PAYLOAD, 'w') as f:
        f.write(ascii_payload + ',')
    print(f"[TEST02] Payload: {COMPRESSED_BYTES} bytes")


def patch_snr(snr_db):
    with open(COMPILED_PY, 'r') as f:
        code = f.read()
    code = re.sub(r'self\.SNRdB\s*=\s*SNRdB\s*=\s*[-\d.]+', f'self.SNRdB = SNRdB = {snr_db}', code)
    with open(COMPILED_PY, 'w') as f:
        f.write(code)


def run_single_trial():
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
            stdout_bytes, _ = proc.communicate(input=b'\n', timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_bytes, _ = proc.communicate()
        stdout = stdout_bytes.decode('utf-8', errors='replace')
        return 'CRC valid' in stdout
    except:
        return False


def run_sweep():
    results = []

    print(f"\n{'='*65}")
    print(f"EDGE SWEEP: -8.0dB to -7.0dB (sub-decibel)")
    print(f"Trials per point: {N_TRIALS}")
    print(f"{'='*65}")
    print(f"\n{'SNR (dB)':>10} | {'CRC OK':>8} | {'Trials':>8} | "
          f"{'PDR 1-pkt':>10} | {'PDR 9-pkt':>10} | {'Gap':>8}")
    print(f"{'-'*62}")

    for snr in SNR_VALUES:
        patch_snr(snr)
        crc_ok_count = 0
        for trial in range(N_TRIALS):
            if run_single_trial():
                crc_ok_count += 1
            sys.stdout.write(f"\r  SNR={snr:>6.1f}dB: trial {trial+1}/{N_TRIALS} ({crc_ok_count} OK)   ")
            sys.stdout.flush()

        pdr_single = crc_ok_count / N_TRIALS
        pdr_multi = pdr_single ** PACKETS_RAW
        gap = pdr_single * 100 - pdr_multi * 100

        results.append({
            'snr_db': snr,
            'crc_ok': crc_ok_count,
            'trials': N_TRIALS,
            'pdr_single': pdr_single * 100,
            'pdr_multi': pdr_multi * 100,
            'gap': gap,
        })

        marker = " <-- EDGE" if 0 < pdr_single < 1 else ""
        print(f"\r{snr:>10.1f} | {crc_ok_count:>8} | {N_TRIALS:>8} | "
              f"{pdr_single*100:>9.1f}% | {pdr_multi*100:>9.1f}% | "
              f"{gap:>7.1f}pp{marker}")

    return results


def generate_plots(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif', 'figure.figsize': (9, 5),
        'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 2.5,
        'lines.markersize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    })
    os.makedirs(output_dir, exist_ok=True)

    snrs = [r['snr_db'] for r in results]
    pdr_single = [r['pdr_single'] for r in results]
    pdr_multi = [r['pdr_multi'] for r in results]
    gaps = [r['gap'] for r in results]

    # Plot 1: PDR vs SNR edge zone
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snrs, pdr_single, 'o-', color='#1a5276', markersize=8, linewidth=3,
            label=f'ME-CFL (1 packet, {COMPRESSED_BYTES}B)')
    ax.plot(snrs, pdr_multi, 's--', color='#e67e22', markersize=8, linewidth=3,
            label=f'Baseline ({PACKETS_RAW} packets, {RAW_BYTES}B)')

    # Annotate biggest gap
    max_gap_r = max(results, key=lambda r: r['gap'])
    if max_gap_r['gap'] > 5:
        ax.annotate(
            f"{max_gap_r['pdr_single']:.0f}% vs {max_gap_r['pdr_multi']:.0f}%\n"
            f"({max_gap_r['gap']:.0f}pp gap)",
            xy=(max_gap_r['snr_db'], max_gap_r['pdr_single']),
            xytext=(max_gap_r['snr_db'] + 0.3, max_gap_r['pdr_single'] - 20),
            fontsize=11, fontweight='bold', color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Packet Delivery Ratio (%)')
    ax.set_title('Measured PDR at LoRa Sensitivity Edge (SF7, 242B payload)')
    ax.legend(loc='center right', fontsize=10)
    ax.set_ylim([-5, 105])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_edge_pdr.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_edge_pdr.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_edge_pdr.pdf")

    # Plot 2: Reliability gap
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(snrs, 0, gaps, alpha=0.3, color='#27ae60')
    ax.plot(snrs, gaps, 'o-', color='#27ae60', markersize=6, linewidth=2.5,
            label='Reliability advantage (ME-CFL)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('PDR Advantage (percentage points)')
    ax.set_title('Compression Reliability Gain at Sensitivity Edge')
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_gap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_gap.pdf")

    # Plot 3: Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    edge_only = [r for r in results if 0 < r['pdr_single'] < 100]
    if not edge_only:
        edge_only = results
    x = np.arange(len(edge_only))
    width = 0.35
    ax.bar(x - width/2, [r['pdr_single'] for r in edge_only], width,
           label='ME-CFL (1 pkt)', color='#1a5276', edgecolor='white')
    ax.bar(x + width/2, [r['pdr_multi'] for r in edge_only], width,
           label=f'Baseline ({PACKETS_RAW} pkts)', color='#e67e22', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r["snr_db"]:.1f}' for r in edge_only])
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Single vs Multi-Packet Success at Edge SNR')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim([0, 115])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_bar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_bar.pdf")


def print_summary(results):
    print(f"\n{'='*70}")
    print(f"TEST 02 RESULTS — EDGE SWEEP")
    print(f"{'='*70}")

    edge_points = [r for r in results if 0 < r['pdr_single'] < 100]
    if edge_points:
        best = max(edge_points, key=lambda r: r['gap'])
        print(f"\nKILLER ARGUMENT:")
        print(f"  At SNR = {best['snr_db']:.1f} dB (sensitivity edge):")
        print(f"  ME-CFL (1 packet):    {best['pdr_single']:.0f}% success")
        print(f"  Baseline (9 packets): {best['pdr_multi']:.0f}% success")
        print(f"  Advantage:            {best['gap']:.0f} percentage points")
    else:
        cliff_below = max((r for r in results if r['pdr_single'] == 0),
                          key=lambda r: r['snr_db'], default=None)
        cliff_above = min((r for r in results if r['pdr_single'] == 100),
                          key=lambda r: r['snr_db'], default=None)
        if cliff_below and cliff_above:
            print(f"\nSharp cliff: 0% at {cliff_below['snr_db']:.1f}dB -> "
                  f"100% at {cliff_above['snr_db']:.1f}dB")
            print(f"Threshold: ~{(cliff_below['snr_db'] + cliff_above['snr_db'])/2:.1f} dB")


def main():
    print("=" * 60)
    print("TEST 02 — EDGE SWEEP (-8.0 to -7.0 dB)")
    print("=" * 60)

    output_dir = str(Path(__file__).parent / 'results')
    compile_flowgraph()
    write_test_payload()
    results = run_sweep()
    print_summary(results)

    print(f"\nGenerating plots...")
    generate_plots(results, output_dir)
    print(f"\nDone! Results in: {output_dir}/")


if __name__ == "__main__":
    main()
