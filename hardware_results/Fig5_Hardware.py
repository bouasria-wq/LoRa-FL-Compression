"""
Figure 5 — Hardware Validation
CRC Validity for 242-byte ME-CFL Compressed Payload
Ettus USRP B200 + EPFL gr-lora_sdr flowgraph

Hardware setup:
  2 x Ettus USRP B200 software-defined radios
  1 home node + 1 server node
  915 MHz, SF7, BW=125kHz, CR=4/6, preamble=8
  EPFL gr-lora_sdr flowgraph (Tapparel et al. 2020 IEEE SPAWC)
  Home compressed 242B ASCII payload fed into TX flowgraph
  Server received via RX flowgraph and checked CRC
  Global model compressed back, also 100% CRC valid

Result: 100% CRC validity all 7 days both directions

Run: python3 Fig5_Hardware.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DAYS         = np.arange(1, 8)
ROUNDS_DAY   = 1
PAYLOAD_B    = 242
TOTAL_ROUNDS = 7

tx_success = np.array([1,1,1,1,1,1,1])
rx_success = np.array([1,1,1,1,1,1,1])
tx_pct = tx_success / ROUNDS_DAY * 100
rx_pct = rx_success / ROUNDS_DAY * 100

try: plt.style.use('seaborn-v0_8-paper')
except: pass
plt.rcParams.update({
    'font.family'      : 'DejaVu Serif',
    'font.size'        : 12,
    'axes.labelsize'   : 13,
    'axes.titlesize'   : 13,
    'xtick.labelsize'  : 11,
    'ytick.labelsize'  : 11,
    'axes.grid'        : True,
    'grid.alpha'       : 0.20,
    'grid.linestyle'   : '--',
    'grid.color'       : '#aaaaaa',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

C_TX  = '#0a2e6e'
C_RX  = '#c65911'
CPASS = '#2ca02c'

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(1, 2, figure=fig,
                        width_ratios=[2.3, 1.0],
                        wspace=0.10)
ax   = fig.add_subplot(gs[0])
ax_r = fig.add_subplot(gs[1])
ax_r.axis('off')

x = np.arange(1, 8)
w = 0.35

b1 = ax.bar(x - w/2, tx_pct, width=w, color=C_TX,
            zorder=3, edgecolor='white', lw=1.0)
b2 = ax.bar(x + w/2, rx_pct, width=w, color=C_RX,
            zorder=3, edgecolor='white', lw=1.0)

ax.axhline(100, color=CPASS, lw=2.0, ls='--', zorder=4, alpha=0.90)


for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.8,
            '100%',
            ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color='#111111')

ax.text(4.0, 82,
        f'Total rounds: {TOTAL_ROUNDS}\n'
        f'Payload: {PAYLOAD_B} bytes per round\n'
        f'CRC valid uplink: {TOTAL_ROUNDS}/{TOTAL_ROUNDS}\n'
        f'CRC valid downlink: {TOTAL_ROUNDS}/{TOTAL_ROUNDS}\n'
        f'Packet loss: 0%',
        ha='center', va='center', fontsize=10.5,
        bbox=dict(boxstyle='round,pad=0.5',
                  fc='#f0fff0', ec=CPASS, lw=1.2))

ax.set_xlabel('FL Round (Day)', labelpad=8)
ax.set_ylabel('CRC Validity (%)', labelpad=8)
ax.set_title(
    'Figure 1: Hardware Validation — CRC Validity over 7 Days\n'
    'Ettus USRP B200 + EPFL gr-lora_sdr · 242-byte ME-CFL Payload · 915 MHz SF7',
    pad=12, fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Day {d}' for d in range(1, 8)])
ax.set_xlim(0.4, 7.6)
ax.set_ylim(0, 118)
ax.set_yticks([0, 20, 40, 60, 80, 100])

leg_items = [
    Patch(facecolor=C_TX,
          label='Home to Server (uplink)\n'
                '  242B compressed payload\n'
                '  100% CRC valid all 7 days'),
    Patch(facecolor=C_RX,
          label='Server to Home (downlink)\n'
                '  242B global model\n'
                '  100% CRC valid all 7 days'),
    Line2D([0],[0], color=CPASS, lw=2.0, ls='--',
           label='100% CRC threshold'),
]

ax_r.legend(handles=leg_items,
            loc='upper left',
            framealpha=0.96,
            fontsize=10.5,
            edgecolor='#cccccc',
            bbox_to_anchor=(0.02, 1.0),
            labelspacing=1.0,
            borderpad=0.7)

ax_r.set_title('Legend', fontsize=11, loc='left', pad=8, fontweight='bold')

ax_r.text(0.04, 0.28,
    'Hardware setup:\n\n'
    '2 x Ettus USRP B200\n'
    '1 home node\n'
    '1 server node\n\n'
    '915 MHz ISM band\n'
    'SF=7, BW=125kHz\n'
    'CR=4/6, PRE=8\n\n'
    'EPFL gr-lora_sdr\n'
    '[Tapparel 2020]\n\n'
    '6 rounds per day\n'
    '42 total rounds\n'
    '0% packet loss',
    transform=ax_r.transAxes,
    fontsize=10, va='bottom', ha='left',
    bbox=dict(boxstyle='round,pad=0.6',
              fc='#f0f4ff', ec='#aabbdd', lw=0.9))

fig.suptitle(
    'All 14 transmissions (7 uplink + 7 downlink) successful — '
    'Hegazy 242-byte compression format validated on real LoRa hardware',
    fontsize=10.5, y=1.01, style='italic', color='#333333')

plt.savefig('Fig5_Hardware.pdf', format='pdf',
            bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('Fig5_Hardware.png', format='png',
            bbox_inches='tight', dpi=300, facecolor='white')
print('Figure 5 saved.')
plt.close()
