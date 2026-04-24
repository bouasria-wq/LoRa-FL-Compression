"""
Figure 2 — Radio Energy per FL Round
Clean version v3 — all review fixes applied
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────
LORA_V=3.3; LORA_I_TX=0.050; LORA_I_RX=0.012; LORA_I_SLEEP=0.0000002
SF=7; BW=125000; CR=2; PRE=8; LORA_MAX_PKT=255

def lora_toa(n_bytes):
    t_sym=(2**SF)/BW; t_pre=(PRE+4.25)*t_sym
    n_pay=max(np.ceil((8*n_bytes-4*SF+28+16)/(4*(SF-2)))*(CR+4),0)
    return t_pre+n_pay*t_sym

WIFI_V=3.3; WIFI_I_TX=0.240; WIFI_I_RX=0.095
WIFI_I_IDLE=0.020; WIFI_ASSOC_S=3.0; WIFI_RATE=1e6

ZB_V=3.3; ZB_I_TX=0.034; ZB_I_RX=0.024; ZB_I_SLEEP=0.000001
ZB_RATE=250000; ZB_MAX_DATA=104; ZB_ACK_S=0.00192; ZB_HOPS=2

ROUND_S=14400; COMPRESSED=242; UNCOMPRESSED=2212

def mJ(V,I,t): return V*I*t*1000

toa_c1=lora_toa(COMPRESSED)
E1_tx=mJ(LORA_V,LORA_I_TX,toa_c1)
E1_rx=mJ(LORA_V,LORA_I_RX,toa_c1)
E1_oh=0.0; n1=1

n2=int(np.ceil(UNCOMPRESSED/LORA_MAX_PKT))
toa_c2=n2*lora_toa(LORA_MAX_PKT)
E2_tx=mJ(LORA_V,LORA_I_TX,toa_c2)
E2_rx=mJ(LORA_V,LORA_I_RX,toa_c2)
E2_oh=0.0

tx_s=UNCOMPRESSED*8/WIFI_RATE; rx_s=tx_s
E3_tx=mJ(WIFI_V,WIFI_I_TX,tx_s)
E3_rx=mJ(WIFI_V,WIFI_I_RX,rx_s)
E3_oh=mJ(WIFI_V,WIFI_I_TX,WIFI_ASSOC_S)+mJ(WIFI_V,WIFI_I_IDLE,2.0)

n4=int(np.ceil(UNCOMPRESSED/ZB_MAX_DATA))
tx_s_zb=n4*(ZB_MAX_DATA*8/ZB_RATE); ack_s_zb=n4*ZB_ACK_S
E4_tx=mJ(ZB_V,ZB_I_TX,tx_s_zb)*ZB_HOPS
E4_rx=mJ(ZB_V,ZB_I_RX,ack_s_zb)
E4_oh=mJ(ZB_V,ZB_I_RX,tx_s_zb*(ZB_HOPS-1))

TX=np.array([E1_tx,E2_tx,E3_tx,E4_tx])
RX=np.array([E1_rx,E2_rx,E3_rx,E4_rx])
OH=np.array([E1_oh,E2_oh,E3_oh,E4_oh])
TOT=TX+RX+OH

SL=np.array([
    mJ(LORA_V,LORA_I_SLEEP,ROUND_S-toa_c1),
    mJ(LORA_V,LORA_I_SLEEP,ROUND_S-toa_c2),
    mJ(LORA_V,LORA_I_SLEEP,ROUND_S-tx_s-rx_s-WIFI_ASSOC_S-2.0),
    mJ(ZB_V,ZB_I_SLEEP,ROUND_S-tx_s_zb-ack_s_zb),
])
TOT_FULL=TOT+SL
BAT_J=2500*3.6*1.2*4*0.85; ROUNDS_DAY=6
bat_days=BAT_J/(TOT_FULL/1000*ROUNDS_DAY)

# ── Style ─────────────────────────────────────────────────────────
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
    'grid.color'       : '#999999',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

C1='#0a2e6e'; C2='#2a6fba'; C3='#c65911'; C4='#1a6b3a'
BASE=[C1,C2,C3,C4]

def lighten(h,f):
    r,g,b=int(h[1:3],16),int(h[3:5],16),int(h[5:7],16)
    return f'#{int(r+(255-r)*f):02x}{int(g+(255-g)*f):02x}{int(b+(255-b)*f):02x}'

def darken(h,f):
    r,g,b=int(h[1:3],16),int(h[3:5],16),int(h[5:7],16)
    return f'#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}'

NAMES=[
    'LoRa-FL\nCompressed\n(242B, 1 pkt)',
    'LoRa-FL\nUncompressed\n(2212B, 9 pkts)',
    'WiFi 802.11n\n(ESP32)',
    'Zigbee 802.15.4\n(CC2530, 2-hop)',
]

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(1, 2, figure=fig,
                        width_ratios=[2.2, 1.1],
                        wspace=0.10)
ax   = fig.add_subplot(gs[0])
ax_r = fig.add_subplot(gs[1])
ax_r.axis('off')

x=np.arange(4); w=0.55

# ── Stacked bars — darker TX, medium RX, light OH ────────────────
# TX — full colour
b_tx=ax.bar(x, TX, width=w,
            color=BASE, zorder=3,
            edgecolor=[darken(c,0.7) for c in BASE], lw=1.2)

# RX — lighter shade, clear contrast
b_rx=ax.bar(x, RX, width=w, bottom=TX,
            color=[lighten(c,0.45) for c in BASE],
            zorder=3, edgecolor=[darken(c,0.7) for c in BASE], lw=1.2)

# OH — even lighter, hatched
b_oh=ax.bar(x, OH, width=w, bottom=TX+RX,
            color=[lighten(c,0.70) for c in BASE],
            zorder=3, edgecolor=[darken(c,0.7) for c in BASE],
            lw=1.2, hatch='xxx')

# ── Total labels above bars — well clear ─────────────────────────
for i,bar in enumerate(b_tx):
    ypos = TOT[i] * 2.8
    ax.text(bar.get_x()+bar.get_width()/2,
            ypos,
            f'{TOT[i]:.1f} mJ',
            ha='center', va='bottom',
            fontsize=11.5, fontweight='bold',
            color='#111111')

# ── White text inside bars — with shadow for contrast ────────────
# FIX 6: stronger contrast for all bar text
info=[
    (n1, f'ToA={toa_c1*1000:.0f}ms'),
    (n2, f'ToA={toa_c2*1000:.0f}ms'),
    (1,  f'TX={tx_s*1000:.0f}ms'),
    (n4, f'{ZB_HOPS}-hop\n{tx_s_zb*1000:.0f}ms'),
]
for i,(bar,(npkt,label)) in enumerate(zip(b_tx,info)):
    if TX[i] > 0.5:
        # Dark outline effect for readability
        bx = bar.get_x()+bar.get_width()/2
        by = TX[i]*0.75
        txt = f'$n_{{pkt}}$={npkt}\n{label}'
        # Shadow
        ax.text(bx+0.008, by-0.008, txt,
                ha='center', va='center',
                fontsize=9.5, color='black', fontweight='bold',
                alpha=0.4, zorder=4)
        # Main text
        ax.text(bx, by, txt,
                ha='center', va='center',
                fontsize=9.5, color='white', fontweight='bold',
                zorder=5)

# ── Zigbee NOT VIABLE — higher up, not covering 25.6 mJ ──────────
zb_x = b_tx[3].get_x() + b_tx[3].get_width()/2
# 25.6 mJ label sits at TOT[3]*2.8, put NOT VIABLE above that
ax.text(zb_x,
        TOT[3] * 18,
        '✗ NOT VIABLE\nPDR = 0.5%\n(multi-hop\ncollapse\nPollin 2008)',
        ha='center', va='bottom',
        fontsize=10, fontweight='bold',
        color='#cc0000',
        bbox=dict(boxstyle='round,pad=0.45',
                  fc='#fff0f0', ec='#cc0000', lw=1.4))

# ── Y axis — fix 3: max at 10^4 ──────────────────────────────────
ax.set_yscale('log')
ax.set_ylim(8, 1e4)

# ── Axes labels ───────────────────────────────────────────────────
ax.set_ylabel('Active Radio Energy per FL Round (mJ) — log scale',
              labelpad=10)
ax.set_title('Figure 3: Radio Energy per FL Round — '
             'LoRa-FL Compressed vs IEEE Baselines',
             pad=12, fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(NAMES, fontsize=11)
ax.set_xlim(-0.6, 3.8)
ax.yaxis.grid(True, which='both', alpha=0.2)

# ── Right panel legend ────────────────────────────────────────────
legend_items=[
    Patch(facecolor=C1, edgecolor=darken(C1,0.7), lw=1.2,
          label=f'LoRa-FL Compressed\n'
                f'  TX: {TX[0]:.1f} mJ\n'
                f'  Total: {TOT[0]:.1f} mJ'),
    Patch(facecolor=C2, edgecolor=darken(C2,0.7), lw=1.2,
          label=f'LoRa-FL Uncompressed\n'
                f'  TX: {TX[1]:.1f} mJ\n'
                f'  Total: {TOT[1]:.1f} mJ'),
    Patch(facecolor=C3, edgecolor=darken(C3,0.7), lw=1.2,
          label=f'WiFi 802.11n (ESP32)\n'
                f'  TX: {TX[2]:.2f} mJ\n'
                f'  Total: {TOT[2]:.1f} mJ'),
    Patch(facecolor=C4, edgecolor=darken(C4,0.7), lw=1.2,
          label=f'Zigbee 802.15.4 (CC2530)\n'
                f'  TX: {TX[3]:.1f} mJ\n'
                f'  Total: {TOT[3]:.1f} mJ\n'
                f'  ✗ PDR=0.5% at N=10 homes'),
    Patch(facecolor='#bbbbbb', alpha=0.8,
          label='RX / ACK energy (lighter shade)'),
    Patch(facecolor='#dddddd', hatch='xxx',
          label='Protocol overhead (hatched)'),
]

ax_r.legend(handles=legend_items,
            loc='upper left',
            framealpha=0.96,
            fontsize=10,
            edgecolor='#cccccc',
            bbox_to_anchor=(0.02, 1.0),
            labelspacing=0.9,
            borderpad=0.7)

ax_r.set_title('Legend & Key Values',
               fontsize=11, loc='left', pad=8, fontweight='bold')

plt.savefig('Fig2_Energy.pdf', format='pdf',
            bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('Fig2_Energy.png', format='png',
            bbox_inches='tight', dpi=300, facecolor='white')
print('Figure 2 saved.')
plt.close()
