"""
Figure 3 — Coverage Analysis
PDR vs Distance — LoRa SF7, WiFi, Zigbee
No SF12, no table, simple title
Visually richer uncertainty bands
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import erfc

np.random.seed(42)

# ── Propagation model ─────────────────────────────────────────────
F_MHZ = 915.0
H_B   = 30.0
H_M   = 1.5
SIGMA = 8.0
N_MC  = 200   # more realisations for richer bands

def hata(d_m, hb=H_B):
    d_km = np.maximum(d_m/1000, 0.001)
    a_hm = ((1.1*np.log10(F_MHZ)-0.7)*H_M
             -(1.56*np.log10(F_MHZ)-0.8))
    return (69.55+26.16*np.log10(F_MHZ)
            -13.82*np.log10(hb)-a_hm
            +(44.9-6.55*np.log10(hb))*np.log10(d_km))

def pdr(margin, sigma=SIGMA):
    return 0.5*erfc(-margin/(np.sqrt(2)*sigma))

def coverage_limit(d, p, thresh=0.90):
    idx = np.where(p < thresh)[0]
    return d[idx[0]] if len(idx)>0 else d[-1]

# Radio parameters
LORA_TX=14.0; LORA_S7=-123.0
WIFI_TX=20.0; WIFI_S=-82.0
ZB_TX=4.5;   ZB_S=-97.0

# Start from 10m to remove awkward gap at start
d_m    = np.logspace(1, np.log10(15000), 600)
PL     = hata(d_m)

m_sf7  = LORA_TX - PL - LORA_S7
m_wifi = WIFI_TX - PL - WIFI_S
m_zb   = ZB_TX   - PL - ZB_S

P_sf7  = pdr(m_sf7)
P_wifi = pdr(m_wifi)
P_zb   = pdr(m_zb)

D_sf7  = coverage_limit(d_m, P_sf7)
D_wifi = coverage_limit(d_m, P_wifi)
D_zb   = coverage_limit(d_m, P_zb)

print(f"Coverage: SF7={D_sf7:.0f}m  WiFi={D_wifi:.0f}m  Zigbee={D_zb:.0f}m")

# MC bands — richer with 200 realisations, multiple percentiles
sh = np.random.normal(0, SIGMA, (N_MC, len(d_m)))

def mc_percentiles(margin):
    mc = np.array([pdr(margin+sh[i], sigma=3) for i in range(N_MC)])
    return (np.percentile(mc, 5,  axis=0),
            np.percentile(mc, 25, axis=0),
            np.percentile(mc, 75, axis=0),
            np.percentile(mc, 95, axis=0))

sf7_p5, sf7_p25, sf7_p75, sf7_p95   = mc_percentiles(m_sf7)
wifi_p5, wifi_p25, wifi_p75, wifi_p95 = mc_percentiles(m_wifi)
zb_p5,  zb_p25,  zb_p75,  zb_p95    = mc_percentiles(m_zb)

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

C_SF7  = '#0a2e6e'
C_WIFI = '#c65911'
C_ZB   = '#1a6b3a'

# ── Layout ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(1, 2, figure=fig,
                        width_ratios=[2.4, 1.0],
                        wspace=0.08)
ax   = fig.add_subplot(gs[0])
ax_r = fig.add_subplot(gs[1])
ax_r.axis('off')

# ── Richer uncertainty bands — 2 layers each ─────────────────────
# Outer band 5-95 percentile (very faint)
ax.fill_between(d_m/1000, sf7_p5,  sf7_p95,  color=C_SF7,  alpha=0.08)
ax.fill_between(d_m/1000, wifi_p5, wifi_p95, color=C_WIFI, alpha=0.08)
ax.fill_between(d_m/1000, zb_p5,   zb_p95,   color=C_ZB,   alpha=0.08)

# Inner band 25-75 percentile (more visible)
ax.fill_between(d_m/1000, sf7_p25,  sf7_p75,  color=C_SF7,  alpha=0.18)
ax.fill_between(d_m/1000, wifi_p25, wifi_p75, color=C_WIFI, alpha=0.18)
ax.fill_between(d_m/1000, zb_p25,   zb_p75,   color=C_ZB,   alpha=0.18)

# ── PDR curves ────────────────────────────────────────────────────
ax.semilogx(d_m/1000, P_sf7,  color=C_SF7,  lw=3.0, ls='-')
ax.semilogx(d_m/1000, P_wifi, color=C_WIFI, lw=2.5, ls='-.')
ax.semilogx(d_m/1000, P_zb,   color=C_ZB,   lw=2.5, ls=':')

# ── PDR=90% threshold ─────────────────────────────────────────────
ax.axhline(0.90, color='#cc0000', lw=2.0, ls='--', zorder=5)
ax.axhspan(-0.02, 0.90, alpha=0.04, color='#cc0000')
ax.text(0.012, 0.04, 'Link failure zone (PDR < 90%)',
        fontsize=10, color='#cc0000', alpha=0.85)

# ── Coverage limit verticals ──────────────────────────────────────
for d_lim, col, lbl, yoff in [
    (D_sf7/1000,  C_SF7,  f'LoRa SF7\n{D_sf7/1000:.2f} km', 0.35),
    (D_wifi/1000, C_WIFI, f'WiFi\n{D_wifi:.0f} m',           0.50),
    (D_zb/1000,   C_ZB,   f'Zigbee\n{D_zb:.0f} m',           0.65),
]:
    ax.axvline(d_lim, color=col, lw=1.4, ls=':', alpha=0.75)
    ax.text(d_lim*1.15, yoff, lbl,
            color=col, fontsize=10, fontweight='bold',
            rotation=90, va='bottom')

# ── Axes ──────────────────────────────────────────────────────────
ax.set_xlabel('Distance from Gateway (km) — log scale', labelpad=8)
ax.set_ylabel('Packet Delivery Ratio (PDR)', labelpad=8)
ax.set_title(
    'Figure 4: Coverage Analysis — LoRa-FL Compressed vs WiFi vs Zigbee',
    pad=12, fontsize=13, fontweight='bold')

ax.set_xlim(d_m[0]/1000, d_m[-1]/1000)
ax.set_ylim(-0.02, 1.08)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

# ── Right panel ───────────────────────────────────────────────────
# Legend top
leg_items = [
    Line2D([0],[0], color=C_SF7,  lw=2.8, ls='-',
           label=f'LoRa SF7 [proposed]\n'
                 f'  $P_{{TX}}$=+{LORA_TX:.0f}dBm, '
                 f'$\\mathcal{{S}}$={LORA_S7:.0f}dBm\n'
                 f'  Coverage: {D_sf7/1000:.2f} km'),
    Line2D([0],[0], color=C_WIFI, lw=2.5, ls='-.',
           label=f'WiFi 802.11n [ESP32]\n'
                 f'  $P_{{TX}}$=+{WIFI_TX:.0f}dBm, '
                 f'$\\mathcal{{S}}$={WIFI_S:.0f}dBm\n'
                 f'  Coverage: {D_wifi:.0f} m'),
    Line2D([0],[0], color=C_ZB,   lw=2.5, ls=':',
           label=f'Zigbee 802.15.4 [CC2530]\n'
                 f'  $P_{{TX}}$=+{ZB_TX:.1f}dBm, '
                 f'$\\mathcal{{S}}$={ZB_S:.0f}dBm\n'
                 f'  Coverage: {D_zb:.0f} m'),
    Line2D([0],[0], color='#cc0000', lw=2.0, ls='--',
           label='PDR = 90% threshold'),
    Patch(facecolor='#888888', alpha=0.25,
          label='Inner band: 25th–75th pct\nOuter band: 5th–95th pct\n'
                r'($N_{MC}$=200 shadowing realisations)'),
]

ax_r.legend(handles=leg_items,
            loc='upper left',
            framealpha=0.96,
            fontsize=10,
            edgecolor='#cccccc',
            bbox_to_anchor=(0.02, 1.0),
            labelspacing=1.0,
            borderpad=0.7)

ax_r.set_title('Legend & Key Values',
               fontsize=11, loc='left', pad=8, fontweight='bold')

# Equation box — lower, no clash with legend
ax_r.text(0.04, 0.22,
    r'$L_{Hata} = 69.55 + 26.16\log f$' + '\n'
    r'$\quad - 13.82\log h_b - a(h_m)$' + '\n'
    r'$\quad + (44.9 - 6.55\log h_b)\log d$' + '\n\n'
    r'$\mathrm{PDR} = \frac{1}{2}\,\mathrm{erfc}'
    r'\!\left(\frac{-\Delta_{SNR}}{\sqrt{2}\,\sigma_{sh}}\right)$' + '\n\n'
    r'$\Delta_{SNR} = P_{TX} - L_{Hata} - \mathcal{S}$' + '\n\n'
    f'915MHz, $h_b$=30m, $h_m$=1.5m\n'
    r'$\sigma_{sh}$=8dB (IEEE 802.15.4-2015)',
    transform=ax_r.transAxes,
    fontsize=10.5, va='bottom', ha='left',
    bbox=dict(boxstyle='round,pad=0.6',
              fc='#f0f4ff', ec='#aabbdd', lw=1.0))

plt.savefig('Fig3_Range.pdf', format='pdf',
            bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('Fig3_Range.png', format='png',
            bbox_inches='tight', dpi=300, facecolor='white')
print('Figure 3 saved.')
plt.close()
