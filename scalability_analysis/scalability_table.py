"""
Scalability Table — black and white complex version
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def lora_pdr(n, toa=0.614, t_win=60.0, n_sf=6):
    return float(np.exp(-2*n*toa/(t_win*n_sf)))

def lora_raw_pdr(n):
    return float(np.exp(-2*n*5.808/(60.0*1)))

def wifi_pdr(n, t_win=60.0):
    if n<=1: return 1.0
    t_frame=2212*8/54e6; n_slots=t_win/t_frame
    log_p=sum(np.log(max(1-k/n_slots,1e-300)) for k in range(n))
    return float(np.clip(np.exp(log_p),0,1))

def zigbee_pdr(n, n_pkts=3, be=3, coord_max=7):
    if n<=1: return 1.0
    if n<=coord_max:
        return float(np.clip((1-1.0/2**be)**(n*n_pkts-1),0,1))
    n_coord=int(np.ceil(n/coord_max))
    return float(zigbee_pdr(coord_max,n_pkts,be,coord_max)**n_coord)

N_NODES=np.arange(1,51)
PDR_lc=np.array([lora_pdr(n)     for n in N_NODES])
PDR_lu=np.array([lora_raw_pdr(n) for n in N_NODES])
PDR_wf=np.array([wifi_pdr(n)     for n in N_NODES])
PDR_zb=np.array([zigbee_pdr(n)   for n in N_NODES])

def max_n(pdr):
    idx=np.where(pdr<0.90)[0]
    return f'{int(N_NODES[idx[0]])}' if len(idx)>0 else '>50'

col_h = [
    'Protocol',
    'Channel\nAccess',
    'Payload\n(bytes)',
    'Packets\nper Round',
    'ToA\n(ms)',
    'PDR\nN=5',
    'PDR\nN=10',
    'PDR\nN=20',
    'PDR\nN=50',
    'Max N\n(PDR≥90%)',
    'FL Viable\nN=10?',
]

rows = [
    ['LoRa-FL\nCompressed',
     'ALOHA\n+6 SFs', '242B', '1', '614ms',
     f'{lora_pdr(5)*100:.1f}%',
     f'{lora_pdr(10)*100:.1f}%',
     f'{lora_pdr(20)*100:.1f}%',
     f'{lora_pdr(50)*100:.1f}%',
     max_n(PDR_lc), 'YES'],
    ['LoRa-FL\nUncompressed',
     'ALOHA\n(1 SF)', '2212B', '9', '5808ms',
     f'{lora_raw_pdr(5)*100:.1f}%',
     f'{lora_raw_pdr(10)*100:.1f}%',
     f'{lora_raw_pdr(20)*100:.1f}%',
     f'{lora_raw_pdr(50)*100:.1f}%',
     max_n(PDR_lu), 'NO'],
    ['WiFi\n802.11n',
     'CSMA-CA\nDCF', '2212B', '1', '0.33ms',
     f'{wifi_pdr(5)*100:.1f}%',
     f'{wifi_pdr(10)*100:.1f}%',
     f'{wifi_pdr(20)*100:.1f}%',
     f'{wifi_pdr(50)*100:.1f}%',
     max_n(PDR_wf)+'*', 'YES*'],
    ['Zigbee\n802.15.4',
     'Slotted\nCSMA-CA', '242B', '3', '9ms',
     f'{zigbee_pdr(5)*100:.1f}%',
     f'{zigbee_pdr(10)*100:.1f}%',
     f'{zigbee_pdr(20)*100:.1f}%',
     f'{zigbee_pdr(50)*100:.1f}%',
     max_n(PDR_zb), 'NO'],
]

try: plt.style.use('seaborn-v0_8-paper')
except: pass
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10})

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor('white')
ax.axis('off')

tbl = ax.table(cellText=rows, colLabels=col_h,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 3.2)

# Header — dark grey
for j in range(len(col_h)):
    tbl[0,j].set_facecolor('#222222')
    tbl[0,j].set_text_props(color='white', fontweight='bold')

# Rows — alternating white and light grey, black text only
for i in range(4):
    rc = '#f0f0f0' if i%2==0 else '#ffffff'
    for j in range(len(col_h)):
        tbl[i+1,j].set_facecolor(rc)
        tbl[i+1,j].set_text_props(color='black')
    # Bold the protocol column
    tbl[i+1,0].set_text_props(color='black', fontweight='bold')
    # Bold FL viable column
    tbl[i+1,10].set_text_props(color='black', fontweight='bold')

ax.set_title(
    'Scalability Analysis — PDR vs Network Size · '
    r'$T_{win}$=60s, 915MHz ISM · '
    'Bor et al. 2016 · Georgiou 2017 · Bianchi 2000 · Pollin 2008\n'
    '*WiFi PDR is high but fails on energy and range — not viable for neighbourhood deployment',
    fontsize=10, pad=10)

plt.savefig('scalability_table.pdf', format='pdf',
            bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('scalability_table.png', format='png',
            bbox_inches='tight', dpi=300, facecolor='white')
print('Saved.')
plt.close()

# ── Also generate range table ──────────────────────────────────────
