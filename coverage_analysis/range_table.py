"""
Range / Coverage Table — complex black and white version
Okumura-Hata 915MHz Edmonton
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc

F_MHZ=915.0; H_B=30.0; H_M=1.5; SIGMA=8.0

def hata(d_m):
    d_km=np.maximum(d_m/1000,0.001)
    a_hm=((1.1*np.log10(F_MHZ)-0.7)*H_M-(1.56*np.log10(F_MHZ)-0.8))
    return (69.55+26.16*np.log10(F_MHZ)-13.82*np.log10(H_B)-a_hm
            +(44.9-6.55*np.log10(H_B))*np.log10(d_km))

def pdr_fn(margin): return 0.5*erfc(-margin/(np.sqrt(2)*SIGMA))

def coverage(tx,s):
    d_m=np.logspace(1,np.log10(15000),600)
    p=pdr_fn(tx-hata(d_m)-s)
    idx=np.where(p<0.90)[0]
    return d_m[idx[0]] if len(idx)>0 else d_m[-1]

def pl_at(tx,s,d):
    return hata(np.array([d]))[0]

def margin_at(tx,s,d):
    return tx - pl_at(tx,s,d) - s

LORA_TX=14.0; LORA_S7=-123.0; LORA_S12=-137.0
WIFI_TX=20.0; WIFI_S=-82.0
ZB_TX=4.5;   ZB_S=-97.0

D_sf7 =coverage(LORA_TX, LORA_S7)
D_sf12=coverage(LORA_TX, LORA_S12)
D_wifi=coverage(WIFI_TX, WIFI_S)
D_zb  =coverage(ZB_TX,   ZB_S)

col_h=[
    'Protocol',
    'TX Power\n(dBm)',
    'Rx Sensitivity\n(dBm)',
    'Link Budget\n(dB)',
    'Path Loss\nat Limit (dB)',
    'Shadowing\nMargin (dB)',
    'Coverage\n(PDR≥90%)',
    'vs LoRa SF7',
    'Datasheet\nReference',
]

rows=[
    ['LoRa SF7\n[proposed]',
     f'+{LORA_TX:.0f}', f'{LORA_S7:.0f}',
     f'{LORA_TX-LORA_S7:.0f}',
     f'{pl_at(LORA_TX,LORA_S7,D_sf7):.1f}',
     f'{margin_at(LORA_TX,LORA_S7,D_sf7):.1f}',
     f'{D_sf7/1000:.2f} km',
     '1.0× (reference)',
     'Semtech SX1276\nDS Table 10'],
    ['LoRa SF12\n[reference]',
     f'+{LORA_TX:.0f}', f'{LORA_S12:.0f}',
     f'{LORA_TX-LORA_S12:.0f}',
     f'{pl_at(LORA_TX,LORA_S12,D_sf12):.1f}',
     f'{margin_at(LORA_TX,LORA_S12,D_sf12):.1f}',
     f'{D_sf12/1000:.2f} km',
     f'{D_sf12/D_sf7:.1f}× more',
     'Semtech SX1276\nDS Table 10'],
    ['WiFi 802.11n\n[ESP32]',
     f'+{WIFI_TX:.0f}', f'{WIFI_S:.0f}',
     f'{WIFI_TX-WIFI_S:.0f}',
     f'{pl_at(WIFI_TX,WIFI_S,D_wifi):.1f}',
     f'{margin_at(WIFI_TX,WIFI_S,D_wifi):.1f}',
     f'{D_wifi:.0f} m',
     f'{D_sf7/D_wifi:.0f}× less',
     'IEEE 802.11n-2009\nTable 20-14'],
    ['Zigbee 802.15.4\n[CC2530]',
     f'+{ZB_TX:.1f}', f'{ZB_S:.0f}',
     f'{ZB_TX-ZB_S:.1f}',
     f'{pl_at(ZB_TX,ZB_S,D_zb):.1f}',
     f'{margin_at(ZB_TX,ZB_S,D_zb):.1f}',
     f'{D_zb:.0f} m',
     f'{D_sf7/D_zb:.0f}× less',
     'TI CC2530\nSWRS084D Table 5.3'],
]

try: plt.style.use('seaborn-v0_8-paper')
except: pass
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10})

fig, ax = plt.subplots(figsize=(17, 5))
fig.patch.set_facecolor('white')
ax.axis('off')

tbl=ax.table(cellText=rows, colLabels=col_h,
             loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 3.2)

for j in range(len(col_h)):
    tbl[0,j].set_facecolor('#222222')
    tbl[0,j].set_text_props(color='white', fontweight='bold')

for i in range(4):
    rc='#f0f0f0' if i%2==0 else '#ffffff'
    for j in range(len(col_h)):
        tbl[i+1,j].set_facecolor(rc)
        tbl[i+1,j].set_text_props(color='black')
    tbl[i+1,0].set_text_props(color='black', fontweight='bold')
    tbl[i+1,6].set_text_props(color='black', fontweight='bold')
    tbl[i+1,7].set_text_props(color='black', fontweight='bold')

ax.set_title(
    'Coverage Analysis — Link Budget Summary · Okumura-Hata 915MHz · Edmonton\n'
    r'$h_b$=30m, $h_m$=1.5m, $\sigma_{sh}$=8dB (IEEE 802.15.4-2015 channel model D) · '
    'Hata (1980) IEEE Trans. Veh. Technol.',
    fontsize=10, pad=10)

plt.savefig('range_table.pdf', format='pdf',
            bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('range_table.png', format='png',
            bbox_inches='tight', dpi=300, facecolor='white')
print('Range table saved.')
plt.close()
