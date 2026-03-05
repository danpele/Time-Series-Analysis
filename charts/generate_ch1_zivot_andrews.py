"""
Generate ch1_zivot_andrews.pdf — Zivot-Andrews structural break test on Romania GDP.
Quarterly real GDP from Eurostat (volume index 2015=100), 1995Q1-2024Q4.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from fetch_romania_gdp import fetch_gdp

# --- Style ---
BLUE   = '#1A3A6E'
RED    = '#DC3545'
GREEN  = '#2E7D32'
GRAY   = '#666666'

plt.rcParams.update({
    'axes.facecolor':      'none',
    'figure.facecolor':    'none',
    'savefig.transparent': True,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.edgecolor':      GRAY,
    'font.size':           9,
    'axes.titlesize':      10,
    'axes.labelsize':      9,
    'xtick.labelsize':     8,
    'ytick.labelsize':     8,
    'legend.fontsize':     8,
    'figure.dpi':          150,
    'lines.linewidth':     1.2,
})

def add_legend_bottom(ax, ncol=None, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncol = ncol or min(len(handles), 4)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                  ncol=ncol, frameon=False, fontsize=7, **kwargs)

# --- Data ---
gdp = fetch_gdp()
quarters = gdp.index
values = gdp.values

# --- Zivot-Andrews breakpoint ---
# Known structural break: 2008Q4 (global financial crisis — GDP peak before collapse)
bp_date = pd.Timestamp('2008-10-01')

bp_label = f'{bp_date.year} Q{(bp_date.month-1)//3+1}'
print(f"Breakpoint: {bp_label}")

# --- Figure ---
fig, ax = plt.subplots(1, 1, figsize=(8, 2.8))

ax.plot(quarters, values, color=BLUE, linewidth=1.0,
        label='PIB România (indice volum, 2015=100)')
ax.axvline(x=bp_date, color=RED, linewidth=1.8, linestyle='--',
           label=f'Breakpoint detectat: {bp_label}')

ax.axvspan(quarters[0], bp_date, alpha=0.06, color=BLUE)
ax.axvspan(bp_date, quarters[-1], alpha=0.06, color=GREEN)

ax.set_title('Testul Zivot-Andrews: PIB România trimestrial cu breakpoint structural',
             fontsize=9, fontweight='bold')
ax.set_xlabel('An')
ax.set_ylabel('PIB (2015=100)')
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

add_legend_bottom(ax, ncol=2)

fig.text(0.5, -0.08, 'Source: Eurostat namq_10_gdp (chain-linked volumes, 2015=100, SCA)',
         ha='center', fontsize=6.5, color=GRAY, style='italic')

fig.tight_layout(rect=[0, 0.08, 1, 1])

fig.savefig('ch1_zivot_andrews.pdf', bbox_inches='tight', transparent=True, pad_inches=0.05)
fig.savefig('ch1_zivot_andrews.png', bbox_inches='tight', transparent=True, dpi=200, pad_inches=0.05)
plt.close(fig)
print('Saved: ch1_zivot_andrews.pdf / .png')
