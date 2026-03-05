"""
Generate ch1_case_gdp_raw.pdf — Romania GDP level (quarterly, Eurostat).
Shows the raw GDP series as non-stationary I(1) with crisis annotations.
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
    'axes.titlesize':      12,
    'axes.labelsize':      10,
    'xtick.labelsize':     9,
    'ytick.labelsize':     9,
    'legend.fontsize':     9,
    'figure.dpi':          150,
    'lines.linewidth':     1.5,
})

# --- Data ---
gdp = fetch_gdp()
quarters = gdp.index
values = gdp.values
mean_val = np.mean(values)

# --- Figure ---
fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

ax.plot(quarters, values, color=BLUE, linewidth=1.8, marker='o', markersize=2.5)
ax.fill_between(quarters, values, alpha=0.08, color=BLUE)

# Mean line
ax.axhline(y=mean_val, color=RED, linewidth=1.2, linestyle='--',
           label=f'Mean = {mean_val:.1f}')

ax.set_title(r'Romania GDP Level — Quarterly 1995–2024, Non-stationary $I(1)$',
             fontsize=12, fontweight='bold', color=RED)
ax.set_ylabel('GDP volume index (2015=100)')
ax.set_xlabel('Year')
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(0, values.max() * 1.12)

# Annotate crises
crises = [
    (pd.Timestamp('1999-01-01'), 'Crisis\n1999'),
    (pd.Timestamp('2009-01-01'), 'GFC\n2009'),
    (pd.Timestamp('2020-04-01'), 'COVID\n2020'),
]
for date, label in crises:
    idx = np.argmin(np.abs(quarters - date))
    val = values[idx]
    ax.annotate(label, xy=(quarters[idx], val),
                xytext=(0, 20), textcoords='offset points',
                fontsize=8, color=RED, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.0))

ax.legend(loc='upper left', frameon=False, fontsize=9)

fig.text(0.5, -0.03,
         'Source: Eurostat namq_10_gdp (chain-linked volumes, 2015=100, SCA)',
         ha='center', fontsize=8, color=GRAY, style='italic')

fig.tight_layout()

fig.savefig('ch1_case_gdp_raw.pdf', bbox_inches='tight', transparent=True, pad_inches=0.05)
fig.savefig('ch1_case_gdp_raw.png', bbox_inches='tight', transparent=True, dpi=200, pad_inches=0.05)
plt.close(fig)
print(f'Quarters: {len(values)}, Mean: {mean_val:.1f}')
print('Saved: ch1_case_gdp_raw.pdf / .png')
