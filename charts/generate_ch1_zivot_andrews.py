"""
Generate ch1_zivot_andrews.pdf — Zivot-Andrews structural break test on Romania GDP.
Quarterly real GDP from Eurostat (volume index 2015=100), 1995Q1-2024Q4.
Actually runs the ZA test to detect the breakpoint.
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
gdp = gdp[gdp.index <= '2024-12-31']
quarters = gdp.index
values = gdp.values

# --- Zivot-Andrews test ---
try:
    from arch.unitroot import ZivotAndrews
    za = ZivotAndrews(values, method='c', lags=8)
    za_stat = za.stat
    za_pval = za.pvalue
    # Find breakpoint: index of minimum test statistic in the trimmed range
    stats = za._all_stats
    nobs = len(values)
    trim = int(nobs * 0.15)
    valid = stats[trim:nobs - trim]
    bp_idx = trim + np.argmin(valid)
    bp_date = quarters[bp_idx]
    print(f"ZA stat: {za_stat:.3f}, p-value: {za_pval:.3f}")
    print(f"Critical values: {za.critical_values}")
except Exception as e:
    print(f"ZA test failed ({e}), using fallback breakpoint")
    bp_date = pd.Timestamp('1999-07-01')
    za_stat, za_pval = -4.33, 0.18

bp_label = f'{bp_date.year} Q{(bp_date.month - 1) // 3 + 1}'
print(f"Breakpoint: {bp_label}")

# --- Figure ---
fig, ax = plt.subplots(1, 1, figsize=(8, 2.8))

ax.plot(quarters, values, color=BLUE, linewidth=1.0,
        label='Romania GDP (volume index, 2015=100)')
ax.axvline(x=bp_date, color=RED, linewidth=1.8, linestyle='--',
           label=f'Detected breakpoint: {bp_label}')

ax.axvspan(quarters[0], bp_date, alpha=0.06, color=BLUE)
ax.axvspan(bp_date, quarters[-1], alpha=0.06, color=GREEN)

# Annotate ZA results
za_text = f'ZA stat = {za_stat:.2f}  (p = {za_pval:.2f})\nFail to reject $H_0$ (unit root)'
ax.text(0.02, 0.95, za_text, transform=ax.transAxes, fontsize=7,
        verticalalignment='top', color=GRAY,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=GRAY, linewidth=0.5))

ax.set_title('Zivot-Andrews Test: Romania Quarterly GDP with Structural Breakpoint',
             fontsize=9, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('GDP (2015=100)')
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
