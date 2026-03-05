"""
Generate ch1_log_diff_gdp.pdf — Romania GDP: log transform + differencing = economic growth.
Three panels: (1) GDP level, (2) log(GDP), (3) Δlog(GDP) = quarterly growth rate.
Real quarterly data from Eurostat.
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
    'xtick.labelsize':     7,
    'ytick.labelsize':     7,
    'legend.fontsize':     7,
    'figure.dpi':          150,
    'lines.linewidth':     1.2,
})

# --- Data ---
gdp = fetch_gdp()
quarters = gdp.index
values = gdp.values
log_gdp = np.log(values)
dlog_gdp = np.diff(log_gdp) * 100
dlog_quarters = quarters[1:]
mean_growth = np.mean(dlog_gdp)

# --- Figure ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.2))

# Panel 1: GDP level
ax1.plot(quarters, values, color=BLUE, linewidth=1.0, marker='o', markersize=1)
ax1.fill_between(quarters, values, alpha=0.08, color=BLUE)
ax1.set_title('PIB nivel', fontsize=10, fontweight='bold', color=BLUE)
ax1.set_ylabel('Indice volum (2015=100)')
ax1.set_xlabel('An')
ax1.annotate('Growing variance\n$\\Rightarrow$ I(1), non-stationary',
             xy=(quarters[10], values[10]), fontsize=6.5, color=RED,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=RED, alpha=0.9))
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel 2: log(GDP)
ax2.plot(quarters, log_gdp, color=GREEN, linewidth=1.0, marker='o', markersize=1)
ax2.fill_between(quarters, log_gdp, alpha=0.08, color=GREEN)
ax2.set_title('log(PIB)', fontsize=10, fontweight='bold', color=GREEN)
ax2.set_ylabel('log PIB')
ax2.set_xlabel('An')
ax2.set_ylim(log_gdp.min() - 0.05, log_gdp.max() + 0.15)
ax2.annotate('Stabilized variance\n(linear trend remains)',
             xy=(quarters[5], log_gdp.max() + 0.05), fontsize=6.5, color=BLUE,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=BLUE, alpha=0.9))
ax2.xaxis.set_major_locator(mdates.YearLocator(10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel 3: Δlog(GDP)
colors = [GREEN if v >= 0 else RED for v in dlog_gdp]
ax3.bar(dlog_quarters, dlog_gdp, width=70, color=colors, alpha=0.75)
ax3.axhline(y=mean_growth, color=GREEN, linewidth=1.2, linestyle='--',
            label=f'Mean = {mean_growth:.1f}%/trim.')
ax3.set_title(r'$\Delta\log$(PIB) = Creștere economică', fontsize=10, fontweight='bold', color=RED)
ax3.set_ylabel('Rată de creștere trimestrială (%)')
ax3.set_xlabel('An')
ax3.xaxis.set_major_locator(mdates.YearLocator(5))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Annotate crises
ax3.annotate('Criză\n2009', xy=(pd.Timestamp('2009-01-01'), -3),
             xytext=(pd.Timestamp('2011-06-01'), -6),
             fontsize=6, color=RED, ha='center',
             arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
ax3.annotate('COVID\n2020', xy=(pd.Timestamp('2020-04-01'), -8),
             xytext=(pd.Timestamp('2022-06-01'), -7),
             fontsize=6, color=RED, ha='center',
             arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

ax3.legend(loc='upper right', fontsize=7, frameon=False)

# Arrow annotations between panels
fig.tight_layout()
for ax_l, ax_r, label, col in [(ax1, ax2, 'ln', GREEN), (ax2, ax3, r'$\Delta$', RED)]:
    fig.text((ax_l.get_position().x1 + ax_r.get_position().x0) / 2, 0.5,
             label, ha='center', va='center', fontsize=14, fontweight='bold', color=col,
             bbox=dict(boxstyle='rarrow,pad=0.3', facecolor='white', edgecolor=col, linewidth=1.5))

fig.suptitle('Romania GDP: log transform + differencing = economic growth',
             fontsize=11, fontweight='bold', y=1.02)
fig.text(0.5, -0.06, 'Source: Eurostat namq_10_gdp (chain-linked volumes, 2015=100, SCA)',
         ha='center', fontsize=7, color=GRAY, style='italic')

fig.savefig('ch1_log_diff_gdp.pdf', bbox_inches='tight', transparent=True, pad_inches=0.05)
fig.savefig('ch1_log_diff_gdp.png', bbox_inches='tight', transparent=True, dpi=200, pad_inches=0.05)
plt.close(fig)
print(f'Mean quarterly growth: {mean_growth:.2f}%')
print('Saved: ch1_log_diff_gdp.pdf / .png')
