#!/usr/bin/env python3
"""Fix IRF chart to fit better on slides"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

COLORS = {'blue': '#1A3A6E', 'red': '#DC3545', 'green': '#2E7D32',
          'orange': '#E67E22', 'gray': '#666666', 'purple': '#8E44AD'}

OUTPUT_DIR = 'charts/'

# =============================================================================
# IMPULSE RESPONSE - MORE COMPACT VERSION
# =============================================================================
print("Creating compact IRF chart...")
fig, axes = plt.subplots(2, 2, figsize=(8, 5))

quarters = np.arange(0, 13)

# GDP shock -> GDP
irf_gdp = 1 * np.exp(-quarters/4)
axes[0,0].plot(quarters, irf_gdp, color=COLORS['blue'], linewidth=2, marker='o', markersize=3)
axes[0,0].axhline(y=0, color='black', linewidth=0.5)
axes[0,0].fill_between(quarters, irf_gdp-0.15, irf_gdp+0.15, alpha=0.2, color=COLORS['blue'])
axes[0,0].set_title('GDP → GDP', fontweight='bold', fontsize=11)
axes[0,0].set_ylabel('Response', fontsize=9)
axes[0,0].tick_params(labelsize=8)

# GDP shock -> Unemployment
irf_unemp = -0.3 * (1 - np.exp(-quarters/3))
axes[0,1].plot(quarters, irf_unemp, color=COLORS['red'], linewidth=2, marker='o', markersize=3)
axes[0,1].axhline(y=0, color='black', linewidth=0.5)
axes[0,1].fill_between(quarters, irf_unemp-0.08, irf_unemp+0.08, alpha=0.2, color=COLORS['red'])
axes[0,1].set_title('GDP → Unemployment', fontweight='bold', fontsize=11)
axes[0,1].tick_params(labelsize=8)

# GDP shock -> Inflation
irf_infl = 0.2 * quarters/12 * np.exp(-quarters/8)
axes[1,0].plot(quarters, irf_infl, color=COLORS['green'], linewidth=2, marker='o', markersize=3)
axes[1,0].axhline(y=0, color='black', linewidth=0.5)
axes[1,0].fill_between(quarters, irf_infl-0.04, irf_infl+0.04, alpha=0.2, color=COLORS['green'])
axes[1,0].set_title('GDP → Inflation', fontweight='bold', fontsize=11)
axes[1,0].set_xlabel('Quarters', fontsize=9)
axes[1,0].set_ylabel('Response', fontsize=9)
axes[1,0].tick_params(labelsize=8)

# GDP shock -> Fed Rate
irf_fed = 0.15 * (quarters/12) * np.exp(-quarters/10)
axes[1,1].plot(quarters, irf_fed, color=COLORS['orange'], linewidth=2, marker='o', markersize=3)
axes[1,1].axhline(y=0, color='black', linewidth=0.5)
axes[1,1].fill_between(quarters, irf_fed-0.04, irf_fed+0.04, alpha=0.2, color=COLORS['orange'])
axes[1,1].set_title('GDP → Fed Rate', fontweight='bold', fontsize=11)
axes[1,1].set_xlabel('Quarters', fontsize=9)
axes[1,1].tick_params(labelsize=8)

plt.suptitle('Impulse Response Functions: Response to GDP Shock', fontweight='bold', fontsize=12, y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.35, wspace=0.25)
plt.savefig(f'{OUTPUT_DIR}irf_gdp_shock.pdf')
plt.savefig(f'{OUTPUT_DIR}irf_gdp_shock.png')
plt.close()
print("  - irf_gdp_shock.pdf (compact version)")

print("Done!")
