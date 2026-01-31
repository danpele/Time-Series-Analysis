"""
TSA_ch3_gdp_levels
==================
Visualize US Real GDP as a non-stationary time series.

Data Source: FRED (GDPC1) - US Real GDP, Quarterly, Seasonally Adjusted
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr

# Chart style settings - Nature journal quality
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

BLUE = '#1A3A6E'

# Get real GDP data from FRED
gdp = pdr.get_data_fred('GDPC1', start='1990-01-01', end='2024-06-30')
gdp_data = gdp['GDPC1'].dropna()

# Create chart
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(gdp_data.index, gdp_data.values, color=BLUE, linewidth=1.5)
ax.set_title('US Real GDP (Quarterly, 1990-2024): A Non-Stationary Time Series',
             fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Billions of 2017 Dollars')
ax.text(0.02, 0.98, 'Source: FRED (GDPC1)', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', style='italic', color='gray')

plt.tight_layout()
plt.savefig('ch3_gdp_levels.pdf', dpi=300, bbox_inches='tight')
plt.show()
