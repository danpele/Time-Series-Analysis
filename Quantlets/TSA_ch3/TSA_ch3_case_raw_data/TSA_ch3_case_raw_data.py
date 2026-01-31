"""
TSA_ch3_case_raw_data
=====================
Case Study: US Real GDP from FRED - Raw Data Visualization

Data Source: FRED (GDPC1) - Real Gross Domestic Product
Period: 1960Q1 - 2024Q3
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
gdp = pdr.get_data_fred('GDPC1', start='1960-01-01', end='2024-09-30')
gdp_data = gdp['GDPC1'].dropna()
print(f"Loaded {len(gdp_data)} observations")

# Create chart
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(gdp_data.index, gdp_data.values, color=BLUE, linewidth=1.5)
ax.set_title('US Real GDP (FRED: GDPC1)', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Billions of Chained 2017 Dollars')
ax.text(0.02, 0.98, 'Source: Federal Reserve Economic Data (FRED)',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        style='italic', color='gray')

plt.tight_layout()
plt.savefig('ch3_case_raw_data.pdf', dpi=300, bbox_inches='tight')
plt.show()
