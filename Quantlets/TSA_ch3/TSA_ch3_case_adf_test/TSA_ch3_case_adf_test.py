"""
TSA_ch3_case_adf_test
=====================
ADF Unit Root Test for US Real GDP

Data Source: FRED (GDPC1)
Tests stationarity of log GDP levels vs first differences (growth rate)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from statsmodels.tsa.stattools import adfuller

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

BLUE, GREEN = '#1A3A6E', '#2E7D32'

# Get data
gdp = pdr.get_data_fred('GDPC1', start='1960-01-01', end='2024-09-30')
gdp_data = gdp['GDPC1'].dropna()
log_gdp = np.log(gdp_data)
diff_gdp = log_gdp.diff().dropna()

# ADF tests
adf_level = adfuller(log_gdp, maxlag=8, regression='ct')
adf_diff = adfuller(diff_gdp, maxlag=8, regression='c')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original series
axes[0].plot(log_gdp.index, log_gdp.values, color=BLUE, linewidth=1.5)
axes[0].set_title('Log GDP (Levels)', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Log(GDP)')
axes[0].text(0.02, 0.98, f'ADF stat: {adf_level[0]:.2f}\np-value: {adf_level[1]:.4f}\nUnit root: Yes',
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# First difference
axes[1].plot(diff_gdp.index, diff_gdp.values * 100, color=GREEN, linewidth=1.5)
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_title('GDP Growth Rate (First Difference of Log)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Percent Change')
axes[1].text(0.02, 0.98, f'ADF stat: {adf_diff[0]:.2f}\np-value: {adf_diff[1]:.4f}\nStationary: Yes',
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('ch3_case_adf_test.pdf', dpi=300, bbox_inches='tight')
plt.show()
