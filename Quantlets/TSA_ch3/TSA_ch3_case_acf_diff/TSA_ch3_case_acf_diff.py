"""
TSA_ch3_case_acf_diff
=====================
ACF/PACF comparison: Before and After Differencing

Data Source: FRED (GDPC1) - US Real GDP
Shows how differencing transforms non-stationary series to stationary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# ACF of levels
plot_acf(log_gdp, lags=20, ax=axes[0, 0], color=BLUE)
axes[0, 0].set_title('ACF: Log GDP (Levels)', fontweight='bold')

# PACF of levels
plot_pacf(log_gdp, lags=20, ax=axes[0, 1], color=BLUE)
axes[0, 1].set_title('PACF: Log GDP (Levels)', fontweight='bold')

# ACF of first difference
plot_acf(diff_gdp, lags=20, ax=axes[1, 0], color=GREEN)
axes[1, 0].set_title('ACF: GDP Growth (Differenced)', fontweight='bold')

# PACF of first difference
plot_pacf(diff_gdp, lags=20, ax=axes[1, 1], color=GREEN)
axes[1, 1].set_title('PACF: GDP Growth (Differenced)', fontweight='bold')

plt.tight_layout()
plt.savefig('ch3_case_acf_diff.pdf', dpi=300, bbox_inches='tight')
plt.show()
