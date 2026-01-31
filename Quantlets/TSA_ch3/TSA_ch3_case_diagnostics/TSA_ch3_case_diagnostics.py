"""
TSA_ch3_case_diagnostics
========================
ARIMA Model Diagnostics: Residual Analysis

Data Source: FRED (GDPC1) - US Real GDP
Checks model adequacy via residual plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

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

BLUE, RED = '#1A3A6E', '#DC3545'

# Get data
gdp = pdr.get_data_fred('GDPC1', start='1960-01-01', end='2024-09-30')
gdp_data = gdp['GDPC1'].dropna()
log_gdp = np.log(gdp_data)

# Fit best model
model = ARIMA(log_gdp, order=(1, 1, 1))
fit = model.fit()
residuals = fit.resid

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residuals over time
axes[0, 0].plot(residuals.index, residuals.values, color=BLUE, linewidth=0.8)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residual')

# Histogram
axes[0, 1].hist(residuals, bins=30, color=BLUE, edgecolor='white', density=True)
axes[0, 1].set_title('Residual Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Density')

# ACF of residuals
plot_acf(residuals, lags=20, ax=axes[1, 0], color=BLUE)
axes[1, 0].set_title('ACF of Residuals', fontweight='bold')

# Q-Q plot with 45-degree reference line
res_sorted = np.sort(residuals)
norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(res_sorted)))
axes[1, 1].scatter(norm_quantiles, (res_sorted - res_sorted.mean()) / res_sorted.std(),
                   color=BLUE, s=20, alpha=0.7)
lims = [min(norm_quantiles.min(), -3), max(norm_quantiles.max(), 3)]
axes[1, 1].plot(lims, lims, color=RED, linewidth=1.5, linestyle='--')
axes[1, 1].set_xlim(lims)
axes[1, 1].set_ylim(lims)
axes[1, 1].set_xlabel('Theoretical Quantiles')
axes[1, 1].set_ylabel('Sample Quantiles')
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
axes[1, 1].set_aspect('equal')

plt.suptitle('Diagnostic Plots: ARIMA(1,1,1)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('ch3_case_diagnostics.pdf', dpi=300, bbox_inches='tight')
plt.show()
