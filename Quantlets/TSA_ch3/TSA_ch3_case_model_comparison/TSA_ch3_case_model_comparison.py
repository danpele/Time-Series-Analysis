"""
TSA_ch3_case_model_comparison
=============================
ARIMA Model Selection using AIC/BIC

Data Source: FRED (GDPC1) - US Real GDP
Compares multiple ARIMA specifications to find best model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from statsmodels.tsa.arima.model import ARIMA

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

BLUE, ORANGE, RED = '#1A3A6E', '#FF8C00', '#DC3545'

# Get data
gdp = pdr.get_data_fred('GDPC1', start='1960-01-01', end='2024-09-30')
gdp_data = gdp['GDPC1'].dropna()
log_gdp = np.log(gdp_data)

# Fit multiple models
models = {
    'ARIMA(0,1,0)': (0, 1, 0),
    'ARIMA(1,1,0)': (1, 1, 0),
    'ARIMA(0,1,1)': (0, 1, 1),
    'ARIMA(1,1,1)': (1, 1, 1),
    'ARIMA(2,1,1)': (2, 1, 1),
}

results = []
for name, order in models.items():
    model = ARIMA(log_gdp, order=order)
    fit = model.fit()
    results.append({'Model': name, 'AIC': fit.aic, 'BIC': fit.bic})

df_results = pd.DataFrame(results)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_results))
width = 0.35

ax.bar(x - width/2, df_results['AIC'], width, label='AIC', color=BLUE)
ax.bar(x + width/2, df_results['BIC'], width, label='BIC', color=ORANGE)

ax.set_xlabel('Model')
ax.set_ylabel('Information Criterion')
ax.set_title('ARIMA Model Comparison: US Real GDP', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df_results['Model'], rotation=15)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Mark best model
best_idx = df_results['AIC'].idxmin()
ax.annotate('Best', xy=(best_idx - width/2, df_results.loc[best_idx, 'AIC']),
            xytext=(best_idx - width/2, df_results.loc[best_idx, 'AIC'] - 30),
            ha='center', fontsize=10, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED))

plt.tight_layout()
plt.savefig('ch3_case_model_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nBest model by AIC: {df_results.loc[best_idx, 'Model']}")
