#!/usr/bin/env python3
"""Generate SARIMA model selection and diagnostics charts for unemployment"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
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
# LOAD DATA
# =============================================================================
print("Loading Unemployment data...")
import pandas_datareader as pdr

unemp = pdr.get_data_fred('UNRATE', start='2010-01-01', end='2025-01-15')
unemp_series = unemp['UNRATE']

# Train data (before COVID)
train_end = '2020-02-01'
train_data = unemp_series[unemp_series.index <= train_end]
print(f"Training data: {len(train_data)} observations")

# =============================================================================
# 1. MODEL SELECTION - AIC/BIC COMPARISON
# =============================================================================
print("\nFitting multiple SARIMA models...")

models_results = []
models_to_try = [
    ((1,1,0), (0,0,0,12), 'ARIMA(1,1,0)'),
    ((1,1,1), (0,0,0,12), 'ARIMA(1,1,1)'),
    ((2,1,1), (0,0,0,12), 'ARIMA(2,1,1)'),
    ((1,1,1), (1,0,0,12), 'SARIMA(1,1,1)(1,0,0)'),
    ((1,1,1), (0,0,1,12), 'SARIMA(1,1,1)(0,0,1)'),
    ((1,1,1), (1,0,1,12), 'SARIMA(1,1,1)(1,0,1)'),
]

for order, seasonal, name in models_to_try:
    try:
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal,
                       enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
        models_results.append({
            'name': name,
            'order': order,
            'seasonal': seasonal,
            'aic': result.aic,
            'bic': result.bic,
            'result': result
        })
        print(f"  {name:25s} AIC: {result.aic:8.2f}  BIC: {result.bic:8.2f}")
    except Exception as e:
        print(f"  {name:25s} Failed: {e}")

# Create comparison chart
fig, ax = plt.subplots(figsize=(9, 4))

names = [m['name'] for m in models_results]
aic_values = [m['aic'] for m in models_results]
bic_values = [m['bic'] for m in models_results]

x = np.arange(len(names))
width = 0.35

bars1 = ax.bar(x - width/2, aic_values, width, label='AIC', color=COLORS['blue'], alpha=0.8)
bars2 = ax.bar(x + width/2, bic_values, width, label='BIC', color=COLORS['orange'], alpha=0.8)

# Mark the best model
best_aic_idx = np.argmin(aic_values)
best_bic_idx = np.argmin(bic_values)
ax.bar(x[best_aic_idx] - width/2, aic_values[best_aic_idx], width, color=COLORS['green'], alpha=0.9)
ax.bar(x[best_bic_idx] + width/2, bic_values[best_bic_idx], width, color=COLORS['green'], alpha=0.9)

ax.set_ylabel('Information Criterion')
ax.set_title('SARIMA Model Selection: AIC and BIC Comparison', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

# Add "Best" annotation
ax.annotate('Best', xy=(x[best_aic_idx] - width/2, aic_values[best_aic_idx]),
            xytext=(x[best_aic_idx] - width/2, aic_values[best_aic_idx] - 15),
            ha='center', fontsize=9, color=COLORS['green'], fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(f'{OUTPUT_DIR}sarima_model_selection.pdf')
plt.savefig(f'{OUTPUT_DIR}sarima_model_selection.png')
plt.close()
print("\n  - sarima_model_selection.pdf")

# =============================================================================
# 2. BEST MODEL DIAGNOSTICS
# =============================================================================
print("\nCreating diagnostics for best model...")

# Use SARIMA(1,1,1)(1,0,0,12) as the best model based on AIC
best_model = models_results[best_aic_idx]
result = best_model['result']
residuals = result.resid

fig, axes = plt.subplots(2, 2, figsize=(10, 5))

# Residuals over time
axes[0, 0].plot(residuals.index, residuals.values, color=COLORS['blue'], linewidth=0.8)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].set_title('Residuals Over Time', fontweight='bold', fontsize=11)
axes[0, 0].set_ylabel('Residual')

# Histogram
axes[0, 1].hist(residuals.values, bins=25, color=COLORS['blue'], alpha=0.7, edgecolor='white')
axes[0, 1].axvline(x=0, color='black', linewidth=1)
axes[0, 1].set_title('Residual Distribution', fontweight='bold', fontsize=11)
axes[0, 1].set_xlabel('Residual')

# ACF of residuals
plot_acf(residuals.dropna().values, ax=axes[1, 0], lags=20, alpha=0.05)
axes[1, 0].set_title('ACF of Residuals', fontweight='bold', fontsize=11)
axes[1, 0].set_xlabel('Lag')

# Q-Q plot
from scipy import stats
stats.probplot(residuals.dropna().values, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality)', fontweight='bold', fontsize=11)
axes[1, 1].get_lines()[0].set_color(COLORS['blue'])
axes[1, 1].get_lines()[1].set_color(COLORS['red'])

plt.suptitle(f'{best_model["name"]} Residual Diagnostics', fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}sarima_diagnostics.pdf')
plt.savefig(f'{OUTPUT_DIR}sarima_diagnostics.png')
plt.close()
print("  - sarima_diagnostics.pdf")

# Ljung-Box test
lb_test = acorr_ljungbox(residuals.dropna(), lags=[10, 20], return_df=True)
print(f"\n  Ljung-Box Test:")
print(f"    Lag 10: p-value = {lb_test['lb_pvalue'].iloc[0]:.4f}")
print(f"    Lag 20: p-value = {lb_test['lb_pvalue'].iloc[1]:.4f}")

# =============================================================================
# 3. SARIMA IN-SAMPLE FIT
# =============================================================================
print("\nCreating in-sample fit chart...")

fitted_values = result.fittedvalues

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_data.index, train_data.values, color=COLORS['blue'], linewidth=1.5, label='Actual')
ax.plot(fitted_values.index, fitted_values.values, color=COLORS['red'], linewidth=1.5,
        linestyle='--', label='SARIMA Fitted', alpha=0.8)
ax.set_title(f'{best_model["name"]}: In-Sample Fit', fontweight='bold', fontsize=13)
ax.set_xlabel('Date')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

# Add model info
textstr = f'AIC = {result.aic:.1f}\nBIC = {result.bic:.1f}'
ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}sarima_fit.pdf')
plt.savefig(f'{OUTPUT_DIR}sarima_fit.png')
plt.close()
print("  - sarima_fit.pdf")

# =============================================================================
# 4. SARIMA FORECAST (showing it fails at COVID)
# =============================================================================
print("\nCreating SARIMA forecast chart...")

test_start = '2020-03-01'
test_data = unemp_series[unemp_series.index >= test_start]

# Forecast
forecast = result.get_forecast(steps=len(test_data))
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

forecast_mean.index = test_data.index
forecast_ci.index = test_data.index

fig, ax = plt.subplots(figsize=(10, 4))

# Plot training data
ax.plot(train_data.index, train_data.values, color=COLORS['blue'], linewidth=1.5, label='Training')

# Plot test actual
ax.plot(test_data.index, test_data.values, color=COLORS['green'], linewidth=2, label='Actual (Test)')

# Plot forecast
ax.plot(forecast_mean.index, forecast_mean.values, color=COLORS['red'], linewidth=2,
        linestyle='--', label='SARIMA Forecast')
ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                color=COLORS['red'], alpha=0.15, label='95% CI')

ax.axvline(x=pd.Timestamp(test_start), color='black', linestyle=':', alpha=0.5)
ax.set_title(f'{best_model["name"]}: Out-of-Sample Forecast vs COVID Reality', fontweight='bold', fontsize=13)
ax.set_xlabel('Date')
ax.set_ylabel('Unemployment Rate (%)')
ax.set_ylim(2, 16)

# Calculate RMSE
rmse = np.sqrt(np.mean((test_data.values - forecast_mean.values)**2))
ax.text(0.72, 0.95, f'Test RMSE = {rmse:.2f}', transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontweight='bold', color=COLORS['red'])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(f'{OUTPUT_DIR}sarima_forecast.pdf')
plt.savefig(f'{OUTPUT_DIR}sarima_forecast.png')
plt.close()
print(f"  - sarima_forecast.pdf (RMSE = {rmse:.2f})")

print("\nDone!")
